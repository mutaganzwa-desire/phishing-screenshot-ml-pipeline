"""FastAPI service: prediction, bulk upload, retraining, monitoring and the UI.

Run locally:  uvicorn src.api:app --port 8000
"""
from __future__ import annotations

import io
import json
import os
import socket
import threading
import time
import traceback
import uuid
import zipfile
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from src import lite
from src import model as M
from src import preprocessing as pp
from src.prediction import get_predictor

STATIC_DIR = Path(__file__).parent / "static"
STATUS_PATH = M.MODELS_DIR / "retrain_status.json"
LOCK_PATH = M.MODELS_DIR / "retrain.lock"
AUTO_RETRAIN = os.getenv("AUTO_RETRAIN", "true").lower() == "true"
RETRAIN_THRESHOLD = int(os.getenv("RETRAIN_THRESHOLD", "30"))
RETRAIN_EPOCHS = int(os.getenv("RETRAIN_EPOCHS", "5"))
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "200"))

START_TIME = time.time()
HOSTNAME = socket.gethostname()
stats = {"requests": 0, "predictions": 0, "errors": 0}
latencies: deque[float] = deque(maxlen=1000)

app = FastAPI(title="Phish-IRIS Screenshot Classifier",
              description="Detects which brand a phishing web page imitates from its screenshot.",
              version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.middleware("http")
async def track(request: Request, call_next):
    t0 = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        stats["errors"] += 1
        raise
    ms = (time.perf_counter() - t0) * 1000
    stats["requests"] += 1
    if request.url.path == "/predict":
        latencies.append(ms)
    response.headers["X-Served-By"] = HOSTNAME
    response.headers["X-Process-Time-ms"] = f"{ms:.1f}"
    return response


@app.on_event("startup")
def _load():
    get_predictor()  # load the model once, before the first request
    # A lock left behind by a container that died mid-retrain would block forever.
    if LOCK_PATH.exists() and time.time() - LOCK_PATH.stat().st_mtime > 3 * 3600:
        LOCK_PATH.unlink(missing_ok=True)


# --------------------------------------------------------------------------- #
# Retrain status shared by all containers through the models/ volume
# --------------------------------------------------------------------------- #
def read_status() -> dict:
    if STATUS_PATH.exists():
        try:
            return json.loads(STATUS_PATH.read_text())
        except json.JSONDecodeError:
            pass
    return {"state": "idle"}


def write_status(**kw):
    tmp = STATUS_PATH.with_suffix(f".{uuid.uuid4().hex}.tmp")
    tmp.write_text(json.dumps(kw, indent=2))
    os.replace(tmp, STATUS_PATH)


def _acquire_lock() -> bool:
    try:
        fd = os.open(LOCK_PATH, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, HOSTNAME.encode())
        os.close(fd)
        return True
    except FileExistsError:
        return False


def run_retrain(trigger: str):
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    log: list[str] = []

    def _log(msg):
        log.append(msg)
        write_status(state="running", trigger=trigger, started_at=started, host=HOSTNAME, log=log)

    try:
        _log(f"Retraining started ({trigger}) on {HOSTNAME}")
        # Full fine-tuning with TensorFlow, or a head-only retrain on small hosts.
        retrain_fn = lite.retrain if get_predictor().backend == "lite" else M.retrain
        entry = retrain_fn(trigger=trigger, epochs=RETRAIN_EPOCHS, log=_log)
        get_predictor().reload_if_changed()
        write_status(state="finished", trigger=trigger, started_at=started, host=HOSTNAME,
                     finished_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                     result=entry, log=log)
    except Exception as exc:
        traceback.print_exc()
        write_status(state="failed", trigger=trigger, started_at=started, host=HOSTNAME,
                     error=str(exc), log=log)
    finally:
        LOCK_PATH.unlink(missing_ok=True)


def start_retrain(trigger: str, background: BackgroundTasks | None = None) -> bool:
    if not _acquire_lock():
        return False
    write_status(state="queued", trigger=trigger, host=HOSTNAME)
    threading.Thread(target=run_retrain, args=(trigger,), daemon=True).start()
    return True


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #
@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health():
    """Model up-time and live serving statistics."""
    p = get_predictor()
    p.reload_if_changed()  # pick up a model promoted by another container
    lat = sorted(latencies)
    pct = (lambda q: round(lat[min(len(lat) - 1, int(q * len(lat)))], 1)) if lat else (lambda q: None)
    return {
        "status": "ok",
        "host": HOSTNAME,
        "started_at": datetime.fromtimestamp(START_TIME, timezone.utc).isoformat(timespec="seconds"),
        "uptime_seconds": round(time.time() - START_TIME, 1),
        "model_version": p.version,
        "backend": p.backend,
        "model_loaded": p.model is not None,
        "num_classes": len(p.class_names),
        "requests": stats["requests"],
        "predictions": stats["predictions"],
        "errors": stats["errors"],
        "latency_ms": {"p50": pct(0.5), "p95": pct(0.95), "mean":
                       round(sum(lat) / len(lat), 1) if lat else None},
        "retrain": {"state": read_status().get("state"), "pending_uploads": sum(M.pending_uploads().values()),
                    "auto_retrain": AUTO_RETRAIN, "threshold": RETRAIN_THRESHOLD},
    }


@app.get("/classes")
def classes():
    names = get_predictor().class_names
    return [{"label": c, "brand": pp.CLASS_LABELS.get(c, c)} for c in names]


@app.post("/predict")
async def predict(file: UploadFile = File(..., description="Screenshot of a web page")):
    data = await file.read()
    if not data:
        raise HTTPException(400, "Empty file")
    try:
        result = get_predictor().predict(data)
    except Exception as exc:
        raise HTTPException(400, f"Could not read image: {exc}")
    stats["predictions"] += 1
    result["filename"] = file.filename
    return result


@app.post("/upload")
async def upload(files: list[UploadFile] = File(...), label: str | None = Form(None)):
    """Bulk upload labelled screenshots for retraining.

    Either send several images together with one `label` (a class name such as
    'paypal'), or send a .zip whose folders are class names
    (paypal/img1.png, other/img2.jpg, ...).
    """
    names = set(get_predictor().class_names)
    if label is not None and label not in names:
        raise HTTPException(400, f"Unknown label '{label}'. Use one of: {sorted(names)}")
    saved, skipped = {}, []
    total_bytes = 0

    def _save(raw: bytes, cls: str, fname: str):
        try:
            img = pp.load_image(raw)
        except Exception:
            skipped.append(fname)
            return
        out = pp.UPLOAD_DIR / cls
        out.mkdir(parents=True, exist_ok=True)
        img.save(out / f"upload_{uuid.uuid4().hex[:10]}.jpg", "JPEG", quality=90)
        saved[cls] = saved.get(cls, 0) + 1

    for f in files:
        raw = await f.read()
        total_bytes += len(raw)
        if total_bytes > MAX_UPLOAD_MB * 1024 * 1024:
            raise HTTPException(413, f"Upload larger than {MAX_UPLOAD_MB} MB")
        if f.filename and f.filename.lower().endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(raw)) as zf:
                for info in zf.infolist():
                    parts = [p for p in Path(info.filename).parts if p not in ("", ".")]
                    if info.is_dir() or Path(info.filename).suffix.lower() not in pp.ALLOWED_EXT \
                            or any(p.startswith("__MACOSX") for p in parts):
                        continue
                    cls = label or (parts[-2].lower() if len(parts) >= 2 else None)
                    if cls not in names:
                        skipped.append(info.filename)
                        continue
                    _save(zf.read(info), cls, info.filename)
        else:
            if label is None:
                raise HTTPException(400, "Provide a 'label' for image uploads, or upload a zip of class folders")
            _save(raw, label, f.filename or "image")

    pending = M.pending_uploads()
    n_pending = sum(pending.values())
    auto = False
    if AUTO_RETRAIN and n_pending >= RETRAIN_THRESHOLD and read_status().get("state") not in ("running", "queued"):
        auto = start_retrain("auto: upload threshold reached")
    return {"saved": saved, "skipped": skipped, "pending_uploads": pending,
            "pending_total": n_pending, "auto_retrain_started": auto,
            "threshold": RETRAIN_THRESHOLD}


@app.post("/retrain")
def retrain():
    """Button-triggered retraining on the current training data plus uploads."""
    if not start_retrain("manual"):
        raise HTTPException(409, "A retraining job is already running")
    return {"started": True, "pending_uploads": M.pending_uploads()}


@app.get("/retrain/status")
def retrain_status():
    return read_status()


@app.get("/metrics")
def metrics():
    """Evaluation metrics of the production model and the version history."""
    m = json.loads(M.METRICS_PATH.read_text()) if M.METRICS_PATH.exists() else {}
    return {"current": m, "registry": M.load_registry()}


@app.get("/insights")
def insights():
    """Dataset statistics for the visualizations (computed by preprocessing)."""
    path = pp.REPORTS_DIR / "dataset_stats.json"
    if not path.exists():
        pp.save_statistics(pp.dataset_statistics())
    return json.loads(path.read_text())


@app.get("/sample/{label}")
def sample(label: str, i: int = 0):
    if label not in get_predictor().class_names:
        raise HTTPException(404, "Unknown class")
    files = pp.list_images(pp.TRAIN_DIR / label)
    if not files:
        raise HTTPException(404, "No images")
    return FileResponse(files[i % len(files)])


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
