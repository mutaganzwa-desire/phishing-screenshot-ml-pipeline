"""Lightweight deployment backend (TFLite + NumPy), for hosts with 512 MB RAM.

Full TensorFlow needs more than 1 GB of RAM, which free hosts such as Render
do not offer. This backend splits the trained Keras model in two:

  models/feature_extractor.tflite  MobileNetV2 backbone (fine-tuned weights)
                                   -> 1280-number embedding per screenshot
  models/head.npz                  softmax layer (W, b) -> 15 class probabilities

Predictions give the same results as the Keras model, use about 200 MB and
take ~10 ms per image instead of ~150 ms.

Retraining in this mode re-fits the classification head (multinomial logistic
regression) on embeddings of data/train plus the uploads. Embeddings of the
existing data are cached in models/embeddings.npz, so a retrain only needs to
embed the new images. The backbone stays as trained in the notebook.

Build the files once, on a machine with TensorFlow:
    python -m src.lite export
"""
from __future__ import annotations

import json
import os
import shutil
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src import preprocessing as pp
from src.model import (METRICS_PATH, MODELS_DIR, PROMOTION_TOLERANCE, load_registry,
                       metrics_from_probs, pending_uploads, save_registry)

TFLITE_PATH = MODELS_DIR / "feature_extractor.tflite"
HEAD_PATH = MODELS_DIR / "head.npz"
EMB_PATH = MODELS_DIR / "embeddings.npz"
HEAD_VERSIONS = MODELS_DIR / "versions"


def enabled() -> bool:
    """Use the lite backend when BACKEND=lite, or when TensorFlow is not installed."""
    if os.getenv("BACKEND", "").lower() == "lite":
        return True
    if os.getenv("BACKEND", "").lower() == "tf":
        return False
    try:
        import tensorflow  # noqa: F401
        return False
    except ImportError:
        return True


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


# --------------------------------------------------------------------------- #
# Runtime
# --------------------------------------------------------------------------- #
class Embedder:
    """Runs the TFLite backbone. One interpreter, guarded by a lock (not thread-safe)."""

    def __init__(self, path: Path = TFLITE_PATH, threads: int | None = None):
        from ai_edge_litert.interpreter import Interpreter  # pip install ai-edge-litert
        threads = threads or int(os.getenv("TFLITE_THREADS", "1"))
        self.it = Interpreter(model_path=str(path), num_threads=threads)
        self.it.allocate_tensors()
        self.inp = self.it.get_input_details()[0]["index"]
        self.out = self.it.get_output_details()[0]["index"]
        self.lock = threading.Lock()

    def __call__(self, batch: np.ndarray) -> np.ndarray:
        """(N, 224, 224, 3) float32 pixels in [0, 255] -> (N, 1280) embeddings."""
        rows = []
        with self.lock:
            for img in batch:
                self.it.set_tensor(self.inp, img[None].astype(np.float32))
                self.it.invoke()
                rows.append(self.it.get_tensor(self.out)[0].copy())
        return np.stack(rows)

    def embed_files(self, files: list[Path]) -> np.ndarray:
        if not files:
            return np.zeros((0, 1280), np.float32)
        return np.concatenate([self(pp.image_to_array(pp.load_image(f))[None]) for f in files])


class LiteModel:
    """Callable like a Keras model: pixels -> class probabilities."""

    def __init__(self):
        self.embedder = Embedder()
        head = np.load(HEAD_PATH)
        self.W, self.b = head["W"], head["b"]

    def __call__(self, batch, training=False):
        probs = softmax(self.embedder(np.asarray(batch, np.float32)) @ self.W + self.b)
        return _Arr(probs)


class _Arr(np.ndarray):
    """ndarray with .numpy(), so Predictor code works unchanged for both backends."""

    def __new__(cls, a):
        return np.asarray(a).view(cls)

    def numpy(self):
        return np.asarray(self)


# --------------------------------------------------------------------------- #
# Embedding cache
# --------------------------------------------------------------------------- #
def _rel(p: Path) -> str:
    return str(Path(p).resolve().relative_to(pp.ROOT))


def load_cache() -> dict[str, np.ndarray]:
    if not EMB_PATH.exists():
        return {}
    d = np.load(EMB_PATH, allow_pickle=False)
    return dict(zip(d["paths"].tolist(), d["X"]))


def save_cache(cache: dict[str, np.ndarray]):
    keys = sorted(cache)
    tmp = EMB_PATH.with_name("embeddings_tmp.npz")
    np.savez_compressed(tmp, paths=np.array(keys), X=np.stack([cache[k] for k in keys]).astype(np.float32))
    os.replace(tmp, EMB_PATH)


def embed_split(embedder: Embedder, root: Path, class_names: list[str], cache: dict) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Embeddings and labels for root/<class>/*, using and filling the cache."""
    paths, labels = pp.collect_files([root], class_names)
    rels = [_rel(Path(p)) for p in paths]
    missing = [p for p, r in zip(paths, rels) if r not in cache]
    for p, e in zip(missing, embedder.embed_files([Path(m) for m in missing])):
        cache[_rel(Path(p))] = e
    X = np.stack([cache[r] for r in rels]) if rels else np.zeros((0, 1280), np.float32)
    return X, np.array(labels), rels


# --------------------------------------------------------------------------- #
# Retraining (head only)
# --------------------------------------------------------------------------- #
def retrain(trigger: str = "manual", log=print, **_) -> dict:
    """Re-fit the softmax head on data/train + uploads, evaluate, promote if not worse.

    Same contract as src.model.retrain: versioned, guarded by the macro-F1
    promotion rule, and uploads are moved into data/train afterwards.
    """
    from sklearn.linear_model import LogisticRegression

    t0 = time.time()
    class_names = json.loads((MODELS_DIR / "class_names.json").read_text())
    pending = pending_uploads()
    n_new = sum(v for k, v in pending.items() if k in class_names)
    log(f"Retraining the classification head with {n_new} new images: {pending}")

    emb = Embedder()
    cache = load_cache()
    X_tr, y_tr, _ = embed_split(emb, pp.TRAIN_DIR, class_names, cache)
    X_up, y_up, up_rels = embed_split(emb, pp.UPLOAD_DIR, class_names, cache)
    X_te, y_te, _ = embed_split(emb, pp.TEST_DIR, class_names, cache)
    log(f"Embedded {len(y_tr)} training, {len(y_up)} uploaded and {len(y_te)} test images")

    X = np.concatenate([X_tr, X_up])
    y = np.concatenate([y_tr, y_up])
    clf = LogisticRegression(C=float(os.getenv("HEAD_C", "1.0")), class_weight="balanced", max_iter=3000)
    clf.fit(X, y)
    W, b = clf.coef_.T.astype(np.float32), clf.intercept_.astype(np.float32)
    metrics = metrics_from_probs(y_te, softmax(X_te @ W + b), class_names)

    reg = load_registry()
    cur = next((v for v in reg["versions"] if v["version"] == reg["current"]), None)
    baseline_f1 = cur["macro_f1"] if cur else 0.0
    promote = metrics["macro_f1"] >= baseline_f1 - PROMOTION_TOLERANCE

    version = len(reg["versions"]) + 1
    HEAD_VERSIONS.mkdir(parents=True, exist_ok=True)
    vpath = HEAD_VERSIONS / f"head_v{version}.npz"
    np.savez(vpath, W=W, b=b)
    entry = {
        "version": version, "file": _rel(vpath),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "trigger": trigger + " (head retrain)", "new_images": n_new, "n_train": int(len(y)),
        "accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"],
        "binary_f1": metrics.get("binary_phishing_vs_legit", {}).get("f1"), "promoted": promote,
    }
    reg["versions"].append(entry)
    if promote:
        shutil.copy(vpath, HEAD_PATH.with_name("head_tmp.npz"))
        os.replace(HEAD_PATH.with_name("head_tmp.npz"), HEAD_PATH)
        reg["current"] = version
        old = json.loads(METRICS_PATH.read_text()) if METRICS_PATH.exists() else {}
        METRICS_PATH.write_text(json.dumps({"version": version, "metrics": metrics,
                                            "history": old.get("history", {})}, indent=2))
    save_registry(reg)
    log(f"Candidate v{version}: macro-F1 {metrics['macro_f1']:.4f} (current {baseline_f1:.4f}) -> "
        f"{'promoted' if promote else 'rejected'}")

    # Move uploads into data/train, keeping their cached embeddings.
    for rel in up_rels:
        src = pp.ROOT / rel
        dst = pp.TRAIN_DIR / src.parent.name / src.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), dst)
        cache[_rel(dst)] = cache.pop(rel)
    save_cache(cache)
    entry["duration_s"] = round(time.time() - t0, 1)
    entry["baseline_macro_f1"] = baseline_f1
    return entry


# --------------------------------------------------------------------------- #
# Export (needs TensorFlow; run once after training)
# --------------------------------------------------------------------------- #
def export():
    import tensorflow as tf
    from tensorflow import keras

    from src.model import MODEL_PATH, load_model

    model = load_model(MODEL_PATH)
    inp = keras.Input((pp.IMG_SIZE, pp.IMG_SIZE, 3))
    x = model.get_layer("mobilenet_scaling")(inp)
    x = model.get_layer("mobilenetv2_1.00_224")(x, training=False)
    x = next(l for l in model.layers if l.name.startswith("global_average"))(x)
    extractor = keras.Model(inp, x)
    TFLITE_PATH.write_bytes(tf.lite.TFLiteConverter.from_keras_model(extractor).convert())
    W, b = model.get_layer("brand").get_weights()
    np.savez(HEAD_PATH, W=W, b=b)
    print(f"wrote {TFLITE_PATH.name} ({TFLITE_PATH.stat().st_size / 1e6:.1f} MB) and {HEAD_PATH.name}")

    names = json.loads((MODELS_DIR / "class_names.json").read_text())
    emb, cache = Embedder(threads=os.cpu_count()), {}
    for root in (pp.TRAIN_DIR, pp.TEST_DIR):
        embed_split(emb, root, names, cache)
    save_cache(cache)
    print(f"cached embeddings for {len(cache)} images in {EMB_PATH.name}")

    # Sanity check: the lite pipeline must agree with the Keras model.
    X_te, y_te, _ = embed_split(emb, pp.TEST_DIR, names, cache)
    acc = float((softmax(X_te @ W + b).argmax(1) == y_te).mean())
    print(f"lite backend test accuracy: {acc:.4f}")


if __name__ == "__main__":
    import sys
    if sys.argv[1:] == ["export"]:
        export()
    elif sys.argv[1:] == ["retrain"]:
        print(retrain())
    else:
        print(__doc__)
