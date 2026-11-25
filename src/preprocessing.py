"""Data acquisition and preprocessing for the Phish-IRIS screenshot classifier.

Pipeline stages covered here:
  1. Data acquisition  - download the Phish-IRIS archive from Mendeley Data.
  2. Data processing   - convert every screenshot to RGB, resize it to a fixed
                         square, and save it under data/<split>/<class>/.
  3. Input pipelines   - tf.data datasets used for training, evaluation and
                         retraining.
  4. Feature extraction - simple, interpretable image features (colour,
                         brightness, edge density, size) for the visualizations.
"""
from __future__ import annotations

import io
import json
import re
import shutil
import zipfile
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageFilter, ImageOps

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"
UPLOAD_DIR = DATA_DIR / "uploads"
REPORTS_DIR = ROOT / "reports"

IMG_SIZE = 224
SEED = 42
ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DATASET_URL = "https://data.mendeley.com/public-api/zip/tk4fkswtj5/download/1"

# Human-readable names for the 15 Phish-IRIS classes.
CLASS_LABELS = {
    "adobe": "Adobe", "alibaba": "Alibaba", "amazon": "Amazon", "apple": "Apple",
    "boa": "Bank of America", "chase": "Chase", "dhl": "DHL", "dropbox": "Dropbox",
    "facebook": "Facebook", "linkedin": "LinkedIn", "microsoft": "Microsoft",
    "other": "Other / legitimate", "paypal": "PayPal", "wellsfargo": "Wells Fargo",
    "yahoo": "Yahoo",
}


# --------------------------------------------------------------------------- #
# 1. Acquisition
# --------------------------------------------------------------------------- #
def download_dataset(dest: Path, url: str = DATASET_URL) -> Path:
    """Download the Phish-IRIS zip archive (~800 MB) if it is not already there."""
    import requests

    dest.mkdir(parents=True, exist_ok=True)
    archive = dest / "phishiris.zip"
    if archive.exists() and archive.stat().st_size > 0:
        return archive
    with requests.get(url, stream=True, timeout=600) as r:
        r.raise_for_status()
        with open(archive, "wb") as fh:
            for chunk in r.iter_content(1 << 20):
                fh.write(chunk)
    return archive


def find_split_dirs(raw_root: Path) -> dict[str, Path]:
    """Locate the original 'train' and 'val' folders inside the extracted archive.

    The archive nests them inconsistently (train is one level deeper than val),
    so we search instead of hard-coding paths.
    """
    found = {}
    for split in ("train", "val"):
        candidates = sorted(p for p in raw_root.rglob(split) if p.is_dir())
        if not candidates:
            raise FileNotFoundError(f"No '{split}' folder found under {raw_root}")
        found[split] = candidates[0]
    return found


# --------------------------------------------------------------------------- #
# 2. Processing
# --------------------------------------------------------------------------- #
def load_image(source, size: int = IMG_SIZE) -> Image.Image:
    """Open an image from a path or raw bytes and return a size x size RGB image.

    Screenshots arrive as PNG (often RGBA) or JPEG at many resolutions.
    Transparent pixels are composited on white (how a browser renders them),
    EXIF orientation is honoured, and the image is resized to the network input.
    """
    if isinstance(source, (bytes, bytearray)):
        img = Image.open(io.BytesIO(source))
    else:
        img = Image.open(source)
    img = ImageOps.exif_transpose(img)
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        background = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(background, img)
    img = img.convert("RGB")
    return img.resize((size, size), Image.Resampling.BILINEAR)


def image_to_array(img: Image.Image) -> np.ndarray:
    """PIL image -> float32 array in [0, 255], shape (H, W, 3).

    Scaling to [-1, 1] for MobileNetV2 happens inside the model, so the same
    raw array works for training, the API and the notebook.
    """
    return np.asarray(img, dtype=np.float32)


def list_images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in ALLOWED_EXT)


def file_hash(path: Path) -> str:
    import hashlib
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def audit_duplicates(splits: dict[str, Path]) -> dict:
    """Find exact duplicate files within and across the original splits.

    Phish-IRIS ships ~400 byte-identical copies, and many test ('val') images
    also appear in train. Left in, they inflate test scores (leakage), so we
    keep one copy per hash, prefer train, and drop hashes with two labels.
    """
    seen: dict[str, tuple[str, str, Path]] = {}
    labels: dict[str, set] = {}
    keep: dict[str, list[tuple[str, Path]]] = {"train": [], "val": []}
    dup_within, leak = 0, 0
    for split in ("train", "val"):
        for class_dir in sorted(p for p in splits[split].iterdir() if p.is_dir()):
            for f in list_images(class_dir):
                h = file_hash(f)
                labels.setdefault(h, set()).add(class_dir.name)
                if h in seen:
                    if seen[h][0] == split:
                        dup_within += 1
                    else:
                        leak += 1
                    continue
                seen[h] = (split, class_dir.name, f)
    conflicting = {h for h, ls in labels.items() if len(ls) > 1}
    for h, (split, cls, f) in seen.items():
        if h not in conflicting:
            keep[split].append((cls, f))
    return {
        "files_total": sum(1 for _ in seen) + dup_within + leak,
        "duplicates_within_split": dup_within,
        "test_images_also_in_train": leak,
        "conflicting_label_hashes": len(conflicting),
        "unique_images": sum(len(v) for v in keep.values()),
        "_keep": keep,
    }


def prepare_split(items: list[tuple[str, Path]], dst: Path, size: int = IMG_SIZE) -> dict[str, int]:
    """Clean and resize one split: (class, file) pairs -> dst/<class>/*.jpg."""
    counts: dict[str, int] = {}
    for cls, f in items:
        out_dir = dst / cls
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            img = load_image(f, size)
        except Exception as exc:  # corrupt files are skipped, not fatal
            print(f"skip {f}: {exc}")
            continue
        # Keep the original extension in the name: the archive has pairs such
        # as "boa (1).png" and "boa (1).jpg" that would otherwise overwrite.
        safe = re.sub(r"[^A-Za-z0-9]+", "_", f.stem).strip("_")
        img.save(out_dir / f"{safe}_{f.suffix.lstrip('.').lower()}.jpg", "JPEG", quality=90)
        counts[cls] = counts.get(cls, 0) + 1
    return dict(sorted(counts.items()))


def dhash(path: Path, size: int = 8) -> int:
    """Difference hash: a 64-bit fingerprint that survives resizing/re-encoding."""
    img = Image.open(path).convert("L").resize((size + 1, size), Image.Resampling.LANCZOS)
    px = np.asarray(img, dtype=np.int16)
    bits = (px[:, 1:] > px[:, :-1]).flatten()
    return int("".join("1" if x else "0" for x in bits), 2)


def near_duplicate_groups(files: list[Path], max_distance: int = 6) -> list[int]:
    """Group near-identical screenshots (same phishing kit captured twice).

    Two images whose dHashes differ in <= max_distance of 64 bits are joined
    (union-find), so a group id per file is returned. The train/test split is
    made by group, so no near-copy of a test image is ever trained on -
    the image equivalent of the hostname-disjoint split in the URL project.
    """
    hashes = np.array([dhash(f) for f in files], dtype=np.uint64)
    bits = np.unpackbits(hashes.view(np.uint8).reshape(-1, 8), axis=1)
    parent = list(range(len(files)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(files)):
        dist = (bits[i + 1:] != bits[i]).sum(axis=1)
        for j in np.nonzero(dist <= max_distance)[0] + i + 1:
            ri, rj = find(i), find(int(j))
            if ri != rj:
                parent[rj] = ri
    return [find(i) for i in range(len(files))]


def prepare_dataset(raw_root: Path, size: int = IMG_SIZE, test_frac: float = 0.3) -> dict:
    """Build deduplicated, group-disjoint data/train and data/test folders.

    1. drop exact duplicates and the one image that carries two labels;
    2. pool the original train/val folders (the original val reused 204 train
       images and held 73% 'other', leaving few brand test images);
    3. group near-duplicates and split ~70/30, stratified by class and
       disjoint by group.
    """
    from sklearn.model_selection import StratifiedGroupKFold

    splits = find_split_dirs(raw_root)
    audit = audit_duplicates(splits)
    keep = audit.pop("_keep")
    items = keep["train"] + keep["val"]
    files = [f for _, f in items]
    y = [c for c, _ in items]
    groups = near_duplicate_groups(files)
    audit["near_duplicate_groups"] = len(set(groups))
    audit["images_in_multi_image_groups"] = int(sum(np.bincount(groups)[groups] > 1))

    n_splits = round(1 / test_frac)
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    train_idx, test_idx = next(sgkf.split(files, y, groups))

    for d in (TRAIN_DIR, TEST_DIR):
        if d.exists():
            shutil.rmtree(d)
    audit["train"] = prepare_split([items[i] for i in train_idx], TRAIN_DIR, size)
    audit["test"] = prepare_split([items[i] for i in test_idx], TEST_DIR, size)
    audit["n_train"], audit["n_test"] = len(train_idx), len(test_idx)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "data_audit.json").write_text(json.dumps(audit, indent=2))
    return audit


def acquire_and_prepare(work_dir: Path) -> dict:
    """End-to-end acquisition: download, extract, clean and resize."""
    archive = download_dataset(work_dir)
    extract_dir = work_dir / "extracted"
    if not extract_dir.exists():
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(extract_dir)
    return prepare_dataset(extract_dir)


# --------------------------------------------------------------------------- #
# 3. tf.data input pipelines
# --------------------------------------------------------------------------- #
def get_class_names(folder: Path = TRAIN_DIR) -> list[str]:
    return sorted(p.name for p in folder.iterdir() if p.is_dir())


def make_dataset(paths: list[str], labels: list[int], batch_size: int = 32,
                 shuffle: bool = False, size: int = IMG_SIZE):
    """Create a batched tf.data pipeline from file paths and integer labels."""
    import tensorflow as tf

    def _load(path, label):
        raw = tf.io.read_file(path)
        img = tf.io.decode_image(raw, channels=3, expand_animations=False)
        img = tf.image.resize(img, (size, size))
        img.set_shape((size, size, 3))
        return tf.cast(img, tf.float32), label

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(len(paths), seed=SEED, reshuffle_each_iteration=True)
    return ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(tf.data.AUTOTUNE)


def collect_files(folders: Iterable[Path], class_names: list[str]) -> tuple[list[str], list[int]]:
    """Collect (path, label) pairs from one or more <root>/<class>/ folders.

    Folders whose class is not in `class_names` are ignored.
    """
    index = {c: i for i, c in enumerate(class_names)}
    paths, labels = [], []
    for root in folders:
        if not root.exists():
            continue
        for class_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            if class_dir.name not in index:
                continue
            for f in list_images(class_dir):
                paths.append(str(f))
                labels.append(index[class_dir.name])
    return paths, labels


# --------------------------------------------------------------------------- #
# 4. Interpretable features for visualizations
# --------------------------------------------------------------------------- #
def image_features(path: Path) -> dict:
    """Simple visual features that describe a screenshot.

    - mean R/G/B and saturation: brand colour palette (e.g. Facebook blue)
    - brightness: light login forms vs dark/photo-heavy pages
    - edge density: visual clutter (text, images, widgets) on the page
    - whitespace ratio: share of near-white pixels (minimal login pages)
    """
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img, dtype=np.float32)
    hsv = np.asarray(img.convert("HSV"), dtype=np.float32)
    edges = np.asarray(img.convert("L").filter(ImageFilter.FIND_EDGES), dtype=np.float32)
    return {
        "mean_r": float(arr[..., 0].mean()),
        "mean_g": float(arr[..., 1].mean()),
        "mean_b": float(arr[..., 2].mean()),
        "brightness": float(arr.mean()),
        "saturation": float(hsv[..., 1].mean()),
        "edge_density": float((edges > 40).mean()),
        "whitespace_ratio": float((arr.min(axis=-1) > 235).mean()),
    }


def dataset_statistics(train_dir: Path = TRAIN_DIR, test_dir: Path = TEST_DIR) -> dict:
    """Per-class counts and mean visual features, saved for the dashboard."""
    stats = {"classes": {}, "label_names": CLASS_LABELS}
    for class_name in get_class_names(train_dir):
        feats = [image_features(p) for p in list_images(train_dir / class_name)]
        mean = {k: round(float(np.mean([f[k] for f in feats])), 4) for k in feats[0]} if feats else {}
        stats["classes"][class_name] = {
            "train": len(feats),
            "test": len(list_images(test_dir / class_name)) if (test_dir / class_name).exists() else 0,
            **mean,
        }
    return stats


def save_statistics(stats: dict, path: Path = REPORTS_DIR / "dataset_stats.json") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stats, indent=2))
    return path


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Download and prepare Phish-IRIS")
    ap.add_argument("--raw", type=Path, help="already-extracted archive folder")
    ap.add_argument("--work", type=Path, default=ROOT / ".cache" / "phishiris")
    args = ap.parse_args()
    summary = prepare_dataset(args.raw) if args.raw else acquire_and_prepare(args.work)
    print(json.dumps(summary, indent=2))
    save_statistics(dataset_statistics())
