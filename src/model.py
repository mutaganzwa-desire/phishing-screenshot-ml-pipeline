"""Model creation, training, evaluation and retraining.

The classifier is MobileNetV2 (ImageNet weights) with a new softmax head for
the 15 Phish-IRIS classes. Training runs in two phases:
  1. feature extraction - the backbone is frozen and only the head learns;
  2. fine-tuning        - the top backbone layers are unfrozen at a low LR.

Retraining warm-starts from the current production model, adds the newly
uploaded images to the training set, evaluates the candidate on the fixed
test set and only promotes it if it is not worse than the current model.
Every model version is recorded in models/registry.json.
"""
from __future__ import annotations

import json
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src import preprocessing as pp

MODELS_DIR = pp.ROOT / "models"
MODEL_NAME = "phish_iris_mobilenetv2"
MODEL_PATH = MODELS_DIR / f"{MODEL_NAME}.h5"
VERSIONS_DIR = MODELS_DIR / "versions"
REGISTRY_PATH = MODELS_DIR / "registry.json"
METRICS_PATH = pp.REPORTS_DIR / "metrics.json"

# A candidate model must reach at least (current macro-F1 - tolerance) to be promoted.
PROMOTION_TOLERANCE = float(os.getenv("PROMOTION_TOLERANCE", "0.01"))


# --------------------------------------------------------------------------- #
# Model creation
# --------------------------------------------------------------------------- #
def build_model(num_classes: int, img_size: int = pp.IMG_SIZE, dropout: float = 0.3):
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    inputs = keras.Input(shape=(img_size, img_size, 3), name="image")
    # Light augmentation only: screenshots are never mirrored or rotated in the
    # wild, so flips would teach the model unrealistic logos.
    aug = keras.Sequential([
        layers.RandomTranslation(0.05, 0.05, fill_mode="nearest"),
        layers.RandomZoom(0.1, fill_mode="nearest"),
        layers.RandomContrast(0.1),
    ], name="augmentation")
    x = aug(inputs)
    x = layers.Rescaling(1.0 / 127.5, offset=-1.0, name="mobilenet_scaling")(x)
    base = keras.applications.MobileNetV2(include_top=False, weights="imagenet",
                                          input_shape=(img_size, img_size, 3))
    base.trainable = False
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="brand",
                           kernel_regularizer=keras.regularizers.l2(1e-4))(x)
    model = keras.Model(inputs, outputs, name=MODEL_NAME)
    return model


def _backbone(model):
    return next(l for l in model.layers if l.name.startswith("mobilenetv2"))


def compile_model(model, lr: float):
    from tensorflow import keras
    model.compile(
        optimizer=keras.optimizers.Adam(lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy", keras.metrics.SparseTopKCategoricalAccuracy(k=3, name="top3_acc")],
    )


def set_finetune(model, n_layers: int = 50):
    """Unfreeze the last `n_layers` of the backbone (BatchNorm stays frozen)."""
    from tensorflow.keras import layers
    base = _backbone(model)
    base.trainable = True
    for layer in base.layers[:-n_layers]:
        layer.trainable = False
    for layer in base.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False


def class_weights(labels: list[int], num_classes: int, power: float = 0.5) -> dict[int, float]:
    """Softened inverse-frequency class weights.

    'other' is ~56% of the data and Amazon <1%. Fully balanced weights
    (power=1) make the model call many legitimate pages a brand; power=0.5
    (inverse square root) still lifts the rare brands without that cost.
    """
    counts = np.bincount(labels, minlength=num_classes).astype(float)
    counts[counts == 0] = 1.0
    w = (counts.max() / counts) ** power
    w = w / (w * counts).sum() * len(labels)  # mean weight per sample = 1
    return {i: float(v) for i, v in enumerate(w)}


def stratified_split(paths, labels, val_frac=0.15, seed=pp.SEED):
    from sklearn.model_selection import train_test_split
    return train_test_split(paths, labels, test_size=val_frac, stratify=labels, random_state=seed)


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def train(train_dirs=(pp.TRAIN_DIR,), head_epochs: int = 25, finetune_epochs: int = 15,
          batch_size: int = 32, model=None, class_names=None, head_lr: float = 1e-3,
          finetune_lr: float = 2e-5, verbose: int = 2):
    """Train (or continue training) a model. Returns (model, history, class_names)."""
    from tensorflow import keras

    class_names = class_names or pp.get_class_names(pp.TRAIN_DIR)
    paths, labels = pp.collect_files(train_dirs, class_names)
    tr_p, va_p, tr_y, va_y = stratified_split(paths, labels)
    train_ds = pp.make_dataset(tr_p, tr_y, batch_size, shuffle=True)
    val_ds = pp.make_dataset(va_p, va_y, batch_size)
    cw = class_weights(tr_y, len(class_names))

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-7),
    ]
    history: dict[str, list] = {}

    def _merge(h, phase):
        for k, v in h.history.items():
            history.setdefault(k, []).extend(float(x) for x in v)
        history.setdefault("phase", []).extend([phase] * len(h.history["loss"]))

    if model is None:
        model = build_model(len(class_names))
        compile_model(model, head_lr)
        h = model.fit(train_ds, validation_data=val_ds, epochs=head_epochs,
                      class_weight=cw, callbacks=callbacks, verbose=verbose)
        _merge(h, "head")

    if finetune_epochs > 0:
        set_finetune(model)
        compile_model(model, finetune_lr)
        h = model.fit(train_ds, validation_data=val_ds, epochs=finetune_epochs,
                      class_weight=cw, callbacks=callbacks, verbose=verbose)
        _merge(h, "finetune")

    history["n_train"] = len(tr_p)
    history["n_val"] = len(va_p)
    return model, history, class_names


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def predict_dir(model, folder: Path, class_names: list[str], batch_size: int = 64):
    paths, labels = pp.collect_files([folder], class_names)
    ds = pp.make_dataset(paths, labels, batch_size)
    probs = model.predict(ds, verbose=0)
    return np.array(labels), probs, paths


def evaluate(model, class_names: list[str], test_dir: Path = pp.TEST_DIR) -> dict:
    """All metrics used in the notebook and stored for the dashboard.

    Multi-class: accuracy, top-3 accuracy, macro / weighted precision, recall,
    F1, Cohen's kappa, log loss, per-class report and confusion matrix.
    Binary view ('phishing' = any brand, 'legitimate' = other): the security
    question the classifier answers in practice, with ROC-AUC on P(not other).
    """
    from sklearn import metrics as m

    y, probs, _ = predict_dir(model, test_dir, class_names)
    pred = probs.argmax(1)
    labels_idx = list(range(len(class_names)))
    p, r, f, _ = m.precision_recall_fscore_support(y, pred, average="macro", zero_division=0)
    pw, rw, fw, _ = m.precision_recall_fscore_support(y, pred, average="weighted", zero_division=0)
    top3 = float(np.mean([yy in np.argsort(pp_)[-3:] for yy, pp_ in zip(y, probs)]))

    other = class_names.index("other") if "other" in class_names else None
    binary = {}
    if other is not None:
        yb = (y != other).astype(int)
        pb = (pred != other).astype(int)
        score = 1.0 - probs[:, other]
        tn, fp, fn, tp = m.confusion_matrix(yb, pb, labels=[0, 1]).ravel()
        binary = {
            "accuracy": float(m.accuracy_score(yb, pb)),
            "precision": float(m.precision_score(yb, pb, zero_division=0)),
            "recall": float(m.recall_score(yb, pb, zero_division=0)),
            "f1": float(m.f1_score(yb, pb, zero_division=0)),
            "roc_auc": float(m.roc_auc_score(yb, score)),
            "average_precision": float(m.average_precision_score(yb, score)),
            "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        }

    return {
        "n_test": int(len(y)),
        "accuracy": float(m.accuracy_score(y, pred)),
        "balanced_accuracy": float(m.balanced_accuracy_score(y, pred)),
        "top3_accuracy": top3,
        "macro_precision": float(p), "macro_recall": float(r), "macro_f1": float(f),
        "weighted_precision": float(pw), "weighted_recall": float(rw), "weighted_f1": float(fw),
        "cohen_kappa": float(m.cohen_kappa_score(y, pred)),
        "log_loss": float(m.log_loss(y, np.clip(probs, 1e-7, 1), labels=labels_idx)),
        "per_class": m.classification_report(y, pred, labels=labels_idx, target_names=class_names,
                                              output_dict=True, zero_division=0),
        "confusion_matrix": m.confusion_matrix(y, pred, labels=labels_idx).tolist(),
        "class_names": class_names,
        "binary_phishing_vs_legit": binary,
    }


# --------------------------------------------------------------------------- #
# Saving, registry and versioning
# --------------------------------------------------------------------------- #
def load_registry() -> dict:
    if REGISTRY_PATH.exists():
        return json.loads(REGISTRY_PATH.read_text())
    return {"current": None, "versions": []}


def save_registry(reg: dict):
    tmp = REGISTRY_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(reg, indent=2))
    os.replace(tmp, REGISTRY_PATH)  # atomic, so other containers never read half a file


def load_model(path: Path = MODEL_PATH):
    from tensorflow import keras
    return keras.models.load_model(path, compile=False)


def save_model(model, class_names, history, metrics, trigger: str, n_new: int = 0,
               promote: bool = True) -> dict:
    """Save a version, record it in the registry and optionally make it current."""
    VERSIONS_DIR.mkdir(parents=True, exist_ok=True)
    reg = load_registry()
    version = len(reg["versions"]) + 1
    vpath = VERSIONS_DIR / f"{MODEL_NAME}_v{version}.h5"
    model.save(vpath, include_optimizer=False)  # ~9 MB instead of ~24 MB
    entry = {
        "version": version,
        "file": str(vpath.relative_to(pp.ROOT)),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "trigger": trigger,
        "new_images": n_new,
        "n_train": history.get("n_train"),
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "binary_f1": metrics.get("binary_phishing_vs_legit", {}).get("f1"),
        "promoted": promote,
    }
    reg["versions"].append(entry)
    if promote:
        tmp = MODEL_PATH.with_suffix(".tmp.h5")
        shutil.copy(vpath, tmp)
        os.replace(tmp, MODEL_PATH)
        (MODELS_DIR / "class_names.json").write_text(json.dumps(class_names))
        reg["current"] = version
        pp.REPORTS_DIR.mkdir(exist_ok=True)
        METRICS_PATH.write_text(json.dumps({"version": version, "metrics": metrics,
                                            "history": history}, indent=2))
    save_registry(reg)
    return entry


# --------------------------------------------------------------------------- #
# Retraining
# --------------------------------------------------------------------------- #
def pending_uploads() -> dict[str, int]:
    """Count labelled images waiting in data/uploads/<class>/."""
    out = {}
    if pp.UPLOAD_DIR.exists():
        for d in sorted(p for p in pp.UPLOAD_DIR.iterdir() if p.is_dir()):
            n = len(pp.list_images(d))
            if n:
                out[d.name] = n
    return out


def retrain(trigger: str = "manual", epochs: int = 5, log=print) -> dict:
    """Retrain on data/train + data/uploads, starting from the current model.

    1. load the production model (warm start, so retraining takes minutes);
    2. fine-tune on the original training data plus the new uploads;
    3. evaluate on the untouched test set;
    4. promote if macro-F1 >= current macro-F1 - PROMOTION_TOLERANCE;
    5. move the uploads into data/train so they become part of the dataset.
    """
    t0 = time.time()
    class_names = json.loads((MODELS_DIR / "class_names.json").read_text())
    pending = pending_uploads()
    n_new = sum(v for k, v in pending.items() if k in class_names)
    log(f"Retraining with {n_new} new images: {pending}")

    # Rebuild the architecture and copy the production weights into it. A model
    # deserialised from .h5 cannot be re-saved after training in Keras 3.
    current = build_model(len(class_names))
    current.set_weights(load_model().get_weights())
    reg = load_registry()
    cur_entry = next((v for v in reg["versions"] if v["version"] == reg["current"]), None)
    baseline_f1 = cur_entry["macro_f1"] if cur_entry else 0.0

    model, history, _ = train(train_dirs=(pp.TRAIN_DIR, pp.UPLOAD_DIR), model=current,
                              class_names=class_names, finetune_epochs=epochs, verbose=0)
    metrics = evaluate(model, class_names)
    promote = metrics["macro_f1"] >= baseline_f1 - PROMOTION_TOLERANCE
    entry = save_model(model, class_names, history, metrics, trigger, n_new, promote)
    log(f"Candidate v{entry['version']}: macro-F1 {metrics['macro_f1']:.4f} "
        f"(current {baseline_f1:.4f}) -> {'promoted' if promote else 'rejected'}")

    # Uploaded images become part of the training data whatever the outcome,
    # so the next retrain still sees them.
    for class_dir in [p for p in pp.UPLOAD_DIR.iterdir() if p.is_dir()]:
        if class_dir.name not in class_names:
            continue
        dst = pp.TRAIN_DIR / class_dir.name
        dst.mkdir(parents=True, exist_ok=True)
        for f in pp.list_images(class_dir):
            shutil.move(str(f), dst / f.name)
    entry["duration_s"] = round(time.time() - t0, 1)
    entry["baseline_macro_f1"] = baseline_f1
    return entry


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["train", "retrain", "evaluate"])
    args = ap.parse_args()
    if args.command == "train":
        model, hist, names = train()
        mets = evaluate(model, names)
        print(json.dumps({k: v for k, v in mets.items() if not isinstance(v, (dict, list))}, indent=2))
        print(save_model(model, names, hist, mets, trigger="initial"))
    elif args.command == "retrain":
        print(retrain())
    else:
        names = json.loads((MODELS_DIR / "class_names.json").read_text())
        print(json.dumps(evaluate(load_model(), names), indent=2)[:3000])
