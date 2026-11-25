"""Prediction: load the production model and classify one screenshot.

The Predictor hot-reloads: before each prediction it checks the registry's
modification time, so when any container (or the retrain job) promotes a new
model to the shared models/ folder, every API container starts using it.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path

import numpy as np

from src import preprocessing as pp
from src.model import MODEL_PATH, MODELS_DIR, REGISTRY_PATH, load_model, load_registry


class Predictor:
    def __init__(self, model_path: Path = MODEL_PATH):
        self.model_path = model_path
        self._lock = threading.Lock()
        self._mtime = None
        self.model = None
        self.class_names: list[str] = []
        self.version = None
        self.reload_if_changed()

    def _stamp(self):
        files = [self.model_path, REGISTRY_PATH]
        return tuple(f.stat().st_mtime if f.exists() else 0 for f in files)

    def reload_if_changed(self) -> bool:
        stamp = self._stamp()
        if stamp == self._mtime and self.model is not None:
            return False
        with self._lock:
            if stamp == self._mtime and self.model is not None:
                return False
            model = load_model(self.model_path)
            self.class_names = json.loads((MODELS_DIR / "class_names.json").read_text())
            self.version = load_registry().get("current")
            self.model = model
            self._mtime = stamp
            # Warm-up call so the first real request is not slow.
            self.model(np.zeros((1, pp.IMG_SIZE, pp.IMG_SIZE, 3), np.float32), training=False)
            return True

    def predict_array(self, batch: np.ndarray) -> np.ndarray:
        self.reload_if_changed()
        return self.model(batch, training=False).numpy()

    def predict(self, image, top_k: int = 3) -> dict:
        """Classify one screenshot given as bytes, a path or a PIL image."""
        img = image if hasattr(image, "resize") and not isinstance(image, (bytes, str, Path)) \
            else pp.load_image(image)
        if img.size != (pp.IMG_SIZE, pp.IMG_SIZE):
            img = img.convert("RGB").resize((pp.IMG_SIZE, pp.IMG_SIZE))
        probs = self.predict_array(pp.image_to_array(img)[None])[0]
        order = np.argsort(probs)[::-1]
        best = self.class_names[order[0]]
        p_other = float(probs[self.class_names.index("other")]) if "other" in self.class_names else 0.0
        return {
            "label": best,
            "brand": pp.CLASS_LABELS.get(best, best),
            "confidence": round(float(probs[order[0]]), 4),
            "verdict": "legitimate / unknown brand" if best == "other"
                       else f"possible phishing page imitating {pp.CLASS_LABELS.get(best, best)}",
            "phishing_score": round(1.0 - p_other, 4),
            "top_k": [{"label": self.class_names[i], "brand": pp.CLASS_LABELS.get(self.class_names[i]),
                       "probability": round(float(probs[i]), 4)} for i in order[:top_k]],
            "model_version": self.version,
        }


_default: Predictor | None = None


def get_predictor() -> Predictor:
    global _default
    if _default is None:
        _default = Predictor()
    return _default


def predict_image(image, top_k: int = 3) -> dict:
    return get_predictor().predict(image, top_k)


if __name__ == "__main__":
    import sys
    for f in sys.argv[1:]:
        print(f, json.dumps(predict_image(Path(f)), indent=2))
