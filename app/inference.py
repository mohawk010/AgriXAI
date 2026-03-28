"""
inference.py — Loads the Keras ResNet50 model once and provides prediction.
"""
from __future__ import annotations

import io
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

# ── Resolve class names from prevention_tips ──────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from utils.prevention_tips import PREVENTION_TIPS  # noqa: E402

CLASS_NAMES: list[str] = sorted(PREVENTION_TIPS.keys())
NUM_CLASSES: int = len(CLASS_NAMES)
IMG_SIZE: int = 224

# ── Model (loaded lazily once) ────────────────────────────────────────────────
_model = None  # type: ignore


def _get_model():
    """Load model on first call; cache afterward."""
    global _model
    if _model is not None:
        return _model

    import tensorflow as tf  # local import so TF is only loaded when needed

    model_path = os.environ.get(
        "MODEL_PATH",
        str(ROOT / "New Plant Diseases Dataset(Augmented)"
                  / "New Plant Diseases Dataset(Augmented)"
                  / "resnet50_best.keras"),
    )
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Keras model not found at: {model_path}")

    print(f"[inference] Loading model from {model_path} …")
    _model = tf.keras.models.load_model(model_path)
    print("[inference] Model loaded.")
    return _model


def preprocess_image(image_bytes: bytes) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert raw image bytes → (preprocessed_batch [1,224,224,3], original_rgb [H,W,3])
    """
    import tensorflow as tf

    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    orig_rgb = np.array(img)

    img_resized = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    arr = np.array(img_resized, dtype=np.float32)
    arr = tf.keras.applications.resnet50.preprocess_input(arr)
    batch = np.expand_dims(arr, axis=0)
    return batch, orig_rgb


def predict(image_bytes: bytes) -> dict:
    """
    Run inference on raw image bytes.
    Returns:
        predicted_class   : str   — top-1 class name
        confidence        : float — top-1 probability (0-1)
        top5              : list[dict] — [{class, probability}, …]
        preprocessed_batch: np.ndarray — for Grad-CAM reuse
        original_rgb      : np.ndarray — for display / overlay
    """
    model = _get_model()
    batch, original_rgb = preprocess_image(image_bytes)

    preds = model.predict(batch, verbose=0)[0]  # (num_classes,)
    top5_idx = np.argsort(preds)[::-1][:5]

    predicted_class = CLASS_NAMES[top5_idx[0]]
    confidence = float(preds[top5_idx[0]])
    top5 = [
        {"class": CLASS_NAMES[i], "probability": float(preds[i])}
        for i in top5_idx
    ]

    return {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "top5": top5,
        "preprocessed_batch": batch,
        "original_rgb": original_rgb,
        "model": model,
    }
