"""
inference.py — Loads the Keras ResNet50 model once and provides prediction.
"""
from __future__ import annotations

import io
import json
import logging
import os
import threading
from pathlib import Path

import numpy as np
from PIL import Image

logger = logging.getLogger("agrixai.inference")

# ── Resolve class names from class_names.json ─────────────────────────────────
ROOT = Path(__file__).parent.parent

_class_names_path = ROOT / "class_names.json"
if not _class_names_path.exists():
    raise FileNotFoundError(
        f"class_names.json not found at {_class_names_path}. "
        "This file must list the 38 class names in the same order as the training dataset."
    )

with open(_class_names_path, "r", encoding="utf-8") as f:
    CLASS_NAMES: list[str] = json.load(f)

NUM_CLASSES: int = len(CLASS_NAMES)
IMG_SIZE: int = 224

# ── Validate class names against prevention_tips at import time ───────────────
import importlib.util
_tips_spec = importlib.util.spec_from_file_location(
    "prevention_tips", str(ROOT / "utils" / "prevention_tips.py")
)
_tips_mod = importlib.util.module_from_spec(_tips_spec)
_tips_spec.loader.exec_module(_tips_mod)
PREVENTION_TIPS = _tips_mod.PREVENTION_TIPS

from app.config import settings

_tips_keys = set(PREVENTION_TIPS.keys())
_class_set = set(CLASS_NAMES)
_missing_tips = _class_set - _tips_keys
_extra_tips = _tips_keys - _class_set

if _missing_tips:
    logger.warning(
        "Classes in class_names.json but NOT in PREVENTION_TIPS: %s",
        sorted(_missing_tips),
    )
if _extra_tips:
    logger.warning(
        "Classes in PREVENTION_TIPS but NOT in class_names.json: %s",
        sorted(_extra_tips),
    )
if not _missing_tips and not _extra_tips:
    logger.info("Class names validated: %d classes match PREVENTION_TIPS ✓", NUM_CLASSES)

# ── Model (loaded lazily once) ────────────────────────────────────────────────
_model = None  # type: ignore
_model_lock = threading.Lock()


def _get_model():
    """Load model on first call; cache afterward. Thread-safe."""
    global _model
    if _model is not None:
        return _model

    with _model_lock:
        if _model is not None:  # double-checked locking
            return _model

        import tensorflow as tf  # local import so TF is only loaded when needed

        # Support both Docker container path and local Windows environment path
        default_docker_path = ROOT / "models" / "resnet50_best.keras"
        default_local_path = (ROOT / "New Plant Diseases Dataset(Augmented)"
                                   / "New Plant Diseases Dataset(Augmented)"
                                   / "resnet50_best.keras")

        if default_docker_path.exists():
            fallback_path = str(default_docker_path)
        else:
            fallback_path = str(default_local_path)

        model_path = settings.model_path or fallback_path
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Keras model not found at: {model_path}")

        logger.info("Loading model from %s …", model_path)
        _model = tf.keras.models.load_model(model_path)
        logger.info("Model loaded successfully.")
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

    import tensorflow as tf
    preds = model(tf.constant(batch), training=False).numpy()[0]  # (num_classes,)
    top5_idx = np.argsort(preds)[::-1][:5]

    predicted_class = CLASS_NAMES[top5_idx[0]]
    confidence = float(preds[top5_idx[0]])
    top5 = [
        {"class": CLASS_NAMES[i], "probability": float(preds[i])}
        for i in top5_idx
    ]

    logger.info("Prediction: %s (%.2f%%)", predicted_class, confidence * 100)

    return {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "top5": top5,
        "preprocessed_batch": batch,
        "original_rgb": original_rgb,
        "model": model,
    }
