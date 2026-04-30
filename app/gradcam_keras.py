"""
gradcam_keras.py — Grad-CAM for the Keras/TF ResNet50 model.
Uses tf.GradientTape to hook into the last conv layer.
"""
from __future__ import annotations

import base64
import io
import logging

import cv2
import numpy as np

logger = logging.getLogger("agrixai.gradcam")

LAST_CONV_LAYER = "conv5_block3_out"


_gradcam_cache: dict = {}

def _make_gradcam_model(model):
    """Build a 2-output model: (last_conv_output, final_predictions). Cache it to avoid rebuilding."""
    key = id(model)
    if key in _gradcam_cache:
        return _gradcam_cache[key]

    import tensorflow as tf

    try:
        # If the model is nested (e.g. contains a 'resnet50' sub-model layer)
        resnet_base = model.get_layer("resnet50")
        last_conv_layer = resnet_base.get_layer(LAST_CONV_LAYER)
        
        conv_model = tf.keras.Model(
            inputs=resnet_base.inputs,
            outputs=[last_conv_layer.output, resnet_base.output]
        )
        
        inputs = tf.keras.Input(shape=model.inputs[0].shape[1:])
        conv_out, x = conv_model(inputs)
        
        for layer in model.layers[1:]:
            x = layer(x)
            
        grad_model = tf.keras.Model(inputs=inputs, outputs=[conv_out, x])
    except ValueError:
        # Fallback if it's already a flat model
        grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[
                model.get_layer(LAST_CONV_LAYER).output,
                model.output,
            ],
        )

    _gradcam_cache[key] = grad_model
    return grad_model


def generate_gradcam(
    model,
    preprocessed_batch: np.ndarray,
    original_rgb: np.ndarray,
    class_idx: int | None = None,
) -> tuple[str, np.ndarray]:
    """
    Generate a Grad-CAM heatmap overlay.

    Returns:
        b64_png : base64-encoded PNG string of the heatmap overlay
        heatmap : raw float32 heatmap [H, W] normalised 0-1
    """
    import tensorflow as tf

    logger.info("Building Grad-CAM model for layer '%s'...", LAST_CONV_LAYER)
    grad_model = _make_gradcam_model(model)

    with tf.GradientTape() as tape:
        inputs = tf.cast(preprocessed_batch, tf.float32)
        conv_outputs, predictions = grad_model(inputs)
        if class_idx is None:
            class_idx = tf.argmax(predictions[0])
        class_channel = predictions[:, class_idx]

    grads = tape.gradient(class_channel, conv_outputs)          # (1, h, w, C)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))        # (C,)
    conv_outputs = conv_outputs[0]                               # (h, w, C)
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]      # (h, w, 1)
    heatmap = tf.squeeze(heatmap)                               # (h, w)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    heatmap = heatmap.numpy()

    if np.max(heatmap) < 1e-6:
        logger.warning("Grad-CAM heatmap is near-zero — model may lack confidence in this prediction.")

    # ── Resize heatmap to original image size ─────────────────────────────
    h, w = original_rgb.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    # ── Overlay on original image ──────────────────────────────────────────
    original_bgr = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2BGR)
    superimposed = cv2.addWeighted(original_bgr, 0.55, heatmap_colored, 0.45, 0)

    # ── Encode to base64 PNG ───────────────────────────────────────────────
    _, buffer = cv2.imencode(".png", superimposed)
    b64 = base64.b64encode(buffer.tobytes()).decode("utf-8")

    logger.info("Grad-CAM heatmap generated (%dx%d)", w, h)
    return b64, heatmap_resized
