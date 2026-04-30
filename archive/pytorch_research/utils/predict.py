"""
predict.py
----------
Single-image inference utility that combines:
  - ResNet50 prediction
  - Confidence score
  - Grad-CAM heatmap
  - Agronomic prevention / treatment tips

Usage:
    from utils.predict import predict_image

    result = predict_image(
        image_path="leaf.jpg",
        model=model,
        class_names=class_names,
        device="cuda",
        gradcam=gradcam_instance,   # optional
    )
    print(result["tips_text"])
"""

import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from typing import Any, Dict, List, Optional

from utils.prevention_tips import get_tips, format_tips

# ImageNet normalization (must match training pipeline)
_INFER_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def predict_image(
    image_path: str,
    model: torch.nn.Module,
    class_names: List[str],
    device: str = "cpu",
    gradcam=None,
    top_k: int = 3,
) -> Dict[str, Any]:
    """Run inference on a single image and return a rich result dictionary.

    Args:
        image_path: Path to the input image file.
        model: Loaded, eval-mode PyTorch model.
        class_names: Ordered list of class names (same order as ImageFolder).
        device: 'cpu' or 'cuda'.
        gradcam: Optional GradCAM instance from visualization.gradcam.
        top_k: Number of top predictions to return.

    Returns:
        A dict with keys:
            predicted_class (str)       — top-1 class name
            confidence      (float)     — top-1 softmax probability (0–1)
            top_k_classes   (list)      — list of (class_name, probability) tuples
            tips            (dict|None) — raw tips dict from PREVENTION_TIPS
            tips_text       (str)       — formatted human-readable tips string
            heatmap         (ndarray|None) — Grad-CAM heatmap array if gradcam was provided
    """
    model.eval()
    model.to(device)

    # ── Load & preprocess image ──────────────────────────────────────────────
    image = Image.open(image_path).convert("RGB")
    input_tensor = _INFER_TRANSFORM(image).unsqueeze(0).to(device)

    # ── Forward pass ─────────────────────────────────────────────────────────
    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = torch.softmax(logits, dim=1)[0]

    top_probs, top_indices = probabilities.topk(min(top_k, len(class_names)))
    top_k_results = [
        (class_names[idx.item()], round(prob.item(), 4))
        for prob, idx in zip(top_probs, top_indices)
    ]

    predicted_class = top_k_results[0][0]
    confidence = top_k_results[0][1]

    # ── Grad-CAM heatmap ─────────────────────────────────────────────────────
    heatmap = None
    if gradcam is not None:
        pred_idx = int(probabilities.argmax().item())
        # GradCAM requires gradients — use a fresh tensor
        input_for_cam = _INFER_TRANSFORM(image).unsqueeze(0).to(device)
        heatmap = gradcam.generate_cam(input_for_cam, target_class=pred_idx)

    # ── Prevention / treatment tips ──────────────────────────────────────────
    tips = get_tips(predicted_class)
    tips_text = format_tips(predicted_class)

    return {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "top_k_classes": top_k_results,
        "tips": tips,
        "tips_text": tips_text,
        "heatmap": heatmap,
    }


def print_prediction_report(result: Dict[str, Any]) -> None:
    """Pretty-print a full prediction report to stdout."""
    print("=" * 60)
    print(f"PREDICTION  : {result['predicted_class']}")
    print(f"CONFIDENCE  : {result['confidence'] * 100:.1f}%")
    print()
    print("TOP PREDICTIONS:")
    for cls, prob in result["top_k_classes"]:
        bar = "#" * int(prob * 40)
        print(f"  {cls:<55} {prob * 100:5.1f}%  {bar}")
    print()
    print(result["tips_text"])
    print("=" * 60)
