import os
import cv2
import numpy as np
import torch
from typing import Iterable


def unnormalize_tensor(img_tensor: torch.Tensor) -> np.ndarray:
    """Convert a torch image tensor (C,H,W) normalized with ImageNet stats back to uint8 BGR image."""
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = img_tensor.cpu().numpy()
    img = np.transpose(img, (1, 2, 0))
    img = (img * std + mean)
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    # convert RGB -> BGR for cv2
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img_bgr


def save_gradcam_visualizations(model: torch.nn.Module, gradcam, dataloader: Iterable, class_names: list,
                                out_dir: str, device='cpu', max_images: int = 200):
    os.makedirs(out_dir, exist_ok=True)
    model.eval()
    model.to(device)
    count = 0
    for inputs, labels in dataloader:
        if count >= max_images:
            break
        inputs = inputs.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            outputs = model(inputs)
            preds = outputs.argmax(dim=1)

        # single-batch assumption
        for i in range(inputs.size(0)):
            inp = inputs[i:i+1]
            true_idx = int(labels[i].argmax().item())
            pred_idx = int(preds[i].item())

            heatmap = gradcam.generate_cam(inp, target_class=pred_idx)
            orig = unnormalize_tensor(inputs[i].cpu())
            overlay = gradcam.overlay_heatmap_on_image(orig, heatmap, alpha=0.5)

            status = 'correct' if pred_idx == true_idx else 'incorrect'
            fname = f'{count:05d}_{status}_true_{class_names[true_idx]}_pred_{class_names[pred_idx]}.jpg'
            cv2.imwrite(os.path.join(out_dir, fname), overlay)
            count += 1
            if count >= max_images:
                break
