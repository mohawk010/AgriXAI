import numpy as np
import cv2
import torch


class GradCAM:
    """Simple Grad-CAM implementation for ResNet-like models.

    Usage:
        cam = GradCAM(model, target_layer)
        heatmap = cam.generate_cam(input_tensor, target_class)
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.model.eval()
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        # register hooks
        def forward_hook(module, inp, out):
            self.activations = out.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        target_layer.register_forward_hook(forward_hook)
        target_layer.register_backward_hook(backward_hook)

    def generate_cam(self, input_tensor: torch.Tensor, target_class: int = None) -> np.ndarray:
        """Generate a CAM heatmap (H x W) in range [0,1].

        Args:
            input_tensor: single image tensor shaped (1,C,H,W)
            target_class: optional target class index. If None, uses predicted class.
        Returns:
            heatmap: numpy array height x width normalized to [0,1]
        """
        self.model.zero_grad()
        input_tensor = input_tensor.requires_grad_(True)
        outputs = self.model(input_tensor)
        if target_class is None:
            target_class = int(outputs.argmax(dim=1).item())

        score = outputs[0, target_class]
        score.backward(retain_graph=True)

        # gradients: [N, C, H, W]
        grads = self.gradients[0]  # [C, H, W]
        activations = self.activations[0]  # [C, H, W]

        weights = grads.mean(dim=(1, 2))  # [C]
        cam = (weights[:, None, None] * activations).sum(dim=0).cpu().numpy()
        cam = np.maximum(cam, 0)
        if cam.max() != 0:
            cam = cam / cam.max()

        return cam

    @staticmethod
    def overlay_heatmap_on_image(img_bgr: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4) -> np.ndarray:
        """Overlay single-channel heatmap (0..1) onto BGR image and return BGR result."""
        h, w = img_bgr.shape[:2]
        heatmap_resized = cv2.resize(heatmap, (w, h))
        heatmap_uint8 = np.uint8(255 * heatmap_resized)
        colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(colored, alpha, img_bgr, 1 - alpha, 0)
        return overlay
