from typing import List, Dict
import time
import torch
from sklearn.metrics import classification_report, confusion_matrix


def classification_report_dict(y_true, y_pred, target_names: List[str]) -> Dict:
    """Return sklearn classification report as a dict."""
    return classification_report(y_true, y_pred, target_names=target_names, output_dict=True)


def confusion_matrix_np(y_true, y_pred):
    return confusion_matrix(y_true, y_pred)


def measure_inference_time(model: torch.nn.Module, dataloader, device='cpu', max_batches=100):
    """Measure average inference time (seconds) per batch and per sample.

    Limits to `max_batches` for speed.
    """
    model.eval()
    model.to(device)
    import time
    total_time = 0.0
    total_samples = 0
    with torch.no_grad():
        for i, (x, _) in enumerate(dataloader):
            if i >= max_batches:
                break
            x = x.to(device)
            t0 = time.time()
            _ = model(x)
            t1 = time.time()
            total_time += (t1 - t0)
            total_samples += x.size(0)

    avg_per_batch = total_time / max(1, min(max_batches, len(dataloader)))
    avg_per_sample = total_time / max(1, total_samples)
    return {'total_time': total_time, 'avg_per_batch': avg_per_batch, 'avg_per_sample': avg_per_sample}
