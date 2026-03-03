import argparse
import os
import time
import torch
import json
import pandas as pd

from models.resnet50 import get_resnet50
from utils.dataset import create_dataloaders
from utils.metrics import measure_inference_time


def load_model(path: str, model_type: str, num_classes: int, device='cpu'):
    """Load model artifact. Supports 'resnet50' (state_dict) or a torchscript file."""
    if model_type == 'resnet50':
        model = get_resnet50(num_classes=num_classes)
        state = torch.load(path, map_location=device)
        # state could be state_dict or full dict
        if isinstance(state, dict) and any(k.startswith('module.') or k.startswith('layer1') for k in state.keys()):
            model.load_state_dict(state)
        else:
            try:
                model.load_state_dict(state)
            except Exception:
                # try loading entire model object
                model = torch.load(path, map_location=device)
    else:
        # attempt to load as torchscript or saved model
        try:
            model = torch.jit.load(path, map_location=device)
        except Exception:
            model = torch.load(path, map_location=device)

    model.to(device)
    model.eval()
    return model


def model_file_size_mb(path: str) -> float:
    return os.path.getsize(path) / (1024.0 * 1024.0)


def main(args):
    _, val_loader, class_names = create_dataloaders(args.data_dir, img_size=args.img_size, batch_size=args.batch_size)
    num_classes = len(class_names)

    device = 'cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu'

    models_info = []
    for path, mtype, name in [(args.model1, args.model1_type, 'model1'), (args.model2, args.model2_type, 'model2')]:
        model = load_model(path, mtype, num_classes, device=device)
        size_mb = model_file_size_mb(path) if os.path.exists(path) else None
        timing = measure_inference_time(model, val_loader, device=device, max_batches=50)
        models_info.append({'path': path, 'type': mtype, 'size_mb': size_mb, 'inference_avg_s_per_sample': timing['avg_per_sample']})

    df = pd.DataFrame(models_info)
    out = os.path.join(args.output_dir, 'model_comparison.csv')
    os.makedirs(args.output_dir, exist_ok=True)
    df.to_csv(out, index=False)
    print('Comparison saved to', out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='.', help='root dataset folder (train/ valid/)')
    parser.add_argument('--model1', type=str, required=True)
    parser.add_argument('--model1-type', type=str, default='resnet50', help='resnet50|torchscript|other')
    parser.add_argument('--model2', type=str, required=True)
    parser.add_argument('--model2-type', type=str, default='resnet50')
    parser.add_argument('--output-dir', type=str, default='artifacts')
    parser.add_argument('--img-size', type=int, default=224)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--no-cuda', action='store_true')
    args = parser.parse_args()
    main(args)
