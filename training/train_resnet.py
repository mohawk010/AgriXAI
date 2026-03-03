import argparse
import json
import os
import time
from typing import Dict

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt

from models.resnet50 import get_resnet50
from utils.dataset import create_dataloaders
from utils.metrics import classification_report_dict


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        _, true = labels.max(1)
        correct += (preds == true).sum().item()
        total += images.size(0)

    return running_loss / total, correct / total


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    y_true = []
    y_pred = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            _, true = labels.max(1)
            correct += (preds == true).sum().item()
            total += images.size(0)
            y_true.extend(true.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())

    avg_loss = running_loss / total
    acc = correct / total
    return avg_loss, acc, y_true, y_pred


def plot_history(history: Dict, out_dir: str):
    epochs = list(range(1, len(history['train_loss']) + 1))
    plt.figure()
    plt.plot(epochs, history['train_loss'], label='train_loss')
    plt.plot(epochs, history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss curves')
    plt.savefig(os.path.join(out_dir, 'loss_curve.png'))
    plt.close()

    plt.figure()
    plt.plot(epochs, history['train_acc'], label='train_acc')
    plt.plot(epochs, history['val_acc'], label='val_acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy curves')
    plt.savefig(os.path.join(out_dir, 'acc_curve.png'))
    plt.close()


def main(args):
    device = 'cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu'
    os.makedirs(args.output_dir, exist_ok=True)

    train_loader, val_loader, class_names = create_dataloaders(args.data_dir, img_size=args.img_size,
                                                              batch_size=args.batch_size, num_workers=args.workers)

    num_classes = len(class_names)
    model = get_resnet50(num_classes=num_classes, dropout=args.dropout,
                         freeze_backbone=args.freeze_backbone, trainable_layers=args.trainable_layers)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    # Only parameters that require_grad will be optimized
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, y_true, y_pred = validate(model, val_loader, criterion, device)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        epoch_time = time.time() - t0
        print(f'Epoch {epoch}/{args.epochs} - {epoch_time:.1f}s - train_loss: {train_loss:.4f} val_loss: {val_loss:.4f}'
              f' train_acc: {train_acc:.4f} val_acc: {val_acc:.4f}')

        # save best model (by val_acc)
        if epoch == 1 or val_acc >= max(history['val_acc'][:-1] + [val_acc]):
            torch.save(model.state_dict(), os.path.join(args.output_dir, 'resnet50_best.pth'))

    total_time = time.time() - start_time
    history['training_time_seconds'] = total_time

    # save history
    with open(os.path.join(args.output_dir, 'resnet_history.json'), 'w') as f:
        json.dump(history, f)

    plot_history(history, args.output_dir)

    # classification report on validation
    report = classification_report_dict(y_true, y_pred, class_names)
    with open(os.path.join(args.output_dir, 'val_classification_report.json'), 'w') as f:
        json.dump(report, f)

    print('Training complete. Artifacts saved to', args.output_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='.', help='root dataset folder (contains train/ and valid/)')
    parser.add_argument('--output-dir', type=str, default='artifacts', help='where to save models and plots')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--img-size', type=int, default=224)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--freeze-backbone', action='store_true')
    parser.add_argument('--trainable-layers', type=int, default=2, help='how many last ResNet blocks to unfreeze')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--no-cuda', action='store_true', help='force cpu')
    args = parser.parse_args()

    main(args)
