import os
from typing import Tuple, List
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def create_dataloaders(root_dir: str, img_size: int = 224, batch_size: int = 32, num_workers: int = 4,
                       val_split: float = 0.1, shuffle: bool = True) -> Tuple[DataLoader, DataLoader, List[str]]:
    """Create train/val dataloaders from an ImageFolder-structured dataset.

    Expects `root_dir` to contain `train` and `valid` (or `val`) subfolders.
    Returns (train_loader, val_loader, class_names).
    """
    train_dir = os.path.join(root_dir, 'train')
    val_dir = os.path.join(root_dir, 'valid')

    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    train_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        normalize,
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        normalize,
    ])

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transforms)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    class_names = train_dataset.classes

    return train_loader, val_loader, class_names
