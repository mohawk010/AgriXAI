import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights


def get_resnet50(num_classes: int, dropout: float = 0.5, freeze_backbone: bool = True, trainable_layers: int = 2):
    """Return a ResNet50 model prepared for fine-tuning.

    Args:
        num_classes: number of output classes
        dropout: dropout probability inserted before final fc
        freeze_backbone: if True, freeze most of the backbone layers
        trainable_layers: number of last ResNet blocks to leave trainable (1..4)
    """
    model = models.resnet50(weights=ResNet50_Weights.DEFAULT)

    # optionally freeze backbone
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

        # Unfreeze last `trainable_layers` blocks in layer4, layer3, etc.
        layers = [model.layer4, model.layer3, model.layer2, model.layer1]
        for i in range(min(trainable_layers, len(layers))):
            for param in layers[i].parameters():
                param.requires_grad = True

    # Replace the classifier (fc). Insert dropout for regularization.
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, num_classes)
    )

    return model
