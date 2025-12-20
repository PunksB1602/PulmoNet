import torch
import torch.nn as nn
from torchvision import models

def get_model(model_name, num_classes=5, pretrained=True):
    # Initialize a torchvision model and replace its head for `num_classes` outputs.
    # `pretrained=True` loads ImageNet weights where available.
    print(f"Initializing {model_name} (Pretrained={pretrained})...")

    model_name = model_name.lower()

    if model_name == 'densenet121':
        weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
        model = models.densenet121(weights=weights)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, num_classes)

    elif model_name == 'resnet50':
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)

    elif model_name == 'efficientnet_b0':
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_classes)

    else:
        raise ValueError(f"Model {model_name} not supported.")

    return model
