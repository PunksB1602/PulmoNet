import torch.nn as nn
from torchvision import models


def get_model(
    model_name: str,
    num_classes: int = 5,
    pretrained: bool = True,
    dropout: float = 0.0,
    verbose: bool = False,
):
    name = model_name.lower().strip()

    if verbose:
        print(f"Initializing {name} (pretrained={pretrained}, num_classes={num_classes})")

    def make_head(in_features: int) -> nn.Module:
        if dropout and dropout > 0:
            return nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, num_classes))
        return nn.Linear(in_features, num_classes)

    if name == "densenet121":
        # New API if available, else old
        try:
            weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
            model = models.densenet121(weights=weights)
        except AttributeError:
            model = models.densenet121(pretrained=pretrained)

        in_features = model.classifier.in_features
        model.classifier = make_head(in_features)

    elif name == "resnet50":
        try:
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            model = models.resnet50(weights=weights)
        except AttributeError:
            model = models.resnet50(pretrained=pretrained)

        in_features = model.fc.in_features
        model.fc = make_head(in_features)

    elif name == "efficientnet_b0":
        try:
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            model = models.efficientnet_b0(weights=weights)
        except AttributeError:
            model = models.efficientnet_b0(pretrained=pretrained)

        # classifier is usually Sequential([Dropout, Linear])
        if isinstance(model.classifier, nn.Sequential):
            # replace last linear, optionally rebuild head for fairness
            in_features = model.classifier[-1].in_features
        else:
            in_features = model.classifier.in_features

        model.classifier = make_head(in_features)

    else:
        raise ValueError("model_name must be one of: densenet121, resnet50, efficientnet_b0")

    return model
