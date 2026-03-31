import torch.nn as nn
import torchvision.models as models


def build_model(num_classes: int = 7, pretrained: bool = True):
    weights = "IMAGENET1K_V1" if pretrained else None
    model = models.resnet50(weights=weights)

    # freeze all
    for p in model.parameters():
        p.requires_grad = False

    # unfreeze layer4
    for p in model.layer4.parameters():
        p.requires_grad = True

    # replace classifier head
    model.fc = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(model.fc.in_features, 256),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(256, num_classes),
    )

    for p in model.fc.parameters():
        p.requires_grad = True

    return model
