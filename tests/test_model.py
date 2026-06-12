import sys
import torch
import yaml

sys.path.append("src")  # noqa: E402
from models.resnet50 import ResNet50FineTuned  # noqa: E402

with open("config/train_config.yaml") as f:
    config = yaml.safe_load(f)

model_config = config["model"]


def test_model_builds():
    """Model should instantiate from the training config."""
    model = ResNet50FineTuned(model_config)
    assert model is not None


def test_forward_pass_shape():
    """Forward pass should return (batch_size, num_classes)."""
    model = ResNet50FineTuned(model_config)
    model.eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (2, model_config["num_classes"])


def test_frozen_layers():
    """Layers listed in freeze_layers should not be trainable."""
    model = ResNet50FineTuned(model_config)
    for name, param in model.backbone.named_parameters():
        for layer in model_config["freeze_layers"]:
            if name.startswith(layer):
                assert not param.requires_grad, f"{name} should be frozen"


def test_fc_head_trainable():
    """Classifier head should be fully trainable."""
    model = ResNet50FineTuned(model_config)
    for param in model.backbone.fc.parameters():
        assert param.requires_grad
