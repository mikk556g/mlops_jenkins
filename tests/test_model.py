import sys
import os
import torch
import yaml
sys.path.append("src")
from models.resnet50 import ResNet50FineTuned  


# Load config 

def load_config(path="config/train_config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


MODEL_CONFIG = load_config()["model"]


# Helper 

def build_test_model(**overrides):
    """Build model with pretrained=False by default to avoid downloading
    ImageNet weights in CI. Pass overrides to change specific fields."""
    cfg = {**MODEL_CONFIG, "pretrained": False, **overrides}
    return ResNet50FineTuned(cfg)


# Construction 

def test_resnet50_builds():
    """Model should instantiate without errors."""
    model = build_test_model()
    assert model is not None


def test_resnet50_custom_num_classes():
    """FC head should output a custom number of classes."""
    model = build_test_model(num_classes=3)
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        y = model(x)
    assert y.shape[1] == 3


def test_resnet50_no_freeze():
    """Model should build when freeze_layers is empty."""
    model = build_test_model(freeze_layers=[])
    assert model is not None


# Frozen / trainable layers

def test_resnet50_frozen_layers():
    """All layers in freeze_layers (from config) should be non-trainable."""
    model = build_test_model()
    freeze_layers = MODEL_CONFIG.get("freeze_layers", [])
    for name, param in model.backbone.named_parameters():
        if any(name.startswith(layer) for layer in freeze_layers):
            assert not param.requires_grad, f"{name} should be frozen"


def test_frozen_layers_from_config():
    """Each freeze_layer should contain at least one frozen parameter."""
    model = build_test_model()
    for layer_name in MODEL_CONFIG.get("freeze_layers", []):
        layer = getattr(model.backbone, layer_name, None)
        assert layer is not None, f"Layer '{layer_name}' not found on backbone"
        frozen = [p for p in layer.parameters() if not p.requires_grad]
        assert len(frozen) > 0, f"'{layer_name}' should have frozen parameters"


def test_resnet50_fc_trainable():
    """Classifier head (fc) should be fully trainable."""
    model = build_test_model()
    trainable = [p for p in model.backbone.fc.parameters() if p.requires_grad]
    assert len(trainable) > 0


def test_layer4_is_trainable():
    """layer4 is not in freeze_layers and should have trainable parameters."""
    model = build_test_model()
    trainable = [p for p in model.backbone.layer4.parameters() if p.requires_grad]
    assert len(trainable) > 0


def test_trainable_params_exist():
    """Model should have at least some trainable parameters overall."""
    model = build_test_model()
    trainable = [p for p in model.parameters() if p.requires_grad]
    assert len(trainable) > 0


# Forward pass 

def test_resnet50_forward_shape():
    """Forward pass should return (batch_size, num_classes)."""
    num_classes = MODEL_CONFIG["num_classes"]
    model = build_test_model()
    model.eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (2, num_classes)


def test_resnet50_forward_batch_size_one():
    """Model should handle a single-image batch."""
    num_classes = MODEL_CONFIG["num_classes"]
    model = build_test_model()
    model.eval()
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (1, num_classes)


def test_resnet50_forward_returns_tensor():
    """Forward pass should return a torch.Tensor."""
    model = build_test_model()
    model.eval()
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        y = model(x)
    assert isinstance(y, torch.Tensor)


# Real CustomDataset (same as train.py) 

import glob
import pytest
from PIL import Image

CONFIG_FULL = load_config()
DATA_PATH = CONFIG_FULL["dataset"]["dataset_path"]
CLASSES = CONFIG_FULL["classes"]

DATA_AVAILABLE = any(
    os.path.isdir(os.path.join(DATA_PATH, c)) for c in CLASSES
)

requires_data = pytest.mark.skipif(
    not DATA_AVAILABLE,
    reason=f"Dataset not found under '{DATA_PATH}/' — run 'dvc pull' first.",
)


class CustomDataset:
    def __init__(self, img_paths, img_labels, transform=None):
        self.img_paths = img_paths
        self.img_labels = img_labels
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        image = Image.open(self.img_paths[idx]).convert("RGB")
        label = torch.tensor(self.img_labels[idx], dtype=torch.long)
        if self.transform:
            image = self.transform(image)
        return image, label


def get_sample_paths_and_labels(max_per_class=5):
    img_paths, img_labels = [], []
    for class_name, class_idx in CLASSES.items():
        class_dir = os.path.join(DATA_PATH, class_name)
        if not os.path.isdir(class_dir):
            continue
        images = glob.glob(os.path.join(class_dir, "*.jpg"))
        images += glob.glob(os.path.join(class_dir, "*.png"))
        images = images[:max_per_class]
        img_paths.extend(images)
        img_labels.extend([class_idx] * len(images))
    return img_paths, img_labels


@requires_data
def test_custom_dataset_len():
    """Dataset should report correct length."""
    paths, labels = get_sample_paths_and_labels()
    ds = CustomDataset(paths, labels)
    assert len(ds) == len(paths)


@requires_data
def test_custom_dataset_getitem_label():
    """__getitem__ should return the correct label."""
    paths, labels = get_sample_paths_and_labels()
    ds = CustomDataset(paths, labels)
    _, label = ds[0]
    assert 0 <= label.item() < MODEL_CONFIG["num_classes"]


@requires_data
def test_custom_dataset_getitem_image_shape():
    """__getitem__ should return a PIL Image (before transform)."""
    paths, labels = get_sample_paths_and_labels()
    ds = CustomDataset(paths, labels)
    image, _ = ds[0]
    assert isinstance(image, Image.Image)
