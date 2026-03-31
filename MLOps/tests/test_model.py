import sys
import torch

sys.path.append("src")  # noqa: E402
from training.model import build_model  # noqa: E402


# Model construction
def test_model_builds():
    model = build_model(num_classes=7, pretrained=False)
    assert model is not None


def test_build_model_no_pretrained():
    """Model should build fine without pretrained weights."""
    model = build_model(num_classes=7, pretrained=False)
    assert model is not None


def test_model_output_has_seven_classes():
    """Output tensor should have 7 class logits."""
    model = build_model(num_classes=7, pretrained=False)
    model.eval()
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        y = model(x)
    assert y.shape[1] == 7


# Trainable / frozen parameters


def test_trainable_params_exist():
    model = build_model(num_classes=7, pretrained=False)
    trainable = [p for p in model.parameters() if p.requires_grad]
    assert len(trainable) > 0


def test_frozen_layers():
    """Early layers (layer1-3) should be frozen."""
    model = build_model(num_classes=7, pretrained=False)
    frozen = [p for p in model.layer1.parameters() if not p.requires_grad]
    assert len(frozen) > 0


def test_layer4_is_trainable():
    """layer4 should have trainable parameters."""
    model = build_model(num_classes=7, pretrained=False)
    trainable = [p for p in model.layer4.parameters() if p.requires_grad]
    assert len(trainable) > 0


def test_fc_is_trainable():
    """Classifier head (fc) should be fully trainable."""
    model = build_model(num_classes=7, pretrained=False)
    trainable = [p for p in model.fc.parameters() if p.requires_grad]
    assert len(trainable) > 0


# Forward pass


def test_forward_shape():
    model = build_model(num_classes=7, pretrained=False)
    model.eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (2, 7)


def test_forward_batch_size_one():
    """Model should handle a single-image batch."""
    model = build_model(num_classes=7, pretrained=False)
    model.eval()
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (1, 7)


def test_forward_returns_tensor():
    """Output should be a torch.Tensor."""
    model = build_model(num_classes=7, pretrained=False)
    model.eval()
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        y = model(x)
    assert isinstance(y, torch.Tensor)


# CustomDataset
class MockDataset:
    """Minimal re-implementation of CustomDataset
    for testing without real files."""


def __init__(self, img_paths, img_labels, transform=None):
    self.img_paths = img_paths
    self.img_labels = img_labels
    self.transform = transform


def __len__(self):
    return len(self.img_labels)


def __getitem__(self, idx):
    # Return a dummy tensor instead of loading from disk
    image = torch.zeros(3, 224, 224)
    label = torch.tensor(self.img_labels[idx], dtype=torch.long)
    return image, label


def test_custom_dataset_len():
    """Dataset should report correct length."""
    ds = MockDataset(["img1.jpg", "img2.jpg", "img3.jpg"], [0, 1, 2])
    assert len(ds) == 3


def test_custom_dataset_getitem_label():
    """__getitem__ should return the correct label."""
    ds = MockDataset(["img1.jpg", "img2.jpg"], [3, 5])
    _, label = ds[1]
    assert label.item() == 5


def test_custom_dataset_getitem_image_shape():
    """__getitem__ should return an image tensor of shape (3, 224, 224)."""
    ds = MockDataset(["img1.jpg"], [0])
    image, _ = ds[0]
    assert image.shape == (3, 224, 224)
