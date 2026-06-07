import sys
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import tempfile
import os
import yaml

sys.path.append("src")  # noqa: E402
from dataset.dataset import CustomDataset  # noqa: E402


# ── Load transforms from config (must match training) ─────────────────────────

with open("config/train_config.yaml") as f:
    config = yaml.safe_load(f)

eval_cfg = config["evaluate"]

val_test_transform = transforms.Compose(
    [
        transforms.Resize(size=eval_cfg["Resize"]["size"]),
        transforms.CenterCrop(size=eval_cfg["CenterCrop"]["size"]),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=eval_cfg["Normalize"]["mean"],
            std=eval_cfg["Normalize"]["std"],
        ),
    ]
)


# ── Helpers ───────────────────────────────────────────────────────────────────


def make_fake_image(path, size=(100, 100)):
    """Create a small synthetic RGB image and save it to disk."""
    arr = np.random.randint(0, 255, (*size, 3), dtype=np.uint8)
    img = Image.fromarray(arr, mode="RGB")
    img.save(path)


def make_dataset(num_images, transform=None):
    """Instantiate a real CustomDataset backed by temporary fake images."""
    tmpdir = tempfile.mkdtemp()
    paths = []
    labels = []
    for i in range(num_images):
        path = os.path.join(tmpdir, f"img_{i}.jpg")
        make_fake_image(path)
        paths.append(path)
        labels.append(i % 7)
    return CustomDataset(paths, labels, transform=transform)


# ── Tests ─────────────────────────────────────────────────────────────────────


def test_dataset_loads_from_disk():
    """CustomDataset should read a real image file via PIL, not return zeros."""
    ds = make_dataset(1)
    image, _ = ds[0]
    assert isinstance(image, Image.Image)


def test_dataset_correct_label():
    """CustomDataset should return the correct label as a long tensor."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "img.jpg")
        make_fake_image(path)
        ds = CustomDataset([path], [4])
        _, label = ds[0]
        assert label.item() == 4
        assert label.dtype == torch.long


def test_transform_output_shape():
    """After val/test transforms, images should be (3, 224, 224)."""
    ds = make_dataset(3, transform=val_test_transform)
    image, _ = ds[0]
    assert image.shape == (3, 224, 224)


def test_dataloader_batch_shape():
    """DataLoader should produce correctly shaped batches."""
    ds = make_dataset(8, transform=val_test_transform)
    loader = DataLoader(ds, batch_size=4, shuffle=False)
    images, labels = next(iter(loader))
    assert images.shape == (4, 3, 224, 224)
    assert labels.shape == (4,)
