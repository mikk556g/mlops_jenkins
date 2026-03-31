import sys
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import tempfile
import os

sys.path.append("src")  # noqa: E402


# ── Helpers


def make_fake_image(path, size=(100, 100)):
    """Create a small synthetic RGB image
    and save it to disk."""
    arr = np.random.randint(0, 255, (*size, 3), dtype=np.uint8)
    img = Image.fromarray(arr, mode="RGB")
    img.save(path)


val_test_transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

train_transform = transforms.Compose(
    [
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


class MockDataset:
    """Dataset that creates real temporary image files for testing."""

    def __init__(self, num_images, transform=None):
        self.tmpdir = tempfile.mkdtemp()
        self.img_paths = []
        self.img_labels = []
        self.transform = transform

        for i in range(num_images):
            path = os.path.join(self.tmpdir, f"img_{i}.jpg")
            make_fake_image(path)
            self.img_paths.append(path)
            self.img_labels.append(i % 7)

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        image = Image.open(self.img_paths[idx]).convert("RGB")
        label = torch.tensor(self.img_labels[idx], dtype=torch.long)
        if self.transform:
            image = self.transform(image)
        return image, label


# ── Transform tests


def test_val_transform_output_size():
    """val_test_transform should produce 224x224 images."""
    ds = MockDataset(3, transform=val_test_transform)
    image, _ = ds[0]
    assert image.shape == (3, 224, 224)


def test_train_transform_output_size():
    """train_transform should produce 224x224 images."""
    ds = MockDataset(3, transform=train_transform)
    image, _ = ds[0]
    assert image.shape == (3, 224, 224)


def test_transform_output_is_tensor():
    """Transform should return a torch.Tensor."""
    ds = MockDataset(3, transform=val_test_transform)
    image, _ = ds[0]
    assert isinstance(image, torch.Tensor)


def test_transform_three_channels():
    """Images should have 3 channels (RGB)."""
    ds = MockDataset(3, transform=val_test_transform)
    image, _ = ds[0]
    assert image.shape[0] == 3


# ── DataLoader tests


def test_dataloader_batch_shape():
    """A batch from DataLoader should have shape
    (batch_size, 3, 224, 224)."""
    ds = MockDataset(8, transform=val_test_transform)
    loader = DataLoader(ds, batch_size=4, shuffle=False)
    images, labels = next(iter(loader))
    assert images.shape == (4, 3, 224, 224)


def test_dataloader_label_range():
    """All labels should be valid class indices
    between 0 and 6."""
    ds = MockDataset(8, transform=val_test_transform)
    loader = DataLoader(ds, batch_size=8, shuffle=False)
    _, labels = next(iter(loader))
    assert labels.min().item() >= 0
    assert labels.max().item() <= 6


def test_dataloader_iterates_without_error():
    """DataLoader should iterate through all batches without raising."""
    ds = MockDataset(10, transform=val_test_transform)
    loader = DataLoader(ds, batch_size=4, shuffle=False)
    for images, labels in loader:
        assert images is not None
        assert labels is not None


def test_dataset_not_empty():
    """Dataset should contain at least one sample."""
    ds = MockDataset(5, transform=val_test_transform)
    assert len(ds) > 0


def test_labels_are_integers():
    """Labels should be long tensors (integers)."""
    ds = MockDataset(4, transform=val_test_transform)
    _, label = ds[0]
    assert label.dtype == torch.long
