import sys
import os
import glob
import torch
import yaml
import pytest
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from PIL import Image
sys.path.append("src")  


# Load config 

def load_config(path="config/train_config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


CONFIG = load_config()
DATA_PATH = CONFIG["dataset"]["dataset_path"]  
CLASSES = CONFIG["classes"]                     
NUM_CLASSES = CONFIG["model"]["num_classes"]    

# Skip all data tests if DVC data has not been pulled yet
DATA_AVAILABLE = any(
    os.path.isdir(os.path.join(DATA_PATH, c)) for c in CLASSES
)
pytestmark = pytest.mark.skipif(
    not DATA_AVAILABLE,
    reason=f"Dataset not found under '{DATA_PATH}/' — run 'dvc pull' first.",
)


# Inline CustomDataset

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


# Build real image paths + labels from data

def get_real_image_paths_and_labels(max_per_class=10):
    """Collect up to max_per_class images per class from data/.
    Returns (img_paths, img_labels)."""
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


# Transforms

val_test_transform = transforms.Compose(
    [
        transforms.Resize(CONFIG["evaluate"]["Resize"]["size"]),
        transforms.CenterCrop(CONFIG["evaluate"]["CenterCrop"]["size"]),
        transforms.ToTensor(),
        transforms.Normalize(
            CONFIG["evaluate"]["Normalize"]["mean"],
            CONFIG["evaluate"]["Normalize"]["std"],
        ),
    ]
)

train_transform = transforms.Compose(
    [
        transforms.RandomResizedCrop(
            CONFIG["train"]["RandomResizedCrop"]["size"],
            scale=CONFIG["train"]["RandomResizedCrop"]["scale"],
        ),
        transforms.RandomHorizontalFlip(CONFIG["train"]["RandomHorizontalFlip"]["p"]),
        transforms.RandomRotation(CONFIG["train"]["RandomRotation"]["degrees"]),
        transforms.ColorJitter(
            brightness=CONFIG["train"]["ColorJitter"]["brightness"],
            contrast=CONFIG["train"]["ColorJitter"]["contrast"],
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            CONFIG["train"]["Normalize"]["mean"],
            CONFIG["train"]["Normalize"]["std"],
        ),
    ]
)


# Fixtures 

IMG_PATHS, IMG_LABELS = get_real_image_paths_and_labels(max_per_class=10)


# Dataset sanity 

def test_dataset_not_empty():
    """data/ should contain images for all 7 classes."""
    assert len(IMG_PATHS) > 0, (
        f"No images found under '{DATA_PATH}/'. "
        "Make sure the dataset is present."
    )


def test_dataset_covers_all_classes():
    """Every class index (0-6) should appear at least once."""
    found = set(IMG_LABELS)
    expected = set(CLASSES.values())
    assert found == expected, f"Missing classes: {expected - found}"


def test_dataset_len():
    """CustomDataset should report correct length."""
    ds = CustomDataset(IMG_PATHS, IMG_LABELS, transform=val_test_transform)
    assert len(ds) == len(IMG_PATHS)


def test_labels_are_integers():
    """Labels should be long tensors."""
    ds = CustomDataset(IMG_PATHS, IMG_LABELS, transform=val_test_transform)
    _, label = ds[0]
    assert label.dtype == torch.long


def test_label_range():
    """All labels should be valid class indices (0 to num_classes-1)."""
    ds = CustomDataset(IMG_PATHS, IMG_LABELS, transform=val_test_transform)
    for _, label in ds:
        assert 0 <= label.item() < NUM_CLASSES


# Transform tests 

def test_val_transform_output_size():
    """val_test_transform should produce (3, 224, 224) tensors."""
    ds = CustomDataset(IMG_PATHS, IMG_LABELS, transform=val_test_transform)
    image, _ = ds[0]
    assert image.shape == (3, 224, 224)


def test_train_transform_output_size():
    """train_transform should produce (3, 224, 224) tensors."""
    ds = CustomDataset(IMG_PATHS, IMG_LABELS, transform=train_transform)
    image, _ = ds[0]
    assert image.shape == (3, 224, 224)


def test_transform_output_is_tensor():
    """Transform output should be a torch.Tensor."""
    ds = CustomDataset(IMG_PATHS, IMG_LABELS, transform=val_test_transform)
    image, _ = ds[0]
    assert isinstance(image, torch.Tensor)


def test_transform_three_channels():
    """Images should have 3 channels (RGB)."""
    ds = CustomDataset(IMG_PATHS, IMG_LABELS, transform=val_test_transform)
    image, _ = ds[0]
    assert image.shape[0] == 3


# DataLoader tests 

def test_dataloader_batch_shape():
    """A batch should have shape (batch_size, 3, 224, 224)."""
    ds = CustomDataset(IMG_PATHS, IMG_LABELS, transform=val_test_transform)
    loader = DataLoader(ds, batch_size=4, shuffle=False)
    images, labels = next(iter(loader))
    assert images.shape[1:] == (3, 224, 224)


def test_dataloader_label_range():
    """All labels in a batch should be valid class indices."""
    ds = CustomDataset(IMG_PATHS, IMG_LABELS, transform=val_test_transform)
    loader = DataLoader(ds, batch_size=len(ds), shuffle=False)
    _, labels = next(iter(loader))
    assert labels.min().item() >= 0
    assert labels.max().item() < NUM_CLASSES


def test_dataloader_iterates_without_error():
    """DataLoader should iterate all batches without raising."""
    ds = CustomDataset(IMG_PATHS, IMG_LABELS, transform=val_test_transform)
    loader = DataLoader(ds, batch_size=4, shuffle=False)
    for images, labels in loader:
        assert images is not None
        assert labels is not None
