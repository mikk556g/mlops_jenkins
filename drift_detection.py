import torch
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader
from PIL import Image
from sklearn.model_selection import train_test_split
import mlflow
import torchdrift
import os
import yaml


mlflow.set_tracking_uri("http://172.24.198.42:5050")
mlflow.set_experiment("lam-resnet50-emotion-classifier")

with open("config/train_config.yaml") as f:
    config = yaml.safe_load(f)

dataset_config = config["dataset"]
classes_to_idx = config["classes"]
dataset_path = dataset_config["dataset_path"]

N_CALIBRATION = 200
N_TEST = 50
P_VALUE_THRESHOLD = 0.05

# Standard transform for calibration and normal test
normal_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# Drifted transform: Gaussian blur simulates poor camera quality
drifted_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.GaussianBlur(kernel_size=23, sigma=(5.0, 10.0)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

image_path_list = []
image_label_list = []

for class_name, class_idx in classes_to_idx.items():
    folder_path = os.path.join(dataset_path, class_name)
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        image_path_list.append(file_path)
        image_label_list.append(class_idx)

# Split data the same way as train.py
train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
    image_path_list,
    image_label_list,
    test_size=dataset_config["test_size"],
    random_state=dataset_config["random_state"],
)
train_paths, _, train_labels, _ = train_test_split(
    train_val_paths,
    train_val_labels,
    test_size=dataset_config["test_size"],
    random_state=dataset_config["random_state"],
)


class SimpleDataset(torch.utils.data.Dataset):
    """Loads images from disk and applies a transform."""
    def __init__(self, paths, transform):
        self.paths = paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        image = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(image)


calibration_dataset = SimpleDataset(train_paths[:N_CALIBRATION], normal_transform)
normal_test_dataset = SimpleDataset(test_paths[:N_TEST], normal_transform)
drifted_test_dataset = SimpleDataset(test_paths[:N_TEST], drifted_transform)

calibration_loader = DataLoader(calibration_dataset, batch_size=32)
normal_test_loader = DataLoader(normal_test_dataset, batch_size=32)
drifted_test_loader = DataLoader(drifted_test_dataset, batch_size=32)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ResNet50 without classification head — extracts 2048-dim features per image
backbone = models.resnet50(pretrained=True)
feature_extractor = torch.nn.Sequential(
    *list(backbone.children())[:-1],
    torch.nn.Flatten()
)
feature_extractor = feature_extractor.to(device)
feature_extractor.eval()


def extract_features(loader):
    """Extracts features from all images in a DataLoader."""
    features = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            feat = feature_extractor(batch)
            features.append(feat.cpu())
    return torch.cat(features, dim=0)


# Fit detector on training data — learns what "normal" looks like
print("Extracting calibration features...")
calibration_features = extract_features(calibration_loader)

print("Fitting drift detector on training data...")
drift_detector = torchdrift.detectors.KernelMMDDriftDetector()
drift_detector.fit(calibration_features)
print("Drift detector fitted.")


def run_drift_test(loader, label):
    """Runs drift detection and returns score, p-value, and drift flag."""
    print(f"\nRunning drift detection on: {label}")
    features = extract_features(loader)
    score = drift_detector(features)
    p_value = drift_detector.compute_p_value(features)
    drift_detected = p_value < P_VALUE_THRESHOLD
    print(f"  Score:          {score:.4f}")
    print(f"  P-value:        {p_value:.4f}")
    print(f"  Drift detected: {drift_detected}")
    return float(score), float(p_value), drift_detected


with mlflow.start_run(run_name="drift_detection"):

    mlflow.log_param("n_calibration_samples", N_CALIBRATION)
    mlflow.log_param("n_test_samples", N_TEST)
    mlflow.log_param("p_value_threshold", P_VALUE_THRESHOLD)
    mlflow.log_param("drift_type_simulated", "GaussianBlur kernel=23")

    # Test 1: normal images — expect no drift
    score_normal, p_normal, drift_normal = run_drift_test(
        normal_test_loader, "Normal test data"
    )
    mlflow.log_metric("normal_mmd_score", score_normal)
    mlflow.log_metric("normal_p_value", p_normal)
    mlflow.log_metric("normal_drift_detected", int(drift_normal))

    # Test 2: blurred images — expect drift
    score_drifted, p_drifted, drift_drifted = run_drift_test(
        drifted_test_loader, "Drifted test data (Gaussian blur)"
    )
    mlflow.log_metric("drifted_mmd_score", score_drifted)
    mlflow.log_metric("drifted_p_value", p_drifted)
    mlflow.log_metric("drifted_drift_detected", int(drift_drifted))

    print(f"\nNormal data  -> drift: {drift_normal} (p={p_normal:.4f})")
    print(f"Drifted data -> drift: {drift_drifted} (p={p_drifted:.4f})")

    mlflow.log_param("normal_data_result", "DRIFT" if drift_normal else "NO DRIFT")
    mlflow.log_param("drifted_data_result", "DRIFT" if drift_drifted else "NO DRIFT")

print("\nDrift detection complete. Results logged to MLflow.")