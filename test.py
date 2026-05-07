import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sea
import os
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient
import yaml
import time

mlflow.set_tracking_uri("http://172.24.198.42:5050")
mlflow.set_experiment("lam-resnet50-emotion-classifier")

with open("config/test_config.yaml") as f:
    config = yaml.safe_load(f)

dataset_config = config["dataset"]

tranforms_config = config["transforms"]

dataset_path = dataset_config["dataset_path"]


classes_to_idx = config["classes"]


image_path_list = []
image_label_list = []


for class_name, class_idx in classes_to_idx.items():
    folder_path = os.path.join(dataset_path, class_name)
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        image_path_list.append(file_path)
        image_label_list.append(class_idx)


train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
    image_path_list,
    image_label_list,
    test_size=dataset_config["test_size"],
    random_state=dataset_config["random_state"],
)
train_paths, val_paths, train_labels, val_labels = train_test_split(
    train_val_paths,
    train_val_labels,
    test_size=dataset_config["test_size"],
    random_state=dataset_config["random_state"],
)


val_test_transform = transforms.Compose(
    [
        transforms.Resize(size=tranforms_config["Resize"]["size"]),
        transforms.CenterCrop(size=tranforms_config["CenterCrop"]["size"]),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=tranforms_config["Normalize"]["mean"],
            std=tranforms_config["Normalize"]["std"],
        ),
    ]
)


class CustomDataset(Dataset):
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


test_set = CustomDataset(test_paths, test_labels, transform=val_test_transform)

test_dataloader = DataLoader(
    test_set,
    batch_size=config["batch_size"],
    shuffle=config["shuffle"],
    num_workers=config["num_workers"],
    pin_memory=config["pin_memory"],
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")


with mlflow.start_run(run_name="evaluation in FP32 and FP16"):

    client = MlflowClient()

    mlflow.log_param("device", str(device))

    # ------- FINDING LATEST STAGING MODEL VERSION ------- #
    staging_versions = client.get_latest_versions(
        "resnet50-emotion-classifier", stages=["Staging"]
    )
    if not staging_versions:
        raise ValueError("No model found in Staging.")

    staging_version = staging_versions[0].version
    print(f"Loading Staging model version: {staging_version}")

    mlflow.log_param("evaluated_staging_version", staging_version)

    # ------- TESTING MODEL IN FP32 ------- #
    model_32 = mlflow.pytorch.load_model(
        f"models:/resnet50-emotion-classifier/{staging_version}"
    )
    model_32 = model_32.to(device)
    model_32.eval()

    all_preds_fp32 = []
    all_labels_fp32 = []

    start_time = time.time()

    with torch.no_grad():
        for X_test, y_test in test_dataloader:
            X_test = X_test.to(device)
            y_test = y_test.to(device)
            output = model_32(X_test)
            _, preds = torch.max(output, 1)
            all_preds_fp32.extend(preds.cpu().numpy())
            all_labels_fp32.extend(y_test.cpu().numpy())

    end_time = time.time()
    fp32_inference_duration = end_time - start_time
    mlflow.log_metric("fp32_inference_duration_seconds", fp32_inference_duration)
    print(
        f"FP32 inference completed in {fp32_inference_duration:.2f} seconds", flush=True
    )

    report_fp32 = classification_report(
        all_labels_fp32,
        all_preds_fp32,
        target_names=classes_to_idx.keys(),
        output_dict=True,
    )
    fp32_accuracy = report_fp32["accuracy"]
    mlflow.log_metric("fp32_test_accuracy", fp32_accuracy)
    print(f"FP32 test Accuracy: {fp32_accuracy}")

    for class_name in classes_to_idx.keys():
        mlflow.log_metric(f"fp32_{class_name}_f1", report_fp32[class_name]["f1-score"])
        mlflow.log_metric(
            f"fp32_{class_name}_precision", report_fp32[class_name]["precision"]
        )
        mlflow.log_metric(
            f"fp32_{class_name}_recall", report_fp32[class_name]["recall"]
        )

    # ------- TESTING MODEL IN FP16 ------- #
    model_16 = mlflow.pytorch.load_model(
        f"models:/resnet50-emotion-classifier/{staging_version}"
    )
    model_16 = model_16.to(device).half()
    model_16.eval()

    all_preds_fp16 = []
    all_labels_fp16 = []

    start_time = time.time()

    with torch.no_grad():
        for X_test, y_test in test_dataloader:
            X_test = X_test.to(device).half()
            y_test = y_test.to(device)
            output = model_16(X_test)
            _, preds = torch.max(output, 1)
            all_preds_fp16.extend(preds.cpu().numpy())
            all_labels_fp16.extend(y_test.cpu().numpy())

    end_time = time.time()
    fp16_inference_duration = end_time - start_time
    mlflow.log_metric("fp16_inference_duration_seconds", fp16_inference_duration)
    print(
        f"FP16 inference completed in {fp16_inference_duration:.2f} seconds", flush=True
    )

    report_fp16 = classification_report(
        all_labels_fp16,
        all_preds_fp16,
        target_names=classes_to_idx.keys(),
        output_dict=True,
    )
    fp16_accuracy = report_fp16["accuracy"]
    mlflow.log_metric("fp16_test_accuracy", fp16_accuracy)
    print(f"FP16 test Accuracy: {fp16_accuracy}")

    for class_name in classes_to_idx.keys():
        mlflow.log_metric(f"fp16_{class_name}_f1", report_fp16[class_name]["f1-score"])
        mlflow.log_metric(
            f"fp16_{class_name}_precision", report_fp16[class_name]["precision"]
        )
        mlflow.log_metric(
            f"fp16_{class_name}_recall", report_fp16[class_name]["recall"]
        )

    # ------- SELECTING THE BEST MODEL ------- #
    threshold = config["accuracy_threshold"]

    if fp32_accuracy < threshold and fp16_accuracy < threshold:
        raise ValueError(
            f"Both models below accuracy threshold ({threshold})."
            f"FP32: {fp32_accuracy:.4f}, FP16: {fp16_accuracy:.4f}"
        )

    if fp16_accuracy > threshold and fp16_accuracy > fp32_accuracy:
        best_model = model_16
        best_type = "fp16"
        best_acc = fp16_accuracy
        best_labels = all_labels_fp16
        best_preds = all_preds_fp16
    else:
        best_model = model_32
        best_type = "fp32"
        best_acc = fp32_accuracy
        best_labels = all_labels_fp32
        best_preds = all_preds_fp32

    print(f"Selected best model: {best_type} with accuracy {best_acc}", flush=True)

    # ------- CONFUSION MATRIX FOR BEST MODEL ------- #
    cm = confusion_matrix(best_labels, best_preds)
    plt.figure(figsize=(8, 6))
    sea.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=classes_to_idx.keys(),
        yticklabels=classes_to_idx.keys(),
    )

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Best Model Confusion Matrix")

    plt.tight_layout()
    plt.savefig("best_model_confusion_matrix.png")
    mlflow.log_artifact("best_model_confusion_matrix.png")
    plt.close()

    # ------- PROMOTE BEST MODEL TO PRODUCTION ------- #
    best_model = best_model.to("cpu")

    model_info = mlflow.pytorch.log_model(
        best_model,
        artifact_path="model",
        registered_model_name="resnet50-emotion-classifier",
    )

    # model_info.registered_model_version is the exact version just created
    new_version = model_info.registered_model_version
    print(f"Newly registered model version: {new_version} ({best_type})")

    client.transition_model_version_stage(
        name="resnet50-emotion-classifier",
        version=new_version,
        stage="Production",
        archive_existing_versions=True,
    )

    # Tag the version with evaluation metadata for traceability
    client.set_model_version_tag(
        name="resnet50-emotion-classifier",
        version=new_version,
        key="precision_type",
        value=best_type,
    )
    client.set_model_version_tag(
        name="resnet50-emotion-classifier",
        version=new_version,
        key="test_accuracy",
        value=str(round(best_acc, 4)),
    )

    print(f"Version {new_version} ({best_type}) promoted to Production.", flush=True)
