import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from collections import Counter
import mlflow
import mlflow.pytorch
import time
import yaml
from mlflow.tracking import MlflowClient
from thop import profile
from carbontracker.tracker import CarbonTracker

from src.models.resnet50 import ResNet50FineTuned
from src.optimizers.adamw import adamw
from src.schedulers.reducelronplateau import reducelronplateau
from src.dataset.dataset import CustomDataset

mlflow.set_tracking_uri("http://172.24.198.42:5050")

mlflow.set_experiment("lam-resnet50-emotion-classifier")

with open("config/train_config.yaml") as f:
    config = yaml.safe_load(f)

dataset_config = config["dataset"]
model_config = config["model"]
train_config = config["train"]
optimizer_config = config["optimizer"]
scheduler_config = config["scheduler"]
evaluate_config = config["evaluate"]

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


train_transform = transforms.Compose(
    [
        transforms.RandomResizedCrop(
            size=train_config["RandomResizedCrop"]["size"],
            scale=tuple(train_config["RandomResizedCrop"]["scale"]),
        ),
        transforms.RandomHorizontalFlip(p=train_config["RandomHorizontalFlip"]["p"]),
        transforms.RandomRotation(degrees=train_config["RandomRotation"]["degrees"]),
        transforms.ColorJitter(
            brightness=train_config["ColorJitter"]["brightness"],
            contrast=train_config["ColorJitter"]["contrast"],
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=train_config["Normalize"]["mean"], std=train_config["Normalize"]["std"]
        ),
    ]
)

val_test_transform = transforms.Compose(
    [
        transforms.Resize(size=evaluate_config["Resize"]["size"]),
        transforms.CenterCrop(size=evaluate_config["CenterCrop"]["size"]),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=evaluate_config["Normalize"]["mean"],
            std=evaluate_config["Normalize"]["std"],
        ),
    ]
)


train_set = CustomDataset(train_paths, train_labels, transform=train_transform)
val_set = CustomDataset(val_paths, val_labels, transform=val_test_transform)

class_counts = Counter(train_labels)
num_classes = len(classes_to_idx)
total_samples = len(train_labels)
weights = [total_samples / (num_classes * class_counts[i]) for i in range(num_classes)]

train_dataloader = DataLoader(
    train_set,
    batch_size=train_config["batch_size"],
    shuffle=train_config["shuffle"],
    num_workers=train_config["num_workers"],
    pin_memory=train_config["pin_memory"],
)

val_dataloader = DataLoader(
    val_set,
    batch_size=train_config["batch_size"],
    shuffle=evaluate_config["shuffle"],
    num_workers=train_config["num_workers"],
    pin_memory=evaluate_config["pin_memory"],
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}", flush=True)

model = ResNet50FineTuned(model_config)
model = model.to(device)

class_weights = torch.tensor(weights, dtype=torch.float).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = adamw(model, model_config, optimizer_config)
scheduler = reducelronplateau(optimizer, scheduler_config)

epochs = train_config["epochs"]
early_stopping_patience = train_config["early_stopping_patience"]

train_loss_list = []
val_loss_list = []

# CarbonTracker setup
tracker = CarbonTracker(
    epochs=epochs, log_dir="carbontracker_logs/", verbose=0, components="gpu"
)


with mlflow.start_run(run_name="training"):
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    flops, params = profile(model, inputs=(dummy_input,))
    mlflow.log_metric("model_params", params)
    mlflow.log_metric("model_flops", flops)

    mlflow.log_param("model", model_config["name"])
    mlflow.log_param("batch_size", train_config["batch_size"])
    mlflow.log_param("epochs", epochs)
    mlflow.log_param("optimizer", optimizer_config["name"])
    mlflow.log_param("learning_rate", optimizer_config["lr"])
    mlflow.log_param("scheduler", scheduler_config["name"])
    mlflow.log_param("amp_enabled", True)

    scaler = GradScaler()
    best_val_loss = float("inf")
    epochs_no_improve = 0
    start_time = time.time()

    for epoch in range(epochs):
        tracker.epoch_start()  # ← CarbonTracker start

        print(f"epoch {epoch+1}/{epochs}", flush=True)
        running_train_loss = 0.0
        running_train_corrects = 0.0

        model.train()
        for X_train, y_train in train_dataloader:
            X_train = X_train.to(device)
            y_train = y_train.to(device)

            optimizer.zero_grad()
            with autocast():
                output_train = model(X_train)
                train_loss = criterion(output_train, y_train)
            scaler.scale(train_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            _, preds = torch.max(output_train, 1)
            running_train_corrects += torch.sum(preds == y_train.data).item()
            running_train_loss += train_loss.item()

        train_epoch_acc = running_train_corrects / len(train_set)
        train_epoch_loss = running_train_loss / len(train_dataloader)
        train_loss_list.append(train_epoch_loss)
        print(
            f"Training loss: {train_epoch_loss:.4f} "
            f"Training accuracy: {train_epoch_acc:.4f}",
            flush=True,
        )

        mlflow.log_metric("train_epoch_loss", train_epoch_loss, step=epoch)
        mlflow.log_metric("train_epoch_accuracy", train_epoch_acc, step=epoch)

        model.eval()
        running_val_loss = 0.0
        running_val_corrects = 0.0
        with torch.no_grad():
            for X_val, y_val in val_dataloader:
                X_val = X_val.to(device)
                y_val = y_val.to(device)
                output_val = model(X_val)
                val_loss = criterion(output_val, y_val)
                running_val_loss += val_loss.item()

                _, preds = torch.max(output_val, 1)
                running_val_corrects += torch.sum(preds == y_val.data).item()

        val_epoch_acc = running_val_corrects / len(val_set)
        val_epoch_loss = running_val_loss / len(val_dataloader)
        val_loss_list.append(val_epoch_loss)

        scheduler.step(val_epoch_loss)

        mlflow.log_metric("val_epoch_loss", val_epoch_loss, step=epoch)
        mlflow.log_metric("val_epoch_accuracy", val_epoch_acc, step=epoch)

        print(
            f"Validation loss: {val_epoch_loss:.4f} "
            f"Validation accuracy: {val_epoch_acc:.4f}",
            flush=True,
        )

        if val_epoch_loss < best_val_loss:
            best_val_loss = val_epoch_loss
            epochs_no_improve = 0  # Resets early stop counter
            torch.save(model.state_dict(), "weights_path.pth")
            print(f" New best loss model saved (val_loss {best_val_loss:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        tracker.epoch_end()  # ← CarbonTracker slut

    tracker.stop()  # ← Gem endelig rapport

    # Log CarbonTracker output til MLflow
    if os.path.exists("carbontracker_logs/"):
        mlflow.log_artifacts("carbontracker_logs/", artifact_path="carbontracker")

    end_time = time.time()
    training_duration = end_time - start_time
    mlflow.log_metric("training_duration_seconds", training_duration)
    print(f"Training completed in {training_duration:.2f} seconds", flush=True)

    # Loss plot
    plt_epochs = range(1, len(train_loss_list) + 1)
    plt.plot(plt_epochs, train_loss_list, label="Train Loss")
    plt.plot(plt_epochs, val_loss_list, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.savefig("loss_plot.png", dpi=300, bbox_inches="tight")
    mlflow.log_figure(plt.gcf(), "loss_plot.png")
    plt.close()

    mlflow.log_artifact("MODEL_CARD.md")

    # Log model to MLFlow
    model.load_state_dict(torch.load("weights_path.pth"))
    mlflow.pytorch.log_model(
        pytorch_model=model,
        artifact_path="model",
        registered_model_name="resnet50-emotion-classifier",
    )

    client = MlflowClient()
    run_id = mlflow.active_run().info.run_id

    results = client.search_model_versions(f"run_id='{run_id}'")
    if not results:
        raise RuntimeError(f"No model version found for run_id={run_id}")
    model_version = results[0].version

    client.transition_model_version_stage(
        name="resnet50-emotion-classifier",
        version=model_version,
        stage="Staging",
        archive_existing_versions=True,
    )

    print(f"Model version {model_version} moved to Staging.", flush=True)
