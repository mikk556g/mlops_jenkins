---
language:
  - en
license: mit
library_name: pytorch
pipeline_tag: image-classification
tags:
  - facial-emotion-recognition
  - pytorch
  - resnet50
  - computer-vision
  - deepspeed
datasets:
  - fer2013
metrics:
  - accuracy
  - f1
co2_eq_emissions:
  emissions: TBD
  source: CarbonTracker
  training_type: pre-training
  geographic_location: Denmark
  hardware_used: NVIDIA GPU
model-index:
  - name: ResNet50 FER Emotion Classifier
    results:
      - task:
          type: image-classification
          name: Image Classification
        dataset:
          name: FER2013
          type: fer2013
        metrics:
          - type: accuracy
            name: Validation Accuracy
            value: 0.151
          - type: accuracy
            name: Training Accuracy
            value: 0.173
---

# Model Card: ResNet50 FER Emotion Classifier

## Model Description

A fine-tuned ResNet50 model trained to classify facial expressions into seven emotion categories. Developed as part of an MLOps course project at Aalborg University, with a full CI/CD pipeline covering training, evaluation, drift detection, and ONNX export.

- **Model type:** Convolutional Neural Network (ResNet50, fine-tuned)
- **Task:** Facial Emotion Recognition — 7-class image classification
- **Input:** RGB facial images, 224×224 pixels
- **Output:** Probability distribution over 7 emotion classes
- **Framework:** PyTorch
- **Developed by:** LAM (Aalborg University MLOps group)
- **License:** MIT

## Uses

### Direct Use

The model classifies a facial image into one of seven discrete emotion categories. It can be used as a standalone classifier given a pre-cropped and aligned frontal face image.

```python
import torch
import mlflow.pytorch
from torchvision import transforms
from PIL import Image

model = mlflow.pytorch.load_model("models:/resnet50-emotion-classifier/<version>")
model.eval()

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

image = Image.open("face.jpg").convert("RGB")
input_tensor = transform(image).unsqueeze(0)

with torch.no_grad():
    logits = model(input_tensor)
    predicted_class = logits.argmax(-1).item()
```

### Downstream Use

The model can be used as a feature extractor in larger affective computing pipelines, or further fine-tuned on domain-specific datasets. An ONNX export and TensorRT FP16 engine are also produced by the pipeline for optimized inference.

### Out-of-Scope Use

- Should **not** be used as a sole decision-maker in high-stakes applications (hiring, healthcare, law enforcement) without human oversight.
- Not intended for surveillance or real-time mass monitoring.
- Performance is not guaranteed on non-frontal faces, occluded images, or video frames.
- Emotion recognition is inherently subjective and culturally dependent.

## Bias, Risks, and Limitations

- **Dataset bias:** FER2013 is known to contain labeling noise and demographic imbalances that may affect performance across groups.
- **Domain shift:** Performance may degrade on images outside the training distribution (e.g., extreme lighting, non-frontal poses, drawings).
- **Class imbalance:** Some emotion classes (e.g., Disgust) are significantly underrepresented in FER2013.
- **Adversarial robustness:** Like most CNNs, the model is not robust to adversarial perturbations.
- **Subjectivity:** Emotion labels are inherently ambiguous and annotator-dependent.

### Recommendations

Evaluate per-class precision, recall, and F1 before any real-world deployment. Apply confidence thresholding and fallback mechanisms in production. Audit outputs across demographic subgroups if the task involves diverse populations.

## Training Data

The model was trained on [FER2013](https://www.kaggle.com/datasets/msambare/fer2013), a publicly available benchmark dataset for facial emotion recognition. It contains 48×48 pixel grayscale images, here converted to RGB for compatibility with the ResNet50 backbone.

- **Classes:** Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise
- **Preprocessing:** Resize to 256px, RandomResizedCrop to 224px, RandomHorizontalFlip, RandomRotation, ColorJitter, Normalize (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
- **Split:** 80% training, 20% validation (stratified); separate held-out test split

Data versioning is managed via **DVC** with a MinIO S3-compatible remote.

## Architecture

ResNet50 pre-trained on ImageNet, with the classification head replaced for 7-class output:

| Component | Details |
|---|---|
| Backbone | ResNet50 (ImageNet pre-trained) |
| Classification Head | Linear(2048 → 7) |
| Loss | CrossEntropyLoss with class-frequency weights |
| Optimizer | AdamW |
| Scheduler | OneCycleLR |
| Total Parameters | ~25M (ResNet50 standard) |

## Training Procedure

### Hyperparameters

Loaded from `config/train_config.yaml`. Key settings:

- **Optimizer:** AdamW
- **Scheduler:** OneCycleLR
- **Data augmentation:** RandomResizedCrop, RandomHorizontalFlip, RandomRotation, ColorJitter
- **Class weighting:** Inverse frequency weighting via CrossEntropyLoss

### Infrastructure

Training is run inside Docker containers on a GPU worker node via a **Jenkins CI/CD pipeline** triggered on every git commit (polling every 5 minutes). The full pipeline stages are:

1. Build Docker image
2. Pull dataset via DVC
3. Run unit tests (pytest)
4. Train model → register to MLflow Staging
5. Evaluate in FP32 and FP16 → best version promoted to Production
6. Drift detection (KernelMMD via TorchDrift)
7. Export to ONNX (opset 11)
8. Push Docker image to private registry

Carbon emissions are tracked during training and evaluation using **CarbonTracker**, with logs uploaded as MLflow artifacts.

## Evaluation

### Testing Data

A held-out test split (same stratified split as training, `random_state` fixed). Evaluation is run in both FP32 and FP16 precision; the better-performing variant is promoted to Production.

### Metrics

- **Accuracy** (overall)
- **Per-class F1, Precision, Recall** (logged to MLflow for all 7 classes)
- **Inference duration** (FP32 vs FP16)
- **Confusion matrix** (logged as artifact)

### Results

| Metric | Value |
|---|---|
| Validation Accuracy | 15.1% |
| Training Accuracy | 17.3% |

> ⚠️ These results reflect an early training run. The low accuracy is consistent with known challenges of FER2013 (high labeling noise, class imbalance) and may improve with extended training or architecture adjustments. The Naive Baseline on noisy data with 7 classes has a random-chance ceiling of ~14.3%, so the model does learn signal — but marginally.

### Promotion Threshold

A configurable `accuracy_threshold` in `config/test_config.yaml` gates Production promotion. If both FP32 and FP16 fall below the threshold, the pipeline raises an error and does not promote.

## Data Drift Detection

The pipeline includes an automated drift detection stage using **TorchDrift** with a Kernel MMD detector:

- **Calibration:** 200 training samples (normal distribution)
- **Test:** 50 samples — one normal batch, one artificially drifted (Gaussian blur, kernel=23)
- **Threshold:** p-value < 0.05 → drift detected
- Results logged to MLflow under `drift_detection` run

## ONNX Export

After Production promotion, the model is exported to ONNX (opset 11) and the artifact is logged back to the same MLflow run that produced the version. A TensorRT FP16 engine (`.engine`) is also generated for optimized GPU inference.

## Environmental Impact

Carbon emissions are estimated using [CarbonTracker](https://github.com/lfwa/carbontracker) and reported per training/evaluation run in MLflow.

- **Hardware:** NVIDIA GPU (GPU Worker 1)
- **Location:** Denmark
- **Emissions:** TBD — logged automatically per run via CarbonTracker

## Technical Specifications

### Model Architecture and Objective

ResNet50 with compound residual blocks pre-trained on ImageNet-1K. The final fully connected layer is replaced with a 7-class linear head. Trained with cross-entropy loss weighted by inverse class frequency to compensate for FER2013 class imbalance.

- **Input:** RGB image, 224×224 px
- **Output:** Softmax probability distribution over 7 emotion classes

### Compute Infrastructure

| Component | Details |
|---|---|
| CI/CD | Jenkins (SCM polling every 5 min) |
| Containerization | Docker (image tagged per git commit hash) |
| Experiment Tracking | MLflow (self-hosted, `http://172.24.198.42:5050`) |
| Model Registry | MLflow Model Registry (Staging → Production lifecycle) |
| Data Versioning | DVC + MinIO S3 remote |
| Inference Formats | PyTorch, ONNX (opset 11), TensorRT FP16 |

### Software

- Python 3.x
- PyTorch
- torchvision
- MLflow 2.10.2
- TorchDrift
- CarbonTracker
- scikit-learn 1.5.2
- ONNX 1.15.0
- TensorRT (cu12) 10.6.0

## Model Card Authors

LAM — MLOps course project group, Aalborg University

## Model Card Contact

See project repository for contact details.
