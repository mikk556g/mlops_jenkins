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
datasets:
- fer2013
- raf-db
metrics:
- accuracy
co2_eq_emissions:
  emissions: 4.858 g CO2eq
  source: CarbonTracker
  training_type: fine-tuning
  geographic_location: Denmark
  hardware_used: NVIDIA RTX 4000 Ada
model-index:
- name: ResNet50 FER Emotion Classifier
  results:
  - task:
      type: image-classification
      name: Image Classification
    dataset:
      name: FER2013 & RAF-DB
      type: fer2013
    metrics:
    - type: accuracy
      name: Validation Accuracy
      value: 0.699
    - type: accuracy
      name: Test Accuracy (FP32)
      value: 0.704
---

# Model Card: ResNet50 FER Emotion Classifier

## Model Description

This model is a fine-tuned ResNet50 trained to classify facial expressions into seven emotion categories. It was developed as part of an MLOps course project at Aalborg University.

- **Model type:** Convolutional Neural Network (ResNet50, fine-tuned)
- **Task:** Facial Emotion Recognition (multi-class classification)
- **Input:** RGB facial images, 224×224 pixels
- **Output:** Probability distribution over 7 emotion classes
- **Framework:** PyTorch

## Training Data

The model was trained on a combination of the FER2013 and RAF-DB datasets, both publicly available benchmarks for facial emotion recognition. FER2013 consists of 48×48 pixel grayscale images converted to RGB, while RAF-DB contains real-world RGB facial images. Together they provide approximately 49,000 images across seven emotion categories. Data augmentation and class balancing were applied during preprocessing to address class imbalance.

- **Datasets:** FER2013 & RAF-DB (~49,000 images combined)
- **Classes:** Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise
- **Train preprocessing:** RandomResizedCrop to 224px, RandomHorizontalFlip, RandomRotation, ColorJitter, Normalize (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
- **Val/test preprocessing:** Resize to 256px, CenterCrop to 224px, Normalize (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
- **Split:** 80% training, 20% validation

## Architecture

ResNet50 pre-trained on ImageNet with a replaced classification head:

| Component | Details |
|---|---|
| Backbone | ResNet50 (ImageNet pre-trained) |
| Classification Head | Linear(2048 → 7) |
| Loss | CrossEntropyLoss with inverse class-frequency weights |
| Optimizer | AdamW |
| Scheduler | OneCycleLR |
| Total Parameters | ~25M |

## Performance

### Accuracy

| Metric | Value |
|---|---|
| Validation Accuracy | 69.9% |
| Training Accuracy | 70.2% |
| Test Accuracy (FP32) | 70.4% |
| Test Accuracy (FP16) | 70.3% |

### Inference Benchmarks (NVIDIA RTX 4000 Ada, FP16)

| Metric | Value |
|---|---|
| FP32 Latency (batch=1) | 1.088 ms/sample |
| FP16 Latency (batch=1) | 0.529 ms/sample |
| Peak Throughput (batch=8) | 2120.6 samples/sec |
| Throughput Saturation Point | batch_size=8 |
| FP32 Inference Duration (~5,742 images) | 13.36 sec |

### Carbon Footprint

| Metric | Value |
|---|---|
| Training CO₂ (4 epochs) | 4.858 g CO₂eq |
| Evaluation CO₂ | 0.099 g CO₂eq |
| Total per pipeline run | ~4.96 g CO₂eq |
| Yearly estimate (100 img/day, weekly retraining) | ~252.9 g CO₂eq ≈ 2.4 km by car |
| Carbon intensity used | 143 gCO₂eq/kWh (Danish grid) |

## MLOps Pipeline

The model is trained and versioned through an automated Jenkins CI/CD pipeline:

- **CI/CD:** Jenkins with SCM polling every 5 minutes (webhook not supported on AAU network)
- **Data versioning:** DVC with MinIO S3 storage
- **Experiment tracking:** MLflow (metrics, artifacts, model registry)
- **Model promotion:** FP32 vs FP16 evaluated; best version promoted to Production
- **Quantization:** FP16 via PyTorch AMP — 2× latency improvement, negligible accuracy loss
- **Drift detection:** TorchDrift (Kernel MMD), calibrated on 200 training images
- **Export:** ONNX (opset 11) logged to MLflow
- **Carbon tracking:** CarbonTracker logs per-epoch GPU power consumption to MLflow
- **Monitoring:** Prometheus + Grafana — p50/p90 latency, request rate, memory, CPU usage; all requests served within 100ms (average response time 7.39 ms)

### Distributed Training (experimental)

Distributed training was evaluated on the AAU AI-Lab cluster (NVIDIA L4 GPUs):

| Config | Throughput (img/s) | VRAM (MB) | Speedup |
|---|---|---|---|
| 1-GPU FP32 | 76.2 | 2,906 | 1.00× |
| 1-GPU AMP (FP16) | 86.5 | 1,669 | 1.14× |
| DDP 2-GPU + AMP | 234.6 | 1,826 | 2.71× |
| DDP 4-GPU + AMP | 263.4 | 1,829 | 3.44× |

AMP reduces VRAM by 43% with negligible accuracy cost (−0.6%). Multi-node DDP (4 GPUs across 2 nodes) was 1.88× slower than single-node due to network bottleneck. DeepSpeed ZeRO increased VRAM usage for this model size and is not recommended. Production pipeline uses single-GPU + AMP as the optimal cost/complexity trade-off.

### Pruning (experimental)

Unstructured L1 pruning was evaluated without finetuning. Accuracy degrades sharply above 50% sparsity. At 90% sparsity, finetuning (5 epochs, AdamW lr=1e-5) recovered accuracy from 14.01% to 65.36%, demonstrating that aggressive compression is feasible with recovery finetuning.

## Limitations and Biases

- FER2013 is known to contain labeling noise, which may affect model reliability
- Performance may vary across demographic groups due to dataset composition
- The model is trained on static images and may not generalize to video frames or non-frontal faces
- Emotion recognition is inherently subjective and culturally dependent

## Ethical Considerations

Facial emotion recognition systems carry significant ethical risks, including potential misuse for surveillance or discrimination. This model is intended strictly for academic purposes and should not be deployed in contexts where automated emotion inference could harm individuals.
