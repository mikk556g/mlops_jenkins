Opdateret med begge datasæt:

```markdown
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

| Metric | Value |
|---|---|
| Validation Accuracy | 69.9% |
| Training Accuracy | 70.2% |
| Test Accuracy (FP32) | 70.4% |
| Test Accuracy (FP16) | 70.3% |
| FP32 Inference Duration | 13.36 sec (~5,742 images) |
| CO2 Training | 4.858 g CO2eq |

## MLOps Pipeline

The model is trained and versioned through an automated Jenkins CI/CD pipeline:

- Data versioning via DVC with MinIO S3 storage
- Experiment tracking and model registry via MLflow
- FP32 vs FP16 evaluation — best version promoted to Production
- Data drift detection via TorchDrift (Kernel MMD)
- ONNX export (opset 11) logged to MLflow
- Carbon emission tracking via CarbonTracker

## Limitations and Biases

- FER2013 is known to contain labeling noise, which may affect model reliability
- Performance may vary across demographic groups due to dataset composition
- The model is trained on static images and may not generalize to video frames or non-frontal faces
- Emotion recognition is inherently subjective and culturally dependent

## Ethical Considerations

Facial emotion recognition systems carry significant ethical risks, including potential misuse for surveillance or discrimination. This model is intended strictly for academic purposes and should not be deployed in contexts where automated emotion inference could harm individuals.
```