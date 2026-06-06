# 2. Continuous ML

In DevOps, multiple branches are used to separate development activities from the production-ready main branch, preventing faulty code from being deployed. MLOps adopts the same principle, as errors in production ML systems can lead to downtime or unreliable predictions. In this project, two branches were created: one for unit test development and one for general feature development, as shown in Fig. X. Branch protection was activated on the main branch to avoid force pushes, mitigating the risk of unwanted repository behaviour or merging buggy code into the protected branch.

To catch simple issues early, local pre-commit hooks were implemented using the `pre-commit` framework. These hooks run predefined steps before a commit is accepted, functioning similarly to unit tests by ensuring that early checks fail fast. The configured hooks include: (1) checking Python code style and syntax errors with Flake8, (2) scanning for hardcoded API keys and secrets using `detect-secrets`, preventing credentials from accidentally entering version history, and (3) rejecting commits containing files above a defined size limit, ensuring that large model files or datasets are handled via DVC rather than committed directly.

## 2.1 CI/CD Pipeline Overview

The implemented Continuous ML pipeline is orchestrated by Jenkins on the AAU MLOps cluster and consists of eight stages, as illustrated in Fig. X. The pipeline triggers automatically on each push via SCM polling every five minutes — webhook-based triggering is not supported on the AAU network.

**Stage 1 — Build Docker Image.** Jenkins builds a Docker image from the repository's `Dockerfile`, which installs all Python dependencies from `requirements.txt`. The image is tagged with the first seven characters of the Git commit hash, directly linking the container to the source code that produced it, ensuring full reproducibility.

**Stage 2 — Pull Dataset.** The FER2013 and RAF-DB datasets are versioned with DVC, using a MinIO S3 instance on the AAU cluster as the remote storage backend. Running `dvc pull` inside the container fetches the exact dataset version recorded in `Dataset.dvc`, ensuring that any past run can be reproduced with the same data.

**Stage 3 — Run Unit Tests.** pytest is executed inside the container. If any test fails, the pipeline is aborted immediately and the Docker image is not pushed to the registry, preventing untested code from propagating further. Code coverage is measured using the `coverage` package and is reported per run.

**Stage 4 — Train Model.** Training runs in the Docker container with GPU access (`--gpus 1`) via `train.py`. All hyperparameters, per-epoch training and validation loss, accuracy, model parameters, FLOPs, training duration, and CarbonTracker energy measurements are logged to MLflow under the experiment `lam-resnet50-emotion-classifier`. The best model checkpoint (highest validation accuracy) is registered in the MLflow Model Registry under `resnet50-emotion-classifier` and placed in **Staging**. The MODEL_CARD.md is logged as an MLflow artifact, linking the model's characteristics directly to the run that produced it.

**Stage 5 — Evaluate Model in FP32 and FP16.** `test.py` loads the latest Staging model version from MLflow and evaluates it on the held-out test set in both FP32 and FP16 precision. Per-class precision, recall, and F1-score are logged to MLflow for both variants. If neither model version exceeds the configured accuracy threshold (`accuracy_threshold` in `test_config.yaml`), the pipeline fails and no new Production version is created. Otherwise, the better-performing variant is promoted to **Production** in the MLflow Model Registry, and a confusion matrix is logged as an artifact.

**Stage 6 — Drift Detection.** `drift_detection.py` runs a Kernel Maximum Mean Discrepancy (MMD) drift detector from TorchDrift. The detector is calibrated on 200 training images and tested against both normal test images and artificially degraded images (Gaussian blur, kernel=23) that simulate camera quality degradation. The resulting MMD scores and p-values are logged to MLflow. Drift is flagged if p < 0.05.

**Stage 7 — Export to ONNX.** `export_onnx.py` loads the Production model and exports it to ONNX format (opset 11), which allows the model to be served independently of PyTorch. The ONNX file is logged back to the same MLflow run that produced the Production model version, co-locating it with its originating run.

**Stage 8 — Push Docker Image.** Once all preceding stages have completed successfully, the Docker image is tagged and pushed to the local container registry at `172.24.198.42:5000`. On success, Jenkins additionally archives any generated model card files as build artifacts. Regardless of outcome, local Docker images are removed to conserve disk space.

What is currently not fully automated is the deployment stage: the trained and evaluated model is available in Production in the MLflow Model Registry, but a dedicated `deploy.py` script that logs the deployment event and serves the model has not yet been integrated as a pipeline stage. Additionally, the pipeline currently uses SCM polling rather than push-triggered webhooks due to network constraints on the AAU cluster.

### 2.1.1 Lineage

Lineage in this pipeline is handled through four mechanisms. First, MLflow records the hyperparameters, metrics, and artifacts for each training run, linking a specific model version to the exact configuration that produced it. Second, the Docker image is tagged with the Git commit hash, linking the deployed container to the source code. Third, DVC tracks the dataset version via a content hash in `Dataset.dvc`, meaning any model can be traced back to the exact data used for training. Fourth, the MODEL_CARD.md is logged directly as an MLflow artifact in the training run, preserving the model's intended characteristics alongside the run that produced it. Together, these mechanisms ensure that a model version in production can be traced back to its code, data, hyperparameters, and configuration. What is not yet fully automated is the logging of the deployment event itself, which would close the lineage chain from training to live serving.

## 2.2 Unit Test Coverage

Unit tests were implemented for the core functionality of the project using pytest, covering the model architecture, optimizer, and scheduler modules in `src/`. The tests are located in `tests/` and configured via `pytest.ini` with `pythonpath = src`. Code coverage was measured using the `coverage` package. Fig. X shows the coverage report.

> **[INDSÆT SCREENSHOT AF CODE COVERAGE HER]**
>
> Kør: `pytest --cov=src --cov-report=term-missing tests/`

It is worth noting that 100% coverage does not guarantee bug-free code — it indicates that all lines of code are at least executed during tests. The focus was therefore placed on testing the core model-loading, forward-pass, and configuration-handling logic rather than maximising the coverage number alone.

## 2.3 Experiment Tracking

MLflow is used as the experiment tracking tool and serves as the central link between training, evaluation, and deployment. For each pipeline run, three MLflow runs are created: `training`, `evaluation in FP32 and FP16`, and `drift_detection`. Together they capture the full lifecycle of a model version.

For the training run, MLflow logs per-epoch train and validation loss and accuracy, final validation accuracy, model parameter count and FLOPs, training duration, and CarbonTracker GPU energy measurements. For the evaluation run, per-class precision, recall, and F1-scores are logged for both FP32 and FP16, along with a confusion matrix of the promoted model. The model registry stores all promoted versions, enabling rollback to any previous Production version if performance degrades.

Key metrics from the most recent completed run are shown in Table X.

| Metric | Value |
|---|---|
| Validation accuracy (training) | 69.9% |
| Test accuracy FP32 | 73.46% |
| Test accuracy FP16 | 73.52% |
| Selected precision type | FP16 |
| FP32 inference duration (~5,742 images) | 13.36 s |
| Training CO₂ (4 epochs) | 4.858 g CO₂eq |

FP16 was selected as the Production precision type, achieving a marginally higher accuracy (73.52% vs. 73.46%) and approximately 2× lower inference latency, with negligible accuracy loss.

> **[INDSÆT SCREENSHOT AF MLFLOW EXPERIMENT VIEW HER]**
> 
> **[INDSÆT SCREENSHOT AF MLFLOW MODEL REGISTRY HER]**
