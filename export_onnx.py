import mlflow
import torch
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("http://172.24.198.42:5050")
client = MlflowClient()


# ------- GET EXPLICIT PRODUCTION VERSION ------- #
production_versions = client.get_latest_versions(
    "resnet50-emotion-classifier", stages=["Production"]
)
if not production_versions:
    raise ValueError("No model found in Production.")

production_version = production_versions[0]
print(f"Exporting Production model version: {production_version.version}")

# Load by explicit version number, not alias
model = mlflow.pytorch.load_model(
    f"models:/resnet50-emotion-classifier/{production_version.version}",
    map_location=torch.device("cpu"),
)
model.eval()


# ------- EXPORT TO ONNX ------- #
dummy_input = torch.randn(1, 3, 224, 224)

torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    export_params=True,
    opset_version=11,
    input_names=["input"],
    output_names=["output"],
)

# ------- LOG ONNX TO THE SAME RUN THAT PRODUCED THIS VERSION ------- #
# production_version.run_id is the run that logged this exact version,
# so the ONNX artifact is co-located with the model that produced it
with mlflow.start_run(run_id=production_version.run_id):
    mlflow.log_artifact("model.onnx")
    mlflow.log_param("onnx_opset_version", 11)
    mlflow.log_param("onnx_exported_from_version", production_version.version)

print(
    f"ONNX export of version {production_version.version}"
    f"Logged to run {production_version.run_id}"
)
