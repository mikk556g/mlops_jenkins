import mlflow
import torch
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("http://172.24.198.42:5050")
client = MlflowClient()

model = mlflow.pytorch.load_model(
    "models:/resnet50-emotion-classifier/Production", map_location=torch.device("cpu")
)
model.eval()

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

with mlflow.start_run(
    run_id=client.get_latest_versions(
        "resnet50-emotion-classifier", stages=["Production"]
    )[0].run_id
):
    mlflow.log_artifact("model.onnx")

print("ONNX export complete and logged to MLflow")
