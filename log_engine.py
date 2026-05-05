import mlflow
import argparse
from mlflow.tracking import MlflowClient

parser = argparse.ArgumentParser()
parser.add_argument("--mlflow-uri", required=True)
parser.add_argument("--engine", required=True)
args = parser.parse_args()

mlflow.set_tracking_uri(args.mlflow_uri)
client = MlflowClient()

with mlflow.start_run(
    run_id=client.get_latest_versions(
        "resnet50-emotion-classifier", stages=["Production"]
    )[0].run_id
):

    mlflow.log_artifact(args.engine)

print("Engine logged to MLflow")
