import mlflow

model = mlflow.pytorch.load_model("models:/resnet50-emotion-classifier/Production")

# Convert and log the .onnx as an artifact on the same or a new MLFlow run
