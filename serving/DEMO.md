# Emotion Classifier — Serving Demo

The `serving/` directory contains a FastAPI application that serves the trained model via a simple web interface. It is included here as a demonstration and is not part of the automated pipeline.

## What it does

Upload a face image (JPEG or PNG) and the app returns the predicted emotion along with a confidence score for each of the seven classes: angry, disgust, fear, happy, neutral, sad, surprise.

## How to run it manually

From the repo root, with a trained `model.onnx` available:

```
cp model.onnx serving/
docker build -t fer-demo serving/
docker run -d --name fer-demo -p 8000:8000 fer-demo
```

Then open `http://localhost:8000` in a browser.

## How to stop it

```
docker stop fer-demo
docker rm fer-demo
```
