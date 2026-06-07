import os
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from preprocess import preprocess_image

app = FastAPI()

# ── Load model at startup ──────────────────────────────────────────
MODEL_PATH = os.getenv("MODEL_PATH", "model.onnx")

# Class order must match training config (config/train_config.yaml):
# angry:0, disgust:1, fear:2, happy:3, neutral:4, sad:5, surprise:6
EMOTION_LABELS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

try:
    session = ort.InferenceSession(MODEL_PATH)
    # dummy inference to verify model loaded correctly
    dummy = np.zeros((1, 3, 224, 224), dtype=np.float32)
    session.run(None, {"input": dummy})
    print(f"Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    raise RuntimeError(f"Failed to load model: {e}")


# ── Routes ─────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_PATH}


@app.get("/", response_class=HTMLResponse)
def frontend():
    with open("static/index.html") as f:
        return f.read()


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(
            status_code=400,
            detail="Only JPEG and PNG images are supported"
        )

    image_bytes = await file.read()

    try:
        input_array = preprocess_image(image_bytes)
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Image preprocessing failed: {e}"
        )

    outputs = session.run(None, {"input": input_array})
    scores = outputs[0][0]

    # softmax
    exp_scores = np.exp(scores - np.max(scores))
    probabilities = exp_scores / exp_scores.sum()

    predicted_index = int(np.argmax(probabilities))

    return {
        "label": EMOTION_LABELS[predicted_index],
        "confidence": float(probabilities[predicted_index]),
        "scores": {
            label: float(prob)
            for label, prob in zip(EMOTION_LABELS, probabilities)
        }
    }
