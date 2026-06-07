FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model and serving code
# model.onnx is copied in by Jenkins before docker build
COPY model.onnx /app/model.onnx
COPY src/serving/ /app/

ENV MODEL_PATH=/app/model.onnx

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
