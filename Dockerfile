FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /project

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt && pip install dvc[s3]

# Copy project files
COPY . .
