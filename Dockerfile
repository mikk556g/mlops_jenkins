FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

WORKDIR /project

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy project files
COPY . .
