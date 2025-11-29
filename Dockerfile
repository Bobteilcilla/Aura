FROM python:3.12-slim

# Environment vars
ENV MODEL_BUCKET=aura_model_data

# Set working directory
WORKDIR /app

# Install OS-level dependencies for OpenCV / Ultralytics / Pillow
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
 && pip install --no-cache-dir numpy==1.26.4  # <--- belt & suspenders for numpy compatibility

# Copy code
COPY package_aura package_aura
COPY models models

# Start backend (PORT is set via docker run)
CMD uvicorn package_aura.api_file:app --host 0.0.0.0 --port ${PORT}
