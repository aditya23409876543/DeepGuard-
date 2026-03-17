FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
# Requires libsndfile1 for Audio/MFCC analysis (soundfile/librosa)
# Requires ffmpeg for PyDub/ImageIO transcoding
RUN apt-get update && apt-get install -y \
    libsndfile1-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install CPU-only PyTorch to save ~3GB of CUDA bloat
COPY backend/requirements.txt .
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy the backend code
COPY backend/ ./backend/

# Expose port (Railway dynamically injects $PORT)
EXPOSE $PORT

# Run via uvicorn
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}
