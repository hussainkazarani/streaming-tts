# R2: File-Based Generation Review
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install ffmpeg (required by torchaudio for reading/writing wav files)
RUN apt-get update && apt-get install -y \
    ffmpeg \ 
    git \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy root requirements
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 1. The core engine logic
COPY backend/tts_engine/models.py ./backend/tts_engine/
# 2. The reference voices for cloning
COPY backend/voice_cloning/voices/ ./backend/voice_cloning/voices/
# 3. The specific script for this review
COPY scripts/review_2.py ./scripts/

# Ensure Python can find the 'backend' package
ENV PYTHONPATH="/app"

# Command to run the local generation script
CMD ["python", "scripts/review_2.py"]