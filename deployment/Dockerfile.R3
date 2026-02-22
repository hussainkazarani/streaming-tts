# R3: High-Performance Streaming TTS Review
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Audio dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 1. Core architecture models
COPY backend/tts_engine/models.py ./backend/tts_engine/
# 2. Standalone Review 3 server and frontend
COPY scripts/review-3/ ./scripts/review-3/
# 3. Static Assets (Favicon, etc.)
COPY assets/ ./assets/

# Ensure module imports resolve correctly from root
ENV PYTHONPATH="/app"

EXPOSE 8000

# Boot the isolated R3 streamer
CMD ["python", "scripts/review-3/review_3.py"]