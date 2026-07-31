FROM python:3.12-slim

# ffmpeg is required by moviepy for encoding/muxing.
# libgl1/libglib2.0-0/libsm6/libxext6/libxrender1 are common runtime deps
# that opencv-python-headless still needs on a slim base image.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Render (and most Docker PaaS hosts) inject $PORT at runtime and route traffic to it.
# Default to 10000 for local `docker run` testing where $PORT isn't set.
ENV PORT=10000
EXPOSE 10000

CMD ["sh", "-c", "uvicorn src.app:app --host 0.0.0.0 --port ${PORT:-10000}"]
