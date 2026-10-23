FROM python:3.12-slim

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg libgl1 libglib2.0-0 libgomp1 tesseract-ocr tesseract-ocr-eng \
 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt requirements-public-hub.txt requirements-runtimes.txt ./
ARG INSTALL_RUNTIMES=0
RUN pip install -r requirements.txt \
 && if [ "$INSTALL_RUNTIMES" = "1" ]; then pip install -r requirements-runtimes.txt; fi
COPY backend ./backend
COPY config ./config
COPY third_party ./third_party
RUN mkdir -p /data/config /data/cache /data/weights /data/workflows /data/benchmarks /data/workspace/assets /data/workspace/previews /imports /exports
ENV VIS_PROJECT_ROOT=/app \
    VIS_DATA_DIR=/data \
    VIS_MODELS_FILE=/app/config/models.yaml
EXPOSE 8000
CMD ["uvicorn", "inference_studio.main:app", "--app-dir", "backend", "--host", "0.0.0.0", "--port", "8000"]
