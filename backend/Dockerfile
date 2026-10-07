FROM python:3.12-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt requirements-runtimes.txt ./
ARG INSTALL_RUNTIMES=0
RUN pip install --no-cache-dir -r requirements.txt && if [ "$INSTALL_RUNTIMES" = "1" ]; then pip install --no-cache-dir -r requirements-runtimes.txt; fi
COPY backend ./backend
COPY config ./config
RUN mkdir -p weights/yolo weights/rfdetr weights/dinov3 weights/onnx workflows workspace/assets workspace/previews
ENV VIS_PROJECT_ROOT=/app PYTHONUNBUFFERED=1
EXPOSE 8000
CMD ["uvicorn", "inference_studio.main:app", "--app-dir", "backend", "--host", "0.0.0.0", "--port", "8000"]
