# Backend with every inference dependency: CUDA 12.1 PyTorch, Ultralytics (YOLO, SAM 2, SAM 3), RF-DETR,
# Transformers (SAM, DINOv3), ONNX Runtime GPU, official IDEA-Research Grounding DINO, the CLIP tokenizer
# SAM 3 needs, and the chaiNNer Standard runtime. Package versions follow requirements-lock.txt, the set
# verified with real GPU inference.
#
# Model weights, the chaiNNer source and Hugging Face caches are added by docker/build-and-push.sh
# (docker cp + commit) so ~7 GB of weights are not duplicated through the build context.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/opt/huggingface \
    YOLO_CONFIG_DIR=/tmp/Ultralytics

RUN apt-get update \
 && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt requirements-runtimes.txt requirements-lock.txt requirements-chainner-standard.txt requirements-prompt-models.txt ./

# One layer: git/g++ are only needed to build Grounding DINO and CLIP from their pinned commits.
RUN apt-get update \
 && apt-get install -y --no-install-recommends git g++ \
 && pip install --extra-index-url https://download.pytorch.org/whl/cu121 -c requirements-lock.txt \
      -r requirements-runtimes.txt -r requirements-chainner-standard.txt \
      "yapf>=0.40,<1" "addict>=2.4,<3" "ftfy>=6,<7" \
 && pip install --no-deps --no-build-isolation -r requirements-prompt-models.txt \
 && python -c "import torch, ultralytics, rfdetr, transformers, groundingdino, clip, onnxruntime, chainner_ext" \
 && apt-get purge -y git g++ && apt-get autoremove -y \
 && rm -rf /var/lib/apt/lists/* /root/.cache

COPY backend ./backend
COPY config ./config
RUN mkdir -p weights workflows benchmarks workspace/assets workspace/previews workspace/reports chainner/src /opt/huggingface/hub

ENV VIS_PROJECT_ROOT=/app \
    VIS_CHAINNER_SRC_DIR=/app/chainner/src
EXPOSE 8000
CMD ["uvicorn", "inference_studio.main:app", "--app-dir", "backend", "--host", "0.0.0.0", "--port", "8000"]
