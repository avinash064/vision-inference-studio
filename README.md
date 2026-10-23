# Vision Inference Studio v0.6.3

Vision Inference Studio is a local, production-oriented computer-vision workbench for building and comparing inference workflows. Its React Flow canvas connects typed preprocessing, model, postprocessing, and preview nodes to a FastAPI execution engine. Images, videos, checkpoints, tensors, and results remain backend-owned; the browser works with opaque IDs and bounded previews.

Version 0.6.3 adds explicit RF-DETR backbone patch-size selection and native official RT-DETRv2 checkpoint execution. It also includes the Public Resource Hub, immutable benchmark experiments, promoted recipes, OCR model families, prompt-based inference, GPU/CPU monitoring, and a Linux desktop package.

Repositories and release:

- Standalone project: [avinash064/vision-inference-studio](https://github.com/avinash064/vision-inference-studio)
- Auto Annotation monorepo: [avinash064/auto-annotation](https://github.com/avinash064/auto-annotation/tree/main/vision-inference-studio)
- Linux packages: [Vision Inference Studio v0.6.3](https://github.com/avinash064/vision-inference-studio/releases/tag/v0.6.3)

## Contents

- [What is new in v0.6.3](#what-is-new-in-v063)
- [Product workflow](#product-workflow)
- [Supported model families](#supported-model-families)
- [RF-DETR patch-size selection](#rf-detr-patch-size-selection)
- [Architecture](#architecture)
- [Install the desktop application](#install-the-desktop-application)
- [Run from source](#run-from-source)
- [Using the application](#using-the-application)
- [Public Hub, experiments, and recipes](#public-hub-experiments-and-recipes)
- [Configuration](#configuration)
- [Storage layout](#storage-layout)
- [API overview](#api-overview)
- [Security model](#security-model)
- [Testing and release verification](#testing-and-release-verification)
- [Troubleshooting](#troubleshooting)
- [Known limitations](#known-limitations)

## What is new in v0.6.3

### RF-DETR architecture selection

The universal Model node now exposes **Backbone patch size** when RF-DETR is selected. Patch size is part of the immutable model identity because it changes the backbone architecture rather than image preprocessing.

- Base supports patch 14.
- Nano, Small, and Medium support patch 16.
- Large supports patch 14 for legacy checkpoints and patch 16 for current checkpoints.
- RF-DETR Large patch 14 explicitly selects `RFDETRLargeDeprecated`.
- RF-DETR Large patch 16 explicitly selects `RFDETRLarge`.
- Invalid size/patch combinations fail during graph validation, before checkpoint loading.
- Older saved workflows without `patch_size` retain the RF-DETR runtime defaults.

### Official RT-DETRv2 execution

Native `.pth` RT-DETRv2 checkpoints are handled by a dedicated adapter based on the official Apache-2.0 RT-DETR PyTorch source vendored under `third_party/rtdetrv2_official/`. These checkpoints are not routed through Ultralytics because the checkpoint structures and model-construction paths are different.

The BMD-45 `R101VD-X` checkpoint is exposed as **RT-DETRv2 (Official PyTorch)** and supports CUDA/CPU device selection and FP16/FP32 execution where the device permits it.

### Model discovery and OCR

The model picker distinguishes RF-DETR, RT-DETR, and RT-DETRv2 checkpoints instead of assigning all DETR-family `.pth` files to RF-DETR. The release also contains:

- current Ultralytics YOLO families plus compatible exported YOLOv7 artifacts;
- PaddleOCR and Tesseract OCR adapters;
- remote OpenAI-compatible vLLM vision/OCR inference;
- OCR result normalization and preview rendering;
- Grounding DINO text-prompt detection and SAM-family segmentation.

### Desktop reliability

The packaged application establishes its stable Electron profile before acquiring the single-instance lock. It preserves workflows, weights, provider settings, benchmarks, and runtime configuration during package upgrades.

## Product workflow

Vision Inference Studio is the experimentation layer in the wider annotation and training loop:

```text
model + preprocessing + inference + postprocessing
                         │
                  freeze a recipe
                         │
                         ▼
              Auto Annotation Studio
        immutable dataset snapshot + recipe
                         │
                         ▼
              CanonicalPrediction → Policy
                         │
                         ▼
                 shadow / audit / HITL
                         │
                         ▼
             trusted annotation → CVAT/export
                         │
                         ▼
                   Training Studio
                         │
                         ▼
                    new checkpoint
                         │
                         └──── back to Vision Studio
```

Vision Studio owns inference experiments and recipe promotion. Auto Annotation Studio consumes promoted recipes and runs them over immutable datasets. Training remains a separate responsibility.

## Supported model families

| Family | Versions and sizes | Input/output | Runtime |
|---|---|---|---|
| YOLO | v3, v5, v6, v7-export, v8, v9, v10, v11, v12, v26 | image/video → detections | Ultralytics |
| RT-DETR | Ultralytics `l`, `x`, custom | image/video → detections | Ultralytics RTDETR |
| RT-DETRv2 | BMD-45 `R101-X` | image/video → detections | official RT-DETR PyTorch |
| RF-DETR | base, nano, small, medium, large | image/video → detections | `rfdetr` |
| DINOv3 | ViT-S, ViT-B, ViT-L | image → spatial feature map | Transformers/PyTorch |
| Grounding DINO | tiny, base | image + text prompt → detections | official IDEA Research runtime |
| Segment Anything | ViT-B, ViT-L, ViT-H | image + point/box → masks | Transformers/PyTorch |
| SAM 2 / 2.1 | tiny, small, base-plus, large | image/video frame + point/box → masks | Ultralytics |
| SAM 3 | base | image + text/point/box → masks | Ultralytics |
| PaddleOCR | PP-OCRv5 server/mobile | image → text and regions | PaddleOCR/PaddlePaddle |
| Tesseract OCR | system v5 | image → text and word regions | pytesseract + Tesseract |
| vLLM Vision/OCR | remote OpenAI-compatible model | image + instruction → text | remote HTTP endpoint |
| Generic ONNX | custom | image → tensor | ONNX Runtime |

The authoritative compatibility manifest is [`config/models.yaml`](config/models.yaml).

### Weight formats

| Family | Accepted artifact |
|---|---|
| YOLO | `.pt`, `.onnx`, `.engine` |
| RT-DETR | `.pt` |
| RT-DETRv2 | `.pth` |
| RF-DETR | `.pth`, `.pt` |
| DINOv3 | `.pth`, `.pt` |
| Grounding DINO | `.pth` |
| SAM | `.safetensors`, `.bin`, `.pth`, `.pt` |
| SAM 2 / SAM 3 | `.pt` |
| Generic ONNX | `.onnx` |

PaddleOCR, Tesseract, and vLLM are weightless from the workflow's perspective. PaddleOCR manages its engine files, Tesseract uses the system installation, and vLLM uses a configured remote endpoint.

## RF-DETR patch-size selection

Choose the model size and patch size that match the checkpoint architecture:

| RF-DETR size | Valid patch sizes | Runtime class |
|---|---:|---|
| Base | 14 | `RFDETRBase` |
| Nano | 16 | `RFDETRNano` |
| Small | 16 | `RFDETRSmall` |
| Medium | 16 | `RFDETRMedium` |
| Large, legacy | 14 | `RFDETRLargeDeprecated` |
| Large, current | 16 | `RFDETRLarge` |

To configure it in the UI:

1. Add or select the **Model** node.
2. Select **RF-DETR** as the family and `v1` as the version.
3. Choose the checkpoint size.
4. Choose **Backbone patch size** from the values presented for that size.
5. Select the registered checkpoint, device, and precision.
6. Select **Validate** before running.

If the wrong architecture is selected, loading may report tensor-shape or positional-encoding mismatches. Return to the Model inspector and choose the checkpoint's real patch size; do not use image-resize controls to compensate.

## Architecture

```text
React + TypeScript + Zustand + React Flow
                  │
                  │ typed nodes, edges, IDs, configuration
                  ▼
           FastAPI HTTP/WebSocket API
                  │
                  ├── AssetStore / WeightStore
                  ├── Public Hub providers and durable downloads
                  ├── WorkflowValidator → ExecutionPlan
                  └── ExecutionEngine → ExecutionScope
                                      │
                         ModelManager immutable cache
                                      │
                              AdapterRegistry
                  ┌──────────┬────────┼────────┬─────────┐
                YOLO      DETR      RF-DETR  SAM/DINO   OCR/vLLM
                  └──────────┴────────┼────────┴─────────┘
                                      │
                         standardized result models
                                      │
                              RendererRegistry
                                      │
                   bounded preview + metrics + events
```

Key boundaries:

- **Nodes** define typed ports and configuration metadata.
- **Validation** checks the DAG, ports, required assets, model compatibility, and configuration before execution.
- **Adapters** own model-specific loading, preprocessing, inference, and result conversion.
- **ModelManager** caches loaded models by immutable identity, including RF-DETR patch size.
- **ExecutionScope** cleans run-owned streams, artifacts, and callbacks on completion, failure, or cancellation.
- **Renderers** draw standardized results; model adapters never draw into source images.
- **Frontend** receives serializable status, metrics, results, and preview URLs rather than tensors or model objects.

### Repository layout

```text
vision-inference-studio/
├── backend/
│   ├── inference_studio/
│   │   ├── adapters/       # model-family integrations
│   │   ├── api/            # HTTP and WebSocket routes
│   │   ├── assets/         # opaque asset and weight storage
│   │   ├── chainner/       # optional preprocessing-node bridge
│   │   ├── features/       # feature artifacts and comparisons
│   │   ├── graph/          # schema, validation, planning, execution
│   │   ├── models/         # manifest, registry, lifecycle manager
│   │   ├── public_hub/     # providers, downloads, experiments, recipes
│   │   ├── renderers/      # detection, segmentation, OCR, feature, tensor
│   │   ├── results/        # canonical inference-result schemas
│   │   └── video/          # sequential video decoding and metrics
│   └── tests/
├── frontend/               # React/Vite desktop UI
├── desktop/                # Electron shell and Linux packaging
├── config/models.yaml      # model-family compatibility manifest
├── third_party/            # attributed official RT-DETRv2 runtime
├── docs/                   # ADRs, acceptance material, screenshots
├── packaging/              # installer and runtime scripts
├── workflows/              # saved/example graphs
└── workspace/              # runtime data; ignored by Git
```

## Install the desktop application

Download the Debian asset from the [v0.6.3 release](https://github.com/avinash064/vision-inference-studio/releases/tag/v0.6.3), then install it:

```bash
sudo apt install ./vision-inference-studio_0.6.3_amd64.deb
```

Launch **Vision Inference Studio** from the desktop application menu, or run:

```bash
vision-inference-studio
```

The installer upgrades the application without removing user weights, workflows, benchmarks, provider credentials, or workspace data.

### Use the required shared Python environment

This workstation is configured to use `/home/avinash/venv`; no second virtual environment is needed. Store the interpreter path once:

```bash
mkdir -p ~/.config/vision-inference-studio
printf '%s\n' '/home/avinash/venv/bin/python' > \
  ~/.config/vision-inference-studio/python-runtime
chmod 600 ~/.config/vision-inference-studio/python-runtime
```

`VIS_PYTHON=/absolute/path/to/python` is a temporary process-level override and takes precedence over the file.

## Run from source

Requirements:

- Python 3.10 or newer;
- Node.js 20 or newer;
- NVIDIA driver and compatible PyTorch build for CUDA inference;
- Tesseract system package when using Tesseract OCR;
- optional runtime packages from the relevant requirement files.

### Backend

```bash
cd vision-inference-studio
PYTHONPATH=backend /home/avinash/venv/bin/uvicorn \
  inference_studio.main:app --host 0.0.0.0 --port 8000
```

Health and interactive API documentation:

- `http://127.0.0.1:8000/health`
- `http://127.0.0.1:8000/docs`

### Frontend

```bash
cd vision-inference-studio/frontend
npm ci
npm run dev -- --host 0.0.0.0 --port 5174
```

Open `http://127.0.0.1:5174`. Set `VITE_API_URL` before starting/building if the API is not at port 8000.

### Optional runtime bundles

```bash
/home/avinash/venv/bin/pip install -r requirements-runtimes.txt
/home/avinash/venv/bin/pip install -r requirements-public-hub.txt
```

Prompt-model packages are pinned separately because some build from source:

```bash
/home/avinash/venv/bin/pip install --no-deps --no-build-isolation \
  -r requirements-prompt-models.txt
```

### Docker baseline

```bash
docker compose up --build -d
```

The frontend is published at port 5174 by default and the API at port 8000. GPU containers require the NVIDIA container runtime and a CUDA-compatible PyTorch build; they are not silently assumed by the baseline compose file.

## Using the application

### Build and run a canvas workflow

1. Open **Canvas**.
2. Add an **Image** or **Video** input node.
3. Select the node and upload or choose a registered asset.
4. Add optional preprocessing nodes from the node library.
5. Add the universal **Model** node.
6. Select family, version, size, patch size where applicable, checkpoint, device, and precision.
7. Configure request controls such as confidence, text prompt, OCR language, point prompt, or box prompt.
8. Add the compatible Detection, Segmentation, OCR, Feature, Tensor, or Video Preview node.
9. Connect ports, then select **Validate**.
10. Select **Run** and monitor node state, animated active edges, streamed previews, timing, and memory metrics.

The Live Output panel is resizable. Its Original/Prediction toggle changes the displayed frame without changing the saved inference result.

### Prompt behavior

- **Grounding DINO:** enter comma-separated concepts for text-prompt detection.
- **SAM v1 and SAM 2:** provide a point or bounding box.
- **SAM 3:** use text, point, or box prompts.
- **PaddleOCR/Tesseract:** select the OCR language.
- **vLLM:** configure the endpoint and served model, then enter an instruction such as `Read all visible text and preserve line breaks`.

### Runtime monitoring

The header displays CPU and GPU utilization indicators. Completed runs report available preprocessing, inference, postprocessing, total latency, detection count, and CUDA memory measurements. Video runs additionally report frame-rate and latency distributions.

## Public Hub, experiments, and recipes

### Credentials

Open **API Keys** to configure Hugging Face, Roboflow, Kaggle, or a vLLM bearer token. Credentials are validated by the backend and stored in an owner-only `.env` file. Saved secret values are never returned to the frontend.

### Discover and download

The **Public Hub** can search provider resources. Dataset cards include a small preview when the provider supplies or permits one. Select **Download & register** to start a durable backend job, then follow progress on **Downloads**.

Downloads are staged and checked before registration. Archives are bounded, checksummed, inspected, and rejected when they contain traversal paths, links, devices, or excessive expansion. Completing a download does not execute model code.

### Benchmark and compare

1. Register a dataset from Public Hub.
2. Create an immutable random, stratified, or manual benchmark.
3. Save two or more canvas workflows.
4. Create an experiment using the same benchmark for every workflow variant.
5. Compare persisted latency, memory, evaluation data, and rendered outputs.
6. Promote the chosen workflow.

A promoted recipe pins the workflow, benchmark, model/checkpoint identities, hashes, runtime requirements, and originating experiment. Auto Annotation Studio consumes promoted recipes rather than provider-specific resources.

## Configuration

Copy the documented template for manual configuration:

```bash
cp .env.example .env
chmod 600 .env
```

Important settings:

| Variable | Purpose |
|---|---|
| `VIS_PYTHON` | Override the desktop Python interpreter |
| `VIS_MAX_UPLOAD_BYTES` | Maximum uploaded artifact size; default supports large SAM checkpoints |
| `VIS_CHAINNER_SRC_DIR` | Path to a local chaiNNer `src` directory |
| `VIS_RECIPE_INPUT_ROOTS` | Allowed server-side input roots for Auto Annotation recipe execution |
| `VIS_FEATURE_STORE_MAX_ENTRIES` | Bound retained feature artifacts by count |
| `VIS_FEATURE_STORE_MAX_BYTES` | Bound retained feature artifacts by memory |
| `VITE_API_URL` | Frontend API base URL at build/development time |
| `HF_TOKEN` | Hugging Face access, including gated models |
| `ROBOFLOW_API_KEY` | Roboflow API access |
| `KAGGLE_API_TOKEN` | Preferred Kaggle token form |
| `KAGGLE_USERNAME`, `KAGGLE_KEY` | Legacy Kaggle credential pair |
| `VLLM_API_KEY` | Optional remote vLLM bearer token |
| `PUBLIC_HUB_CACHE` | Public Hub download/cache root |
| `PUBLIC_HUB_MAX_CACHE_GB` | Cache capacity bound |
| `PUBLIC_HUB_MAX_CONCURRENT_DOWNLOADS` | Concurrent download bound |
| `PUBLIC_HUB_MAX_EXTRACT_GB` | Maximum expanded archive size |
| `PUBLIC_HUB_MAX_FILE_GB` | Maximum single extracted file size |

Never commit `.env`, API tokens, downloaded weights, or imported datasets. The included ignore rules cover normal runtime locations.

## Storage layout

Packaged Linux application data:

```text
~/.config/vision-inference-studio/
├── .env                  # provider secrets, mode 0600
├── python-runtime        # optional external interpreter path
├── chainner-settings.json
└── window-state.json

~/.local/share/vision-inference-studio/
├── weights/
├── workflows/
├── workspace/
├── benchmarks/
├── runtime/
└── logs/
```

In source mode, defaults are project-relative unless overridden by `VIS_*_DIR` settings. Runtime uploads, weights, generated previews, provider caches, and local workflows are excluded from Git.

## API overview

The backend exposes versioned HTTP routes and run-specific WebSockets. Major groups include:

| Area | Representative routes |
|---|---|
| Health/runtime | `GET /health`, `/api/v1/devices`, `/api/v1/system/metrics` |
| Definitions | `GET /api/v1/nodes`, `/api/v1/models` |
| Assets/weights | `POST /api/v1/assets/image`, `/api/v1/assets/video`, `/api/v1/weights` |
| Workflows | `GET/POST /api/v1/workflows`, `POST /api/v1/workflows/validate`, `/run` |
| Execution control | `POST /api/v1/executions/{id}/pause`, `/resume`, `/stop` |
| Execution events | `WS /api/v1/executions/{id}` |
| Public Hub | `/api/v1/public-hub/providers`, `/datasets/search`, `/models/search`, `/downloads` |
| Credentials | `GET/PUT/DELETE /api/v1/public-hub/credentials/{provider}` |
| Experiments | `GET/POST /api/v1/experiments`, start/cancel/promote, results |
| Recipes | `GET /api/v1/recipes`, `POST /api/v1/recipes/execute` |
| Features | `GET /api/v1/features/stats` |

Use `http://127.0.0.1:8000/docs` for the generated request/response schemas.

## Security model

- Uploaded assets and weights receive opaque IDs; workflows never need client filesystem paths.
- Resolved files must remain inside configured storage roots.
- Upload is separate from model loading and execution.
- File type, size, existence, and decodability/readability are checked.
- Public downloads are treated as data. Downloaded Python is never imported and `trust_remote_code` is not enabled.
- Public archives reject absolute paths, traversal paths, links, device entries, oversized members, and excessive expansion.
- Provider secrets remain backend-only and are never returned in API responses, workflow files, experiments, or recipes.
- Browser-controlled arbitrary local paths are denied.
- Workflows contain JSON-safe configuration and opaque IDs, not tensors, model objects, frames, or file handles.
- There are no arbitrary Python, shell, `eval`, or `exec` canvas nodes.
- Run-owned streams and feature artifacts are released on success, error, or cancellation.

PyTorch and third-party checkpoint formats should still be treated as trusted artifacts. Prefer official sources and safetensors where supported.

## Testing and release verification

### Normal test suite

```bash
cd vision-inference-studio
PYTHONPATH=backend /home/avinash/venv/bin/pytest -q backend/tests
```

v0.6.3 verification result:

```text
120 passed, 9 skipped
```

The skipped integration tests require external checkpoints, provider access, video fixtures, or GPU-specific configuration.

### Frontend and desktop

```bash
cd frontend
npm ci
npm run build

cd ../desktop
npm ci
npm run desktop:make:linux
```

The v0.6.3 release passed frontend TypeScript/production bundling, desktop TypeScript compilation, Electron packaging, Debian packaging, and installed-package startup.

### Real model acceptance

- BMD RF-DETR Base, patch 14: CUDA FP16 load and inference passed.
- BMD RT-DETRv2 R101VD-X: CUDA FP16 load and inference passed.
- The application started its packaged backend using `/home/avinash/venv/bin/python`.

### Release checksums

```text
68c600d015d2829b13bbba92a5d4a23f1027dc482f32be21fe8fffa10509c336  vision-inference-studio_0.6.3_amd64.deb
1f7d52647f4824cf1d54cfb05a1522997ec93e27bffa8f8cb3f752438f54e576  Vision Inference Studio-linux-x64-0.6.3.zip
```

Verify a downloaded file with `sha256sum` before installation.

## Troubleshooting

### The RT-DETRv2 checkpoint appears as RF-DETR

Refresh the Downloads/Assets view after upgrading to v0.6.3. Choose **RT-DETRv2 (Official PyTorch)** for the BMD-45 `weights/RT-DETRv2/best.pth` checkpoint. RT-DETRv2 and RF-DETR are separate families.

### RF-DETR reports shape or positional-encoding errors

Verify the checkpoint size and backbone patch size. Base uses 14; Nano/Small/Medium use 16; Large can use legacy 14 or current 16.

### CUDA is unavailable

Confirm the selected interpreter is the requested environment and that its PyTorch build sees the GPU:

```bash
/home/avinash/venv/bin/python -c \
  "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Use CPU/FP32 if the machine has no compatible CUDA runtime.

### Desktop exits immediately when launched from a developer shell

Electron must not be forced into Node mode. Clear the development-only variable before launching:

```bash
unset ELECTRON_RUN_AS_NODE
vision-inference-studio
```

### Backend fails to start

Inspect:

```text
~/.local/share/vision-inference-studio/logs/electron.log
~/.local/share/vision-inference-studio/logs/backend.log
```

Confirm that `~/.config/vision-inference-studio/python-runtime` contains an existing executable and that the environment has the required packages.

### Port conflict

The source backend defaults to 8000 and the frontend to 5174. Auto Annotation commonly uses port 8002. Stop the conflicting process or select another development port.

### A model family is unavailable

Optional imports are lazy. The rest of the application remains available when one engine is missing. Install that family's requirement bundle into the configured Python environment and restart the backend.

## Known limitations

- Training is outside Vision Studio; this project performs inference experimentation and recipe promotion.
- RTSP, distributed scheduling, remote GPU workers, and arbitrary shell/code nodes are not included.
- Generic ONNX currently assumes a single NCHW image input and exposes the first output tensor.
- SAM 2/3 video tracking memory and automatic Grounding-DINO-to-SAM chaining are not implemented.
- Some prompt-model dependencies build from source and are not part of the default prebuilt desktop runtime.
- Grounding DINO can run without its compiled CUDA attention operator but is slower.
- vLLM is a remote OpenAI-compatible integration; Vision Studio does not install or operate a vLLM server.
- Public-provider availability, rate limits, licenses, and gated-model approval remain external concerns.
- Accuracy metrics are shown only when compatible ground-truth geometry is present.

## Documentation and attribution

- Architecture decisions: [`docs/adr/`](docs/adr/)
- RF-DETR acceptance material: [`docs/acceptance/rfdetr-feature-runtime/`](docs/acceptance/rfdetr-feature-runtime/)
- Release notes and checksums: [`release-0.6.3/`](release-0.6.3/)
- Official RT-DETRv2 attribution: [`third_party/rtdetrv2_official/UPSTREAM.md`](third_party/rtdetrv2_official/UPSTREAM.md)

The application is currently marked `UNLICENSED`. The vendored official RT-DETRv2 source retains its upstream Apache-2.0 license and attribution.
