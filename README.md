# Vision Inference Studio v0.2

A local, node-based computer-vision inference workbench built independently with React Flow and FastAPI. Compose a typed DAG, upload assets and model weights through opaque IDs, validate the workflow, and run image or sequential video inference while results stream back over WebSocket.

> Status: real image and sequential video inference are complete for RF-DETR and YOLO on the local CUDA GPU. Both families use the same universal Model node, standardized detections, renderer, and graph executor. Browser-driven video acceptance includes live previews, pause/resume/cancel, 100-frame benchmarks, bounded preview retention, and a 300-frame memory run. DINOv3 remains outside the verified milestone.

## Architecture

```text
React + Zustand
      │ node/edge/config JSON
      ▼
FastAPI ── asset/weight stores (opaque IDs only)
      │
      ▼
WorkflowValidator ── types, ports, config, assets, weights, DAG
      │
      ▼
Topological planner → ExecutionEngine → WebSocket events
                              │
                              ▼
                       ModelManager cache
                              │
                    AdapterRegistry / ModelAdapter
                    ┌─────────┼─────────┬──────────┐
                  YOLO     RF-DETR    DINOv3     ONNX
                    └─────────┼─────────┴──────────┘
                              ▼
                     standard InferenceResult
                              │
                              ▼
                       RendererRegistry
                       detection / feature / tensor

Video asset → VideoReader → frame scheduler → cached ModelManager handle
                                      │
                                      ▼
                         DetectionRenderer → bounded JPEG slots
                                      │
                                      ▼
                          latest-frame WebSocket delivery
```

The five boundaries are explicit:

- **Nodes** describe typed ports and parameters; the browser fetches them from `GET /api/v1/nodes`.
- **Adapters** are the only model-specific layer and expose load, preprocess, infer, postprocess, unload, and metadata operations.
- **Execution** validates and topologically runs a backend-owned DAG. Canvas position never affects order.
- **Renderers** convert standardized results into previews. Adapters never draw.
- **Frontend** knows nodes, ports, edges, configuration, and status—never PyTorch, CUDA tensors, or model internals.

## Repository tree

```text
vision-inference-studio/
├── backend/
│   ├── Dockerfile
│   ├── inference_studio/
│   │   ├── adapters/          # base contract, factory, YOLO/RF-DETR/DINOv3/ONNX
│   │   ├── api/               # versioned HTTP and WebSocket routes
│   │   ├── assets/            # secure asset records/storage
│   │   ├── graph/             # schema, validation, planning, execution, persistence
│   │   ├── models/            # manifest, registry, weights, lifecycle manager
│   │   ├── nodes/             # definitions, registry, runtime values
│   │   ├── renderers/         # detection, feature, tensor, registry
│   │   ├── results/           # standardized Pydantic results/metrics
│   │   ├── video/             # sequential reader and benchmark collector
│   │   ├── config.py
│   │   └── main.py
│   └── tests/
├── config/models.yaml
├── frontend/
│   ├── src/
│   │   ├── api/
│   │   ├── components/{canvas,nodes,preview,properties,sidebar}/
│   │   ├── store/
│   │   ├── types/
│   │   ├── App.tsx
│   │   └── styles.css
│   ├── Dockerfile
│   └── package.json
├── weights/{yolo,rfdetr,dinov3,onnx}/
├── workflows/examples/
├── workspace/{assets,previews}/
├── docker-compose.yml
├── Makefile
├── pyproject.toml
├── requirements.txt
└── requirements-runtimes.txt
```

Runtime uploads, generated previews, locally saved workflows, virtual environments, and model weights are git-ignored.

## Dependencies

Backend baseline: Python 3.10+, FastAPI, Pydantic 2, Uvicorn, OpenCV headless, NumPy, httpx, PyYAML, multipart upload support, and WebSockets.

Optional inference bundle: PyTorch, Ultralytics, ONNX Runtime GPU, RF-DETR, and Transformers. They are isolated in `requirements-runtimes.txt` so API/graph development and unit tests do not force several gigabytes of ML packages onto every machine.

Frontend: React 18, TypeScript, Vite, `@xyflow/react`, Zustand, Radix Tooltip, and Lucide icons.

## Local setup

Backend (this workstation uses the shared environment requested for the project):

```bash
cd vision-inference-studio
PYTHONPATH=backend /home/avinash/venv/bin/uvicorn \
  inference_studio.main:app --host 0.0.0.0 --port 8000
```

For a fresh machine, create a Python 3.10+ environment, install `requirements-dev.txt` and `requirements-runtimes.txt`, then substitute that environment's Python/Uvicorn paths. Use the PyTorch wheel appropriate for the installed CUDA driver.

Frontend (Node 20+):

```bash
cd vision-inference-studio/frontend
npm ci
npm run dev -- --host 0.0.0.0
```

On this workstation, where Node is provided by Docker, the exact production-build and preview commands are:

```bash
cd vision-inference-studio
docker run --rm -v "$PWD/frontend:/app" -w /app node:22-alpine npm run build
/home/avinash/venv/bin/python -m http.server 5174 \
  --bind 127.0.0.1 --directory frontend/dist
```

Open `http://localhost:5173` for Vite development or `http://127.0.0.1:5174` for the production preview above. The default API address is `http://<browser-host>:8000`; override it with `VITE_API_URL` at build/dev time.

Docker baseline (graph/API/UI; build with runtime packages separately for GPU inference):

```bash
docker compose up --build -d
```

The backend Dockerfile supports `--build-arg INSTALL_RUNTIMES=1`. NVIDIA container runtime and a CUDA-compatible PyTorch wheel are host-specific and deliberately not silently assumed.

## Using the studio

1. Drag **Image** or **Video** from the library (or double-click the canvas), select it, and upload an asset.
2. Add the single universal **Model** node. Choose family → version → size; incompatible downstream fields reset when family/version changes.
3. Upload a compatible weight. The browser receives an internal ID, never a server path.
4. Choose device, precision, and confidence.
5. Add Detection, Feature, Tensor, or Video Preview as appropriate and connect typed handles.
6. Select **Validate**, then **Run**. Nodes visibly progress through waiting → loading/running → success/error, active edges animate, and results stream into the dock.
7. Toggle **Original** / **Prediction** in the dock and inspect preprocessing, inference, postprocessing, total time, object count, GPU allocated/reserved/peak memory, and device name.
8. Video decoding is sequential; Pause/Resume/Stop operate at frame checkpoints without buffering the whole file.

For video, configure the Video source with `frame_stride`, warm-up frames, and an optional benchmark-frame limit. A stride of 1 infers every source frame; a stride of 5 infers source frames 0, 5, 10, and so on while retaining source timestamps. Models are loaded once by the Model node and reused for every frame.

Saved example shapes are under `workflows/examples/`; replace their placeholder asset and weight IDs with values returned by uploads.

## Supported nodes

| Category | Nodes | Input → output |
|---|---|---|
| Sources | Image, Video | asset ID → `image` / `video` |
| Model | Universal Model | `image` or `video` → registry-derived result |
| Processing | Resize, Crop, Threshold | `image` → `image` |
| Outputs | Detection Preview | `detections` → rendered image |
| Outputs | Feature Preview | `feature_map`/`embedding` → heatmap |
| Outputs | Tensor Preview | `tensor` → raw tensor card |
| Outputs | Video Preview | `video` model stream → streamed frames |

The shared type vocabulary also reserves frame, bboxes, mask, segmentation, classification, heatmap, depth, weights, number, string, and boolean for compatible future nodes.

## Supported model registry

| Family | Versions | Sizes | Result | Adapter |
|---|---|---|---|---|
| YOLO | v8, v11, v12 | n, s, m, l, x, custom | detections | Ultralytics |
| RF-DETR | v1 | nano, small, medium, base, large | detections | `rfdetr` |
| DINOv3 | official | vit-s, vit-b, vit-l | feature map | Transformers/PyTorch |
| Generic ONNX | generic | custom | tensor | ONNX Runtime |

DINOv3 returns a feature result and is never represented as an object detector.

## API

```text
GET  /health
GET  /api/v1/nodes
GET  /api/v1/models
GET  /api/v1/models/loaded
GET  /api/v1/devices
GET  /api/v1/benchmarks
GET  /api/v1/benchmarks/{run_id}
GET  /api/v1/weights
POST /api/v1/weights
POST /api/v1/models/load
POST /api/v1/models/unload
POST /api/v1/assets/image
POST /api/v1/assets/video
POST /api/v1/workflows/validate
POST /api/v1/workflows/run
GET  /api/v1/workflows
POST /api/v1/workflows
GET  /api/v1/workflows/{workflow_id}
POST /api/v1/executions/{run_id}/pause
POST /api/v1/executions/{run_id}/resume
POST /api/v1/executions/{run_id}/stop
WS   /api/v1/executions/{run_id}
```

Interactive API documentation is at `http://localhost:8000/docs`.

## Adding an adapter

1. Implement `ModelAdapter` in `backend/inference_studio/adapters/`. Keep all runtime-specific imports lazy so other adapters still work if that runtime is absent.
2. Return a standard `DetectionResult`, `FeatureResult`, or `TensorResult`; never draw in the adapter.
3. Register the adapter once in `AdapterRegistry`.
4. Add the family, compatible extensions, versions/sizes, adapter key, and result type to `config/models.yaml`.
5. Add lifecycle and serialization tests. No API family branch is necessary.

## Adding a node

1. Add its `NodeDefinition` to the backend node registry with typed ports and config fields.
2. Add its backend operation to the executor (or a dedicated executor callable as the library grows).
3. If it introduces a result type, add a Pydantic result and renderer registration.
4. Add validation/execution tests. The generic React node and property inspector render server metadata automatically; specialized visuals are optional.

## Security model

- Uploads are streamed into family/kind-specific directories and assigned random opaque IDs.
- Filenames are reduced to their basename; client paths are never accepted or opened.
- IDs use strict formats and resolved paths must remain inside their store.
- Extension, size, family, readability/decodability, and existence are checked.
- Upload does not load a model. Model deserialization only happens after an explicit load/run action.
- DINO state dict loading uses PyTorch `weights_only=True`.
- There are no Python-code, shell-command, dynamic import, `eval`, or `exec` nodes.
- Workflows serialize only JSON-safe configuration and IDs—not tensors, model objects, frames, or file handles.

Treat model files as trusted artifacts: some third-party runtimes (notably Ultralytics/PyTorch checkpoint compatibility paths) may deserialize formats with broader behavior than safetensors. Use weights from a trusted source and isolate the backend in production.

## Tests

```bash
PYTHONPATH=backend /home/avinash/venv/bin/pytest -q backend/tests
cd frontend && npm run build
```

The suite covers the node/model registries, typed valid and invalid connections, cycles and topological ordering, path-safe weight/asset registration, model caching/unload, image graph execution, video metadata, sequential read/seek, stride and timestamps, frame-result serialization, benchmark aggregation, model reuse, bounded previews, pause/resume/cancel, and renderer selection. GPU/model integration remains separate because it needs runtime-specific weights and drivers.

## Video runtime and benchmark behavior

- `VideoReader` owns OpenCV open, sequential decode, seek, metadata, and close behavior. Frames are never accumulated in memory.
- The executor owns stride scheduling and calls the existing frame-level `adapter.predict(image)` contract. Adapters contain no video loops.
- The Model node loads or reuses one cached handle before frame iteration. Confidence can change between runs without checkpoint deserialization.
- Warm-up frames are excluded from benchmark distributions. CUDA timing remains synchronized in the adapter.
- Live preview uses ten rotating JPEG slots per run. WebSocket subscribers retain only the newest pending frame while lifecycle/control events remain lossless.
- Pause is acknowledged at the next frame boundary. Resume continues with the existing reader position. Cancel closes the reader, persists the partial benchmark, and leaves the reusable model loaded.
- Benchmark JSON is written atomically under `workspace/benchmarks/`; the UI retains completed YOLO/RF-DETR reports in a comparison table.
- Latency distributions include count, mean, median/p50, p90, p95, p99, minimum, and maximum for decode, preprocessing, inference, postprocessing, rendering, publish queueing, and total frame time.

## Acceptance status

| Check | Result in this workspace |
|---|---|
| Node library → typed React Flow → inspector → validate/run | Implemented; production TypeScript build passes |
| Browser-driven React Flow acceptance | **Passed** in headless Chrome by selecting each saved workflow and clicking the UI Run button. Both runs reached `COMPLETED`, delivered 14 lifecycle events, loaded their preview image, and displayed device/timing/VRAM metrics. |
| Backend graph execution and rendered preview | Passed with real image decode, real adapters, standardized results, separate renderer, and original/prediction URLs |
| YOLO real inference | **Passed on CUDA** using `best.pt` and the saved `wf_yoloacceptance` graph. It produced 4 detections on `cam04.jpg`; cold total was 683.1 ms with 87.8 MB allocated, 350 MB reserved, and 262.3 MB peak GPU memory. The prediction preview was visually checked. |
| RF-DETR real inference | **Passed on CUDA** using the official legacy `rf-detr-large.pth` and saved `wf_rfdetracceptance` graph. It produced 1 `truck` detection at 0.489 confidence; cold total was 749.8 ms with 534.6 MB allocated, 820 MB reserved, and 682.9 MB peak GPU memory. The prediction preview was visually checked. |
| DINOv3 real feature output | Not run; no compatible DINOv3 checkpoint was available |
| YOLO real video inference | **Passed on CUDA** on the 1920×1080, 20 FPS `cam04_44m_acceptance.mp4`: 2 warm-up + 100 measured frames, 27.7 ms median / 31.0 ms p95 inference, 37.4 ms median end-to-end, 35.9 model FPS, 26.6 end-to-end FPS, and 262 MB peak allocated VRAM |
| RF-DETR real video inference | **Passed on CUDA** using the identical video graph and source: 2 warm-up + 100 measured frames, 83.1 ms median / 100.8 ms p95 inference, 94.5 ms median end-to-end, 11.6 model FPS, 10.3 end-to-end FPS, and 762 MB peak allocated VRAM |
| Video controls | Browser-driven pause held the displayed frame stable, resume advanced from the preserved position, and cached-model stop runs reached `CANCELLED` for both families |
| Long-run memory/retention | A 320-frame real 1080p clip processed 302 inferred frames (300 measured). RSS grew 14.6 MB after warm-up/model reuse, GPU allocation remained bounded, and only ten preview files were retained for the completed run |

The normal suite passes in `/home/avinash/venv` (`18 passed`, with 4 runtime-gated image/video integrations skipped when their environment variables are absent). Dedicated real image adapter checks and browser-driven image/video CUDA runs have passed. This completes the scoped v0.2 image-and-video milestone; it does not imply DINOv3 or any excluded platform capability.

With `VIS_TEST_VIDEO`, both checkpoint variables, CUDA, and a 100-frame target configured, the dedicated real-video integration module passes both families (`2 passed`), loading each checkpoint once and applying the same frame-level adapter contract.

## Known limitations and deliberate deviations

- API routes and UI widgets are grouped by cohesive modules rather than creating a one-file stub for every diagram entry; runtime separation is preserved.
- The preview dock provides an original/prediction toggle rather than a simultaneous side-by-side layout.
- Execution events include both a category in `type` and an explicit lifecycle name in `event`.
- Pause/cancel checkpoints occur between video frames or graph nodes; an individual third-party inference call cannot be pre-empted safely.
- Generic ONNX v0.1 assumes a single NCHW image input and exposes the first output tensor.
- The installed RF-DETR runtime recognizes the available large checkpoint as a legacy architecture and automatically selects its deprecated compatibility configuration; this is functional but emits an upstream warning.
- DINOv3 Transformers model metadata may require initial network access unless already cached; the supplied local state dict is still explicitly applied.
- Remote execution, Tailscale, Jetsons, GCP, RTSP, SAM, annotation/export, training, arbitrary code/shell nodes, scheduling, and distributed workers are intentionally absent.

## Conclusion

Vision Inference Studio v0.2 completes the local real-time media milestone: the same model-independent graph now handles both `Image → Model → Detection Preview` and `Video → Model → Video Preview` for real YOLO and RF-DETR CUDA inference. Video decoding, scheduling, rendering, preview transport, lifecycle controls, metrics, and persistence stay outside the adapters, preserving the abstraction proven in v0.1.

The video claim is backed by actual 1920×1080 camera footage, real checkpoints, 100-frame browser-driven runs for each model, visible live predictions, frame-boundary pause/resume, clean cancellation, cached model reuse, percentile benchmark reports, and a separate 300-frame memory validation. Preview storage and subscriber backpressure are bounded rather than proportional to video duration.

The current conclusion remains deliberately narrow. It verifies local RF-DETR/YOLO image and file-video inference only. DINOv3, RTSP, Tailscale, remote Jetsons, GCP, SAM, annotation/export, training, deployment, multi-GPU scheduling, and distributed workers remain unimplemented or unverified.
