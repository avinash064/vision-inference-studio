# Vision Inference Studio v0.1

A local, node-based computer-vision inference workbench built independently with React Flow and FastAPI. Compose a typed DAG, upload assets and model weights through opaque IDs, validate the workflow, and run image or sequential video inference while results stream back over WebSocket.

> Status: the first real Image → Universal Model → Detection Preview milestone is complete for both RF-DETR and YOLO on the local CUDA GPU. The same saved three-node graph was exercised through the backend with real checkpoints; browser acceptance is documented in [Acceptance status](#acceptance-status). DINOv3 and real video remain outside this milestone and are not claimed as verified.

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

The suite covers the node/model registries, typed valid and invalid connections, cycles and topological ordering, path-safe weight/asset registration, model caching/unload, graph execution, standard result serialization, and renderer selection. GPU/model integration is intentionally separate because it needs runtime-specific weights and drivers.

## Acceptance status

| Check | Result in this workspace |
|---|---|
| Node library → typed React Flow → inspector → validate/run | Implemented; production TypeScript build passes |
| Browser-driven React Flow acceptance | **Passed** in headless Chrome by selecting each saved workflow and clicking the UI Run button. Both runs reached `COMPLETED`, delivered 14 lifecycle events, loaded their preview image, and displayed device/timing/VRAM metrics. |
| Backend graph execution and rendered preview | Passed with real image decode, real adapters, standardized results, separate renderer, and original/prediction URLs |
| YOLO real inference | **Passed on CUDA** using `best.pt` and the saved `wf_yoloacceptance` graph. It produced 4 detections on `cam04.jpg`; cold total was 683.1 ms with 87.8 MB allocated, 350 MB reserved, and 262.3 MB peak GPU memory. The prediction preview was visually checked. |
| RF-DETR real inference | **Passed on CUDA** using the official legacy `rf-detr-large.pth` and saved `wf_rfdetracceptance` graph. It produced 1 `truck` detection at 0.489 confidence; cold total was 749.8 ms with 534.6 MB allocated, 820 MB reserved, and 682.9 MB peak GPU memory. The prediction preview was visually checked. |
| DINOv3 real feature output | Not run; no compatible DINOv3 checkpoint was available |
| Real video inference | Pipeline implemented with sequential OpenCV decode and WebSocket frames; not run against a real installed model runtime |

The normal suite passes in `/home/avinash/venv` (`14 passed`, with 2 runtime-gated tests skipped when their environment variables are absent). The dedicated real RF-DETR adapter integration test also passes on CUDA. This completes the scoped image-detection milestone; it does not imply DINOv3 or video acceptance.

## Known limitations and deliberate deviations

- API routes and UI widgets are grouped by cohesive modules rather than creating a one-file stub for every diagram entry; runtime separation is preserved.
- The preview dock provides an original/prediction toggle rather than a simultaneous side-by-side layout.
- Execution events include both a category in `type` and an explicit lifecycle name in `event`.
- Pause/stop checkpoints occur between video frames or graph nodes; an individual third-party inference call cannot be pre-empted safely.
- Generic ONNX v0.1 assumes a single NCHW image input and exposes the first output tensor.
- The installed RF-DETR runtime recognizes the available large checkpoint as a legacy architecture and automatically selects its deprecated compatibility configuration; this is functional but emits an upstream warning.
- DINOv3 Transformers model metadata may require initial network access unless already cached; the supplied local state dict is still explicitly applied.
- Remote execution, Tailscale, Jetsons, GCP, RTSP, SAM, annotation/export, training, arbitrary code/shell nodes, scheduling, and distributed workers are intentionally absent.
