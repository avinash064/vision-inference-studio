# Vision Inference Studio v0.1

A local, node-based computer-vision inference workbench built independently with React Flow and FastAPI. Compose a typed DAG, upload assets and model weights through opaque IDs, validate the workflow, and run image or sequential video inference while results stream back over WebSocket.

> Status: production-oriented MVP implementation. Mocked end-to-end execution and the complete non-GPU suite pass. Real YOLO, RF-DETR, DINOv3, and video acceptance results are reported honestly in [Acceptance status](#acceptance-status); a model is not claimed as verified unless its runtime and compatible weights were actually available.

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

Backend:

```bash
cd vision-inference-studio
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements-dev.txt

# Required for real models; use the PyTorch wheel/index appropriate for your CUDA driver.
pip install -r requirements-runtimes.txt

uvicorn inference_studio.main:app \
  --app-dir backend --host 0.0.0.0 --port 8000 --reload
```

Frontend (Node 20+):

```bash
cd vision-inference-studio/frontend
npm ci
npm run dev -- --host 0.0.0.0
```

Open `http://localhost:5173`. The default API address is `http://<browser-host>:8000`; override it with `VITE_API_URL` at build/dev time.

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
6. Select **Validate**, then **Run**. Active nodes highlight while results and metrics stream into the dock.
7. Video decoding is sequential; Pause/Resume/Stop operate at frame checkpoints without buffering the whole file.

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
PYTHONPATH=backend .venv/bin/pytest -q
cd frontend && npm run build
```

The suite covers the node/model registries, typed valid and invalid connections, cycles and topological ordering, path-safe weight/asset registration, model caching/unload, graph execution, standard result serialization, and renderer selection. GPU/model integration is intentionally separate because it needs runtime-specific weights and drivers.

## Acceptance status

| Check | Result in this workspace |
|---|---|
| Node library → typed React Flow → inspector → validate/run | Implemented; production TypeScript build passes |
| Backend graph execution and rendered preview | Passed end-to-end with a mock adapter and real image decoding/rendering |
| YOLO real inference | **Passed on CPU** through upload registration → model manager → Ultralytics adapter → standardized detections → renderer. The local `best.pt` produced 2 detections (`Black_PP` 0.81, `Color_PP` 0.52) in 1,859 ms total on the 1920×1200 test image. The rendered preview was visually checked. |
| RF-DETR real inference | Not run; no compatible RF-DETR checkpoint was available |
| DINOv3 real feature output | Not run; no compatible DINOv3 checkpoint was available |
| Real video inference | Pipeline implemented with sequential OpenCV decode and WebSocket frames; not run against a real installed model runtime |

The optional real-checkpoint test also passed (`1 passed in 3.25s`) with `VIS_YOLO_WEIGHT`, `VIS_TEST_IMAGE`, and CPU configured. The full acceptance test is still **not declared complete** because RF-DETR, DINOv3, real video, and an interactive browser-driven React Flow run could not all be exercised with the artifacts available in this workspace.

## Known limitations and deliberate deviations

- API routes and UI widgets are grouped by cohesive modules rather than creating a one-file stub for every diagram entry; runtime separation is preserved.
- The preview dock shows the rendered image and all timing/object/VRAM metrics, but does not yet provide a side-by-side original/result toggle.
- Execution events use a consistent `{type, state, ...}` envelope; they carry the same lifecycle data as the example `{event: ...}` envelope.
- Pause/stop checkpoints occur between video frames or graph nodes; an individual third-party inference call cannot be pre-empted safely.
- Generic ONNX v0.1 assumes a single NCHW image input and exposes the first output tensor.
- RF-DETR constructor names differ across package releases; the adapter fails with a clear compatibility error if the installed package does not expose a supported local-weight argument.
- DINOv3 Transformers model metadata may require initial network access unless already cached; the supplied local state dict is still explicitly applied.
- Remote execution, Tailscale, Jetsons, GCP, RTSP, SAM, annotation/export, training, arbitrary code/shell nodes, scheduling, and distributed workers are intentionally absent.
