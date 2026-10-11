# Vision Inference Studio v0.5

A local, node-based computer-vision inference workbench built independently with React Flow and FastAPI. Compose a typed DAG, upload assets and model weights through opaque IDs, validate the workflow, and run image or sequential video inference while results stream back over WebSocket.

> Status: v0.5 adds concurrency-safe cached models, run-scoped cleanup, metadata-driven chaiNNer values, persisted package settings, lazy streams, and a GPU-preserving DINOv3 feature runtime with normalize, difference, feature-preview, and heatmap-preview nodes. Existing v0.4 prompt models, image workflows, and the dedicated file-video engine remain compatible.

## Screenshots

![Vision Inference Studio running Grounding DINO text-prompt detection on an image](docs/images/studio-image-grounding-dino.png)

*Image workflow: Image → Model (official Grounding DINO, text prompt `person, tie`) → Detection Preview. The Inspector shows the model and prompt settings; Live Output shows the labelled boxes with timing and GPU memory (RTX 4050, FP16).*

![Vision Inference Studio streaming SAM 2.1 segmentation over a video](docs/images/studio-video-sam2.png)

*Video workflow: Video → Model (SAM 2.1 base-plus, box prompt) → Video Preview, captured mid-run. Each 1080p frame is segmented at about 13 FPS; Pause / Resume / Stop act at frame boundaries.*

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
ExecutionPlan → ExecutionEngine + ExecutionScope → WebSocket events
            │                 │
            │                 ├── ChainnerNodeExecutor → typed lazy StreamValue
            │                 │                    regular / generator / transformer / collector
            │                 └── FeatureStore → normalize / difference / heatmap
                              │
                              ▼
                ModelManager immutable-key cache
                              │
                    AdapterRegistry / ModelAdapter
                    ┌─────────┼─────────┬──────────┬────────────────┬──────┐
                  YOLO     RF-DETR    DINOv3     ONNX       Grounding DINO  SAM / SAM 2 / SAM 3
                    └─────────┼─────────┴──────────┴────────────────┴──────┘
                              ▼
             InferenceResult or FeatureArtifact
                              │
                              ▼
                       RendererRegistry
                 detection / segmentation / feature / tensor

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

## What changed in v0.5

- `ModelConfig` remains the workflow/API shape, but execution splits it into immutable `ModelKey` identity and per-call `InferenceRequest` settings. Cached `LoadedModel` objects contain only identity/checkpoint data. A load lock prevents duplicate deserialization, while a separate adapter-sized semaphore (default `1`) serializes unsafe inference handles without leaking confidence or prompt state.
- Every run owns an `ExecutionScope`. Node callbacks execute LIFO when that node ends; chain callbacks, open streams, and run-owned feature artifacts execute/release LIFO when the workflow completes, errors, or is cancelled. Cached model handles remain reusable across runs.
- Runtime values explicitly distinguish images, tensors, features, heatmaps, streams, video, and model results. `ChainnerValueAdapter` consults the declared Navi port type, so a 2D/3D NumPy array is no longer assumed to be an image.
- chaiNNer generators, regular stream mapping, and transformers are lazy. Collectors consume one item at a time, stream length is retained when supplied, cancellation/pause checks occur between items, and non-replayable fan-out is rejected with `non_replayable_stream_fanout` instead of hidden materialization.
- chaiNNer package defaults are discovered from the imported registry. User overrides persist in `VIS_CONFIG_DIR/chainner-settings.json`, are editable from the header settings dialog, and are deep-snapshotted at run start.
- DINOv3 now returns a spatial `[Hf, Wf, C]` `FeatureArtifact`. Patch tokens are separated from CLS/register tokens using processed dimensions and model patch size. The full tensor stays on its backend device; WebSocket/API payloads contain only an opaque feature ID, shape/device/dtype/memory metadata, summary statistics, and inference metrics.
- `Feature Normalize` supports `none` and channel-wise L2 normalization. `Feature Difference` supports raw, absolute, L2, and cosine modes after exact feature-space compatibility checks. Raw/L2/cosine keep `delta = current - reference`; absolute emits the absolute delta. Heatmaps are interpolated only while rendering.
- Feature storage is bounded by `VIS_FEATURE_STORE_MAX_ENTRIES` and `VIS_FEATURE_STORE_MAX_BYTES`. Planner consumer counts drive reference release, and the run scope guarantees final cleanup. `GET /api/v1/features/stats` exposes counts and CPU/GPU bytes, never tensors.

## Repository tree

```text
vision-inference-studio/
├── backend/
│   ├── Dockerfile
│   ├── inference_studio/
│   │   ├── adapters/          # YOLO/RF-DETR/DINOv3/ONNX/Grounding DINO/SAM/SAM 2/SAM 3
│   │   ├── api/               # versioned HTTP and WebSocket routes
│   │   ├── assets/            # secure asset records/storage
│   │   ├── chainner/          # dynamic registry/type/image/execution compatibility
│   │   ├── features/          # feature schemas/store/ops/visualization public API
│   │   ├── graph/             # schema, validation, planning, execution, persistence
│   │   ├── models/            # manifest, registry, weights, lifecycle manager
│   │   ├── nodes/             # definitions, registry, runtime values
│   │   ├── renderers/         # detection, feature, tensor, registry
│   │   ├── results/           # standardized Pydantic results/metrics
│   │   ├── runtime/           # explicit values, streams, artifacts, execution scope
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
├── weights/{yolo,rfdetr,dinov3,grounding-dino,sam,sam2,sam3,onnx}/
├── workflows/examples/
├── workspace/{assets,previews,reports,chainner-storage}/
├── docker-compose.yml
├── Makefile
├── pyproject.toml
├── requirements.txt
├── requirements-chainner-standard.txt
├── requirements-prompt-models.txt
└── requirements-runtimes.txt
```

Runtime uploads, generated previews, locally saved workflows, virtual environments, and model weights are git-ignored.

## Dependencies

Backend baseline: Python 3.10+, FastAPI, Pydantic 2, Uvicorn, OpenCV headless, NumPy, httpx, PyYAML, multipart upload support, and WebSockets.

Optional inference bundle: PyTorch, Ultralytics, ONNX Runtime GPU, RF-DETR, Transformers, and Safetensors. Transformers provides the SAM (v1) and DINOv3 runtime APIs; Ultralytics runs YOLO, SAM 2/2.1 and SAM 3. `requirements-prompt-models.txt` adds the official [IDEA-Research Grounding DINO](https://github.com/IDEA-Research/GroundingDINO) package and the CLIP tokenizer SAM 3 needs (pinned commits, built from source). They are isolated in `requirements-runtimes.txt` so API/graph development and unit tests do not force several gigabytes of ML packages onto every machine.

Local chaiNNer Standard dependencies are listed in `requirements-chainner-standard.txt`. NCNN, TensorRT, and other heavy optional packages are deliberately not installed automatically. Missing optional packages make affected nodes visibly unavailable without preventing Vision or Standard nodes from loading.

Frontend: React 18, TypeScript, Vite, `@xyflow/react`, Zustand, Radix Tooltip, and Lucide icons.

## Local setup

Backend (this workstation uses the shared environment requested for the project):

```bash
cd vision-inference-studio
PYTHONPATH=backend /home/avinash/venv/bin/uvicorn \
  inference_studio.main:app --host 0.0.0.0 --port 8000
```

Prompt models (Grounding DINO, SAM 2, SAM 3) on top of the runtime bundle:

```bash
/home/avinash/venv/bin/pip install --no-deps --no-build-isolation -r requirements-prompt-models.txt
```

Grounding DINO installs without a CUDA toolkit; when its compiled ops are absent the adapter runs the package's PyTorch multi-scale deformable attention on the GPU and adapts its BERT wrapper to Transformers 5.

Hugging Face access (gated `facebook/sam3`, and the `bert-base-uncased` text encoder Grounding DINO downloads on first load) uses a token in a git-ignored `.env` at the project root, which the backend loads at startup without overriding variables already set (`VIS_ENV_FILE` points elsewhere):

```bash
echo 'HF_TOKEN=hf_...' > .env && chmod 600 .env
```

Point the backend at the extracted chaiNNer source without embedding a workstation-specific path in application code:

```bash
export VIS_CHAINNER_SRC_DIR=/path/to/local/chainner/src
/home/avinash/venv/bin/python -m pip install -r requirements-chainner-standard.txt
```

When `VIS_CHAINNER_SRC_DIR` is unset, development builds also detect a sibling `src/` directory. Discovery writes its full compatibility record to `workspace/reports/chainner-node-import.json`.

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
4. Choose device, precision, and confidence. Grounding DINO also requires a comma-separated text prompt such as `skip car, truck, person`. SAM and SAM 2 require either a point prompt (`x,y`) or box prompt (`x1,y1,x2,y2`) in source-image pixels. SAM 3 accepts a text prompt (every instance of each comma-separated concept, labeled by concept) or a point/box prompt (one object). The inspector shows only the thresholds a family uses (`controls` in `config/models.yaml`).
5. Add Detection, Segmentation, Feature, Tensor, or Video Preview as appropriate and connect typed handles.
6. Select **Validate**, then **Run**. Nodes visibly progress through waiting → loading/running → success/error, active edges animate, and results stream into the dock.
7. Toggle **Original** / **Prediction** in the dock and inspect preprocessing, inference, postprocessing, total time, object count, GPU allocated/reserved/peak memory, and device name.
8. Video decoding is sequential; Pause/Resume/Stop operate at frame checkpoints without buffering the whole file.
9. Resize **Live Output** by dragging its top grip to change height or its right-side divider to change width. The selected dimensions persist across reloads and desktop restarts.
10. Watch the header's CPU and GPU rings for utilization; their detail lines show RAM, VRAM, and GPU temperature and refresh every 1.5 seconds.

The Node Library groups imported entries from registry metadata under CHAINNER STANDARD, PYTORCH, ONNX, NCNN, TENSORRT, and EXTERNAL. Search covers name, description, package, category, group, and schema ID. Imported controls and conditional enum groups are rendered by the same generic Inspector used by native nodes. Images cross the Vision/chaiNNer boundary in memory (`uint8` ↔ normalized `float32`) without PNG/JPEG serialization.

For video, configure the Video source with `frame_stride`, warm-up frames, and an optional benchmark-frame limit. A stride of 1 infers every source frame; a stride of 5 infers source frames 0, 5, 10, and so on while retaining source timestamps. Models are loaded once by the Model node and reused for every frame.

Saved example shapes are under `workflows/examples/`; replace their placeholder asset and weight IDs with values returned by uploads. `image-chainner-resize.json` demonstrates imported preprocessing with a direct image preview, `image-grounding-dino.json` demonstrates text-prompt detection, and `image-sam.json` demonstrates box-prompt segmentation.

## Supported nodes

| Category | Nodes | Input → output |
|---|---|---|
| Sources | Image, Video | asset ID → `image` / `video` |
| Model | Universal Model | `image` or `video` → registry-derived result |
| Processing | Resize, Crop, Threshold | `image` → `image` |
| chaiNNer imported | Registry-derived Standard/PyTorch/ONNX/NCNN/TensorRT/External nodes | Native typed values, including sequences |
| Outputs | Image Preview | processed `image` → live preview |
| Outputs | Detection Preview | `detections` → rendered image |
| Outputs | Segmentation Preview | `segmentation` → translucent masks and boundaries |
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
| Grounding DINO | official | tiny (Swin-T OGC), base (Swin-B) | text-prompt detections | official IDEA-Research `groundingdino` |
| Segment Anything | v1 | vit-b, vit-l, vit-h | point/box-prompt segmentation | Transformers/PyTorch |
| Segment Anything 2 | 2.1, 2.0 | tiny, small, base-plus, large | point/box-prompt segmentation | Ultralytics |
| Segment Anything 3 | 3.0 | base | text-prompt concept segmentation, point/box segmentation | Ultralytics |
| Generic ONNX | generic | custom | tensor | ONNX Runtime |

DINOv3 returns a feature map and is never represented as an object detector. Grounding DINO turns text concepts into labeled boxes. SAM is promptable segmentation rather than text detection: use a foreground/background point or bounding box. Combining Grounding DINO boxes with SAM masks remains out of scope; v0.5 does not pretend that SAM itself accepts text.

SAM (v1) initializes its matching Hugging Face architecture/processor and then applies the uploaded local checkpoint using data-only loading (`weights_only=True` or Safetensors). Uploaded checkpoints must match the selected size.

Checkpoints for the other prompt families are the official releases, uploaded as-is:

- **Grounding DINO**: `groundingdino_swint_ogc.pth` (tiny) or `groundingdino_swinb_cogcoor.pth` (base) from the IDEA-Research GitHub releases. Loaded data-only (`weights_only=True`); a checkpoint for the other size is rejected.
- **SAM 2 / 2.1**: Meta's `sam2.1_hiera_{tiny,small,base_plus,large}.pt` (or `sam2_hiera_*.pt` for 2.0), or the Ultralytics mirrors `sam2.1_{t,s,b,l}.pt`. Ultralytics selects the SAM 2.1 architecture from the file name, so the adapter exposes the stored upload under its canonical name while building; the state dict loads strictly, so a version/size mismatch fails.
- **SAM 3**: `sam3.pt` from the gated [`facebook/sam3`](https://huggingface.co/facebook/sam3) repository (3.45 GB). Text prompts run at SAM 3's native 1008 px; point/box prompts run at 644 px because the interactive predictor exceeds a 6 GB GPU at 840 px and above. Only one of the two predictors stays resident, so switching prompt mode reloads (~10 s). The default upload limit is 5 GiB so SAM ViT-H fits; override it with `VIS_MAX_UPLOAD_BYTES` when needed.

## API

```text
GET  /health
GET  /api/v1/nodes
GET  /api/v1/models
GET  /api/v1/models/loaded
GET  /api/v1/devices
GET  /api/v1/system/metrics
GET  /api/v1/features/stats
GET  /api/v1/chainner/report
GET  /api/v1/chainner/settings
PUT  /api/v1/chainner/settings
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
2. Return a standard `DetectionResult`, `SegmentationResult`, `FeatureResult`, or `TensorResult`; never draw in the adapter.
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

The suite covers the v0.4 graph/video/model behavior plus immutable model identity, request isolation, inference serialization, execution-scope cleanup, explicit runtime values, metadata-driven ndarray conversion, settings persistence/snapshots, lazy single-pass streams, stream fan-out rejection, feature-store refcounts/capacity/run cleanup, feature-space matching, all four difference modes, identity/localized-change behavior, and non-square DINO patch reconstruction. GPU/model integration remains separate because it needs runtime-specific weights and drivers.

## Video runtime and benchmark behavior

- `VideoReader` owns OpenCV open, sequential decode, seek, metadata, and close behavior. Frames are never accumulated in memory.
- The executor owns stride scheduling and calls the frame-level request-aware adapter contract. Adapters contain no video loops.
- The Model node loads or reuses one immutable-key cached handle before frame iteration. Confidence/prompts remain per-request and can change without checkpoint deserialization.
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
| RF-DETR real inference | **Passed on CUDA** using the official legacy `rf-detr-large.pth` and saved `wf_rfdetracceptance` graph. It produced 1 detection at 0.489 confidence (COCO `train`, a false positive on this industrial scene; it was shown as `truck` until the rfdetr 1.6 COCO label shift was fixed in the adapter); cold total was 749.8 ms with 534.6 MB allocated, 820 MB reserved, and 682.9 MB peak GPU memory. The prediction preview was visually checked. |
| DINOv3 real feature output | Not run; no compatible DINOv3 checkpoint was available |
| DINOv3 feature unit acceptance | Passed with backend-device tensors, non-square `2 × 3 × 4` reconstruction, special-token exclusion, opaque results, feature compatibility, identity, and localized-change tests |
| Cached-model concurrency | Passed: one load for concurrent same-key requests, independent prompt/confidence values, and maximum active inference of one with the default semaphore |
| Lazy chaiNNer sequence | Passed using real imported Range → Enumerate → Sequence Length with 100,000 items; traced peak was 0.93 MB after startup and no sequence list was materialized |
| Feature operations (synthetic CUDA) | Passed on RTX 4050 at `32 × 48 × 384` FP16 (1,179,648 bytes): warm median normalize 0.135 ms, cosine difference 0.388 ms; same-feature max distance `2.38e-7`; localized max `0.319` |
| FeatureStore stress | Passed 50 put/release cycles; final entries, CPU bytes, and GPU bytes were all zero |
| SAM prompt segmentation | **Passed on CUDA/FP16** through the API with the `facebook/sam-vit-base` safetensors checkpoint: box prompt → bus mask 1.00 (602 ms cold), point prompt → person mask 0.86 (144 ms). |
| Grounding DINO (official) real inference | **Passed on CUDA/FP16** through the API with `groundingdino_swint_ogc.pth`: on Ultralytics' `bus.jpg`, "bus, person" gave bus 0.85 and all four people 0.63–0.83 in 534 ms (2.0 GB peak). A Swin-T checkpoint selected as `base` is rejected. |
| SAM 2.1 real inference | **Passed on CUDA/FP16** through the API with Meta's `sam2.1_hiera_base_plus.pt`: box prompt → bus mask 0.98 (108 ms), point prompt → person mask 0.94 (80 ms), 2.5 GB peak. The same checkpoint selected as version 2.0 is rejected. |
| SAM 3 real inference | **Passed on CUDA/FP16** through the API with `facebook/sam3` `sam3.pt`: text "person, bus" → four person masks 0.95–0.98 and the bus 0.97 (901 ms, 3.7 GB peak); box → bus 0.98; point → person 0.65 (3.8 GB peak). Meta's own `sam3` 0.1.4 package returns no detections under torch 2.5.1, so the adapter uses Ultralytics. |
| YOLO real video inference | **Passed on CUDA** on the 1920×1080, 20 FPS `cam04_44m_acceptance.mp4`: 2 warm-up + 100 measured frames, 27.7 ms median / 31.0 ms p95 inference, 37.4 ms median end-to-end, 35.9 model FPS, 26.6 end-to-end FPS, and 262 MB peak allocated VRAM |
| RF-DETR real video inference | **Passed on CUDA** using the identical video graph and source: 2 warm-up + 100 measured frames, 83.1 ms median / 100.8 ms p95 inference, 94.5 ms median end-to-end, 11.6 model FPS, 10.3 end-to-end FPS, and 762 MB peak allocated VRAM |
| Video controls | Browser-driven pause held the displayed frame stable, resume advanced from the preserved position, and cached-model stop runs reached `CANCELLED` for both families |
| Long-run memory/retention | A 320-frame real 1080p clip processed 302 inferred frames (300 measured). RSS grew 14.6 MB after warm-up/model reuse, GPU allocation remained bounded, and only ten preview files were retained for the completed run |

The current normal suite passes in `/home/avinash/venv` (`46 passed, 5 optional real-model tests skipped`), and the React production build passes under Node 22. With the available local weights configured, the dedicated real CUDA integration set passes YOLO and RF-DETR image inference plus ten video frames for each family (`4 passed`). A real imported Range → Enumerate → Sequence Length generator/transformer/collector chain also passes through the lazy compatibility executor.

With `VIS_TEST_VIDEO`, both checkpoint variables, CUDA, and a 100-frame target configured, the dedicated real-video integration module passes both families (`2 passed`), loading each checkpoint once and applying the same frame-level adapter contract.

## Known limitations and deliberate deviations

- API routes and UI widgets are grouped by cohesive modules rather than creating a one-file stub for every diagram entry; runtime separation is preserved.
- The preview dock provides an original/prediction toggle rather than a simultaneous side-by-side layout.
- Execution events include both a category in `type` and an explicit lifecycle name in `event`.
- Pause/cancel checkpoints occur between video frames or graph nodes; an individual third-party inference call cannot be pre-empted safely.
- Generic ONNX v0.1 assumes a single NCHW image input and exposes the first output tensor.
- The installed RF-DETR runtime recognizes the available large checkpoint as a legacy architecture and automatically selects its deprecated compatibility configuration; this is functional but emits an upstream warning.
- DINOv3 Transformers model metadata may require initial network access unless already cached; the supplied local state dict is still explicitly applied.
- A compatible DINOv3 checkpoint is not present in this workspace, so real-camera identity/change metrics and DINO GPU latency remain unmeasured. The artifact/token/operation path is covered with synthetic backend tensors; optional real tests should use `VIS_DINO_WEIGHTS`, `VIS_DINO_TEST_REFERENCE`, and `VIS_DINO_TEST_CURRENT` when a checkpoint is supplied.
- SAM (v1) may need initial Hugging Face access for processor/config metadata, and Grounding DINO downloads its `bert-base-uncased` text encoder on first load, unless already cached. Only uploaded local model state is applied as the selected checkpoint.
- Grounding DINO without its compiled CUDA ops uses the package's PyTorch deformable attention (same result, slower); install with a CUDA toolkit to build the kernel.
- The prompt-model packages build from source and are not yet part of the packaged desktop runtime, which installs prebuilt wheels only.
- Loading SAM 3 needs about 6–7 GB of system RAM while Ultralytics reads the 3.45 GB checkpoint.
- SAM, SAM 2 and SAM 3 segment single images or video frames independently. Automatic mask generation, SAM 2/3 video tracking memory, and Grounding-DINO-to-SAM box chaining are not yet implemented.
- The current local chaiNNer report has 17 unavailable registrations because their optional PyTorch restoration, NCNN, or TensorRT dependencies are not installed. They remain searchable and show the exact import failure.
- Common enum-dependent chaiNNer input groups are dynamic. Type-expression-only visibility conditions are preserved as metadata but remain visible because the browser does not yet evaluate the full Navi type language.
- Remote execution, Tailscale, Jetsons, GCP, RTSP, annotation/export, training, arbitrary code/shell nodes, scheduling, and distributed workers are intentionally absent.

## Conclusion

Vision Inference Studio v0.5 turns the v0.4 graph into a safer long-lived execution runtime. Model cache identity can no longer be overwritten by another request, chaiNNer resources now obey node/run lifetimes, streams stay lazy, and ndarray semantics come from port metadata. The feature path retains real DINO tensors behind bounded opaque references and supports continuous normalized delta/heatmap analysis without serializing the tensor or silently comparing incompatible feature spaces.

Acceptance is **PASS for the implemented v0.5 source milestone**: 46 tests pass, the production frontend builds, real YOLO/RF-DETR image and file-video CUDA regressions pass, the 100,000-item imported chaiNNer stream stays bounded, and synthetic CUDA feature identity/change/memory checks pass. Real DINO checkpoint/camera acceptance is the one hardware-artifact limitation and is explicitly not claimed.

RTSP, Tailscale, remote Jetsons, GCP, annotation/export, training, Mahalanobis analysis, clustering, thresholds, tracking, scheduling, multi-GPU coordination, and distributed workers remain intentionally out of scope.
