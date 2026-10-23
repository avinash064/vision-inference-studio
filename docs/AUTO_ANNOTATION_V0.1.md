# Auto Annotation v0.1

Auto Annotation v0.1 is the production-oriented local batch-inference module integrated into Vision Inference Studio v0.6.3. It executes an exact promoted Vision recipe over an approved image folder or a streamed video, applies an explicit retention policy, and produces a validated, portable dataset package.

## Responsibility boundary

Vision Studio remains the owner of model, preprocessing, inference, postprocessing, experiment comparison, and recipe promotion. Auto Annotation owns batch source traversal, deterministic frame sampling, policy evaluation, item checkpoints, retained-media export, annotation conversion, packaging, and validation.

It does not implement training, custom review/HITL, CVAT synchronization, RTSP ingest, cloud SDK uploads, a second model runtime, arbitrary Python, Kafka, or Redis.

```text
Canvas experiment
      │
      ▼
promoted immutable recipe ── preflight
      │
      ▼
image folder / streamed video ── deterministic sampling
      │
      ▼
existing ExecutionEngine + ModelManager
      │
      ▼
CanonicalPrediction ── deterministic retention
      │
      ▼
retained media ── canonical / COCO / YOLO
      │
      ▼
manifest + metadata + checksums ── package validator
```

## Operator guide

### 1. Prepare or pull a promoted recipe

Build and validate a Canvas workflow with one image source, one model node, compatible preprocessing and postprocessing, and an optional compatible preview. You can either evaluate variants through Experiments and promote the chosen variant, or open **AA New Run** and choose **Pull current Canvas into AA**. The direct action is explicit and creates a new read-only recipe snapshot; Auto Annotation never executes the mutable live Canvas graph. It freezes the workflow, model/checkpoint identity, checkpoint SHA-256, preprocessing hash, inference hash, and postprocessing hash. Experiment-promoted recipes additionally retain benchmark identity and experiment provenance.

Auto Annotation only lists `PROMOTED` recipes. Full runs require an explicit `recipe_id` and `recipe_version`; they never silently select the latest version.

### 2. Approve local source roots

Set `VIS_AUTO_ANNOTATION_INPUT_ROOTS` to an `os.pathsep`-separated set of directories. On Linux this is a colon-separated list:

```bash
export VIS_AUTO_ANNOTATION_INPUT_ROOTS=/data/camera-a:/data/inspection-images
```

Image-folder paths and direct video paths are resolved and must remain inside these roots. Uploaded video uses the existing opaque `AssetStore` ID and does not need a browser-visible path. Preparation hashes the ordered image inventory or complete video. The snapshot hash is durable and a changed source is rejected on preparation or resume; start a new run for changed media.

### 3. Inspect and preview

Open **AA New Run**.

- Image folders report source count, readable count, failures, dimensions, and a bounded inventory. Recursive traversal is optional; order is the stable relative-path order.
- Video reports codec, dimensions, nominal FPS, duration, total frames, and decoder time base.
- Video sampling preview decodes representative selected frames and stores only a bounded preview cache.
- Inference preview sends representative images through the actual promoted graph, displays its rendered result when available, and reports prediction count, classes, confidence values, latency, and the retention decision.

Preview is advisory and does not create a durable package run.

### 4. Choose video sampling

Sampling policies are deterministic and stateful:

| Mode | Behavior |
|---|---|
| Target FPS | Keeps the first decoded frame in each `1 / target_fps` timestamp bucket |
| Interval seconds | Keeps the first decoded frame in each fixed time interval |
| Every N frames | Keeps every Nth frame, anchored at the configured start time |

Optional start/end times bound the selection. The execution pass uses decoded presentation timestamps when OpenCV exposes them and falls back to the nominal frame clock otherwise. The video is decoded sequentially; the complete video and all sampled frames are not loaded into memory.

### 5. Choose retention

Retention evaluation order is fixed:

1. minimum confidence;
2. minimum bounding-box area;
3. optional class match;
4. total or per-class count rule;
5. global minimum time since the last retained video frame.

Modes are all sampled items, any qualifying prediction, minimum total object count, class present, and minimum class count. All successfully inferred samples are retained by `ALL_SAMPLED`, including samples with zero predictions. Temporal spacing is global within the run, not per class.

Retention changes the data distribution. Operators should compare source, sampled, inferred, retained, discarded, and failed counts and preserve `run.json` alongside the dataset.

### 6. Monitor, cancel, and resume

Run lifecycle:

```text
CREATED → VALIDATING → PREPARING → RUNNING
        → FINALIZING_MEDIA → EXPORTING_ANNOTATIONS
        → BUILDING_MANIFEST → VALIDATING_PACKAGE → COMPLETED
```

Cancellation moves through `CANCELLING` to `CANCELLED` at a safe item/frame boundary. Run and item documents are written with atomic replacement. On application startup, active work is requeued; a run in `CANCELLING` becomes `CANCELLED`. Resume reuses terminal `EXPORTED`, `DISCARDED`, and `FAILED` item checkpoints, so those inputs are not duplicated.

Individual unreadable files and isolated inference errors are recorded and processing continues. Recipe/checkpoint/device failures are systemic. Three consecutive inference failures are also treated as systemic and stop the run.

### 7. Use the package

Completed runs expose **Open output folder** in the desktop app. In a browser this button copies the path. The upload section only generates a CLI command; it never executes it:

```bash
gcloud storage rsync -r /path/to/AA-aa_<id> gs://bucket/prefix/AA-aa_<id>
aws s3 sync /path/to/AA-aa_<id> s3://bucket/prefix/AA-aa_<id>
```

Review the destination and command before running it in a terminal with your own cloud credentials.

## Exact recipe preflight

Preflight checks:

- requested recipe/version exists and is `PROMOTED`;
- stored workflow matches its SHA-256 hash;
- exactly one model node exists;
- model family, version, size, and patch size are supported;
- required adapter/runtime dependency is installed;
- checkpoint exists, belongs to the family, and matches the promoted SHA-256;
- selected CUDA device is available and CPU is not paired with FP16;
- prompt, OCR, endpoint, input-port, node-availability, and graph rules pass;
- at least one image-source node can accept the injected batch item.

The resulting `RecipePin` is stored in both durable run state and `recipe.json`.

## Canonical prediction contract

Every prediction includes:

- schema version and deterministic prediction ID;
- source item ID;
- category ID/name and confidence;
- geometry type with XYXY bounding box and/or polygons;
- adapter-specific attributes such as OCR text or segmentation area;
- complete recipe/workflow/model/checkpoint provenance;
- portable image or video-frame provenance, including frame index, timestamp, PTS, and time base where applicable.

Current conversions are:

| Runtime result | Canonical output |
|---|---|
| Detection | `BBOX`, XYXY coordinates |
| Segmentation | `POLYGON`, source polygons plus XYXY envelope |
| OCR | `BBOX` plus text and language attributes |
| Feature/tensor | No annotation predictions |

An unsupported runtime result type fails that item explicitly instead of guessing a conversion.

## Export semantics

Canonical JSONL is the authoritative model-independent export. COCO uses stable, sorted categories with 1-based category IDs and converts XYXY boxes to XYWH. Segmentation polygons are preserved. YOLO uses stable, sorted 0-based class IDs, normalized detection boxes, and normalized polygon rows for segmentation.

Only items in `EXPORTED` state are included. Annotation exports are produced after media finalization. `manifest.jsonl` is created after annotations and lists exactly the final media set. Package metadata and checksums are then written, and validation must pass before `ready_for_upload` becomes durable.

## Storage and recovery

Defaults:

```text
<workspace>/auto-annotation/
├── runs/                  # aa_<id>.json
├── items/                 # aai_<run>_<source>.json
└── packages/
    └── AA-aa_<id>/

<workspace>/auto-annotation-previews/ # bounded sampling previews
```

Override state and package roots independently with `VIS_AUTO_ANNOTATION_ROOT` and `VIS_AUTO_ANNOTATION_OUTPUT_ROOT`. Keep state and packages on reliable local storage with enough capacity for the worst-case retained-media estimate.

## Visual review

The **AA Review** page is a read-only view over durable canonical predictions and exported media. Select a run, filter media by annotation presence, navigate paged thumbnails, zoom the image, and toggle overlays or labels. Bounding boxes, polygons, keypoints, confidence, OCR transcription attributes, frame index, and presentation timestamp are rendered directly from the canonical item record. Reviewing an item never rewrites the immutable package.

## HTTP API

All routes are under `/api/v1/auto-annotation`:

| Method and route | Purpose |
|---|---|
| `GET /recipes` | List promoted recipe versions |
| `POST /api/v1/recipes/from-canvas` | Validate and freeze the user-selected current Canvas as an immutable recipe |
| `POST /recipes/{id}/versions/{version}/preflight` | Validate exact immutable execution inputs |
| `POST /sources/inspect` | Inspect folder/video metadata |
| `POST /previews/sampling` | Estimate and render representative sampled frames |
| `POST /previews/inference` | Run representative items through the actual recipe |
| `POST /runs` | Create and schedule a durable run |
| `GET /runs` | List runs and progress |
| `GET /runs/{id}` | Run detail, failures, and completed validation status |
| `GET /runs/{id}/review-items` | Page through exported media with optional prediction filtering |
| `GET /runs/{id}/items/{item_id}` | Return canonical predictions and review metadata for one item |
| `GET /runs/{id}/items/{item_id}/image` | Serve the run-scoped exported image for read-only review |
| `POST /runs/{id}/cancel` | Request safe cancellation |
| `POST /runs/{id}/resume` | Resume recoverable cancelled/error work |
| `POST /runs/{id}/upload-command` | Generate, but do not execute, a GCS/S3 command |

Interactive schemas are available at `/docs`.

## Package validation

The validator rejects a package when:

- required metadata or manifest files are missing;
- manifest media count differs from exported item state;
- a relative path escapes the package or a media file is absent;
- an image cannot be decoded or dimensions differ;
- a checksum differs;
- canonical confidence, category, or provenance is invalid;
- a COCO annotation references a missing image/category or has an invalid box;
- YOLO coordinates are absent or outside `[0, 1]`.

The validator is intentionally local and deterministic. It does not judge annotation accuracy.

## Development and verification

Use the workstation's configured environment:

```bash
cd vision-inference-studio
PYTHONPATH=backend /home/avinash/venv/bin/pytest -q backend/tests
cd frontend && npm run build
cd ../desktop && npm run build
```

Focused tests are in `backend/tests/test_auto_annotation.py` and cover sampling, retention, canonical conversion, approved roots, package exports, tamper detection, and API version preflight.

The v0.1 acceptance pass also used a temporary promoted YOLO v11 recipe with the real local checkpoint on CUDA/FP16 and a real 1920×1080 image. The complete manager produced three predictions, all three export formats, a final manifest and checksums, and a package validation result of `valid: true`. The temporary run did not modify saved recipes or workspace data.

## v0.1 limitations

- OpenCV is the current decoder boundary. PTS/time-base fidelity depends on its container/backend support.
- Review/HITL and CVAT import are downstream concerns; this module creates candidate annotation packages.
- Public Hub imported datasets are not direct v0.1 sources; use an approved image folder.
- Mask RLE and keypoint adapters are represented by the canonical schema but not emitted by current adapters.
- YOLO segmentation is written as normalized polygon rows; mixed geometry consumers must support the corresponding YOLO task.
- The run executor is local and process-managed; it is not a distributed scheduler.
