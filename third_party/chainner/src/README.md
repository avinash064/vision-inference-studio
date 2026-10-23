# chaiNNer Processing Backend Source

This directory contains a Python source snapshot of the chaiNNer-style node-processing backend. It is a graph execution service for image processing, model inference, batch workflows, video handling, and external Stable Diffusion operations. The service discovers Python node definitions at startup, exposes their schemas over HTTP, accepts serialized processing graphs, executes those graphs in a worker process, and streams progress and preview information with Server-Sent Events (SSE).

This README documents the code that is present in this `src` directory. It is intentionally separate from the repository-level README, which describes the surrounding auto-annotation workspace.

> **Source snapshot status:** this directory contains the backend source, bundled bootstrap wheels, and node implementations. It does **not** contain the corresponding desktop/web frontend, a `pyproject.toml`, or a conventional `requirements.txt`. It should therefore be treated as an extracted backend component rather than a complete upstream application checkout.

## Contents

- [Capabilities](#capabilities)
- [Architecture](#architecture)
- [Directory map](#directory-map)
- [Requirements](#requirements)
- [Installation and startup](#installation-and-startup)
- [Command-line options](#command-line-options)
- [Startup lifecycle](#startup-lifecycle)
- [HTTP API](#http-api)
- [Graph request format](#graph-request-format)
- [SSE events](#sse-events)
- [Packages and inference backends](#packages-and-inference-backends)
- [Execution model](#execution-model)
- [Storage, logs, and downloaded tools](#storage-logs-and-downloaded-tools)
- [Adding a package or node](#adding-a-package-or-node)
- [Validation and testing](#validation-and-testing)
- [Security and deployment notes](#security-and-deployment-notes)
- [Troubleshooting](#troubleshooting)
- [Known limitations](#known-limitations)

## Capabilities

The source provides:

- A typed, decorator-driven node registration API.
- JSON graph parsing, validation through registered schemas, optimization, and execution.
- Regular, generator, collector, and transformer node kinds for both one-shot and sequence workflows.
- Individual-node execution for interactive previews.
- In-memory output caching, including reuse of individual preview results by a later graph run.
- Pause, resume, cancellation, and worker-process restart support.
- SSE streams for node lifecycle, progress, preview/broadcast data, errors, and dependency-installation progress.
- Runtime package discovery and optional dependency installation/uninstallation.
- Image operations implemented with NumPy, OpenCV, Pillow, SciPy, Numba, PyMatting, and native extensions.
- Neural-network execution through PyTorch, ONNX Runtime, NCNN/Vulkan, and NVIDIA TensorRT.
- Video frame input/output with FFmpeg.
- Calls to an external AUTOMATIC1111 Stable Diffusion Web UI API.
- CPU, RAM, and NVIDIA VRAM usage reporting.

The snapshot contains 363 Python files and 179 statically declared node registration decorators. The exact number of nodes exposed at runtime can be lower when optional imports, platform checks, GPU checks, or package dependencies are unavailable.

## Architecture

```text
Client / frontend
      |
      | HTTP + SSE
      v
server_host.py  (public host process)
      |
      | loopback HTTP proxy
      v
server.py       (restartable worker process)
      |
      +-- api/             node/package registry and schema contracts
      +-- chain/           graph parsing, optimization, and output caching
      +-- process.py       stable executor
      +-- process_new.py   experimental executor
      +-- packages/        node declarations grouped by backend
      +-- nodes/           shared input/output types and implementations
```

The two-process design separates the public API from execution state:

1. `run.py` bootstraps the server dependencies and imports `server_host.py`.
2. The host chooses an unused local port and starts `server.py` as a child process.
3. The worker imports package definitions, recursively discovers node modules, and builds the runtime registry.
4. The host checks declared dependencies and may stop the worker, install packages into the active Python environment, and restart it.
5. Once the worker reports healthy and its nodes have loaded, the host marks the backend ready.
6. Most requests are proxied to the worker. Host-only operations include dependency management, system statistics, setup events, and shutdown.

The worker is deliberately restartable. If normal cancellation fails, the host's `/kill` route restarts the worker process to recover a clean execution state.

## Directory map

| Path | Responsibility |
| --- | --- |
| `run.py` | Primary entry point. Imports the dependency bootstrap and launches the host server. |
| `server_host.py` | Public Sanic application, worker lifecycle, dependency operations, setup SSE, and system information. |
| `server.py` | Internal Sanic worker, node discovery, graph execution, individual previews, cache, and execution SSE. |
| `server_process_helper.py` | Selects a free loopback port, starts/stops the worker, proxies HTTP, and relays SSE. |
| `server_config.py` | Command-line argument model and parser shared by host and worker. |
| `api/` | Public contracts for packages, categories, node registration, typed inputs/outputs, settings, iterator metadata, and node context. |
| `chain/` | Graph node/edge structures, JSON deserialization, optimization passes, input storage, and output-cache policy. |
| `process.py` | Stable graph executor. |
| `process_new.py` | Alternative executor selected by `useExperimentalFeatures`. |
| `process_common.py` | Shared execution IDs and output representations. |
| `progress_controller.py` | Pause, resume, abort, and nested progress abstractions. |
| `events.py` | Event type definitions, async queues, filtering, and throttled progress delivery. |
| `packages/` | Built-in package metadata, settings, features, categories, groups, and registered node functions. |
| `nodes/properties/` | Reusable input/output definitions for images, files, NumPy values, and model formats. |
| `nodes/impl/` | Shared implementations for color, filters, tiling, resizing, video, inference backends, model conversion, and related operations. |
| `nodes/groups.py` | Helpers for conditional, optional, linked, and nested input layouts. |
| `navi.py` | JSON expression builders for the node type system used by the frontend. |
| `dependencies/` | Installed-package detection, disk-space checks, pip orchestration, and bundled bootstrap wheels. |
| `gpu.py` | NVIDIA/NVML discovery, architecture checks, FP16 support, and VRAM statistics. |
| `logger.py` | Host/worker console logging and rotating log files. |
| `texconv/` | Bundled Microsoft `texconv.exe` integration for DDS files on Windows. |

## Requirements

### Python

Use CPython 3.10 or 3.11. The source uses Python 3.10 features such as PEP 604 union types and `zip(..., strict=...)`. Python 3.11 is the safest practical choice for the pinned scientific stack in this snapshot; some pinned packages, notably Numba 0.57.1, may not support newer Python releases.

### System requirements

- Internet access is normally required on the first launch because missing dependencies are installed with pip.
- Sufficient disk space is required for scientific and model packages. GPU-enabled PyTorch is estimated by the code at roughly 2 GB before extraction overhead.
- An NVIDIA driver is required for CUDA PyTorch, ONNX Runtime GPU, or TensorRT.
- Vulkan-capable drivers are required for GPU-accelerated NCNN.
- Video nodes download an integrated FFmpeg build into backend storage when it is first needed.
- DDS conversion through the included `texconv` binary is Windows-only.
- External Stable Diffusion nodes require a separately running AUTOMATIC1111 Web UI with its API enabled.

## Installation and startup

### 1. Enter this source directory

The source uses top-level imports such as `import api`, so run it with `src` as the current directory or ensure this directory is on `PYTHONPATH`.

```bash
cd "/home/avinash/Downloads/Auto annotation-20260608T151220Z-3-001/Auto annotation/src"
```

### 2. Activate an isolated Python environment

Using a dedicated environment is strongly recommended because the bootstrap installs and may upgrade packages in the interpreter that launches it.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

If the existing environment requested for this workspace is appropriate, activate it instead:

```bash
source /home/avinash/venv/bin/activate
```

Check its version before launch:

```bash
python --version
python -m pip --version
```

### 3. Start the backend

```bash
python run.py 8000 --dev
```

The public API is then available at `http://127.0.0.1:8000` under Sanic's default host behavior. Confirm it from another terminal:

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/status
```

Expected health response:

```json
{"status":"ok"}
```

`/health` only confirms that the host process is listening. Use `/status` to determine whether package setup has completed and the worker is ready.

### Dependency bootstrap behavior

Importing `dependencies.install_server_deps` from `run.py` immediately checks the active environment and installs missing bootstrap dependencies. The bootstrap set includes:

- Sanic 23.3.0 and Sanic-Cors 2.2.0
- aiofiles, html5tagger, sanic-routing, tracerite, and websockets
- aiohttp and psutil
- pynvml
- typing-extensions
- `chainner-pip`, used for dependency installation with progress reporting

Several pure-Python wheels are bundled in `dependencies/whls/`; packages not available there are fetched from package indexes. Standard image dependencies are checked automatically during host setup. Other backend dependencies are installed when requested through the package API or when `--install-builtin-packages` is supplied.

To preinstall every declared built-in backend, run:

```bash
python run.py 8000 --install-builtin-packages
```

This can download several gigabytes and is unnecessary if only the standard CPU image nodes are needed.

## Command-line options

The same parser is used by the host and worker.

| Argument | Default | Description |
| --- | --- | --- |
| `port` | `8000` | Public host port. The worker receives a separate dynamically allocated port. |
| `--close-after-start` | off | Close after setup completes. Intended for CI/startup validation. |
| `--install-builtin-packages` | off | Install dependencies declared by all built-in packages. |
| `--error-on-failed-node` | off | Treat a failed node import as a startup error. |
| `--storage-dir PATH` | platform app-data path, then temp fallback | Persistent directory available through `NodeContext.storage_dir`. |
| `--logs-dir PATH` | platform-specific chaiNNer log directory | Directory for `host.log` and `worker.log`. |
| `--trace` | off | Enable VizTracer around full graph execution. VizTracer must already be installed. |
| `--dev` | off | Print prefixed worker output and disable file logging. |

Example with explicit writable directories:

```bash
mkdir -p ./runtime/storage ./runtime/logs
python run.py 8123 \
  --storage-dir ./runtime/storage \
  --logs-dir ./runtime/logs
```

CI-oriented startup check:

```bash
python run.py 8000 --close-after-start --error-on-failed-node
```

## Startup lifecycle

The setup flow matters when integrating a frontend:

1. The host starts and `/health` becomes available.
2. The host creates the internal worker on a dynamically selected port.
3. The worker imports these packages: standard, PyTorch, NCNN, ONNX, TensorRT, and external.
4. Each package directory is scanned recursively for non-underscore Python files.
5. Successfully imported decorators populate the node registry.
6. The host retrieves package metadata and determines which dependencies are missing or outdated.
7. If installation is needed, it stops the worker, installs dependencies, and starts the worker again.
8. The host waits for node loading and changes `/status` from `ready: false` to `ready: true`.

Subscribe to `/setup-sse` before or during setup to display `backend-started`, `backend-status`, and `package-install-status` updates.

## HTTP API

Unless noted otherwise, clients call the public host port. Most routes are forwarded unchanged to the internal worker.

| Method | Route | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Lightweight host health check; does not wait for node setup. |
| `GET` | `/status` | Host readiness and nested worker execution status. |
| `GET` | `/nodes` | All registered node schemas, categories, groups, inputs, outputs, type expressions, and UI metadata. |
| `GET` | `/packages` | Package metadata. Add `?hideInternal=false` to include the standard package. |
| `GET` | `/installed-dependencies` | Installed versions for dependencies declared by known packages. |
| `GET` | `/features` | Runtime state of feature checks, such as external services. |
| `GET` | `/python-info` | Active Python executable and version. |
| `GET` | `/system-usage` | CPU, RAM, and discovered NVIDIA VRAM percentages. |
| `GET` | `/sse` | Long-lived graph execution event stream. |
| `GET` | `/setup-sse` | Long-lived setup and dependency progress stream. |
| `POST` | `/run` | Parse, optimize, and execute a full graph. The HTTP request remains open until execution ends. |
| `POST` | `/run/individual` | Execute one regular or generator node, primarily for interactive previews. |
| `POST` | `/clear-cache/individual` | Remove one individual-preview result from the worker cache. |
| `POST` | `/pause` | Pause the active full-graph executor. |
| `POST` | `/resume` | Resume the active full-graph executor. |
| `POST` | `/kill` | Abort execution; the host restarts the worker if regular cancellation fails. |
| `POST` | `/packages/install` | Install dependencies for the specified package IDs, restarting the worker. |
| `POST` | `/packages/uninstall` | Uninstall dependencies for the specified package IDs, restarting the worker. |
| `POST` | `/shutdown` | Stop the worker and public host. |

### Status response

```json
{
  "ready": true,
  "worker": {
    "executor": "ready"
  }
}
```

Worker executor states are `ready`, `running`, `paused`, and `killing`.

### Install or uninstall package dependencies

```bash
curl -X POST http://127.0.0.1:8000/packages/install \
  -H 'Content-Type: application/json' \
  -d '{"packages":["chaiNNer_onnx"]}'
```

Use the same body with `/packages/uninstall`. Valid package IDs in this snapshot are:

- `chaiNNer_standard`
- `chaiNNer_pytorch`
- `chaiNNer_ncnn`
- `chaiNNer_onnx`
- `chaiNNer_tensorrt`
- `chaiNNer_external`

### Execution control

```bash
curl -X POST http://127.0.0.1:8000/pause
curl -X POST http://127.0.0.1:8000/resume
curl -X POST http://127.0.0.1:8000/kill
```

Successful execution-control responses use `{"type":"success"}`. Other response types include `no-executor`, `already-running`, and `error`.

## Graph request format

`POST /run` expects this envelope:

```json
{
  "data": [],
  "options": {},
  "sendBroadcastData": true,
  "useExperimentalFeatures": false
}
```

Fields:

- `data`: serialized graph nodes.
- `options`: per-package settings keyed by package ID.
- `sendBroadcastData`: whether output preview metadata should be emitted through `node-broadcast` events.
- `useExperimentalFeatures`: selects `process_new.py` when true and `process.py` when false or absent.

Each graph node has this shape:

```json
{
  "id": "node-1",
  "schemaId": "chainner:utility:pass_through",
  "nodeType": "regularNode",
  "parent": null,
  "inputs": [
    {"type": "value", "value": "hello"},
    {"type": "value", "value": null}
  ]
}
```

`nodeType` is interpreted as follows:

- `generator` creates a sequence-producing node.
- `collector` consumes a sequence.
- `transformer` consumes and emits sequences.
- Any other value creates a regular function node; clients should send `regularNode` for clarity.

An input is either a literal value:

```json
{"type":"value","value":42}
```

or an edge from an earlier node's output index:

```json
{"type":"edge","id":"source-node-id","index":0}
```

Input array positions and output indexes must match the schema returned by `/nodes`. The parser converts indexes to the registered input/output IDs. The serialized `parent` field is currently accepted by the typed request model but is not used by `chain/json.py`; iterator relationships are inferred from graph edges and node kinds.

Before execution, optimization repeatedly applies up to ten passes that:

- remove nodes with no downstream consumer and no declared side effects;
- resolve statically selected `Switch` nodes;
- remove eligible one-input/one-output pass-through nodes; and
- remove conditionals with constant conditions or identical branches.

This means a demonstration graph containing only a pure pass-through node can be optimized away. Real graphs should terminate in a side-effecting node such as a file writer when full execution is expected.

### Individual-node request

`POST /run/individual` accepts:

```json
{
  "id": "preview-node-1",
  "schemaId": "some:registered:schema",
  "inputs": ["value matching input 0"],
  "options": {},
  "useExperimentalFeatures": false
}
```

Only regular and generator nodes can be run individually. Collector and transformer nodes are rejected. A non-generator result is retained in the worker's individual cache and may be supplied to a later full execution. Clear it with:

```bash
curl -X POST http://127.0.0.1:8000/clear-cache/individual \
  -H 'Content-Type: application/json' \
  -d '{"id":"preview-node-1"}'
```

## SSE events

Listen to execution events with:

```bash
curl -N http://127.0.0.1:8000/sse
```

Each message follows standard SSE framing:

```text
event: node-start
data: {"nodeId":"node-1"}
```

### Execution stream: `/sse`

| Event | Important data | Meaning |
| --- | --- | --- |
| `chain-start` | `nodes` | IDs that will execute, excluding cache hits. |
| `node-start` | `nodeId` | A node began execution. |
| `node-progress` | `nodeId`, `progress`, `index`, `total`, `eta` | Iteration or custom node progress. `progress` is clamped to 0–1. |
| `node-broadcast` | `nodeId`, `data`, `types`, `sequenceTypes` | Preview/broadcast output plus inferred type information. |
| `node-finish` | `nodeId`, `executionTime` | Node completed; time is reported in seconds. |
| `execution-error` | `message`, `exception`, `exceptionTrace`, `source` | Execution failure, optionally with node/schema/input context. |

Progress messages are deduplicated per node and normally flushed no more often than every 50 ms. State-changing events flush pending progress first.

### Setup stream: `/setup-sse`

| Event | Important data | Meaning |
| --- | --- | --- |
| `backend-started` | `null` | Host setup began. |
| `backend-status` | `message`, `progress`, `statusProgress` | Overall startup and node-loading progress. |
| `package-install-status` | `message`, `progress`, `statusProgress` | Dependency collection/download/install progress. |

`statusProgress` is optional nested progress, used primarily for the current download.

## Packages and inference backends

### Standard

The `chaiNNer_standard` package is always considered for dependency installation. It provides image I/O, creation, resizing, cropping, filtering, adjustments, channel operations, compositing, material-texture utilities, batch processing, iteration, logic, text/number utilities, and video frame workflows.

Its pinned dependencies include NumPy 1.24.4, OpenCV 4.8.0.76, Pillow 9.2.0, AVIF support, `ffmpeg-python`, Requests, `google-re2`, SciPy 1.9.3, `wcmatch`, Numba 0.57.1, PyMatting 1.1.10, and `chainner_ext` 0.3.10.

### PyTorch

Provides model load/save, image upscaling, inpainting, face restoration, guided upscale, color correction, model interpolation/conversion, and batch model loading.

The code selects PyTorch 2.7.0 and TorchVision 0.22.0 builds based on the machine:

- Apple Silicon uses the normal macOS packages and enables MPS fallback.
- NVIDIA Pascal and older devices select CUDA 12.6 wheels.
- Newer NVIDIA devices select CUDA 12.8 wheels.
- Systems without a detected NVIDIA GPU select CPU wheels.

GPU builds have a large disk footprint. PyTorch and TorchVision have automatic updates disabled in package metadata.

### ONNX

Provides ONNX model loading/saving, optimization, metadata, interpolation, conversion to NCNN, background removal, batch loading, and image upscaling.

Runtime selection is platform-aware:

- NVIDIA detected: `onnxruntime-gpu` 1.17.1 from the configured CUDA-oriented index.
- Windows without NVIDIA: `onnxruntime-directml` 1.17.1.
- Other systems: CPU `onnxruntime` 1.17.1.

### NCNN

Provides `.param`/`.bin` model I/O, Vulkan image upscaling, model interpolation, model-scale inspection, and batch loading. It declares `ncnn-vulkan` 2023.6.18 and is the backend intended to support modern AMD as well as NVIDIA GPUs through Vulkan.

### TensorRT

Provides NVIDIA TensorRT engine load/save, engine building, engine inspection, and inference. The package declares TensorRT 10.15.1.29 and CUDA Python 13.1.1. It is marked disabled when no NVIDIA GPU is detected, and its node category is only created for a supported non-Apple-Silicon NVIDIA environment.

### External Stable Diffusion

Provides text-to-image, image-to-image, inpaint, outpaint, upscale, and CLIP interrogation against an AUTOMATIC1111 Web UI instance.

Start AUTOMATIC1111 with `--api`, then optionally configure:

| Environment variable | Default | Meaning |
| --- | --- | --- |
| `STABLE_DIFFUSION_PROTOCOL` | auto-detected | `http` or `https`. |
| `STABLE_DIFFUSION_HOST` | `127.0.0.1` | Web UI host. |
| `STABLE_DIFFUSION_PORT` | auto-detected | Web UI API port. |
| `STABLE_DIFFUSION_REQUEST_TIMEOUT` | `600` | Request timeout in seconds. |

## Execution model

### Registration and discovery

Each package creates a `Package`, adds one or more `Category` objects, then adds `NodeGroup` objects. Individual functions are registered with a group's `@register(...)` decorator. Registration assigns stable numeric input/output IDs where explicit IDs are absent and stores the function plus its schema metadata in the global registry.

At worker setup, `PackageRegistry.load_nodes()` walks every registered package directory. Files whose names begin with `_` are skipped. Import failures are grouped and logged; with `--error-on-failed-node`, startup fails instead of exposing the successfully loaded subset.

### Typed inputs and outputs

`api.BaseInput` and `api.BaseOutput` carry both Python-side enforcement and frontend-facing type expressions. The `navi.py` helpers represent literals, intervals, unions, intersections, named structural types, field access, function calls, and matches as JSON-compatible values.

Node inputs can also be:

- optional;
- lazy, so evaluation is deferred until the node requests the value;
- fused to an output for pass-through behavior;
- nested in UI groups; or
- adapted and converted with type expressions.

Outputs can publish lightweight broadcast data and a more specific runtime type for previews without serializing the entire Python value.

### Iterators

The four node kinds support sequence workflows:

- **regularNode**: a normal function node.
- **generator**: emits a sequence and optional non-sequence values.
- **collector**: consumes a sequence and produces final values or side effects.
- **transformer**: transforms one sequence into another.

Iterator metadata declares which ordinary inputs/outputs belong to the sequence and can provide sequence length and item types to the client.

### Cache behavior

The executor computes cache strategies from graph topology:

- values needed throughout an iteration are cached for the entire execution;
- ordinary outputs are retained for the number of downstream reads; and
- unconsumed values are not cached.

Full-run caches are cleared after completion. Individual previews have a separate worker-level cache keyed by node ID. Generator preview values are deliberately excluded so a later run cannot resume stale generator state.

### Concurrency and progress

The worker owns a four-thread `ThreadPoolExecutor` for blocking node work and broadcast generation. `NodeContext` exposes package settings, shared storage, cancellation checks, pause handling, custom progress, and node/chain cleanup callbacks.

Only one full-graph executor may run at a time. Individual preview executors are tracked separately by node ID; a new preview for the same ID cancels its predecessor. A full run waits for active individual previews to finish before it begins.

## Storage, logs, and downloaded tools

### Storage directory

The worker selects storage in this order:

1. `--storage-dir` supplied by the caller.
2. `appdirs.user_data_dir(roaming=True)/chaiNNer/backend-storage` when `appdirs` is available.
3. `<system temporary directory>/chaiNNer/backend-storage` as a fallback.

Node code can access this path through `NodeContext.storage_dir`. Integrated FFmpeg is downloaded beneath this storage directory the first time a video node requires it.

For reproducible operation, always pass an explicit persistent storage directory. The temp fallback may be cleaned by the operating system.

### Logs

Without `--dev`, host and worker write separate rotating files:

- Linux: `~/.config/chaiNNer/logs`
- macOS: `~/Library/Application Support/chaiNNer/logs`
- Windows: `%APPDATA%/chaiNNer/logs`
- Explicit override: `--logs-dir PATH`

Each file rotates at 10 MB and retains five backups. `--dev` disables file logging and prints worker output with a `[Worker]` prefix.

### Traces

When `--trace` is enabled, full graph execution is wrapped with VizTracer and written to `../traces/trace_<execution-id>.json` relative to the process working directory. VizTracer is not part of the bootstrap dependency set and must be installed separately.

## Adding a package or node

### Package structure

The built-in layout follows this pattern:

```text
packages/chaiNNer_example/
├── __init__.py              # package and category declarations
└── example/
    ├── __init__.py          # node-group declarations
    └── operations/
        └── double_value.py  # registered node
```

Package declaration:

```python
# packages/chaiNNer_example/__init__.py
from api import add_package

package = add_package(
    __file__,
    id="chaiNNer_example",
    name="Example",
    description="Example nodes.",
)

example_category = package.add_category(
    name="Example",
    description="Example operations.",
    icon="BsGear",
    color="#4A90E2",
)
```

Group declaration:

```python
# packages/chaiNNer_example/example/__init__.py
from .. import example_category

math_group = example_category.add_node_group("Math")
```

Node declaration:

```python
# packages/chaiNNer_example/example/operations/double_value.py
from nodes.properties.inputs import NumberInput
from nodes.properties.outputs import NumberOutput

from .. import math_group


@math_group.register(
    schema_id="example:math:double",
    name="Double",
    description="Multiply a value by two.",
    inputs=[NumberInput("Value")],
    outputs=[NumberOutput("Result")],
)
def double_node(value: float) -> float:
    return value * 2
```

To make a new top-level package load automatically, add its import to `import_packages()` in `server.py`. Package directories are scanned recursively, so the registered module itself does not need a central import once its package is known.

### Registration rules

- Use a globally unique, stable `schema_id`; existing IDs generally follow `namespace:category:operation`.
- Match function parameters to flattened input order and return annotations to output order.
- Declare `side_effects=True` for writers or external actions so graph optimization does not remove them as dead nodes.
- Use `node_context=True` when the function needs `NodeContext` for settings, storage, progress, cancellation, or cleanup.
- Use `features` to associate a node with a runtime capability check.
- Keep optional Python imports inside the relevant package/module boundary so missing optional dependencies do not prevent unrelated packages from loading.
- Add iterator input/output metadata only for generator, collector, or transformer nodes; registration asserts the permitted combinations.

## Validation and testing

No automated test directory or project-level test configuration is included in this source snapshot. The following read-only compile check was run while preparing this README and all 363 Python files compiled successfully:

```bash
python3 - <<'PY'
from pathlib import Path

for path in Path('.').rglob('*.py'):
    compile(path.read_text(encoding='utf-8'), str(path), 'exec')

print('All Python sources compiled successfully.')
PY
```

Recommended validation levels:

1. **Syntax:** run the compile check above.
2. **Bootstrap:** run with `--close-after-start`.
3. **Strict node discovery:** add `--error-on-failed-node`.
4. **All built-ins:** add `--install-builtin-packages`, understanding that this changes the environment and can download several gigabytes.
5. **Runtime smoke:** start normally, wait for `ready: true`, query `/nodes`, subscribe to `/sse`, and execute a small graph ending in a side-effecting node.

Useful smoke commands:

```bash
curl -fsS http://127.0.0.1:8000/health
curl -fsS http://127.0.0.1:8000/status
curl -fsS http://127.0.0.1:8000/python-info
curl -fsS 'http://127.0.0.1:8000/packages?hideInternal=false'
curl -fsS http://127.0.0.1:8000/nodes > /tmp/chainner-nodes.json
```

## Security and deployment notes

This backend is designed for a trusted local frontend, not as a hardened multi-user network service.

- There is no authentication or authorization.
- CORS is enabled broadly on both host and worker Sanic applications.
- Package install/uninstall routes execute package-management commands in the active Python environment.
- Graphs can read and write paths through file nodes and can call external services.
- Model and media operations can consume substantial CPU, RAM, VRAM, disk, and network bandwidth.
- `/shutdown` stops the service, and `/kill` may replace the worker process.
- Errors sent over SSE can include Python tracebacks and input context.

Keep the service bound to a trusted local interface. If remote access is required, put it behind a separately configured authenticated reverse proxy, restrict request size and routes, isolate the Python environment and OS account, and constrain filesystem/network access at the operating-system or container level.

Do not run the service with elevated privileges. Do not expose dependency-management endpoints to untrusted clients.

## Troubleshooting

### `python: command not found`

This system may expose the interpreter as `python3` until a virtual environment is activated:

```bash
python3 --version
source /home/avinash/venv/bin/activate
python --version
```

### `/health` succeeds but `/status` is not ready

The host listens before node imports and dependency checks finish. Watch setup events and logs:

```bash
curl -N http://127.0.0.1:8000/setup-sse
tail -f ~/.config/chaiNNer/logs/host.log
tail -f ~/.config/chaiNNer/logs/worker.log
```

Large PyTorch or TensorRT installs can make setup take several minutes.

### A package is listed but its nodes are missing

Run the strict startup check to expose import failures:

```bash
python run.py 8000 --close-after-start --error-on-failed-node
```

Then inspect `worker.log`. By default, non-check import failures are logged and the successfully imported node subset remains available.

### Dependency installation changes the wrong environment

`run.py` installs into `sys.executable`, the interpreter used to launch the backend. Verify it before starting:

```bash
python -c 'import sys; print(sys.executable)'
curl http://127.0.0.1:8000/python-info
```

Stop the server, activate the intended environment, and relaunch if the paths do not match.

### Insufficient disk space

The installer estimates download sizes, multiplies them to allow for extraction, and adds a 100 MB buffer. Free space or choose an environment/storage volume with more capacity. CUDA PyTorch and TensorRT are the largest declared packages.

### NVIDIA GPU is not detected

The bootstrap includes `pynvml`, but NVML also requires a functioning NVIDIA driver. Check the driver with `nvidia-smi`. If detection fails, PyTorch falls back to CPU dependencies, ONNX selects a non-CUDA runtime, TensorRT is disabled, and VRAM rows are omitted from `/system-usage`.

### NCNN fails on a GPU

NCNN uses Vulkan and depends on working Vulkan drivers. Verify the device and driver with platform Vulkan tools. The code itself describes NCNN/Vulkan GPU execution as experimental and failures can be hardware/driver-specific.

### Video nodes fail on first use

Confirm the storage directory is writable and the machine can reach the FFmpeg download URL. Integrated binaries are downloaded and extracted under `<storage-dir>/ffmpeg`. On macOS the setup also attempts quarantine removal and ad-hoc code signing.

### External Stable Diffusion feature is disabled

Start AUTOMATIC1111 with `--api`, verify its host/port, and set `STABLE_DIFFUSION_PROTOCOL`, `STABLE_DIFFUSION_HOST`, and `STABLE_DIFFUSION_PORT` explicitly if auto-detection cannot find it.

### Port is already in use

Choose another public port:

```bash
python run.py 8123 --dev
```

The internal worker port is selected dynamically and does not need to be configured.

## Known limitations

- This is a backend-only snapshot; the intended UI/client is absent.
- There is no single lockfile or project manifest for reproducing the full environment. Dependencies are distributed between the bootstrap list and package declarations.
- Runtime bootstrap mutates the active Python environment.
- Pinned dependencies span different release eras and may not all support the newest Python versions.
- The API is implementation-oriented and has no version prefix or generated OpenAPI contract.
- Requests are represented with `TypedDict` but do not pass through a dedicated external schema-validation layer before graph construction.
- Only one full graph can execute at a time.
- Execution caches are in memory and are lost when the worker restarts.
- Worker shutdown calls terminate and then kill without a graceful wait interval, so in-flight node cleanup is not guaranteed during forced restart.
- DDS `texconv` support is Windows-only.
- TensorRT is NVIDIA-only; NCNN Vulkan behavior depends on the installed GPU driver; CPU PyTorch can be very slow.
- The source retains both stable and experimental executors, which increases the maintenance and testing surface.

## Source entry points for further study

For a focused code review, read these files in order:

1. `run.py` — bootstrap and top-level launch.
2. `server_host.py` — public lifecycle and route surface.
3. `server_process_helper.py` — host/worker boundary.
4. `server.py` — registry loading and execution endpoints.
5. `api/api.py` and `api/node_data.py` — extension API.
6. `chain/json.py`, `chain/chain.py`, and `chain/optimize.py` — graph representation.
7. `process.py` and `process_new.py` — execution engines.
8. `events.py` — observable runtime protocol.
9. `packages/chaiNNer_standard/` — representative node declarations.
10. `nodes/impl/` — reusable processing and inference implementations.

