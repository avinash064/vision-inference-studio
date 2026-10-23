# Vision Inference Studio on Windows

Vision Inference Studio runs on Windows with the same architecture as on Linux. An Electron desktop shell starts a local FastAPI backend, which runs workflows, recipes and Auto Annotation through the same execution engine and model adapters. Only the parts that touch the operating system differ: packaging, starting the Python runtime, native dependencies, and file locations. Workflows, recipes, weights metadata and annotation packages are interchangeable between Windows and Linux.

> **Status: first Windows build.** The installer, runtime installer and backend are built and tested by CI on Windows. GPU inference, clean-machine installation and upgrades must be checked on real Windows hardware before production use. See [acceptance/windows/REPORT.md](acceptance/windows/REPORT.md) for what has and hasn't been verified.

## Requirements

| | |
|---|---|
| OS | Windows 11 (Windows 10 22H2 should work but is untested), **x64 only**. ARM64 and 32-bit Windows aren't supported. |
| Disk | About 1 GB for the app, plus about 10 GB for the inference runtime, plus your weights and outputs |
| RAM | 16 GB recommended (SAM 3 needs 6–7 GB while loading) |
| GPU (optional) | NVIDIA GPU with a driver that supports **CUDA 12.1** (driver 527.41 or newer). The installer never includes or changes the display driver. Without a GPU, CPU-compatible models still run. |
| Network | Needed once, on first launch, to download the inference runtime (about 4 GB) |

You don't need to install Python, Node.js or npm.

## Install

1. Download `Vision-Inference-Studio-Setup-<version>.exe` and its `SHA256SUMS` file, then check the download:
   ```powershell
   (Get-FileHash .\Vision-Inference-Studio-Setup-0.6.3.exe -Algorithm SHA256).Hash.ToLower()
   ```
   The printed value must match the line in `SHA256SUMS`.
2. Run the installer. It installs for your user only (no administrator rights) and adds Start Menu and desktop shortcuts. Unsigned test builds trigger a SmartScreen warning ("More info → Run anyway").
3. On first launch, click **Install Runtime**. The app uses its bundled Python 3.12.15 to create a private runtime with PyTorch (CUDA 12.1) and the model libraries, installed from prebuilt wheels pinned by `requirements-lock.txt`. This takes 10–30 minutes depending on your connection. **Runtime → Repair Runtime** and **Reinstall Runtime** in the app menu fix a damaged runtime.

### Why the runtime is installed on first launch

The CUDA build of PyTorch for Windows is 2.4 GB on its own. Windows installer formats (Squirrel.Windows, NSIS) can't reliably carry more than about 2 GB, so a fully offline single `.exe` isn't practical. The installer therefore ships the pinned Python interpreter, and the runtime is installed into your user profile on first launch. Linux works the same way. An offline "runtime pack" is a possible future option.

## Where things are stored

Nothing mutable is written inside the installation folder, which Squirrel replaces on every upgrade.

| Data | Location |
|---|---|
| Application (replaced on upgrade) | `%LOCALAPPDATA%\vision_inference_studio\app-<version>\` |
| Settings, `.env` secrets, chaiNNer settings, window state, approved source folders | `%APPDATA%\Vision Inference Studio\` |
| Weights, workflows, workspace, benchmarks, Public Hub downloads, Auto Annotation packages | `%LOCALAPPDATA%\Vision Inference Studio\` |
| Managed Python runtime | `%LOCALAPPDATA%\Vision Inference Studio\runtime\python\` |
| Logs (`electron.log`, `backend.log`, `runtime-setup.log`) | `%LOCALAPPDATA%\Vision Inference Studio\logs\` |

On Linux the equivalents are `~/.config/vision-inference-studio/` and `~/.local/share/vision-inference-studio/`.

## Choosing the Python runtime

The desktop app picks its Python in this order:

1. **`VIS_PYTHON`**, an environment variable pointing to a `python.exe` (for development or a custom environment);
2. **a configured runtime**: a file named `python-runtime` in `%APPDATA%\Vision Inference Studio\` that contains the path to a `python.exe`;
3. **the managed runtime** that **Install Runtime** creates (the default).

The bundled interpreter in the installation is used only to create the managed runtime. **Runtime → Status** in the menu, and the **Runtime** (CPU icon) panel in the header, show the active source, Python, PyTorch and CUDA versions, the GPU, and the status of each model family. The managed runtime's internal path is not shown.

A custom environment must contain the packages from `requirements-desktop.txt` and `requirements-public-hub.txt`, plus the prebuilt wheels in `resources\wheels`.

## CUDA troubleshooting

The Runtime panel shows either **CUDA READY** with the GPU name and memory, or **CUDA UNAVAILABLE** with the reason:

| Reason shown | What to do |
|---|---|
| "No usable NVIDIA GPU/driver detected" | Install or update the NVIDIA driver (527.41+). Run `nvidia-smi` in PowerShell to confirm the driver works. |
| "This PyTorch build has no CUDA support" | The runtime has a CPU-only PyTorch, usually a custom `VIS_PYTHON` environment. Use the managed runtime, or install `torch==2.5.1+cu121` from `https://download.pytorch.org/whl/cu121`. |
| "Native library failed to load (DLL load failed …)" | Install the latest **Microsoft Visual C++ Redistributable (x64)**, then **Runtime → Repair Runtime**. |
| "Python package not installed (…)" | **Runtime → Repair Runtime**. |

CUDA being unavailable never stops the app from starting.

## Model families and optional engines

Each model family reports one of these statuses in the Runtime panel:

| Status | Meaning |
|---|---|
| `AVAILABLE` | Its libraries import correctly. This is not the same as having been validated on Windows; see the acceptance report. |
| `DEPENDENCY_MISSING` | A package or native library is missing or failed to load. The detail line says which. |
| `CONFIGURATION_REQUIRED` | An external program is needed (Tesseract). |
| `UNSUPPORTED_PLATFORM` | Known not to work on Windows. No family is currently marked this way. |

One unavailable family never stops the others or the app.

- **Tesseract OCR** is optional and not bundled. Install it (for example the UB Mannheim Windows build). The app finds it in `%ProgramFiles%\Tesseract-OCR\`, `%LOCALAPPDATA%\Programs\Tesseract-OCR\`, on `PATH`, or wherever `VIS_TESSERACT_CMD` points.
- **PaddleOCR** uses the official Windows wheels (`paddlepaddle` 3.x). It's optional: if it fails to load, only the PaddleOCR family is unavailable.
- **vLLM** is a remote HTTP adapter and works the same on every OS.
- **Grounding DINO** uses its PyTorch fallback (no compiled CUDA ops), the same as Linux.

## Video

Video decoding uses OpenCV. Its Windows wheels include FFmpeg, so no separate FFmpeg install is needed for image or video inference, frame sampling or Auto Annotation. Some imported chaiNNer video nodes call an external `ffmpeg.exe`. For those, put `ffmpeg.exe` on `PATH` or set `VIS_FFMPEG`. The app doesn't redistribute FFmpeg executables.

If a video won't open, check that it plays in a standard player. Very unusual codecs may not be supported by OpenCV's FFmpeg build.

## Auto Annotation

Auto Annotation works the same as on Linux:
- **Choosing sources:** use **Browse…** to pick a video (**Open File**) or an image folder (**Select Folder**) with the native Windows dialogs. You don't need to type paths. Picking a file or folder in the dialog approves it as a source. The approval is remembered across restarts, and web content can't approve paths on its own.
- **Long runs:** runs can take hours. If you close the window while a run is active, the app asks whether to **Keep Running in Background** (it stays in the system tray until you choose Quit), **Cancel Run and Exit**, or **Return**. Choosing **Quit** while a run is active also asks first.
- **Output:** packages are written to `%LOCALAPPDATA%\Vision Inference Studio\workspace\auto-annotation\packages\AA-<run id>\`.

Every file inside a package uses **relative POSIX paths** (`images/aai_….jpg`): `manifest.jsonl`, COCO `file_name`, YOLO files, canonical JSONL and recipe metadata. No Windows drive paths are written, so a package made on Windows uploads to Linux-hosted CVAT unchanged. Package validation rejects absolute paths from either operating system.

### Uploading to cloud storage

The app doesn't upload anything itself. It generates commands for you to review and copy, in PowerShell form on Windows and bash form on Linux:

```powershell
gcloud storage rsync -r `
  'D:\AutoAnnotation\AA-aa_…' `
  'gs://bucket/auto-annotation/AA-aa_…'

aws s3 sync `
  'D:\AutoAnnotation\AA-aa_…' `
  's3://bucket/auto-annotation/AA-aa_…'
```

Single quotes mean PowerShell never expands `$` inside paths.

## Network and security

- The backend listens only on `127.0.0.1`, on a free port chosen at each launch. It never listens on `0.0.0.0`, so Windows Firewall shouldn't prompt.
- Every request needs a per-launch session token. Provider secrets stay in `%APPDATA%\Vision Inference Studio\.env` and are never sent to the browser.
- The app never runs shell commands built from browser input. Downloaded model code is never imported, and `trust_remote_code` stays disabled.
- When the app exits, it asks the backend to shut down gracefully, so models unload and GPU memory is released. Only if that fails does it force-stop **its own** backend process tree. Other Python processes are never touched.

## Upgrading and uninstalling

- **Upgrading:** run the newer `Setup.exe`. Squirrel installs the new version next to the old one and removes the old one. Settings, weights, workflows, recipes, benchmarks and Auto Annotation history stay in `%APPDATA%`/`%LOCALAPPDATA%`. The managed runtime is reused and updated in place.
- **Uninstalling** (Settings → Apps → Vision Inference Studio) removes the application and its shortcuts only. Your data is **kept**: weights, workflows, Auto Annotation outputs and the runtime. To remove everything, also delete `%APPDATA%\Vision Inference Studio` and `%LOCALAPPDATA%\Vision Inference Studio` after uninstalling.

## Building the Windows installer

CI builds it on `windows-latest` ([.github/workflows/windows-desktop.yml](../.github/workflows/windows-desktop.yml)). The job runs the backend tests on Windows, builds the frontend and desktop, packages with Electron Forge's Squirrel maker, smoke-tests the packaged app, and uploads the installer with its `SHA256SUMS`.

To build locally on Windows (Node 22 and Git required):

```powershell
cd frontend; npm ci; npm run build; cd ..\desktop; npm ci; npm test
$env:VIS_CHAINNER_SRC_DIR = "C:\path\to\chainner\src"
node scripts\stage-resources.mjs          # downloads and checksum-verifies the Windows CPython 3.12.15
npx electron-forge make --platform=win32 --arch=x64
```

Output: `desktop\out\make\squirrel.windows\x64\Vision-Inference-Studio-Setup-<version>.exe`, plus a portable zip under `desktop\out\make\zip\win32\x64\`.

The app folder can also be packaged from Linux (`electron-forge package --platform=win32`), but the Squirrel installer step needs Windows.

### Code signing

Local builds are unsigned. Release builds are signed when CI has the `WINDOWS_CERTIFICATE_BASE64` (a base64 `.pfx`) and `WINDOWS_CERTIFICATE_PASSWORD` secrets. Certificates and keys are never committed.

## Known limitations

- **Unverified on real hardware:** GPU acceptance (RF-DETR, RT-DETRv2, YOLO on CUDA), the clean-machine install and the upgrade test still need a real Windows 11 + NVIDIA machine.
- **Online first launch:** the first launch needs internet access to install the runtime. An offline runtime pack isn't implemented yet.
- **Tesseract** isn't bundled.
- **SmartScreen:** unsigned builds show a SmartScreen warning.
- **Architecture:** x64 only.
