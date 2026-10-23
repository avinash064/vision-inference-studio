# Windows x64 desktop: acceptance report

**Version:** 0.6.3 (branch `windows-desktop`)
**Verdict:** the Windows **packaging** is complete and verified on real Windows (GitHub Actions `windows-latest`). **Production support is not yet established.** GPU inference, clean-machine installation, upgrades and Auto Annotation on Windows still need a real Windows 11 machine with an NVIDIA GPU (section 5).

## 1. What was built

| Item | Value |
|---|---|
| Installer | `Vision-Inference-Studio-Setup-0.6.3.exe` (Squirrel.Windows, per user, no administrator rights) |
| Other outputs | `vision_inference_studio-0.6.3-full.nupkg`, `RELEASES`, portable `Vision Inference Studio-win32-x64-0.6.3.zip`, `SHA256SUMS`, smoke report, Windows test results, Windows dependency resolution |
| Built by | `.github/workflows/windows-desktop.yml` on `windows-latest` |
| Bundled interpreter | CPython **3.12.15**, python-build-standalone 20261003, `x86_64-pc-windows-msvc`. SHA-256 `6fba7f2a…0402` is verified at staging |
| Managed runtime | Installed on first launch into `%LOCALAPPDATA%\Vision Inference Studio\runtime\python` from `requirements-lock.txt` (PyTorch 2.5.1+cu121), using the same lock and requirements hash as Linux |
| Code signing | Supported through CI secrets; this build is **unsigned** |

SHA-256 of the build artifacts are in [section 6](#6-checksums).

### Why the CUDA runtime isn't inside the installer

`torch-2.5.1+cu121-cp312-cp312-win_amd64.whl` alone is **2449 MB**. Squirrel.Windows and NSIS installers can't reliably exceed about 2 GB. So the installer ships the pinned interpreter, and the runtime is installed on first launch, which is the same model Linux uses. An offline runtime pack is a future option.

## 2. Dependency verification (Windows x64)

- **Full pinned set resolves for Windows:** `uv pip compile requirements-desktop.txt -c requirements-lock.txt --python-platform x86_64-pc-windows-msvc --python-version 3.12 --only-binary :all:` gives **157 packages**, all prebuilt wheels, with **zero version differences** from the Linux lock. CI repeats this and attaches `requirements-windows-resolved.txt`.
- **Genuine platform differences:** no `nvidia-*`, `triton` or `uvloop` on Windows (their packages' own markers exclude them), plus `colorama` and `tzdata`. No separate Windows requirements file is needed.
- **Key Windows wheels confirmed:** `torch`/`torchvision` 2.5.1+cu121 and 0.20.1+cu121, `paddlepaddle` 3.3.1, `paddleocr` 3.7.0, `onnxruntime-gpu` 1.26.0, `opencv` (includes FFmpeg), `chainner_ext` 0.3.10, `numba`, `rfdetr` 1.6.5.post1, `ultralytics` 8.4.50.

## 3. Tests

| Suite | Where | Result |
|---|---|---|
| Backend pytest | Linux (`/home/avinash/venv`) | **183 passed, 9 skipped** (48 new tests in `test_windows_support.py`) |
| Backend pytest | **Windows** (`windows-latest`, CPU PyTorch, same pinned versions) | **181 passed, 11 skipped**. The extra skips are tests for engines that aren't installed on the CPU-only runner. |
| Desktop `node:test` (paths, runtime priority, spawn arguments, port, exit-guard, launcher) | Linux and **Windows** | **10/10** (the POSIX-only fake-backend test skips on Windows) |
| Frontend type-check and production build | Linux and **Windows** | pass |
| Desktop TypeScript | Linux and **Windows** | pass |

**Windows-specific coverage:**
- runtime path resolution and priority (`VIS_PYTHON` > `python-runtime` file > managed);
- `%APPDATA%`/`%LOCALAPPDATA%` mapping, with spaces, Unicode and drive letters;
- Windows-invalid and reserved file names;
- absolute-path detection for both OS styles;
- portable package paths;
- COCO and manifest path checks;
- bash and PowerShell upload commands, with control-character rejection;
- backend spawn arguments (no shell);
- watchdog never calling `os.kill` on Windows;
- graceful shutdown;
- free loopback port;
- active-run exit guard;
- runtime diagnostics (no secrets, managed path hidden);
- shell-only path approval;
- archive drive-letter and Windows-name rejection;
- installer hash parity with bash and Node;
- stale lock recovery.

## 4. Packaged-app smoke test on real Windows

`packaging/windows/smoke_test.py` runs against the packaged app built by CI. Data folders contain spaces and `ü`.

| Check | Result on `windows-latest` |
|---|---|
| Bundled `python.exe` installs a managed runtime (core profile) | pass (Python 3.12.15, manifest `win32`/`x64`) |
| Backend starts the way Electron starts it, with `VIS_PARENT_PID` set to the test process | pass: healthy, and **the test process survived** (the old `os.kill(pid, 0)` watchdog would have terminated it) |
| `/health` requires the session token | pass (401 without it) |
| `/api/v1/runtime/diagnostics` | `windows`, `x64`, `3.12.15`, source `managed`, path hidden. **CUDA UNAVAILABLE** with reason "PyTorch unavailable", as expected for the core profile. Families report `DEPENDENCY_MISSING` with causes; `vllm` `AVAILABLE` |
| Graceful shutdown via `POST /api/v1/desktop/shutdown` | pass: "Application shutdown complete", "Finished server process" |
| Packaged `vision-inference-studio.exe` launch | pass: "Desktop 0.6.3 starting; platform=win32-x64; packaged=true", then "Backend ready on 127.0.0.1:…" |

The smoke test found Windows bugs, all now fixed:
1. the runtime installer crashed on non-ASCII paths (cp1252 console under `-I`), fixed with `-X utf8` and UTF-8 streams;
2. `os.getloadavg()` doesn't exist on Windows, which broke the system-metrics endpoint, fixed with `psutil.getloadavg()`;
3. Windows Git rewrote line endings, changing the vendored-source hashes, fixed with `.gitattributes`;
4. a POSIX-only file-mode assertion in a test, now checked only on POSIX.

## 5. Not verified (requires a real Windows 11 + NVIDIA machine)

| Definition-of-done item | Status |
|---|---|
| Clean-machine install (no Python, Node or npm), Start Menu entry, first-launch **Install Runtime** with the full CUDA profile | **not run**: CI runners aren't clean end-user machines, and the full runtime is 4 GB |
| CUDA detected; RF-DETR, RT-DETRv2 and YOLO CUDA/FP16 inference | **not run**: CI has no GPU. The CUDA wheels are confirmed to exist and resolve, but importing isn't the same as GPU support |
| Image and video inference, Auto Annotation (5 FPS sampling, retention, temporal spacing, exports, manifest, package validation) | **not run on Windows**. Covered by the cross-platform test suite, which passes on Windows, but not by a real Windows model run |
| Workflows, weights, recipes and Auto Annotation history persisting across restarts and an installer upgrade | **not run** (the upgrade needs two installer versions) |
| Uninstall keeping user data | by design (`src/squirrel.ts` only removes shortcuts; data is outside the install folder); **not run** |

**Suggested procedure on the target machine:**
1. Install `Setup.exe`.
2. Click **Install Runtime**.
3. Open the **Runtime** panel and check for CUDA READY.
4. Run the RF-DETR / RT-DETRv2 / YOLO acceptance workflows.
5. Run Auto Annotation on a real MP4 (5 FPS, class ≥ 1, confidence threshold, minimum spacing) and validate the package.
6. Restart, and confirm history is still there.
7. Install the next version's `Setup.exe` over it, and confirm data survives.

## 6. Checksums

Final CI run `40007592617` on `windows-desktop` (commit `3400e7c`). Every step passed. Artifact `vision-inference-studio-windows-x64`, 522 MB in total (installer, nupkg, zip, reports):

```text
818f0bff9c7eae3fa19ea757ff10aa8361f49142cf8de917f66eb2a3f303ded7  Vision-Inference-Studio-Setup-0.6.3.exe
b31a878e1c0497a26f4adb5e39197f5d83a33ec6d9c63288f34b4b38d5752887  vision_inference_studio-0.6.3-full.nupkg
5d056882f544769dc78d737e343d5bb232bc88437b3aaddb5bcce9ce6eb4c522  Vision Inference Studio-win32-x64-0.6.3.zip
0c5199716afd44be46e7611ffa139af5f5e0368f45c520ae6079722789c3e91c  RELEASES
```

These are CI test builds. The artifact expires after 30 days. A release should be cut from a tagged commit.

## 7. Linux regression

- **`.deb` build:** builds (`vision-inference-studio_0.6.3_amd64.deb`, **121 MB**, down from about 185 MB, because `app.asar` no longer duplicates resources).
- **`VIS_PYTHON=/home/avinash/venv/bin/python`:** the packaged Linux app starts with `runtime_source=env_override`. The backend is ready in 4.5 s, and quitting shuts it down gracefully with no orphan process.
- **Unchanged on Linux:** the `python-runtime` file and the bash `setup-runtime.sh` path.

## 8. Known Windows limitations

- **Online first launch:** the first launch must download the runtime (about 4 GB, needs internet).
- **Unsigned build:** this build shows a SmartScreen warning.
- **Tesseract:** optional external dependency, found via `%ProgramFiles%` / `PATH` / `VIS_TESSERACT_CMD`.
- **Architecture:** x64 only.
- **chaiNNer FFmpeg nodes:** chaiNNer video nodes that call an external `ffmpeg.exe` need FFmpeg on `PATH` or `VIS_FFMPEG`. The studio's own video path uses OpenCV's bundled FFmpeg.
