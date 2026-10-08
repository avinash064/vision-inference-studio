# Vision Inference Studio 0.3.0 — Linux install guide

Vision Inference Studio is a desktop app for running image and video inference workflows
(YOLO, RF-DETR, DINOv3, ONNX) locally, on an NVIDIA GPU or the CPU.

This bundle is a **test build for 64-bit Linux (x86_64)**.

**New in 0.3.0:**
- **No system Python needed.** The app ships its own Python 3.12.15, so it works the same whether the
  machine has Python 3.10, 3.14 or no Python at all.
- **Pinned packages.** The inference runtime installs exact package versions from a lock file that was
  tested with real YOLO and RF-DETR inference.

---

## 1. What's in this bundle

| File | Purpose |
|---|---|
| `vision-inference-studio_0.3.0_amd64.deb` | Installer for Ubuntu / Debian (**Option A**) |
| `vision-inference-studio-0.3.0-linux-x64.tar.gz` | Portable build with `install.sh` for any distro, no root needed (**Option B**) |
| `vision-inference-studio-weights.tar` | Model weights used in testing: YOLO (`best.pt`) and RF-DETR large |
| `SHA256SUMS` | Checksums to confirm the files copied correctly |
| `README.md` | This guide |

You only need **one** of Option A or Option B. The weights file works with both.

---

## 2. Requirements

| Requirement | Details |
|---|---|
| OS | 64-bit Linux (x86_64) with a desktop session (GNOME, KDE, …) and glibc 2.28 or newer (Ubuntu 20.04+, Debian 11+, Fedora 29+) |
| Internet | Needed on **first launch** to download the inference runtime (PyTorch and CUDA libraries, about 4 GB download) |
| Disk | About **15 GB** free: the installed runtime is about 10 GB (mostly NVIDIA CUDA libraries) plus 1.6 GB of weights |
| GPU (optional) | NVIDIA GPU with driver **525 or newer** (`nvidia-smi` should work). Without one, the app uses the CPU. |

Python is **not** required.

Check the GPU driver (optional):

```bash
nvidia-smi
```

---

## 3. Unpack and verify

```bash
unzip vision-inference-studio-0.3.0-linux-x64-bundle.zip
cd vision-inference-studio-0.3.0-linux-x64-bundle
sha256sum -c SHA256SUMS
```

All three lines should end in `OK`.

---

## 4. Install

### Option A — Debian package (Ubuntu / Debian, recommended)

```bash
sudo apt install ./vision-inference-studio_0.3.0_amd64.deb
```

This replaces 0.2.0 if it is installed.

Then import the model weights (once per user; skip this if you already did it for 0.2.0):

```bash
mkdir -p ~/.local/share/vision-inference-studio
tar -xf vision-inference-studio-weights.tar -C ~/.local/share/vision-inference-studio
```

Start **Vision Inference Studio** from the application menu.

### Option B — Portable install (any distro, per user)

```bash
tar -xzf vision-inference-studio-0.3.0-linux-x64.tar.gz
cd vision-inference-studio-0.3.0-linux-x64
./install.sh --weights ../vision-inference-studio-weights.tar
```

`install.sh` does the following:

- Copies the app to `~/.local/opt/vision-inference-studio`.
- Creates a launcher at `~/.local/bin/vision-inference-studio` and an application-menu entry.
- Imports the weights.
- On Ubuntu 23.10 and later, **asks for your sudo password once** to set up Chromium's sandbox
  helper. If you skip this, the app still runs, but with `--no-sandbox` (fine for testing).

Start **Vision Inference Studio** from the application menu, or run:

```bash
~/.local/bin/vision-inference-studio
```

Optional `install.sh` flags:

| Flag | Effect |
|---|---|
| `--weights <file or folder>` | Import weights from a `.tar` file or a folder |
| `--setup-runtime` | Create the inference runtime now, in the terminal, instead of on first launch |
| `--prefix <dir>` | Install somewhere other than `~/.local/opt/vision-inference-studio` |

---

## 5. First launch: runtime setup

On the first start the app shows **Inference Runtime Setup**.

1. Click **Install Runtime**.
2. Wait while it downloads and installs PyTorch, Ultralytics, RF-DETR and ONNX Runtime.
   This takes **5–20 minutes**, depending on your connection.
3. When it finishes, the main studio window opens and shows **BACKEND ONLINE**.

The runtime is installed only for your user, in `~/.local/share/vision-inference-studio/runtime`.
You only do this once.

**Upgrading from 0.2.0:** the app shows **Runtime update required**. Click **Update Runtime**. The old
runtime was built with the system Python and is replaced automatically. Your weights, workflows and
benchmarks are kept.

Check GPU support with **Runtime → Status** in the menu bar. The status should show CUDA as available
if an NVIDIA GPU and driver are present.

---

## 6. Quick test

1. Add an **Image** node from the library and upload a test image.
2. Add a **Model** node. Choose family **YOLO** or **RF-DETR**, then pick the imported weight.
3. Set the device to the GPU (CUDA) or the CPU.
4. Add a **Detection** preview node and connect the nodes.
5. Click **Validate**, then **Run**.
6. In the bottom dock, toggle **Original / Prediction**. Check the detection count, the timings and the
   GPU memory figures.

Expected result: detections drawn on the image, with timing and GPU metrics shown.
For RF-DETR large, a "deprecated model configuration" warning in the logs is normal.

---

## 7. Troubleshooting

| Problem | Fix |
|---|---|
| Runtime setup fails while downloading | Check your internet connection or proxy, then use **Runtime → Repair Runtime**. Details are in `runtime-setup.log`. |
| App window never appears (portable install) | Run `~/.local/bin/vision-inference-studio` in a terminal and read the error. A sandbox message means `chrome-sandbox` is not set up; rerun `./install.sh` and enter your sudo password, or keep using the launcher (it falls back to `--no-sandbox`). |
| `bad option: --no-sandbox` when starting from a VS Code terminal | VS Code sets `ELECTRON_RUN_AS_NODE`. Start the app from the application menu, or run `env -u ELECTRON_RUN_AS_NODE vision-inference-studio`. |
| **Backend could not start** screen | Click **Retry**. If it fails again, use **Runtime → Repair Runtime** and check `backend.log`. |
| Portable app moved to another folder and the backend fails | The runtime is tied to the app's location. Use **Runtime → Reinstall Runtime**, or rerun `./install.sh`. |
| CUDA not available in **Runtime → Status** | Make sure `nvidia-smi` works and the driver is 525 or newer, then use **Runtime → Reinstall Runtime**. |
| Weights not listed in the Model node | Confirm that `~/.local/share/vision-inference-studio/weights/yolo/` and `.../rfdetr/` contain `wgt_*.json` and weight files, then restart the app. You can also upload weights from inside the app. |

### Log files

Help → **Open Logs Folder**, or look in `~/.local/share/vision-inference-studio/logs/`:

- `runtime-setup.log` — runtime installation
- `backend.log` — inference service
- `electron.log` — desktop app start-up

If you report a problem, please send these three files along with the output of `nvidia-smi`.

---

## 8. Where things are stored

| Path | Contents |
|---|---|
| `~/.local/share/vision-inference-studio/runtime/` | Inference runtime (Python packages) |
| `~/.local/share/vision-inference-studio/weights/` | Model weights |
| `~/.local/share/vision-inference-studio/workflows/` | Saved workflows |
| `~/.local/share/vision-inference-studio/workspace/` | Uploaded images/videos and previews |
| `~/.local/share/vision-inference-studio/benchmarks/` | Benchmark results |
| `~/.local/share/vision-inference-studio/logs/` | Logs |
| `~/.config/vision-inference-studio/` | Window and app settings |

---

## 9. Uninstall

**Option A (.deb):**

```bash
sudo apt remove vision-inference-studio
```

**Option B (portable):**

```bash
cd vision-inference-studio-0.3.0-linux-x64
./uninstall.sh            # removes the app, keeps your data
./uninstall.sh --purge    # also removes runtime, weights, workflows and logs
```

Uninstalling keeps your data folder (`~/.local/share/vision-inference-studio`). To remove everything:

```bash
rm -rf ~/.local/share/vision-inference-studio ~/.config/vision-inference-studio
```
