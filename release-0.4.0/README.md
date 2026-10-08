# Vision Inference Studio 0.4.0 — Linux install guide

Vision Inference Studio is a desktop app for building and running image and video inference workflows
locally, on an NVIDIA GPU or the CPU.

This bundle is a **test build for 64-bit Linux (x86_64)**. Everything is included: the app, its own
Python, and the weights for every model below.

**Models included**

| Model | What it does | Prompt |
|---|---|---|
| YOLO (custom `best.pt`) | Object detection | — |
| RF-DETR large | Object detection | — |
| Grounding DINO (Swin-T, official IDEA-Research) | Finds objects you name in text | text |
| SAM (ViT-B) | Segments the object you point at | point, box |
| SAM 2.1 (Hiera B+) | Segments the object you point at | point, box |
| SAM 3 | Segments every instance of a named concept, or the object you point at | text, point, box |

Also included: the chaiNNer image-processing nodes (162 nodes).

**New in 0.4.0**

- **Grounding DINO, SAM 2.1 and SAM 3**, all tested on an NVIDIA GPU.
- **chaiNNer nodes** in the node library.
- **No setup for the models.** Weights come with the bundle and appear in the app without uploading.
- **No system Python needed** (as in 0.3.0). The app ships Python 3.12.15 and installs exact, tested
  package versions.

---

## 1. What's in this bundle

| Item | Purpose |
|---|---|
| `vision-inference-studio_0.4.0_amd64.deb` | Installer for Ubuntu / Debian (**Option A**) |
| `vision-inference-studio-0.4.0-linux-x64.tar.gz` | Portable build with `install.sh`, for any distro, no root needed (**Option B**) |
| `data/weights/` | Weights for all six models (about 6.2 GB) |
| `data/huggingface/` | Files SAM and Grounding DINO need, so they work without internet |
| `SHA256SUMS` | Checksums to confirm the files copied correctly |
| `README.md` | This guide |

You only need **one** of Option A or Option B. Both use the `data/` folder.

---

## 2. Requirements

| Requirement | Details |
|---|---|
| OS | 64-bit Linux (x86_64) with a desktop session and glibc 2.28 or newer (Ubuntu 20.04+, Debian 11+, Fedora 29+) |
| Internet | Needed on **first launch** to download the inference runtime (PyTorch and CUDA libraries, about 4 GB) |
| Disk | About **20 GB** free: about 10 GB for the runtime plus 7 GB for the model data |
| RAM | 16 GB recommended. Loading SAM 3 briefly needs 6–7 GB. |
| GPU (optional) | NVIDIA GPU with driver **525 or newer** (`nvidia-smi` should work). SAM 3 needs about 4 GB of GPU memory; the other models need 1–2.5 GB. Without a GPU the app uses the CPU (much slower). |

Python is **not** required.

---

## 3. Unpack and verify

```bash
unzip vision-inference-studio-0.4.0-linux-x64-bundle.zip
cd vision-inference-studio-0.4.0-linux-x64-bundle
sha256sum -c SHA256SUMS
```

Every line should end in `OK`.

---

## 4. Install

### Option A — Debian package (Ubuntu / Debian, recommended)

```bash
sudo apt install ./vision-inference-studio_0.4.0_amd64.deb
mkdir -p ~/.local/share/vision-inference-studio
cp -a data/. ~/.local/share/vision-inference-studio/
```

The `cp` step installs the model data for your user. Repeat it for each user who will run the app.
The package replaces 0.2.0 or 0.3.0 if one of them is installed.

Start **Vision Inference Studio** from the application menu.

### Option B — Portable install (any distro, per user)

```bash
tar -xzf vision-inference-studio-0.4.0-linux-x64.tar.gz
cd vision-inference-studio-0.4.0-linux-x64
./install.sh --data ../data
```

`install.sh` does the following:

- Copies the app to `~/.local/opt/vision-inference-studio`.
- Creates a launcher at `~/.local/bin/vision-inference-studio` and an application-menu entry.
- Copies the model data into `~/.local/share/vision-inference-studio`.
- On Ubuntu 23.10 and later, **asks for your sudo password once** to set up Chromium's sandbox helper.
  If you skip this, the app still runs, but with `--no-sandbox` (fine for testing).

Start **Vision Inference Studio** from the application menu, or run:

```bash
~/.local/bin/vision-inference-studio
```

Optional `install.sh` flags:

| Flag | Effect |
|---|---|
| `--data <dir>` | Copy a bundle `data/` folder (weights plus Hugging Face files) |
| `--setup-runtime` | Create the inference runtime now, in the terminal, instead of on first launch |
| `--prefix <dir>` | Install somewhere other than `~/.local/opt/vision-inference-studio` |

---

## 5. First launch: runtime setup

On the first start the app shows **Inference Runtime Setup**.

1. Click **Install Runtime**.
2. Wait while it downloads and installs PyTorch, Ultralytics, RF-DETR, ONNX Runtime and the chaiNNer
   support packages. It then installs Grounding DINO and the CLIP tokenizer for SAM 3 from files
   included in the app. This takes **5–20 minutes**, depending on your connection.
3. When it finishes, the main studio window opens and shows **BACKEND ONLINE**.

The runtime is installed only for your user, in `~/.local/share/vision-inference-studio/runtime`.
You only do this once.

**Upgrading from 0.2.0 or 0.3.0:** the app shows **Runtime update required**. Click
**Update Runtime**. Your workflows and benchmarks are kept.

Check GPU support with **Runtime → Status** in the menu bar. The status should show CUDA as available.

---

## 6. Quick test

1. Add an **Image** node from the library and upload a test image.
2. Add a **Model** node. Choose a family; its weight is already listed under **Weights**.
3. Set the device to the GPU (CUDA) and precision to `fp16`.
4. Set the prompt for prompt models:
   - **Grounding DINO:** text, such as `person, car`.
   - **SAM 3:** text such as `person` to find every person, or a point `x,y` or box `x1,y1,x2,y2`
     in image pixels for one object.
   - **SAM / SAM 2.1:** a point or a box.
5. Connect a **Detection Preview** (YOLO, RF-DETR, Grounding DINO) or a **Segmentation Preview**
   (SAM, SAM 2.1, SAM 3).
6. Click **Validate**, then **Run**.

Expected result: boxes or masks drawn on the image, with timing and GPU metrics in the bottom dock.

Notes:

- For RF-DETR large, a "deprecated model configuration" warning in the logs is normal.
- SAM 3 takes about 10 seconds to load. Switching between its text and point/box modes reloads it.

---

## 7. Troubleshooting

| Problem | Fix |
|---|---|
| Runtime setup fails while downloading | Check your internet connection or proxy, then use **Runtime → Repair Runtime**. Details are in `runtime-setup.log`. |
| A model's **Weights** list is empty | The `data/` copy step was skipped. Run the `cp -a data/. ~/.local/share/vision-inference-studio/` command from section 4 and restart the app. |
| App window never appears (portable install) | Run `~/.local/bin/vision-inference-studio` in a terminal and read the error. A sandbox message means `chrome-sandbox` is not set up; rerun `./install.sh` and enter your sudo password, or keep using the launcher (it falls back to `--no-sandbox`). |
| `bad option: --no-sandbox` when starting from a VS Code terminal | VS Code sets `ELECTRON_RUN_AS_NODE`. Start the app from the application menu, or run `env -u ELECTRON_RUN_AS_NODE vision-inference-studio`. |
| **Backend could not start** screen | Click **Retry**. If it fails again, use **Runtime → Repair Runtime** and check `backend.log`. |
| "CUDA out of memory" with SAM 3 | Close other GPU programs. Other models stay loaded after a run, so restart the backend (**Runtime → Restart Backend**) before running SAM 3. |
| App killed while loading SAM 3 | Not enough RAM; close other programs (loading SAM 3 briefly needs 6–7 GB). |
| CUDA not available in **Runtime → Status** | Make sure `nvidia-smi` works and the driver is 525 or newer, then use **Runtime → Reinstall Runtime**. |

### Log files

Help → **Open Logs Folder**, or look in `~/.local/share/vision-inference-studio/logs/`:

- `runtime-setup.log` — runtime installation
- `backend.log` — inference service
- `electron.log` — desktop app start-up

If you report a problem, please send these three files along with the output of `nvidia-smi`.

---

## 8. Hugging Face token (optional)

The included models don't need one. To let the app download gated Hugging Face models, put your token
in `~/.config/vision-inference-studio/.env`:

```bash
mkdir -p ~/.config/vision-inference-studio
echo 'HF_TOKEN=hf_...' > ~/.config/vision-inference-studio/.env
chmod 600 ~/.config/vision-inference-studio/.env
```

---

## 9. Where things are stored

| Path | Contents |
|---|---|
| `~/.local/share/vision-inference-studio/runtime/` | Inference runtime (Python packages) |
| `~/.local/share/vision-inference-studio/weights/` | Model weights |
| `~/.local/share/vision-inference-studio/huggingface/` | Hugging Face model files |
| `~/.local/share/vision-inference-studio/workflows/` | Saved workflows |
| `~/.local/share/vision-inference-studio/workspace/` | Uploaded images/videos and previews |
| `~/.local/share/vision-inference-studio/benchmarks/` | Benchmark results |
| `~/.local/share/vision-inference-studio/logs/` | Logs |
| `~/.config/vision-inference-studio/` | Window and app settings, optional `.env` |

---

## 10. Uninstall

**Option A (.deb):**

```bash
sudo apt remove vision-inference-studio
```

**Option B (portable):**

```bash
cd vision-inference-studio-0.4.0-linux-x64
./uninstall.sh            # removes the app, keeps your data
./uninstall.sh --purge    # also removes runtime, weights, workflows and logs
```

Uninstalling keeps your data folder (`~/.local/share/vision-inference-studio`). To remove everything:

```bash
rm -rf ~/.local/share/vision-inference-studio ~/.config/vision-inference-studio
```
