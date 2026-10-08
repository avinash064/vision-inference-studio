Vision Inference Studio 0.4.0 - Linux x86_64 test build
=======================================================

Models included: YOLO, RF-DETR, SAM, SAM 2.1, SAM 3, Grounding DINO (+ chaiNNer nodes)

Requirements on the test machine
  - 64-bit Linux (x86_64) with a desktop session, glibc 2.28+ (Ubuntu 20.04+, Debian 11+)
  - No system Python needed     the app ships its own Python 3.12.15
  - Internet on first launch    the app downloads its inference runtime (about 4 GB)
  - Optional: NVIDIA GPU + driver 525+ for CUDA (otherwise CPU); SAM 3 needs about 4 GB of GPU memory
  - About 20 GB free disk space (runtime about 10 GB, model weights about 7 GB)
  - 16 GB RAM recommended (loading SAM 3 briefly needs 6-7 GB)

Option A - Debian package (Ubuntu/Debian, needs sudo)
  sudo apt install ./vision-inference-studio_0.4.0_amd64.deb
  mkdir -p ~/.local/share/vision-inference-studio
  cp -a ../data/. ~/.local/share/vision-inference-studio/
  Launch "Vision Inference Studio" from the app menu.
  Remove with: sudo apt remove vision-inference-studio

Option B - Portable, per-user install (any distro)
  tar -xzf vision-inference-studio-0.4.0-linux-x64.tar.gz
  cd vision-inference-studio-0.4.0-linux-x64
  ./install.sh --data ../data
  Launch from the app menu or run ~/.local/bin/vision-inference-studio
  On Ubuntu 23.10+ install.sh asks for sudo once to set up Chromium's sandbox helper.
  Remove with: ./uninstall.sh   (add --purge to also delete runtime, weights and data)

  Extra install.sh options:
    --setup-runtime   create the inference runtime now, in the terminal, instead of on first launch
    --prefix DIR      install somewhere other than ~/.local/opt/vision-inference-studio

First launch
  The app shows a "Runtime Setup" screen and installs its inference runtime
  into ~/.local/share/vision-inference-studio/runtime. Your system Python is not changed.
  All models in data/ appear in the Model node's Weights list without uploading.
  Use Runtime > Status to check that CUDA is detected.

Hugging Face token (optional)
  Not needed for the included models. To let the app download other gated models, put
    HF_TOKEN=hf_...
  in ~/.config/vision-inference-studio/.env

Where things live
  Runtime, weights, workflows, benchmarks, logs:  ~/.local/share/vision-inference-studio/
  Window and Electron settings, optional .env:    ~/.config/vision-inference-studio/
  Logs to send back if something fails:           ~/.local/share/vision-inference-studio/logs/
                                                  (runtime-setup.log, backend.log, electron.log)
