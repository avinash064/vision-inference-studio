Vision Inference Studio 0.3.0 - Linux x86_64 test build
=======================================================

Requirements on the test machine
  - 64-bit Linux (x86_64) with a desktop session, glibc 2.28+ (Ubuntu 20.04+, Debian 11+)
  - No system Python needed     the app ships its own Python 3.12.15
  - Internet on first launch    the app downloads its inference runtime
                                (PyTorch, Ultralytics, RF-DETR, ONNX Runtime; several GB)
  - Optional: NVIDIA GPU + driver for CUDA (otherwise CPU)
  - About 15 GB free disk space (the runtime alone is about 10 GB)

Option A - Debian package (Ubuntu/Debian, needs sudo)
  sudo apt install ./vision-inference-studio_0.3.0_amd64.deb
  Launch "Vision Inference Studio" from the app menu.
  Remove with: sudo apt remove vision-inference-studio

Option B - Portable, per-user install (any distro)
  tar -xzf vision-inference-studio-0.3.0-linux-x64.tar.gz
  cd vision-inference-studio-0.3.0-linux-x64
  ./install.sh --weights ../vision-inference-studio-weights.tar
  Launch from the app menu or run ~/.local/bin/vision-inference-studio
  On Ubuntu 23.10+ install.sh asks for sudo once to set up Chromium's sandbox helper.
  Remove with: ./uninstall.sh   (add --purge to also delete runtime, weights and data)

  Extra install.sh options:
    --setup-runtime   create the inference runtime now, in the terminal, instead of on first launch
    --prefix DIR      install somewhere other than ~/.local/opt/vision-inference-studio

Model weights (both options)
  vision-inference-studio-weights.tar holds the YOLO and RF-DETR weights used in testing.
  With Option A, import them once with:
    mkdir -p ~/.local/share/vision-inference-studio
    tar -xf vision-inference-studio-weights.tar -C ~/.local/share/vision-inference-studio
  Or upload weight files from inside the app.

First launch
  The app shows a "Runtime Setup" screen and installs its inference runtime
  into ~/.local/share/vision-inference-studio/runtime. Your system Python is not changed.
  Use Runtime > Status to check that CUDA is detected.

Where things live
  Runtime, weights, workflows, benchmarks, logs:  ~/.local/share/vision-inference-studio/
  Window and Electron settings:                   ~/.config/vision-inference-studio/
  Logs to send back if something fails:           ~/.local/share/vision-inference-studio/logs/
                                                  (runtime-setup.log, backend.log, electron.log)
