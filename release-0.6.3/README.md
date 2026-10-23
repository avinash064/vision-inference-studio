# Vision Inference Studio v0.6.3

Vision Inference Studio v0.6.3 is a production-oriented Linux desktop release for local image/video inference workflows, model comparison, immutable experiments, and recipe promotion.

## Release highlights

- Added an RF-DETR **Backbone patch size** control to the universal Model node.
- Validated RF-DETR architecture combinations before model loading:
  - Base: patch 14
  - Nano, Small, Medium: patch 16
  - Large legacy: patch 14 via `RFDETRLargeDeprecated`
  - Large current: patch 16 via `RFDETRLarge`
- Added native execution for the BMD-45 RT-DETRv2 R101VD-X `.pth` checkpoint using the official PyTorch runtime.
- Kept RT-DETRv2, Ultralytics RT-DETR, and RF-DETR as distinct model families.
- Added PaddleOCR, Tesseract, and remote OpenAI-compatible vLLM vision/OCR adapters and OCR previews.
- Included the expanded YOLO family manifest, SAM-family prompting, Grounding DINO text prompting, Public Hub credentials/downloads/previews, immutable experiments, and promoted recipes.
- Fixed packaged Electron profile initialization so upgrades and single-instance handling use the stable application configuration directory.
- Preserved support for the configured `/home/avinash/venv/bin/python` runtime.

## Install

Download `vision-inference-studio_0.6.3_amd64.deb`, then run:

```bash
sudo apt install ./vision-inference-studio_0.6.3_amd64.deb
```

Existing weights, workflows, benchmarks, provider settings, and workspace data are preserved during upgrade.

## Verification

- Backend: **120 passed, 9 skipped**.
- Frontend TypeScript and production bundle: passed.
- Desktop TypeScript and Linux packaging: passed.
- Installed Debian application startup: passed.
- Real BMD RF-DETR Base patch-14 CUDA FP16 load/inference: passed.
- Real BMD RT-DETRv2 R101VD-X CUDA FP16 load/inference: passed.

## SHA-256

```text
68c600d015d2829b13bbba92a5d4a23f1027dc482f32be21fe8fffa10509c336  vision-inference-studio_0.6.3_amd64.deb
1f7d52647f4824cf1d54cfb05a1522997ec93e27bffa8f8cb3f752438f54e576  Vision Inference Studio-linux-x64-0.6.3.zip
```

See the project-level [`README.md`](../README.md) for the complete model matrix, setup, workflow guide, configuration reference, security model, API overview, and troubleshooting.
