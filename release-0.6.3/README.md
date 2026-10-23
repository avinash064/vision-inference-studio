# Vision Inference Studio v0.6.3

Production-oriented Linux desktop release for local image/video inference workflows.

## Highlights

- Native official RT-DETRv2 R101VD-X execution for the BMD-45 `.pth` checkpoint.
- RF-DETR backbone patch-size control with validated size compatibility:
  - Base: patch 14
  - Nano, Small, Medium: patch 16
  - Large: patch 14 legacy or patch 16 current
- Explicit Large runtime selection (`RFDETRLargeDeprecated` for patch 14,
  `RFDETRLarge` for patch 16).
- Existing OCR, YOLO, SAM, Grounding DINO, Public Hub, experiments, recipes,
  and local `/home/avinash/venv` runtime support remain intact.

## Verification

- Backend: 120 passed, 9 skipped.
- Frontend TypeScript and production bundle: passed.
- Desktop TypeScript and Debian packaging: passed.
- Real BMD RF-DETR Base patch-14 CUDA FP16 load/inference: passed.
- Real BMD RT-DETRv2 R101VD-X CUDA FP16 load/inference: passed.

Install the Debian asset with:

```bash
sudo apt install ./vision-inference-studio_0.6.3_amd64.deb
```

The application can continue to use `/home/avinash/venv/bin/python` via
`~/.config/vision-inference-studio/python-runtime`.
