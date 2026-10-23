# Official RT-DETRv2 runtime

This directory vendors the Python inference implementation and configuration
files from [`lyuwenyu/RT-DETR`](https://github.com/lyuwenyu/RT-DETR), used to
load native RT-DETRv2 `.pth` state dictionaries such as the BMD-45 R101VD-X
checkpoint.

- Upstream project: `https://github.com/lyuwenyu/RT-DETR`
- Imported from: upstream `main`, 2026-10-22
- License: Apache License 2.0 (see `LICENSE` in this directory)
- Local integration: the upstream `src` package directory was renamed to
  `rtdetrv2_official` to avoid collisions with unrelated top-level `src`
  packages. Model code and configuration contents are otherwise unchanged.

