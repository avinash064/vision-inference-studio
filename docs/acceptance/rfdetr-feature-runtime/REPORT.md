# RF-DETR Feature Runtime — Acceptance Report

Date: 2026-10-11 · Machine: RTX 4050 Laptop (6 GB), torch 2.5.1+cu121, `/home/avinash/venv` · Precision: fp32 · Device: `cuda:0`

Real model weights and real camera frames were used for every acceptance number below. No mocked tensors appear in this report. Raw evidence: `acceptance.json`, `graph.json` and `regression.json`. The scripts that produced them are in this folder.

| Provider | Weight | SHA-256 |
|---|---|---|
| RF-DETR v2.2 (`RFDETRLargeDeprecated`) | `wgt_rfdetrlargelegacy` (rf-detr-large.pth) | `86879a31a42ea4c0…f529cba` |
| RF-DETR Architecture B (`RFDETRSegLarge`, 504, patch 12) | `wgt_25f75bfb824b441989bd35410b886a7b` | `f1edd91a3b41a0d1…64332ccb` |
| MAH geometry | `geo_archb_b2_frozen` (μ [256], Cholesky [256,256], LedoitWolf) | Bound to the Architecture B SHA above |

Inputs:
- Camera pair: cam04 video frames 0 and 100, 1080×1920. Fixed camera, not registered.
- Localized change: a solid patch at y 360–540, x 960–1200 pasted onto frame 0.

## 1. Feature shape and source parity

| Provider:stage | Shape | dtype / device | MB | Parity vs original `Skip_car` source |
|---|---|---|---|---|
| v2.2:fine_80 | 80×80×384 | float32 / cuda:0 | 9.375 | max \|Δ\| = **0.0** vs `candidate_attention/rfdetr_xor_runtime.py` |
| v2.2:coarse_20 | 20×20×384 | float32 / cuda:0 | 0.586 | max \|Δ\| = **0.0** |
| arch-b:backbone_42 | 42×42×256 | float32 / cuda:0 | 1.72 (5.42 with shared srcs/masks/poss) | max \|Δ\| = **0.0** vs `rfdetr_rsd_v1/extract_rfdetr_features.py` |
| MAH map | 42×42 | float32 | — | max relative Δ = **0.0** vs B1 `mahalanobis_map` |

- Vendored modules are byte-identical to the source: SHA-256 prefixes are pinned and checked by `test_vendored_source_files_are_unmodified`.
- v2.2 levels are selected by shape. Strides are 7.0 (fine) and 28.0 (coarse).
- Architecture B runs one backbone forward. The semantic heads (`pred_masks` [1,200,126,126], `pred_logits` [1,200,4]) were run from the same cached `srcs/masks/poss`, with no second backbone pass.

## 2. Identity (same frame as reference and current)

| Provider | L2 max | Cosine max | Raw Δ max | XOR max | MAH |
|---|---|---|---|---|---|
| v2.2 fine | 0 | 0 | 0 | 0 | — |
| v2.2 coarse | 0 | 0 | 0 | 0 | — |
| arch-b | 0 | 0 | 0 | — | flat 12.672 at every token (= μᵀΣ⁻¹μ, the expected μ offset; range 0) |

The graph runs through the live API give the same values (`identity_v22_xor`, `identity_archb_mah` in `graph.json`).

## 3. Localized change (changed region vs the rest)

| Provider / map | Changed mean | Unchanged mean | Ratio | Tokens in patch |
|---|---|---|---|---|
| v2.2 fine L2 | 5.283 | 0.355 | **14.9×** | 140 |
| v2.2 fine XOR (p90) | 0.162 | 0.011 | **14.4×** | 140 |
| v2.2 coarse L2 | 4.034 | 0.460 | **8.8×** | 12 |
| v2.2 coarse XOR | 0.112 | 0.014 | **7.7×** | 12 |
| arch-b L2 | 2.234 | 0.274 | **8.2×** | 42 |
| arch-b MAH | 967.1 | 37.8 | **25.6×** | 42 |

## 4. Real camera pair (frame 0 → frame 100): heatmap statistics

| Map | min | mean | median | p90 | p95 | p99 | max |
|---|---|---|---|---|---|---|---|
| v2.2 fine L2 | 0.473 | 2.699 | 1.995 | 5.674 | 7.109 | 9.469 | 12.004 |
| v2.2 fine cosine | 0.0007 | 0.031 | 0.012 | 0.091 | 0.141 | 0.240 | 0.382 |
| v2.2 fine XOR | 0.005 | 0.086 | 0.065 | 0.182 | 0.227 | 0.299 | 0.367 |
| v2.2 coarse L2 | 0.488 | 2.652 | 2.118 | 5.578 | 6.569 | 9.587 | 10.317 |
| v2.2 coarse XOR | 0.008 | 0.081 | 0.065 | 0.169 | 0.198 | 0.253 | 0.297 |
| arch-b L2 | 0.245 | 0.923 | 0.675 | 1.978 | 2.675 | 3.408 | 4.142 |
| arch-b cosine | 0.004 | 0.074 | 0.027 | 0.241 | 0.397 | 0.527 | 0.619 |
| arch-b MAH | 26.67 | 233.7 | 111.3 | 702.4 | 981.2 | 1435.0 | 1896.0 |

- **XOR:** the map is continuous. The 0.32 threshold is not applied.
- **MAH:** τ_MAH = 589.3056640625 is not applied. It is stored as geometry metadata only; for information, 221 of 1764 tokens are ≥ τ.
- **Graph vs direct numbers:** the live-API graph runs match these numbers to within about 1%. The API input is a JPEG (q95) upload, not the raw decoded frame.
- **Visual check:** `heat_sheet.jpg` shows both providers concentrating the change in the same upper-right region.

> **Caveat:** `geo_archb_b2_frozen` was fitted on deltas to a *registered* reference. This pair is unregistered fixed-camera frames, so the absolute MAH levels show what the runtime computes; they do not calibrate τ for this camera.

## 5. Per-stage performance (warm, CUDA-synchronised, ms)

| Stage | v2.2 fine_80 | v2.2 coarse_20 | arch-b backbone_42 |
|---|---|---|---|
| Preprocess | 3.71 | 4.02 | 7.38 |
| Forward | 70.72 | 63.25 | 18.99 |
| **extract_ms** | **74.43** | **67.27** | **26.37** |
| Store put | 0.07 | 0.07 | 0.05 |
| **difference_ms** (L2) | **0.64** | 0.35 | 0.37 |
| Normalize | 0.43 | 0.37 | 0.35 |
| XOR (p90 reference quantiles) | 14.31 | 0.72 | — |
| MAH (float64 Cholesky solve) | — | — | 6.94 |
| **preview_ms** (heatmap rendered to 1920×1080) | **11.89** | 9.70 | 6.27 |

- **Cold first forward:** v2.2 took about 200 ms; Architecture B took 21 ms.
- **First L2 call:** the first L2 call on the 80×80 grid took 83 ms, which is one-off CUDA kernel warm-up. Steady state is the 0.64 ms in the table.
- **Graph wall time:** a full graph runs in 0.1–0.5 s once models are loaded.

## 6. Model reuse

| Check | Result |
|---|---|
| v2.2, across fine_80 and coarse_20, reference and current, 5 graph runs plus direct calls | `load_count` **1** |
| Architecture B, across reference, current, MAH and semantic heads | `load_count` **1** |
| Cache key (live `/models/loaded`) | `feature:rfdetr:v2.2:RFDETRLargeDeprecated:wgt_rfdetrlargelegacy:cuda:0:fp32` |

- The cache key holds family / version / variant / weight / device / precision only. It contains no stage, confidence or reference/current.
- The RF-DETR *detector* on the same `.pth` is a separate cache entry (`rfdetr:v1:large:…:fp16`), and both coexist.

## 7. Feature-store stress (50 iterations: extract reference and current → put → L2 → put → release)

| Provider | Store at end | Peak entries | CUDA before | After iter 1 | Steady median | After | Growth iter 10→50 |
|---|---|---|---|---|---|---|---|
| v2.2 fine_80 | 0 entries / 0 B | 3 | 720.1 MB | 729.5 MB | 729.5 MB | 729.5 MB | **0.0 MB** |
| arch-b | 0 entries / 0 B | 3 | 729.5 MB | 723.6 MB | 723.6 MB | 723.6 MB | **0.0 MB** |

- The constant +9.4 MB in the v2.2 run is the last hook capture held by the runtime. It does not grow.
- The live backend store held 0 entries / 0 bytes after every graph run (peak 4 entries, 39 MB GPU).

## 8. Requirement checklist

| Requirement | Status |
|---|---|
| v2.2 provider: fine_80 80×80×384, coarse_20 20×20×384, shape-based level discovery, source code reused verbatim | PASS |
| Architecture B: 504 px, patch 12, backbone_42 42×42×256, one forward, `semantic_outputs()` reuses srcs/masks/poss | PASS |
| DINOv3 still supported | PASS (unit and `test_features_v05` tests; real run skipped, see §9) |
| Feature store: put/get/retain/release/release_run/stats, `feat_<uuid>`, stays on CUDA, env limits, `FeatureStoreCapacityExceeded`, no silent eviction, shared forward bytes counted | PASS |
| Model cache key excludes stage, confidence and reference/current; one load serves both | PASS |
| Feature Extract stage options depend on provider (`options_by`, inspector resets dependent fields) | PASS |
| Feature Difference raw/absolute/l2/cosine; `FeatureSpaceMismatch` names both spaces; never resizes | PASS |
| Feature Normalize returns a new artifact (unit-norm error ≤ 1.8e-7) | PASS |
| Activation XOR: v2.2 only (validator and runtime), percentile 90, continuous, no 0.32 threshold | PASS |
| MAH: Architecture B only, frozen μ and Cholesky by opaque `geometry_id` (no client paths), Cholesky solve without an inverse, `MahGeometryMismatch`, τ not applied | PASS |
| Heatmap and Feature Preview interpolate for display only | PASS |
| The 5 new test files | PASS: 31 unit tests plus 2 real-model tests |
| Real acceptance: identity, localized change, camera pair, reuse, stress, per-stage timing | PASS |

## 9. Regression

| Check | Result |
|---|---|
| Full backend pytest, real v2.2 and Architecture B tests enabled | **79 passed**, 5 skipped (the other real-weight tests, run separately below) |
| Real integration (`VIS_TEST_DEVICE=cuda:0`): YOLO image, RF-DETR image, YOLO video, RF-DETR video | **4 passed** |
| Real DINOv3 integration | **Not run.** `weights/dinov3/` is empty on this machine |
| Frontend `tsc -b --noEmit` and `vite build` | PASS (433.9 kB JS) |
| Live API: YOLO image (4 detections), RF-DETR detector (bus 0.97, 5 detections), chaiNNer invert → YOLO, YOLO video (10 frames) | PASS, no errors |
| Packaging: wheel built from `pyproject.toml` contains all 4 `vendor/` modules and no geometry files; `stage-resources.mjs` copies the whole `inference_studio/` tree; no new runtime dependency | PASS |
| Full `.deb` rebuild | PASS for build and contents: `release-0.5.0/vision-inference-studio_0.5.0_amd64.deb` (176 MB, SHA-256 `e8ac0ed9…59c67136`) contains the feature runtime and the new frontend, with no weights, geometries, `.env` or HF token. **Not yet installed and launched** |

The integration tests have a test-order issue that predates this work: `test_yolo_real` defaults to `cpu`, and Ultralytics' `select_device("cpu")` sets `CUDA_VISIBLE_DEVICES=""` for the whole process. As a result, a CUDA video test that runs afterwards fails unless `VIS_TEST_DEVICE=cuda:0` is set.

## Verdict

**RF-DETR feature runtime: PASS.** Every functional and real-model acceptance criterion passed on the actual v2.2 and Architecture B feature paths. Parity with the original source modules is bit-exact.

**Overall: PARTIAL.** Two items remain:
1. The real DINOv3 run: no DINOv3 weights are available on this machine.
2. Installing and launching the 0.5.0 `.deb`: it is built and its contents are checked, but it has not been installed or run.

Both become PASS once those two runs succeed.
