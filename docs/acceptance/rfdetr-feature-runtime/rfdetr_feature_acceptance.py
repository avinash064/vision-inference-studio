"""Real-model acceptance for the RF-DETR feature runtime (v2.2 fine/coarse, Architecture B). No mocked tensors."""
import asyncio
import importlib.util
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

PROJECT = Path(sys.argv[1])
OUT = Path(sys.argv[2])
sys.path.insert(0, str(PROJECT / "backend"))

from inference_studio.adapters import AdapterRegistry  # noqa: E402
from inference_studio.config import Settings  # noqa: E402
from inference_studio.features.geometry import MahalanobisGeometryStore  # noqa: E402
from inference_studio.features.ops import activation_xor, difference, mahalanobis_difference, normalize  # noqa: E402
from inference_studio.features.store import FeatureStore, artifact_nbytes  # noqa: E402
from inference_studio.features.visualization import render_heatmap  # noqa: E402
from inference_studio.models import ModelManager, ModelRegistry, WeightStore  # noqa: E402

V22_WEIGHT, ARCHB_WEIGHT, GEOMETRY = "wgt_rfdetrlargelegacy", "wgt_25f75bfb824b441989bd35410b886a7b", "geo_archb_b2_frozen"
DEVICE, PRECISION = "cuda:0", "fp32"
VIDEO = PROJECT / "workspace/assets/ast_bffa02e274db4d168647328af3067108.mp4"
SOURCE = Path("/home/avinash/Skip_car/experiments")
report: dict = {"device": torch.cuda.get_device_name(0), "torch": torch.__version__, "precision": PRECISION}


def sync():
    torch.cuda.synchronize()
    return time.perf_counter()


def frames(indices):
    cap = cv2.VideoCapture(str(VIDEO))
    out = {}
    for index in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = cap.read()
        assert ok, index
        out[index] = frame
    cap.release()
    return out


def stats(heat):
    return {k: round(v, 6) for k, v in heat.statistics.items()}


def region_ratio(heat_tensor, rect, source_hw):
    h, w = heat_tensor.shape
    y0, y1, x0, x1 = rect
    sh, sw = source_hw
    mask = torch.zeros((h, w), dtype=torch.bool, device=heat_tensor.device)
    mask[int(y0 / sh * h): max(int(np.ceil(y1 / sh * h)), int(y0 / sh * h) + 1), int(x0 / sw * w): max(int(np.ceil(x1 / sw * w)), int(x0 / sw * w) + 1)] = True
    values = heat_tensor.float()
    changed, unchanged = float(values[mask].mean()), float(values[~mask].mean())
    return {"changed_mean": changed, "unchanged_mean": unchanged, "ratio": changed / (unchanged + 1e-8), "changed_tokens": int(mask.sum())}


async def main():
    settings = Settings.load()
    registry = ModelRegistry.load(settings.models_file)
    weights = WeightStore(settings.weights_dir, registry, settings.max_upload_bytes)
    manager = ModelManager(registry, weights, AdapterRegistry())
    geometry = MahalanobisGeometryStore(settings.geometries_dir).get(GEOMETRY)
    store = FeatureStore(max_entries=64, max_bytes=2 * 1024**3)
    pair = frames([0, 100])
    reference_img, current_img = pair[0], pair[100]
    sh, sw = reference_img.shape[:2]
    rect = (sh // 3, sh // 3 + sh // 6, sw // 2, sw // 2 + sw // 8)  # y0, y1, x0, x1
    patched = reference_img.copy()
    patched[rect[0]:rect[1], rect[2]:rect[3]] = (40, 200, 40)
    report["inputs"] = {"source_shape": list(reference_img.shape), "pair": "cam04_44m_acceptance.mp4 frames 0 and 100 (fixed camera, not registered)", "localized_patch_px": {"y": [rect[0], rect[1]], "x": [rect[2], rect[3]]}}

    for provider, weight_id, stage in (("rfdetr-v2.2", V22_WEIGHT, "fine_80"), ("rfdetr-v2.2", V22_WEIGHT, "coarse_20"), ("rfdetr-arch-b", ARCHB_WEIGHT, "backbone_42")):
        name = f"{provider}:{stage}"
        await manager.extract_feature(provider, weight_id, reference_img, stage, DEVICE, PRECISION)  # warm-up
        ref = await manager.extract_feature(provider, weight_id, reference_img, stage, DEVICE, PRECISION)
        ref2 = await manager.extract_feature(provider, weight_id, reference_img, stage, DEVICE, PRECISION)
        cur = await manager.extract_feature(provider, weight_id, current_img, stage, DEVICE, PRECISION)
        loc = await manager.extract_feature(provider, weight_id, patched, stage, DEVICE, PRECISION)
        t = sync(); value = store.put(ref, "acceptance"); store_ms = (sync() - t) * 1000; store.release(value)
        row = {
            "feature_shape": list(ref.tensor.shape), "dtype": str(ref.tensor.dtype), "device": str(ref.tensor.device),
            "feature_MB": ref.metadata.memory_bytes / 2**20, "store_bytes_incl_shared": artifact_nbytes(ref),
            "checkpoint_sha256": ref.metadata.signature.checkpoint_sha256, "signature": ref.metadata.signature.model_dump(),
            "tap": ref.metrics["tap"], "levels": ref.metrics.get("levels"),
            "timing_ms": {k: round(ref.metrics[k], 2) for k in ("feature_preprocess_ms", "feature_forward_ms", "feature_extract_ms")} | {"feature_store_ms": round(store_ms, 3)},
            "load_count": manager.feature_load_count(provider, weight_id, DEVICE, PRECISION),
        }
        t = sync(); _, identity_l2 = difference(ref, ref2, "l2"); delta_ms = (sync() - t) * 1000
        row["identity"] = {"l2": stats(identity_l2)}
        _, identity_cos = difference(ref, ref2, "cosine")
        raw_delta, _ = difference(ref, ref2, "raw")
        row["identity"]["cosine"] = stats(identity_cos)
        row["identity"]["raw_delta_max_abs"] = float(raw_delta.tensor.abs().max())
        row["identity"]["raw_delta_norm"] = float(torch.linalg.vector_norm(raw_delta.tensor.float()))
        _, local_l2 = difference(ref, loc, "l2")
        row["localized"] = {"l2": stats(local_l2) | region_ratio(local_l2.tensor, rect, (sh, sw))}
        t = sync(); _, real_l2 = difference(ref, cur, "l2"); real_delta_ms = (sync() - t) * 1000
        _, real_cos = difference(ref, cur, "cosine")
        t = sync(); canvas = render_heatmap(real_l2.values, sw, sh); preview_ms = (sync() - t) * 1000
        row["real_pair"] = {"l2": stats(real_l2), "cosine": stats(real_cos)}
        row["timing_ms"] |= {"delta_ms": round(real_delta_ms, 3), "identity_delta_ms": round(delta_ms, 3), "preview_ms": round(preview_ms, 2)}
        t = sync(); normed = normalize(cur); row["timing_ms"]["normalize_ms"] = round((sync() - t) * 1000, 3)
        row["normalize_unit_norm_max_err"] = float((torch.linalg.vector_norm(normed.tensor.float(), dim=-1) - 1).abs().max())
        if provider == "rfdetr-v2.2":
            t = sync(); xor = activation_xor(ref, cur); row["timing_ms"]["xor_ms"] = round((sync() - t) * 1000, 3)
            row["identity"]["activation_xor"] = stats(activation_xor(ref, ref2))
            local_xor = activation_xor(ref, loc)
            row["localized"]["activation_xor"] = stats(local_xor) | region_ratio(local_xor.tensor, rect, (sh, sw))
            row["real_pair"]["activation_xor"] = stats(xor)
        else:
            t = sync(); mah = mahalanobis_difference(ref, cur, geometry); row["timing_ms"]["mahalanobis_ms"] = round((sync() - t) * 1000, 3)
            identity_mah = mahalanobis_difference(ref, ref2, geometry)
            local_mah = mahalanobis_difference(ref, loc, geometry)
            row["identity"]["mahalanobis"] = stats(identity_mah)
            row["localized"]["mahalanobis"] = stats(local_mah) | region_ratio(local_mah.tensor, rect, (sh, sw))
            row["real_pair"]["mahalanobis"] = stats(mah)
            row["real_pair"]["mahalanobis_tokens_ge_tau_preset (info only)"] = int((mah.tensor >= geometry.metadata["tau_mah_preset"]).sum())
            row["geometry"] = {"id": GEOMETRY, "checkpoint_sha256": geometry.signature.checkpoint_sha256, "fitting_method": geometry.fitting_method, "tau_mah_preset": geometry.metadata["tau_mah_preset"], "tau_applied": False}
            semantic = manager._feature_models[manager.feature_model_id(manager.feature_key(provider, weight_id, DEVICE, PRECISION))]
            extractor, loaded = semantic.extractor, semantic.loaded
            heads = extractor.semantic_outputs(loaded, cur.shared_forward)
            row["shared_forward_semantic_heads"] = {"pred_masks": list(heads["pred_masks"].shape), "pred_logits": list(heads["pred_logits"].shape), "second_backbone_pass": False}
        cv2.imwrite(str(OUT.parent / f"heat_{provider}_{stage}.jpg"), cv2.resize(canvas, (960, 540)))
        report[name] = row
        del ref, ref2, cur, loc

    # Parity against the ORIGINAL source modules in Skip_car (not the vendored copies).
    def load_module(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module  # dataclasses resolve cls.__module__ through sys.modules
        spec.loader.exec_module(module)
        return module

    original_v22 = load_module("orig_rfdetr_xor_runtime", SOURCE / "v2_2_deployable_package_20260929/candidate_attention/rfdetr_xor_runtime.py")
    v22_path = Path(weights.get(V22_WEIGHT, "rfdetr").path)
    original = original_v22.RFDETRBoundaryFeatures(v22_path, torch.device(DEVICE), (560, 560))
    levels = original(reference_img)
    studio_fine = await manager.extract_feature("rfdetr-v2.2", V22_WEIGHT, reference_img, "fine_80", DEVICE, PRECISION)
    studio_coarse = await manager.extract_feature("rfdetr-v2.2", V22_WEIGHT, reference_img, "coarse_20", DEVICE, PRECISION)
    report["parity_v22_vs_original_source"] = {
        "fine_max_abs_diff": float((levels[0][0].permute(1, 2, 0) - studio_fine.tensor).abs().max()),
        "coarse_max_abs_diff": float((levels[1][0].permute(1, 2, 0) - studio_coarse.tensor).abs().max()),
        "original_level_shapes": [list(level.shape) for level in levels],
    }
    del original, levels
    original_b = load_module("orig_extract_rfdetr_features", SOURCE / "rfdetr_rsd_v1/extract_rfdetr_features.py")
    b_path = Path(weights.get(ARCHB_WEIGHT, "rfdetr").path)
    frozen = original_b.load_frozen_model(str(b_path), DEVICE)
    tensor, _ = original_b.preprocess_image(reference_img, device=frozen.device)
    srcs, _, _ = original_b.extract_backbone_feature(frozen, tensor)
    studio_b = await manager.extract_feature("rfdetr-arch-b", ARCHB_WEIGHT, reference_img, "backbone_42", DEVICE, PRECISION)
    original_b_cur = original_b.extract_backbone_feature(frozen, original_b.preprocess_image(current_img, device=frozen.device)[0])[0][0]
    covariance = load_module("orig_b1_covariance_shim", PROJECT / "backend/inference_studio/features/extractors/vendor/b1_mahalanobis.py")
    delta = (original_b_cur[0] - srcs[0][0]).permute(1, 2, 0).float().cpu().numpy()
    from types import SimpleNamespace
    b1 = covariance.mahalanobis_map(delta, SimpleNamespace(mu=np.asarray(geometry.mu_clear, dtype=np.float64), cholesky_lower=np.asarray(geometry.cholesky)))
    studio_cur = await manager.extract_feature("rfdetr-arch-b", ARCHB_WEIGHT, current_img, "backbone_42", DEVICE, PRECISION)
    studio_mah = mahalanobis_difference(studio_b, studio_cur, geometry).tensor.cpu().numpy()
    report["parity_arch_b_vs_original_source"] = {
        "feature_max_abs_diff": float((srcs[0][0].permute(1, 2, 0) - studio_b.tensor).abs().max()),
        "mahalanobis_vs_B1_mahalanobis_map_max_rel_diff": float(np.max(np.abs(studio_mah - b1) / np.maximum(np.abs(b1), 1e-6))),
    }
    del frozen, srcs
    torch.cuda.empty_cache()

    # Stress: 50 iterations of reference, current, difference, release.
    stress = {}
    for provider, weight_id, stage in (("rfdetr-v2.2", V22_WEIGHT, "fine_80"), ("rfdetr-arch-b", ARCHB_WEIGHT, "backbone_42")):
        stress_store = FeatureStore(max_entries=64, max_bytes=2 * 1024**3)
        torch.cuda.synchronize(); before = torch.cuda.memory_allocated()
        steady = []
        for iteration in range(50):
            ref = await manager.extract_feature(provider, weight_id, reference_img, stage, DEVICE, PRECISION)
            cur = await manager.extract_feature(provider, weight_id, current_img, stage, DEVICE, PRECISION)
            r, c = stress_store.put(ref, f"stress{iteration}"), stress_store.put(cur, f"stress{iteration}")
            delta, heat = difference(stress_store.get(r), stress_store.get(c), "l2")
            d = stress_store.put(delta, f"stress{iteration}")
            for value in (r, c, d):
                stress_store.release(value)
            del ref, cur, delta, heat
            torch.cuda.synchronize()
            steady.append(torch.cuda.memory_allocated())
        stress[f"{provider}:{stage}"] = {
            "iterations": 50, "final_store": stress_store.stats(),
            "cuda_allocated_MB": {"before": before / 2**20, "after_iter_1": steady[0] / 2**20, "steady_median": float(np.median(steady)) / 2**20, "after": steady[-1] / 2**20},
            "growth_iter10_to_50_MB": (steady[-1] - steady[9]) / 2**20,
        }
    report["stress"] = stress
    report["load_counts_final"] = {
        "rfdetr-v2.2": manager.feature_load_count("rfdetr-v2.2", V22_WEIGHT, DEVICE, PRECISION),
        "rfdetr-arch-b": manager.feature_load_count("rfdetr-arch-b", ARCHB_WEIGHT, DEVICE, PRECISION),
    }
    OUT.write_text(json.dumps(report, indent=2, default=str))
    print("ACCEPTANCE_JSON", OUT)


asyncio.run(main())
