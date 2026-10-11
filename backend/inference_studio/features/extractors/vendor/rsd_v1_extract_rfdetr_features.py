"""Core frozen RF-DETR-Seg-Large feature extraction and substitution-wiring module for RSD-V1.

Every other RSD-V1 script imports from here rather than reimplementing model access. This is
the ONLY place the checkpoint gets loaded and the ONLY place the backbone-output substitution
point is defined.

Architecture ground truth (see AUDIT.md Finding 1 -- empirically traced, not assumed):

    features, poss = backbone(nested_tensor)      # features: 1 dense level, (1,256,42,42)
    srcs = [f.decompose()[0] for f in features]    # masks = [f.decompose()[1] for f in features]
    hs, ref_unsigmoid, hs_enc, ref_enc = transformer(srcs, masks, poss, refpoint_embed_weight, query_feat_weight)
    outputs_class = class_embed(hs)
    outputs_masks = segmentation_head(srcs[0], [hs], image_size)[0]
    results = postprocess(outputs, target_sizes)

`class_embed`, `segmentation_head`, and the transformer decoder together are what this
experiment calls "the frozen semantic head" -- they require `hs` (transformer decoder
output), not raw backbone features, so B/C/D substitute the TENSOR THAT FEEDS THE
TRANSFORMER (`srcs[0]`) rather than attempting to feed anything directly into class_embed/
segmentation_head. Everything downstream of that substitution point runs completely
unmodified and frozen.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as F  # matches rfdetr/detr.py's own import exactly
from rfdetr import RFDETRSegLarge

CHECKPOINT_PATH = "/media/avinash/Expansion/skip/rfdetr_seg_train_output/checkpoint_best_ema.pth"
CHECKPOINT_SHA256 = "f1edd91a3b41a0d140363a50d494323327b53ad05b3a66352cb3ecdf64332ccb"
NUM_CLASSES = 3
CLASS_NAMES = ["skip_car", "wheel", "cable_wire"]  # RF-DETR's own trained taxonomy --
# NOT RSD-V1's PERSON/CABLE/SKIP/OTHER change-cause taxonomy. A0/A1 evaluate against THIS
# taxonomy (ordinary semantic recognition); the change-cause taxonomy is a separate label
# space applied at the candidate/dataset level (build_change_semantic_dataset.py), never
# produced by this model directly.
RESOLUTION = 504
MEANS = [0.485, 0.456, 0.406]
STDS = [0.229, 0.224, 0.225]
FEATURE_CHANNELS = 256
FEATURE_SPATIAL = 42


@dataclass
class FrozenModel:
    model: RFDETRSegLarge  # the RFDETR wrapper (has .predict(), .model, .class_names)
    net: torch.nn.Module  # model.model.model -- the raw LWDETR nn.Module
    device: torch.device


def load_frozen_model(checkpoint_path: str = CHECKPOINT_PATH, device: Optional[str] = None) -> FrozenModel:
    """Loads RFDETRSegLarge, freezes every parameter, and asserts it structurally."""
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = RFDETRSegLarge(pretrain_weights=checkpoint_path, num_classes=NUM_CLASSES)
    net = model.model.model.to(dev).eval()
    for p in net.parameters():
        p.requires_grad_(False)
    assert_backbone_frozen(net)
    assert all(not p.requires_grad for p in net.parameters()), "RSD-V1 requires the ENTIRE base model frozen"
    return FrozenModel(model=model, net=net, device=dev)


def assert_backbone_frozen(net: torch.nn.Module) -> None:
    """Hard runtime assertion required by the RSD-V1 spec (Section 2)."""
    assert all(not p.requires_grad for p in net.backbone.parameters()), (
        "RSD-V1 safety violation: backbone has a trainable parameter. No backbone gradient is allowed."
    )


def preprocess_image(image_bgr: np.ndarray, resolution: int = RESOLUTION, device: Optional[torch.device] = None) -> tuple[torch.Tensor, tuple[int, int]]:
    """Replicates RFDETR.predict()'s own preprocessing exactly (traced from rfdetr/detr.py):
    RGB, [0,1] float, square resize to (resolution, resolution), ImageNet normalize.
    Returns (batched_tensor[1,3,res,res], (orig_h, orig_w))."""
    h, w = image_bgr.shape[:2]
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).float().div_(255.0)
    dev = device or torch.device("cpu")
    tensor = tensor.to(dev)
    tensor = F.resize(tensor, [resolution, resolution])
    tensor = F.normalize(tensor, MEANS, STDS)  # exact same call rfdetr/detr.py's predict() makes
    return tensor.unsqueeze(0), (h, w)


def extract_backbone_feature(fm: FrozenModel, image_tensor: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    """Runs ONLY the backbone (frozen). Returns (srcs, masks, poss) exactly as the real
    forward() unpacks them -- srcs is a length-1 list, srcs[0].shape == (1, 256, 42, 42)."""
    from rfdetr.utilities.tensors import NestedTensor

    b, _, h, w = image_tensor.shape
    pad_mask = torch.zeros((b, h, w), dtype=torch.bool, device=image_tensor.device)
    nested = NestedTensor(image_tensor, pad_mask)
    with torch.inference_mode():
        features, poss = fm.net.backbone(nested)
    srcs, masks = [], []
    for feat in features:
        src, mask = feat.decompose()
        srcs.append(src)
        masks.append(mask)
    assert len(srcs) == 1, f"RSD-V1 assumes a single feature tap; found {len(srcs)} (see AUDIT.md Finding 1)"
    assert tuple(srcs[0].shape) == (b, FEATURE_CHANNELS, FEATURE_SPATIAL, FEATURE_SPATIAL), (
        f"Unexpected backbone feature shape {tuple(srcs[0].shape)}; AUDIT.md's empirical "
        f"finding was (B,{FEATURE_CHANNELS},{FEATURE_SPATIAL},{FEATURE_SPATIAL})"
    )
    return srcs, masks, poss


def run_heads_from_srcs(fm: FrozenModel, srcs: list[torch.Tensor], masks: list[torch.Tensor], poss: list[torch.Tensor], image_size: tuple[int, int]) -> dict:
    """Runs the FROZEN transformer decoder + class_embed + segmentation_head starting from
    `srcs` (either the model's own real backbone output, for A0, or a substituted tensor of
    the identical shape, for B/C/D). Nothing in this function's own parameters is ever
    trained by RSD-V1 -- only the tensor fed in as `srcs` differs between variants."""
    net = fm.net
    refpoint_embed_weight = net.refpoint_embed.weight[: net.num_queries]
    query_feat_weight = net.query_feat.weight[: net.num_queries]
    with torch.inference_mode():
        hs, ref_unsigmoid, hs_enc, ref_enc = net.transformer(srcs, masks, poss, refpoint_embed_weight, query_feat_weight)
        if net.bbox_reparam:
            outputs_coord_delta = net.bbox_embed(hs)
            outputs_coord_cxcy = outputs_coord_delta[..., :2] * ref_unsigmoid[..., 2:] + ref_unsigmoid[..., :2]
            outputs_coord_wh = outputs_coord_delta[..., 2:].exp() * ref_unsigmoid[..., 2:]
            outputs_coord = torch.cat([outputs_coord_cxcy, outputs_coord_wh], dim=-1)
        else:
            outputs_coord = (net.bbox_embed(hs) + ref_unsigmoid).sigmoid()
        outputs_class = net.class_embed(hs)
        outputs_masks = None
        if net.segmentation_head is not None:
            # hs is (num_decoder_layers, B, N, C) -- pass it UNWRAPPED so the head's own
            # zip(self.blocks, query_features) iterates hs's leading (layer) dimension
            # naturally (traced exactly from lwdetr.py's real forward(), NOT forward_export's
            # [hs]-wrapped convention, which is for a different, single-layer-only path).
            outputs_masks = net.segmentation_head(srcs[0], hs, image_size)
    # forward()'s own convention: take the FINAL decoder layer's predictions
    return {
        "pred_logits": outputs_class[-1],
        "pred_boxes": outputs_coord[-1],
        "pred_masks": outputs_masks[-1] if outputs_masks is not None else None,
    }


def postprocess_outputs(fm: FrozenModel, outputs: dict, orig_size: tuple[int, int]) -> object:
    """Runs the SAME frozen PostProcess module model.predict() itself uses -- top-k select,
    box denormalization, mask upsample-to-original-size + threshold."""
    target_sizes = torch.tensor([orig_size], device=fm.device)
    with torch.inference_mode():
        results = fm.model.model.postprocess(outputs, target_sizes=target_sizes)
    return results[0]


def run_full_pipeline(fm: FrozenModel, image_bgr: np.ndarray, substitute_srcs: Optional[torch.Tensor] = None) -> object:
    """One-call convenience wrapper. substitute_srcs=None -> A0 (real backbone path,
    should parity-match model.predict()). substitute_srcs=<tensor of shape (1,256,42,42)>
    -> B/C/D (residual/adapted tensor substituted in place of the real backbone output,
    everything downstream frozen and unmodified)."""
    image_tensor, orig_size = preprocess_image(image_bgr, device=fm.device)
    srcs, masks, poss = extract_backbone_feature(fm, image_tensor)
    if substitute_srcs is not None:
        assert substitute_srcs.shape == srcs[0].shape, (
            f"substitute_srcs shape {tuple(substitute_srcs.shape)} != real backbone shape {tuple(srcs[0].shape)}"
        )
        srcs = [substitute_srcs]
    outputs = run_heads_from_srcs(fm, srcs, masks, poss, (RESOLUTION, RESOLUTION))
    return postprocess_outputs(fm, outputs, orig_size)


def save_freeze_manifest(fm: FrozenModel, out_path: Path, extra: Optional[dict] = None) -> None:
    payload = {
        "checkpoint_path": CHECKPOINT_PATH,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "num_classes": NUM_CLASSES,
        "class_names": CLASS_NAMES,
        "resolution": RESOLUTION,
        "feature_tap": {"num_levels": 1, "channels": FEATURE_CHANNELS, "spatial": FEATURE_SPATIAL, "dtype": "float32"},
        "normalization": {"means": MEANS, "stds": STDS},
        "backbone_frozen": all(not p.requires_grad for p in fm.net.backbone.parameters()),
        "all_params_frozen": all(not p.requires_grad for p in fm.net.parameters()),
    }
    if extra:
        payload.update(extra)
    out_path.write_text(json.dumps(payload, indent=2))
