"""Grounding DINO through the official IDEA-Research implementation.

https://github.com/IDEA-Research/GroundingDINO — loads the released checkpoints
(``groundingdino_swint_ogc.pth`` / ``groundingdino_swinb_cogcoor.pth``) directly.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from inference_studio.results import Detection, DetectionResult

from .base import LoadedModel, ModelAdapter, ModelConfig


@dataclass
class GroundingDinoHandle:
    network: Any
    transform: Any


_RUNTIME_PATCHED = False


def _patch_runtime() -> None:
    """Make the official code run on hosts without its compiled ops or with Transformers 5."""

    global _RUNTIME_PATCHED
    if _RUNTIME_PATCHED:
        return
    import torch
    from groundingdino.models.GroundingDINO import bertwarper
    from groundingdino.models.GroundingDINO import ms_deform_attn as msda

    if not hasattr(msda, "_C"):
        # The CUDA kernel needs nvcc at install time. Its pure-PyTorch reference implementation
        # computes the same multi-scale deformable attention on any device.
        class TorchMultiScaleDeformableAttn:
            @staticmethod
            def apply(value, spatial_shapes, level_start_index, sampling_locations, attention_weights, im2col_step):
                return msda.multi_scale_deformable_attn_pytorch(value, spatial_shapes, sampling_locations, attention_weights)

        msda.MultiScaleDeformableAttnFunction = TorchMultiScaleDeformableAttn

    from transformers import BertModel

    if not hasattr(BertModel, "get_head_mask"):
        # Transformers 5 removed the BertModel helpers the wrapper copies. Grounding DINO only consumes
        # last_hidden_state, so run embeddings + encoder directly with an additive self-attention mask.
        from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions

        def __init__(self, bert_model):
            torch.nn.Module.__init__(self)
            self.config = bert_model.config
            self.embeddings = bert_model.embeddings
            self.encoder = bert_model.encoder
            self.pooler = bert_model.pooler

        def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, **_):
            if token_type_ids is None:
                token_type_ids = torch.zeros_like(input_ids)
            hidden = self.embeddings(input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids)
            if attention_mask is None:
                attention_mask = torch.ones(input_ids.shape, device=input_ids.device)
            mask = attention_mask[:, None, :, :] if attention_mask.dim() == 3 else attention_mask[:, None, None, :]
            additive = (1.0 - mask.to(hidden.dtype)) * torch.finfo(hidden.dtype).min
            sequence = self.encoder(hidden, attention_mask=additive).last_hidden_state
            pooled = self.pooler(sequence) if self.pooler is not None else None
            return BaseModelOutputWithPoolingAndCrossAttentions(last_hidden_state=sequence, pooler_output=pooled)

        bertwarper.BertModelWarper.__init__ = __init__
        bertwarper.BertModelWarper.forward = forward

    _RUNTIME_PATCHED = True


class GroundingDinoAdapter(ModelAdapter):
    name = "grounding-dino"
    _configs = {
        "tiny": "GroundingDINO_SwinT_OGC.py",
        "base": "GroundingDINO_SwinB_cfg.py",
    }

    @staticmethod
    def normalize_prompt(prompt: str) -> str:
        phrases = [item.strip().lower().rstrip(".") for item in prompt.replace(";", ",").split(",")]
        phrases = [item for item in phrases if item]
        if not phrases:
            raise ValueError("Grounding DINO requires a text prompt")
        return " . ".join(phrases) + " ."

    def load(self, config: ModelConfig, weight_path: Path) -> LoadedModel:
        try:
            import groundingdino
            import groundingdino.datasets.transforms as T
            import torch
            from groundingdino.models import build_model
            from groundingdino.util.slconfig import SLConfig
            from groundingdino.util.utils import clean_state_dict
        except ImportError as exc:
            raise RuntimeError("Grounding DINO requires the official package: pip install git+https://github.com/IDEA-Research/GroundingDINO.git") from exc

        _patch_runtime()
        args = SLConfig.fromfile(os.path.join(os.path.dirname(groundingdino.__file__), "config", self._configs[config.size]))
        args.device = config.device
        network = build_model(args)
        checkpoint = torch.load(weight_path, map_location="cpu", weights_only=True)
        state = clean_state_dict(checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint)
        try:
            incompatible = network.load_state_dict(state, strict=False)
        except RuntimeError as exc:  # tensor shape mismatch: a Swin-B checkpoint for Swin-T or vice versa
            raise RuntimeError(f"Checkpoint does not match Grounding DINO {config.size} ({self._configs[config.size].removesuffix('.py')})") from exc
        # Released checkpoints also carry the training-only denoising label embedding, and BERT's
        # position_ids buffer is non-persistent in newer Transformers; anything else means a wrong size.
        missing = [key for key in incompatible.missing_keys if not key.endswith("position_ids")]
        unexpected = [key for key in incompatible.unexpected_keys if key != "label_enc.weight" and not key.endswith("position_ids")]
        if missing or unexpected:
            raise RuntimeError(
                f"Checkpoint does not match Grounding DINO {config.size}: "
                f"{len(missing)} missing and {len(unexpected)} unexpected weights"
            )
        network.to(config.device).eval()
        transform = T.Compose([T.RandomResize([800], max_size=1333), T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        return LoadedModel(handle=GroundingDinoHandle(network, transform), config=config, weight_path=weight_path)

    def preprocess(self, model: LoadedModel, input_data: np.ndarray) -> dict[str, Any]:
        from PIL import Image

        rgb = Image.fromarray(cv2.cvtColor(input_data, cv2.COLOR_BGR2RGB))
        image, _ = model.handle.transform(rgb, None)
        return {
            "image": image.to(model.config.device),
            "caption": self.normalize_prompt(model.config.text_prompt),
            "size": (int(input_data.shape[0]), int(input_data.shape[1])),
        }

    def infer(self, model: LoadedModel, tensor: dict[str, Any]):
        import torch

        half = model.config.precision == "fp16" and model.config.device.startswith("cuda")
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16, enabled=half):
            output = model.handle.network(tensor["image"][None], captions=[tensor["caption"]])
        return output["pred_logits"].float().sigmoid()[0], output["pred_boxes"].float()[0], tensor["caption"], tensor["size"]

    @staticmethod
    def to_detections(boxes_cxcywh: np.ndarray, scores: np.ndarray, phrases: list[str], height: int, width: int) -> list[Detection]:
        detections = []
        for index, ((cx, cy, w, h), score) in enumerate(zip(boxes_cxcywh, scores)):
            x1, y1 = np.clip([(cx - w / 2) * width, (cy - h / 2) * height], 0, [width, height])
            x2, y2 = np.clip([(cx + w / 2) * width, (cy + h / 2) * height], 0, [width, height])
            label = phrases[index] if index < len(phrases) and phrases[index] else "prompt"
            detections.append(Detection(bbox=[float(x1), float(y1), float(x2), float(y2)], class_id=index, label=label, confidence=float(np.clip(score, 0, 1))))
        return detections

    def postprocess(self, model: LoadedModel, output) -> DetectionResult:
        from groundingdino.util.utils import get_phrases_from_posmap

        logits, boxes, caption, (height, width) = output
        keep = logits.max(dim=1)[0] > model.config.confidence
        logits, boxes = logits[keep], boxes[keep]
        tokenizer = model.handle.network.tokenizer
        tokenized = tokenizer(caption)
        phrases = [
            get_phrases_from_posmap(row > model.config.text_threshold, tokenized, tokenizer).replace(".", "").strip()
            for row in logits
        ]
        scores = logits.max(dim=1)[0].cpu().numpy()
        return DetectionResult(detections=self.to_detections(boxes.cpu().numpy(), scores, phrases, height, width))

    def unload(self, model: LoadedModel) -> None:
        del model.handle.network
        super().unload(model)
