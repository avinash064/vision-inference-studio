from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from inference_studio.results import OcrResult, OcrSpan

from .base import InferenceRequest, LoadedModel, ModelAdapter, ModelKey


class PaddleOcrAdapter(ModelAdapter):
    name = "paddleocr"

    def load(self, key: ModelKey, weight_path: Path) -> LoadedModel:
        try:
            from paddleocr import PaddleOCR
        except ImportError as exc:
            raise RuntimeError("PaddleOCR runtime is not installed; install requirements-runtimes.txt") from exc
        variant = "mobile" if key.size == "mobile" else "server"
        language = "en" if key.ocr_language.lower() in {"eng", "english"} else key.ocr_language
        device = key.device.replace("cuda", "gpu", 1) if key.device.startswith("cuda") else key.device
        options = dict(
            lang=language,
            ocr_version="PP-OCRv5",
            text_detection_model_name=f"PP-OCRv5_{variant}_det",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            device=device,
        )
        # The server recognizer is the high-accuracy Chinese/English model.
        # For mobile, allowing PaddleOCR to select the recognizer preserves its
        # language-specific models (for example en_/latin_/devanagari_).
        if variant == "server":
            options["text_recognition_model_name"] = "PP-OCRv5_server_rec"
        handle = PaddleOCR(**options)
        return LoadedModel(handle=handle, key=key, weight_path=weight_path)

    def preprocess(self, model: LoadedModel, request: InferenceRequest, input_data: np.ndarray) -> np.ndarray:
        return input_data

    def infer(self, model: LoadedModel, request: InferenceRequest, tensor: np.ndarray):
        return list(model.handle.predict(tensor, text_rec_score_thresh=request.confidence))

    @staticmethod
    def _payload(value: Any) -> dict[str, Any]:
        candidate = getattr(value, "json", value)
        if callable(candidate):
            candidate = candidate()
        if isinstance(candidate, dict):
            nested = candidate.get("res")
            return nested if isinstance(nested, dict) else candidate
        return {}

    @staticmethod
    def _bbox(box: Any) -> list[float] | None:
        values = np.asarray(box, dtype=float)
        if values.size == 4:
            return values.reshape(-1).tolist()
        if values.ndim == 2 and values.shape[1] >= 2:
            return [float(values[:, 0].min()), float(values[:, 1].min()), float(values[:, 0].max()), float(values[:, 1].max())]
        return None

    def postprocess(self, model: LoadedModel, request: InferenceRequest, output) -> OcrResult:
        spans: list[OcrSpan] = []
        for item in output:
            payload = self._payload(item)
            texts = payload.get("rec_texts", [])
            scores = payload.get("rec_scores", [])
            boxes = payload.get("rec_boxes")
            if boxes is None:
                boxes = payload.get("rec_polys", [])
            for index, text in enumerate(texts):
                confidence = float(scores[index]) if index < len(scores) else 1.0
                clean = str(text).strip()
                if clean and confidence >= request.confidence:
                    spans.append(OcrSpan(text=clean, confidence=max(0.0, min(1.0, confidence)), bbox=self._bbox(boxes[index]) if index < len(boxes) else None))
        return OcrResult(text="\n".join(span.text for span in spans), spans=spans, language=model.key.ocr_language)
