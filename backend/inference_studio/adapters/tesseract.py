from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from inference_studio.results import OcrResult, OcrSpan

from .base import InferenceRequest, LoadedModel, ModelAdapter, ModelKey


class TesseractAdapter(ModelAdapter):
    name = "tesseract"

    def load(self, key: ModelKey, weight_path: Path) -> LoadedModel:
        try:
            import pytesseract
            pytesseract.get_tesseract_version()
        except ImportError as exc:
            raise RuntimeError("pytesseract is not installed; install requirements-runtimes.txt") from exc
        except Exception as exc:
            raise RuntimeError("Tesseract OCR executable is not installed or is not on PATH") from exc
        return LoadedModel(handle=pytesseract, key=key, weight_path=weight_path)

    def preprocess(self, model: LoadedModel, request: InferenceRequest, input_data: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(input_data, cv2.COLOR_BGR2RGB)

    def infer(self, model: LoadedModel, request: InferenceRequest, tensor: np.ndarray):
        language = "eng" if model.key.ocr_language.lower() in {"en", "english"} else model.key.ocr_language
        return model.handle.image_to_data(tensor, lang=language, output_type=model.handle.Output.DICT)

    def postprocess(self, model: LoadedModel, request: InferenceRequest, output) -> OcrResult:
        spans: list[OcrSpan] = []
        texts = output.get("text", [])
        for index, value in enumerate(texts):
            text = str(value).strip()
            try:
                confidence = float(output.get("conf", [])[index]) / 100.0
            except (IndexError, TypeError, ValueError):
                confidence = -1.0
            if not text or confidence < request.confidence:
                continue
            x = float(output["left"][index]); y = float(output["top"][index])
            width = float(output["width"][index]); height = float(output["height"][index])
            spans.append(OcrSpan(text=text, confidence=max(0.0, min(1.0, confidence)), bbox=[x, y, x + width, y + height]))
        return OcrResult(text=" ".join(span.text for span in spans), spans=spans, language=model.key.ocr_language)
