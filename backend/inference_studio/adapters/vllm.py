from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Any

import cv2
import httpx
import numpy as np

from inference_studio.results import OcrResult, OcrSpan

from .base import InferenceRequest, LoadedModel, ModelAdapter, ModelKey


class VllmAdapter(ModelAdapter):
    """OpenAI-compatible multimodal vLLM endpoint normalized as OCR text."""

    name = "vllm"
    max_concurrency = 4

    def load(self, key: ModelKey, weight_path: Path) -> LoadedModel:
        endpoint = key.endpoint_url.rstrip("/")
        if endpoint.endswith("/v1"):
            endpoint += "/chat/completions"
        elif not endpoint.endswith("/chat/completions"):
            endpoint += "/v1/chat/completions"
        handle = {"endpoint": endpoint, "model": key.endpoint_model, "api_key": os.getenv("VLLM_API_KEY", "")}
        return LoadedModel(handle=handle, key=key, weight_path=weight_path)

    def preprocess(self, model: LoadedModel, request: InferenceRequest, input_data: np.ndarray) -> str:
        ok, encoded = cv2.imencode(".jpg", input_data, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not ok:
            raise RuntimeError("Could not encode image for the vLLM endpoint")
        return base64.b64encode(encoded.tobytes()).decode("ascii")

    def infer(self, model: LoadedModel, request: InferenceRequest, tensor: str) -> dict[str, Any]:
        headers = {"Content-Type": "application/json"}
        if model.handle["api_key"]:
            headers["Authorization"] = f"Bearer {model.handle['api_key']}"
        payload = {
            "model": model.handle["model"],
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": request.text_prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{tensor}"}},
            ]}],
            "temperature": 0,
        }
        response = httpx.post(model.handle["endpoint"], headers=headers, json=payload, timeout=120.0)
        response.raise_for_status()
        return response.json()

    @staticmethod
    def _content(value: Any) -> str:
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, list):
            return "\n".join(str(item.get("text", "")).strip() for item in value if isinstance(item, dict) and item.get("text")).strip()
        return str(value or "").strip()

    def postprocess(self, model: LoadedModel, request: InferenceRequest, output: dict[str, Any]) -> OcrResult:
        try:
            text = self._content(output["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("vLLM returned an invalid OpenAI-compatible chat response") from exc
        spans = [OcrSpan(text=text, confidence=1.0)] if text else []
        return OcrResult(text=text, spans=spans, language=model.key.ocr_language or None)
