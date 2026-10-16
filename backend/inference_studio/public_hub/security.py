from __future__ import annotations

import os


SECRET_ENVIRONMENTS = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "ROBOFLOW_API_KEY", "KAGGLE_API_TOKEN", "KAGGLE_KEY")


def redact_error(error: Exception | str) -> str:
    value = str(error)
    for name in SECRET_ENVIRONMENTS:
        secret = os.getenv(name)
        if secret:
            value = value.replace(secret, "[REDACTED]")
    return value[:2000]
