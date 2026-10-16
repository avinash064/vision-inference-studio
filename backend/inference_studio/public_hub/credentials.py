from __future__ import annotations

import os
import re
from pathlib import Path

from .registry import ExternalProviderRegistry
from .schemas import AuthState, ProviderCredentialUpdate


PROVIDER_KEYS = {
    "huggingface": ("HF_TOKEN",),
    "roboflow": ("ROBOFLOW_API_KEY",),
    "kaggle": ("KAGGLE_API_TOKEN", "KAGGLE_USERNAME", "KAGGLE_KEY"),
}


class CredentialError(ValueError):
    pass


class ProviderCredentialStore:
    """Server-only provider credentials persisted to a git-ignored env file."""

    def __init__(self, env_file: Path, providers: ExternalProviderRegistry) -> None:
        self.env_file, self.providers = env_file.resolve(), providers

    def status(self) -> list[dict]:
        return [{
            "provider": value["id"], "display_name": value["display_name"],
            "auth_state": value["auth_state"], "configured": value["auth_state"] != AuthState.NOT_CONFIGURED.value,
            "fields": self.fields(value["id"]),
        } for value in self.providers.list()]

    @staticmethod
    def fields(provider: str) -> list[dict]:
        if provider == "kaggle":
            return [{"id": "api_token", "label": "Kaggle API token", "secret": True, "placeholder": "KGAT_…"}]
        if provider == "roboflow":
            return [{"id": "api_key", "label": "Roboflow API key", "secret": True, "placeholder": "Enter API key"}]
        if provider == "huggingface":
            return [{"id": "token", "label": "Hugging Face token", "secret": True, "placeholder": "hf_…"}]
        return []

    def update(self, provider: str, payload: ProviderCredentialUpdate) -> dict:
        if provider not in PROVIDER_KEYS:
            raise CredentialError("Unknown credential provider")
        values = self._values(provider, payload)
        previous = {key: os.environ.get(key) for key in PROVIDER_KEYS[provider]}
        try:
            for key in PROVIDER_KEYS[provider]:
                os.environ.pop(key, None)
            os.environ.update(values)
            state = self.providers.get(provider).test_auth() if payload.test_before_save else AuthState.CONFIGURED
            if state != AuthState.CONFIGURED:
                raise CredentialError(f"{provider} authentication failed; credentials were not saved")
            self._write(values, set(PROVIDER_KEYS[provider]) - set(values))
        except Exception:
            for key, value in previous.items():
                if value is None: os.environ.pop(key, None)
                else: os.environ[key] = value
            raise
        return {"provider": provider, "auth_state": state.value, "configured": True}

    def remove(self, provider: str) -> dict:
        if provider not in PROVIDER_KEYS:
            raise CredentialError("Unknown credential provider")
        keys = set(PROVIDER_KEYS[provider])
        for key in keys:
            os.environ.pop(key, None)
        self._write({}, keys)
        return {"provider": provider, "auth_state": AuthState.NOT_CONFIGURED.value, "configured": False}

    @staticmethod
    def _secret(value) -> str | None:
        if value is None: return None
        result = value.get_secret_value().strip()
        if not result or len(result) > 4096 or any(character in result for character in "\r\n'\""):
            raise CredentialError("Credential contains unsupported characters")
        return result

    def _values(self, provider: str, payload: ProviderCredentialUpdate) -> dict[str, str]:
        if provider == "huggingface":
            value = self._secret(payload.token)
            if not value: raise CredentialError("Hugging Face token is required")
            return {"HF_TOKEN": value}
        if provider == "roboflow":
            value = self._secret(payload.api_key)
            if not value: raise CredentialError("Roboflow API key is required")
            return {"ROBOFLOW_API_KEY": value}
        value = self._secret(payload.api_token)
        if not value: raise CredentialError("Kaggle API token is required")
        return {"KAGGLE_API_TOKEN": value}

    def _write(self, updates: dict[str, str], removals: set[str]) -> None:
        lines = self.env_file.read_text(encoding="utf-8").splitlines() if self.env_file.is_file() else ["# Vision Inference Studio server-side secrets"]
        managed = set(updates) | removals
        pattern = re.compile(r"^(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)=")
        output, written = [], set()
        for line in lines:
            match = pattern.match(line.strip())
            key = match.group(1) if match else None
            if key in managed:
                if key in updates and key not in written:
                    output.append(f"{key}='{updates[key]}'"); written.add(key)
                continue
            output.append(line)
        for key, value in updates.items():
            if key not in written: output.append(f"{key}='{value}'")
        self.env_file.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.env_file.with_suffix(self.env_file.suffix + ".tmp")
        temporary.write_text("\n".join(output).rstrip() + "\n", encoding="utf-8")
        os.chmod(temporary, 0o600)
        temporary.replace(self.env_file)
        os.chmod(self.env_file, 0o600)
