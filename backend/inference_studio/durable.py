from __future__ import annotations

import json
import re
import threading
from pathlib import Path
from typing import Any, Callable


class JsonDocumentStore:
    """Small file-backed durable store with atomic replace and process-local locking."""

    def __init__(self, root: Path, prefix: str) -> None:
        self.root = root.resolve()
        self.prefix = prefix
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    def _path(self, identity: str) -> Path:
        if not re.fullmatch(rf"{re.escape(self.prefix)}[A-Za-z0-9_-]+", identity):
            raise ValueError(f"Invalid {self.prefix} identity")
        return self.root / f"{identity}.json"

    def put(self, identity: str, value: dict[str, Any], *, create_only: bool = False) -> dict[str, Any]:
        with self._lock:
            target = self._path(identity)
            if create_only and target.exists():
                raise FileExistsError(identity)
            temporary = target.with_suffix(".tmp")
            temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str), encoding="utf-8")
            temporary.replace(target)
            return value

    def get(self, identity: str) -> dict[str, Any]:
        with self._lock:
            target = self._path(identity)
            if not target.is_file():
                raise KeyError(identity)
            return json.loads(target.read_text(encoding="utf-8"))

    def list(self) -> list[dict[str, Any]]:
        with self._lock:
            values = []
            for path in sorted(self.root.glob(f"{self.prefix}*.json")):
                try:
                    values.append(json.loads(path.read_text(encoding="utf-8")))
                except (OSError, json.JSONDecodeError):
                    continue
            return values

    def update(self, identity: str, mutate: Callable[[dict[str, Any]], dict[str, Any]]) -> dict[str, Any]:
        with self._lock:
            return self.put(identity, mutate(self.get(identity)))

    def delete(self, identity: str) -> None:
        with self._lock:
            self._path(identity).unlink(missing_ok=True)
