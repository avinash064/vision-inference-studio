from __future__ import annotations

import json
import threading
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Any


class ChainnerSettingsStore:
    def __init__(self, path: Path, registry: Any = None) -> None:
        self.path, self._lock = path, threading.RLock()
        self.registry = registry
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            values = self.defaults()
            if self.path.is_file():
                try:
                    overrides = json.loads(self.path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as exc:
                    raise ValueError(f"Invalid chaiNNer settings file: {exc}") from exc
                if not isinstance(overrides, dict):
                    raise ValueError("chaiNNer settings must be a JSON object")
                for package, package_values in overrides.items():
                    if isinstance(package_values, dict):
                        values.setdefault(package, {}).update(package_values)
                    else:  # backward-compatible flat test/single-package data
                        values[package] = package_values
            return deepcopy(values)

    def defaults(self) -> dict[str, dict[str, Any]]:
        if self.registry is None:
            return {}
        return {package.id: {setting.key: setting.default for setting in package.settings} for package in self.registry.packages.values()}

    def definitions(self) -> dict[str, list[dict[str, Any]]]:
        if self.registry is None:
            return {}
        return {package.id: [asdict(setting) for setting in package.settings] for package in self.registry.packages.values() if package.settings}

    def update(self, values: dict[str, Any]) -> dict[str, Any]:
        json.dumps(values)
        with self._lock:
            temporary = self.path.with_suffix(".tmp")
            temporary.write_text(json.dumps(values, indent=2, sort_keys=True), encoding="utf-8")
            temporary.replace(self.path)
        return self.snapshot()
