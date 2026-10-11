"""Frozen Mahalanobis geometry artifacts, addressed by opaque IDs (workflows never carry file paths).

A geometry is a weights-only ``torch.save`` dictionary holding ``mu_clear [256]``, the lower Cholesky factor
``[256, 256]`` of the clear-delta covariance, the exact ``FeatureSpaceSignature`` it was fitted in, and provenance
(checkpoint hash, calibration corpus, fitting method, source files). Geometries are never refitted at run time.
"""

from __future__ import annotations

import re
import threading
from pathlib import Path
from typing import Any

from inference_studio.features.ops import MahalanobisGeometry, MahGeometryMismatch, load_mahalanobis_geometry
from inference_studio.features.schema import FeatureSpaceSignature

GEOMETRY_ID = re.compile(r"^geo_[a-z0-9_]{3,64}$")


class MahalanobisGeometryStore:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, tuple[float, MahalanobisGeometry]] = {}
        self._lock = threading.Lock()

    def path(self, geometry_id: str) -> Path:
        if not GEOMETRY_ID.fullmatch(str(geometry_id)):
            raise MahGeometryMismatch(f"Invalid geometry id: {geometry_id!r}")
        path = (self.root / f"{geometry_id}.pt").resolve()
        if path.parent != self.root:
            raise MahGeometryMismatch("Geometry path escapes the geometry store")
        return path

    def get(self, geometry_id: str) -> MahalanobisGeometry:
        path = self.path(geometry_id)
        if not path.is_file():
            raise MahGeometryMismatch(f"Unknown Mahalanobis geometry: {geometry_id}")
        mtime = path.stat().st_mtime
        with self._lock:
            cached = self._cache.get(geometry_id)
            if cached is None or cached[0] != mtime:
                cached = (mtime, load_mahalanobis_geometry(path))
                self._cache[geometry_id] = cached
            return cached[1]

    def list(self) -> list[dict[str, Any]]:
        rows = []
        for path in sorted(self.root.glob("geo_*.pt")):
            try:
                geometry = self.get(path.stem)
            except MahGeometryMismatch:
                continue
            rows.append({
                "geometry_id": path.stem,
                "display_name": geometry.metadata.get("display_name", path.stem),
                "feature_space": geometry.signature.display(),
                "checkpoint_sha256": geometry.signature.checkpoint_sha256,
                "calibration_corpus": geometry.calibration_corpus,
                "fitting_method": geometry.fitting_method,
                "tau_mah_preset": geometry.metadata.get("tau_mah_preset"),
            })
        return rows

    def save(
        self,
        geometry_id: str,
        mu_clear: Any,
        cholesky: Any,
        signature: FeatureSpaceSignature,
        metadata: dict[str, Any],
    ) -> Path:
        import numpy as np
        import torch

        mu = np.asarray(mu_clear, dtype=np.float64).reshape(-1)
        lower = np.asarray(cholesky, dtype=np.float64)
        channels = signature.channels
        if mu.shape != (channels,) or lower.shape != (channels, channels):
            raise MahGeometryMismatch(f"Geometry must be mu[{channels}] and cholesky[{channels},{channels}]; got {mu.shape}, {lower.shape}")
        if not np.allclose(lower, np.tril(lower)) or not np.all(np.diag(lower) > 0):
            raise MahGeometryMismatch("cholesky must be lower-triangular with a positive diagonal")
        for key in ("calibration_corpus", "fitting_method"):
            if not metadata.get(key):
                raise MahGeometryMismatch(f"Geometry metadata requires {key}")
        path = self.path(geometry_id)
        torch.save({
            "mu_clear": torch.from_numpy(mu),
            "cholesky": torch.from_numpy(lower),
            "feature_signature": signature.model_dump(mode="json"),
            "metadata": dict(metadata),
        }, path)
        with self._lock:
            self._cache.pop(geometry_id, None)
        return path
