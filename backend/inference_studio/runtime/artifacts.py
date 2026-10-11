from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from inference_studio.features.schema import FeatureMetadata, FeatureSpaceSignature


@dataclass
class FeatureArtifact:
    """Backend-only tensor artifact; only its opaque FeatureValue may enter a graph."""

    tensor: Any
    metadata: FeatureMetadata
    summary: dict[str, float] = field(default_factory=dict)
    metrics: Any = None
    # Architecture B keeps tensors from its one backbone forward here so a
    # future semantic branch can reuse them without running the backbone twice.
    shared_forward: Any = field(default=None, repr=False)


def __getattr__(name: str):
    """Keep the v0.5 import path while the store lives in features/store.py."""
    if name == "FeatureStore":
        from inference_studio.features.store import FeatureStore
        return FeatureStore
    raise AttributeError(name)


__all__ = ["FeatureArtifact", "FeatureMetadata", "FeatureSpaceSignature", "FeatureStore"]
