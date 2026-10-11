from __future__ import annotations

import os
import threading
import uuid
from collections import OrderedDict
from dataclasses import dataclass

from inference_studio.runtime.artifacts import FeatureArtifact
from inference_studio.runtime.values import FeatureValue


class FeatureStoreCapacityExceeded(MemoryError):
    pass


class FeatureNotFound(ValueError):
    pass


@dataclass
class _Entry:
    artifact: FeatureArtifact
    run_id: str
    refs: int
    nbytes: int


def artifact_nbytes(artifact: FeatureArtifact) -> int:
    """Bytes the store keeps alive: the feature plus any shared forward tensors (Architecture B srcs/masks/poss)."""

    total = int(artifact.metadata.memory_bytes)
    shared = getattr(artifact, "shared_forward", None)
    if shared is not None and hasattr(shared, "tensors"):
        total += sum(int(t.numel() * t.element_size()) for t in shared.tensors() if hasattr(t, "numel"))
    return total


class FeatureStore:
    """Bounded process-local owner of CPU and CUDA feature tensors."""

    def __init__(self, max_entries: int | None = None, max_bytes: int | None = None) -> None:
        self.max_entries = max_entries if max_entries is not None else int(os.getenv("VIS_FEATURE_STORE_MAX_ENTRIES", "64"))
        self.max_bytes = max_bytes if max_bytes is not None else int(os.getenv("VIS_FEATURE_STORE_MAX_BYTES", str(2 * 1024**3)))
        if self.max_entries < 1 or self.max_bytes < 1:
            raise ValueError("FeatureStore limits must be positive")
        self._entries: OrderedDict[str, _Entry] = OrderedDict()
        self._bytes = 0
        self._peak_entries = 0
        self._peak_cpu_bytes = 0
        self._peak_gpu_bytes = 0
        self._lock = threading.RLock()

    @staticmethod
    def _id(value: str | FeatureValue) -> str:
        return value.feature_id if isinstance(value, FeatureValue) else value

    def put(self, artifact: FeatureArtifact, run_id: str, refs: int = 1, **legacy: int) -> FeatureValue:
        if "refcount" in legacy:
            refs = legacy["refcount"]
        required = artifact_nbytes(artifact)
        with self._lock:
            if len(self._entries) + 1 > self.max_entries or self._bytes + required > self.max_bytes:
                raise FeatureStoreCapacityExceeded(
                    "feature_store_capacity_exceeded: adding "
                    f"{required} bytes would exceed {self.max_entries} entries/"
                    f"{self.max_bytes} bytes; in-use tensors were not evicted"
                )
            feature_id = f"feat_{uuid.uuid4().hex}"
            self._entries[feature_id] = _Entry(artifact, run_id, max(1, int(refs)), required)
            self._bytes += required
            self._record_peaks()
            return FeatureValue(feature_id, artifact.metadata, artifact.summary)

    def get(self, feature_id: str | FeatureValue) -> FeatureArtifact:
        key = self._id(feature_id)
        with self._lock:
            try:
                entry = self._entries[key]
            except KeyError as exc:
                raise FeatureNotFound(f"feature_not_found: {key}") from exc
            self._entries.move_to_end(key)
            return entry.artifact

    def retain(self, feature_id: str | FeatureValue, count: int = 1) -> None:
        key = self._id(feature_id)
        with self._lock:
            if key not in self._entries:
                raise FeatureNotFound(f"feature_not_found: {key}")
            self._entries[key].refs += count

    def release(self, feature_id: str | FeatureValue, count: int = 1) -> None:
        key = self._id(feature_id)
        with self._lock:
            entry = self._entries.get(key)
            if entry is not None:
                entry.refs -= count
                if entry.refs <= 0:
                    self._drop(key)

    def release_run(self, run_id: str) -> None:
        with self._lock:
            for feature_id in [key for key, value in self._entries.items() if value.run_id == run_id]:
                self._drop(feature_id)

    def stats(self) -> dict[str, int]:
        with self._lock:
            cpu_bytes, gpu_bytes = self._device_bytes()
            return {
                "entries": len(self._entries), "bytes": self._bytes,
                "cpu_bytes": cpu_bytes, "gpu_bytes": gpu_bytes,
                "peak_entries": self._peak_entries,
                "peak_cpu_bytes": self._peak_cpu_bytes,
                "peak_gpu_bytes": self._peak_gpu_bytes,
                "max_entries": self.max_entries, "max_bytes": self.max_bytes,
            }

    def _device_bytes(self) -> tuple[int, int]:
        cpu = sum(
            entry.nbytes
            for entry in self._entries.values()
            if not entry.artifact.metadata.device.startswith("cuda")
        )
        return cpu, self._bytes - cpu

    def _record_peaks(self) -> None:
        cpu, gpu = self._device_bytes()
        self._peak_entries = max(self._peak_entries, len(self._entries))
        self._peak_cpu_bytes = max(self._peak_cpu_bytes, cpu)
        self._peak_gpu_bytes = max(self._peak_gpu_bytes, gpu)

    def _drop(self, feature_id: str) -> None:
        entry = self._entries.pop(feature_id, None)
        if entry is not None:
            self._bytes -= entry.nbytes
