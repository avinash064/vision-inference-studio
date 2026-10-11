from __future__ import annotations

import re
from types import SimpleNamespace

import pytest
import torch

from inference_studio.features.store import FeatureNotFound, FeatureStore, FeatureStoreCapacityExceeded, artifact_nbytes

from .feature_helpers import arch_b, v22


def test_put_get_retain_release_and_feature_ids():
    store = FeatureStore(max_entries=8, max_bytes=1 << 30)
    value = store.put(v22("fine_80"), run_id="run_a", refs=2)
    assert re.fullmatch(r"feat_[0-9a-f]{32}", value.feature_id)
    assert tuple(store.get(value.feature_id).tensor.shape) == (80, 80, 384)
    store.retain(value.feature_id)          # refs 3
    store.release(value.feature_id, 2)      # refs 1
    assert store.stats()["entries"] == 1
    store.release(value.feature_id)
    assert store.stats()["entries"] == 0 and store.stats()["bytes"] == 0
    with pytest.raises(FeatureNotFound):
        store.get(value.feature_id)


def test_release_run_drops_only_that_run():
    store = FeatureStore(max_entries=8, max_bytes=1 << 30)
    store.put(v22("coarse_20"), "run_a", refs=5)
    kept = store.put(v22("coarse_20", seed=1), "run_b")
    store.release_run("run_a")
    stats = store.stats()
    assert stats["entries"] == 1 and stats["peak_entries"] == 2
    assert store.get(kept.feature_id) is not None


def test_capacity_is_enforced_without_evicting_in_use_tensors():
    one = v22("coarse_20")  # 20*20*384*4 bytes
    store = FeatureStore(max_entries=2, max_bytes=10 * artifact_nbytes(one))
    first = store.put(one, "run")
    store.put(v22("coarse_20", seed=1), "run")
    with pytest.raises(FeatureStoreCapacityExceeded):
        store.put(v22("coarse_20", seed=2), "run")
    assert store.get(first.feature_id) is not None  # nothing silently removed
    byte_store = FeatureStore(max_entries=64, max_bytes=artifact_nbytes(one) + 1)
    byte_store.put(one, "run")
    with pytest.raises(FeatureStoreCapacityExceeded):
        byte_store.put(v22("coarse_20", seed=3), "run")


def test_limits_come_from_environment(monkeypatch):
    monkeypatch.setenv("VIS_FEATURE_STORE_MAX_ENTRIES", "3")
    monkeypatch.setenv("VIS_FEATURE_STORE_MAX_BYTES", "12345")
    store = FeatureStore()
    assert (store.max_entries, store.max_bytes) == (3, 12345)


def test_shared_forward_tensors_are_counted():
    feature = arch_b()
    srcs = [torch.zeros(1, 256, 42, 42)]
    poss = [torch.zeros(1, 256, 42, 42)]
    masks = [torch.zeros(1, 42, 42, dtype=torch.bool)]
    feature.shared_forward = SimpleNamespace(tensors=lambda: [*srcs, *masks, *poss])
    expected = feature.metadata.memory_bytes + 2 * 256 * 42 * 42 * 4 + 42 * 42
    assert artifact_nbytes(feature) == expected
    store = FeatureStore(max_entries=4, max_bytes=1 << 30)
    value = store.put(feature, "run")
    assert store.stats()["bytes"] == expected
    store.release(value.feature_id)
    assert store.stats()["bytes"] == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_features_stay_on_cuda_and_are_accounted_as_gpu():
    store = FeatureStore(max_entries=4, max_bytes=1 << 30)
    value = store.put(v22("fine_80", device="cuda:0"), "run")
    assert store.get(value.feature_id).tensor.is_cuda
    stats = store.stats()
    assert stats["gpu_bytes"] == 80 * 80 * 384 * 4 and stats["cpu_bytes"] == 0
    store.release_run("run")
    assert store.stats()["entries"] == 0
