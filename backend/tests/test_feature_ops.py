from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from inference_studio.features.extractors.rfdetr_arch_b import architecture_b_signature
from inference_studio.features.extractors.vendor.b1_mahalanobis import mahalanobis_map
from inference_studio.features.extractors.vendor.rfdetr_feature_xor import (
    fixed_threshold_feature_xor_maps,
    reference_channel_thresholds,
)
from inference_studio.features.ops import (
    FeatureSpaceMismatch,
    MahalanobisGeometry,
    MahGeometryMismatch,
    activation_xor,
    difference,
    mahalanobis_difference,
    normalize,
)

from .feature_helpers import CKPT_A, CKPT_B, arch_b, artifact, v22


def _geometry(checkpoint: str = CKPT_A, seed: int = 0) -> MahalanobisGeometry:
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(256, 256))
    sigma = a @ a.T / 256 + np.eye(256) * 1e-3
    return MahalanobisGeometry(
        mu_clear=rng.normal(scale=0.05, size=256), cholesky=np.linalg.cholesky(sigma),
        signature=architecture_b_signature(checkpoint), calibration_corpus="synthetic", fitting_method="synthetic",
        metadata={"tau_mah_preset": 589.3056640625},
    )


def test_difference_modes_match_their_formulas():
    reference, current = v22("coarse_20", seed=1), v22("coarse_20", seed=2)
    raw = current.tensor - reference.tensor
    delta, heat = difference(reference, current, "raw")
    assert torch.equal(delta.tensor, raw)
    delta, _ = difference(reference, current, "absolute")
    assert torch.equal(delta.tensor, raw.abs())
    delta, heat = difference(reference, current, "l2")
    assert torch.equal(delta.tensor, raw) and torch.allclose(heat.tensor, torch.linalg.vector_norm(raw, dim=-1))
    delta, heat = difference(reference, current, "cosine")
    expected = 1.0 - torch.nn.functional.cosine_similarity(current.tensor, reference.tensor, dim=-1, eps=1e-8)
    assert torch.equal(delta.tensor, raw) and torch.allclose(heat.tensor, expected)
    assert heat.mode == "cosine" and (heat.feature_height, heat.feature_width) == (20, 20)


def test_normalize_returns_a_new_artifact_and_never_mutates_the_source():
    source = v22("coarse_20", seed=3)
    before = source.tensor.clone()
    normalized = normalize(source)
    assert normalized is not source and torch.equal(source.tensor, before)
    norms = torch.linalg.vector_norm(normalized.tensor, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
    assert normalized.metadata.history[-1] == "normalize:l2_channel"


def test_activation_xor_equals_the_vendored_v22_source():
    reference, current = v22("fine_80", seed=4), v22("fine_80", seed=5)
    heat = activation_xor(reference, current, percentile=90)
    ref_bchw = reference.tensor.permute(2, 0, 1).unsqueeze(0)
    cur_bchw = current.tensor.permute(2, 0, 1).unsqueeze(0)
    (expected,) = fixed_threshold_feature_xor_maps([cur_bchw], [ref_bchw], reference_channel_thresholds([ref_bchw], 90))
    assert np.array_equal(heat.tensor.cpu().numpy(), expected)
    assert heat.mode == "activation_xor" and 0.0 <= heat.statistics["max"] <= 1.0


def test_activation_xor_identity_is_zero_and_is_v22_only():
    reference = v22("coarse_20", seed=6)
    assert float(activation_xor(reference, reference).tensor.max()) == 0.0
    with pytest.raises(FeatureSpaceMismatch):
        activation_xor(arch_b(seed=1), arch_b(seed=2))
    with pytest.raises(ValueError):
        activation_xor(reference, reference, percentile=100)  # the source requires 0 < percentile < 100


def test_mahalanobis_matches_b1_float64_reference_implementation():
    reference, current = arch_b(seed=7), arch_b(seed=8)
    geometry = _geometry()
    heat = mahalanobis_difference(reference, current, geometry)
    delta = (current.tensor.float() - reference.tensor.float()).numpy()
    fitted = SimpleNamespace(mu=np.asarray(geometry.mu_clear, dtype=np.float64), cholesky_lower=geometry.cholesky)
    expected = mahalanobis_map(delta, fitted)
    assert heat.tensor.dtype == torch.float32 and heat.mode == "mahalanobis"
    np.testing.assert_allclose(heat.tensor.numpy(), expected, rtol=1e-6)


def test_mahalanobis_requires_architecture_b_and_matching_geometry():
    reference, current = arch_b(seed=1), arch_b(seed=2)
    with pytest.raises(MahGeometryMismatch, match="does not match"):
        mahalanobis_difference(reference, current, _geometry(checkpoint=CKPT_B))
    with pytest.raises(FeatureSpaceMismatch):
        mahalanobis_difference(v22("fine_80"), v22("fine_80", seed=1), _geometry())


def test_identity_mahalanobis_is_the_mu_offset_not_zero():
    reference = arch_b(seed=9)
    geometry = _geometry()
    heat = mahalanobis_difference(reference, reference, geometry)
    y = np.linalg.solve(geometry.cholesky, -np.asarray(geometry.mu_clear))
    np.testing.assert_allclose(heat.tensor.numpy(), np.full((42, 42), np.sum(y ** 2), dtype=np.float32), rtol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_operations_stay_on_cuda():
    reference, current = v22("fine_80", seed=1, device="cuda:0"), v22("fine_80", seed=2, device="cuda:0")
    delta, heat = difference(reference, current, "l2")
    assert delta.tensor.is_cuda and heat.tensor.is_cuda
    assert normalize(reference).tensor.is_cuda
    assert activation_xor(reference, current).tensor.is_cuda
    mah = mahalanobis_difference(arch_b(seed=1, device="cuda:0"), arch_b(seed=2, device="cuda:0"), _geometry())
    assert mah.tensor.is_cuda
    cpu = artifact(reference.tensor.cpu(), reference.metadata.signature)
    with pytest.raises(FeatureSpaceMismatch, match="devices"):
        difference(cpu, current, "l2")
