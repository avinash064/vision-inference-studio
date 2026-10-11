from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from inference_studio.features.extractors import vendor
from inference_studio.features.extractors.rfdetr_arch_b import architecture_b_signature
from inference_studio.features.geometry import MahalanobisGeometryStore
from inference_studio.features.ops import FeatureSpaceMismatch, MahGeometryMismatch, difference, mahalanobis_difference
from inference_studio.graph import WorkflowValidator
from inference_studio.graph.schema import Edge, NodeInstance, Workflow
from inference_studio.nodes import NodeRegistry

from .feature_helpers import CKPT_A, CKPT_B, arch_b, v22


def test_matching_v22_fine_features_are_compatible():
    delta, heat = difference(v22("fine_80", seed=1), v22("fine_80", seed=2), "l2")
    assert tuple(delta.tensor.shape) == (80, 80, 384) and tuple(heat.tensor.shape) == (80, 80)


@pytest.mark.parametrize("reference,current", [
    (lambda: v22("fine_80"), lambda: v22("coarse_20")),                       # v2.2 fine vs coarse
    (lambda: v22("fine_80"), lambda: arch_b()),                               # v2.2 vs Architecture B
    (lambda: arch_b(checkpoint=CKPT_A), lambda: arch_b(checkpoint=CKPT_B)),  # same arch, other checkpoint
    (lambda: arch_b(), lambda: v22("fine_80")),                               # 42x42x256 vs 80x80x384
])
def test_incompatible_feature_spaces_fail_without_resizing(reference, current):
    with pytest.raises(FeatureSpaceMismatch, match="Cannot compare"):
        difference(reference(), current(), "l2")


def test_mismatch_message_names_both_feature_spaces():
    with pytest.raises(FeatureSpaceMismatch) as error:
        difference(v22("fine_80"), arch_b(), "l2")
    message = str(error.value)
    assert "rfdetr/v2.2/RFDETRLargeDeprecated/fine_80 [80,80,384]" in message
    assert "rfdetr/architecture-b/seglarge/backbone_42 [42,42,256]" in message


def test_vendored_source_files_are_unmodified():
    root = Path(vendor.__file__).parent
    for name, prefix in vendor.VENDORED_SHA256_PREFIX.items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest().startswith(prefix), name


def test_architecture_b_signature_encodes_seglarge_504_patch12():
    signature = architecture_b_signature(CKPT_A)
    assert (signature.input_height, signature.input_width, signature.patch_size, signature.stride) == (504, 504, 12, 12.0)
    assert (signature.spatial_height, signature.spatial_width, signature.channels) == (42, 42, 256)
    assert "F.resize(504x504)" in signature.normalization and signature.preprocessing_signature == "rfdetr-arch-b-seglarge-504"


def _lower(seed: int = 0):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(256, 256))
    return rng.normal(scale=0.05, size=256), np.linalg.cholesky(a @ a.T / 256 + np.eye(256) * 1e-3)


def test_geometry_store_roundtrip_ids_and_signature_validation(tmp_path):
    store = MahalanobisGeometryStore(tmp_path / "geometries")
    mu, lower = _lower()
    meta = {"calibration_corpus": "synthetic clear set", "fitting_method": "test", "tau_mah_preset": 589.3056640625}
    store.save("geo_test_a", mu, lower, architecture_b_signature(CKPT_A), meta)
    geometry = store.get("geo_test_a")
    assert geometry.signature.checkpoint_sha256 == CKPT_A and geometry.metadata["tau_mah_preset"] == 589.3056640625
    assert [row["geometry_id"] for row in store.list()] == ["geo_test_a"]
    heat = mahalanobis_difference(arch_b(seed=1), arch_b(seed=2), geometry)
    assert heat.mode == "mahalanobis"  # continuous map; tau is metadata only, never applied
    assert set(np.unique(heat.tensor.numpy())) != {0.0, 1.0}
    with pytest.raises(MahGeometryMismatch):
        mahalanobis_difference(arch_b(checkpoint=CKPT_B, seed=1), arch_b(checkpoint=CKPT_B, seed=2), geometry)
    for bad in ("../geo_test_a", "geo_TEST", "/etc/passwd", "wgt_123"):
        with pytest.raises(MahGeometryMismatch):
            store.get(bad)
    with pytest.raises(MahGeometryMismatch):
        store.save("geo_bad_shape", mu[:10], lower, architecture_b_signature(CKPT_A), meta)
    with pytest.raises(MahGeometryMismatch):
        store.save("geo_upper", mu, lower.T, architecture_b_signature(CKPT_A), meta)


def _feature_graph(provider: str, operation: str, config: dict) -> Workflow:
    extract = {"provider": provider, "weight_id": "wgt_x", "device": "cpu", "precision": "fp32", "stage": "fine_80"}
    return Workflow(
        nodes=[
            NodeInstance(id="ref_img", type="image_source", config={}), NodeInstance(id="cur_img", type="image_source", config={}),
            NodeInstance(id="ref", type="feature_extract", config=extract), NodeInstance(id="cur", type="feature_extract", config=extract),
            NodeInstance(id="norm", type="feature_normalize", config={"mode": "l2_channel"}),
            NodeInstance(id="op", type=operation, config=config), NodeInstance(id="view", type="heatmap_preview", config={}),
        ],
        edges=[
            Edge(id="1", source="ref_img", source_handle="image", target="ref", target_handle="image"),
            Edge(id="2", source="cur_img", source_handle="image", target="cur", target_handle="image"),
            Edge(id="3", source="ref", source_handle="feature_map", target="norm", target_handle="feature"),
            Edge(id="4", source="norm", source_handle="feature", target="op", target_handle="reference"),
            Edge(id="5", source="cur", source_handle="feature_map", target="op", target_handle="current"),
            Edge(id="6", source="op", source_handle="heatmap", target="view", target_handle="heatmap"),
        ],
    )


def test_validator_rejects_feature_operations_on_the_wrong_provider(model_registry, stores):
    weights, assets = stores
    validator = WorkflowValidator(NodeRegistry(), model_registry, weights, assets)
    issues = validator.validate(_feature_graph("rfdetr-arch-b", "rfdetr_activation_xor", {"percentile": 90})).issues
    assert any(issue.code == "incompatible_feature_provider" and issue.node_id == "op" for issue in issues)
    issues = validator.validate(_feature_graph("rfdetr-v2.2", "mahalanobis_difference", {"geometry_id": "../x"})).issues
    codes = {issue.code for issue in issues if issue.node_id == "op"}
    assert {"incompatible_feature_provider", "invalid_geometry"} <= codes
    issues = validator.validate(_feature_graph("rfdetr-v2.2", "rfdetr_activation_xor", {"percentile": 90})).issues
    assert not any(issue.code == "incompatible_feature_provider" for issue in issues)


def test_feature_extract_stage_options_depend_on_provider():
    definition = NodeRegistry().get("feature_extract")
    stage = next(field for field in definition.config if field.id == "stage")
    assert stage.options_by["field"] == "provider"
    assert stage.options_by["options"] == {"dinov3": ["last_hidden_state"], "rfdetr-v2.2": ["fine_80", "coarse_20"], "rfdetr-arch-b": ["backbone_42"]}
    mah = NodeRegistry().get("mahalanobis_difference")
    assert [field.id for field in mah.config] == ["geometry_id"]  # no client file paths
