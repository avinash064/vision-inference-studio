from __future__ import annotations

from inference_studio.graph import Edge, Workflow, WorkflowValidator, topological_sort
from inference_studio.nodes import NodeInstance, NodeRegistry


def model_config(weight_id: str):
    return {"family": "yolo", "version": "v11", "size": "n", "weight_id": weight_id, "device": "cpu", "precision": "fp32", "confidence": 0.25}


def valid_workflow(weight_id: str, asset_id: str) -> Workflow:
    return Workflow(nodes=[NodeInstance(id="source", type="image_source", config={"asset_id": asset_id}), NodeInstance(id="model", type="model", config=model_config(weight_id)), NodeInstance(id="preview", type="detection_preview")], edges=[Edge(id="one", source="source", source_handle="image", target="model", target_handle="input"), Edge(id="two", source="model", source_handle="output", target="preview", target_handle="result")])


def test_typed_workflow_validation_and_topological_order(model_registry, stores, registered_files):
    weights, assets = stores
    weight, asset = registered_files
    workflow = valid_workflow(weight.weight_id, asset.asset_id)
    result = WorkflowValidator(NodeRegistry(), model_registry, weights, assets).validate(workflow)
    assert result.valid, result.issues
    assert result.order == ["source", "model", "preview"]


def test_type_mismatch_is_reported(model_registry, stores, registered_files):
    weights, assets = stores
    _, asset = registered_files
    workflow = Workflow(nodes=[NodeInstance(id="source", type="image_source", config={"asset_id": asset.asset_id}), NodeInstance(id="preview", type="feature_preview")], edges=[Edge(id="bad", source="source", source_handle="image", target="preview", target_handle="result")])
    result = WorkflowValidator(NodeRegistry(), model_registry, weights, assets).validate(workflow)
    assert not result.valid
    assert "type_mismatch" in {issue.code for issue in result.issues}


def test_cycle_is_detected(model_registry, stores):
    weights, assets = stores
    workflow = Workflow(nodes=[NodeInstance(id="a", type="resize", config={"width": 20, "height": 20}), NodeInstance(id="b", type="resize", config={"width": 20, "height": 20})], edges=[Edge(id="ab", source="a", source_handle="image", target="b", target_handle="image"), Edge(id="ba", source="b", source_handle="image", target="a", target_handle="image")])
    result = WorkflowValidator(NodeRegistry(), model_registry, weights, assets).validate(workflow)
    assert "cycle" in {issue.code for issue in result.issues}
    try:
        topological_sort(workflow)
        assert False, "cycle should fail"
    except ValueError:
        pass


def test_prompt_model_validation_requires_family_specific_prompt(model_registry, stores, registered_files):
    weights, assets = stores
    _, asset = registered_files
    grounding_weight = weights.save(__import__("io").BytesIO(b"grounding"), "groundingdino_swint_ogc.pth", "grounding-dino")
    workflow = Workflow(
        nodes=[
            NodeInstance(id="source", type="image_source", config={"asset_id": asset.asset_id}),
            NodeInstance(id="model", type="model", config={"family": "grounding-dino", "version": "official", "size": "tiny", "weight_id": grounding_weight.weight_id, "device": "cpu", "precision": "fp32", "prompt_mode": "text", "text_prompt": ""}),
            NodeInstance(id="preview", type="detection_preview"),
        ],
        edges=[
            Edge(id="one", source="source", source_handle="image", target="model", target_handle="input"),
            Edge(id="two", source="model", source_handle="output", target="preview", target_handle="result"),
        ],
    )
    result = WorkflowValidator(NodeRegistry(), model_registry, weights, assets).validate(workflow)
    assert not result.valid
    assert any("Text prompt is required" in issue.message for issue in result.issues)

    workflow.nodes[1].config["text_prompt"] = "skip car, truck"
    assert WorkflowValidator(NodeRegistry(), model_registry, weights, assets).validate(workflow).valid


def test_sam_output_connects_only_to_segmentation_preview(model_registry, stores, registered_files):
    weights, assets = stores
    _, asset = registered_files
    sam_weight = weights.save(__import__("io").BytesIO(b"sam"), "model.safetensors", "sam")
    workflow = Workflow(
        nodes=[
            NodeInstance(id="source", type="image_source", config={"asset_id": asset.asset_id}),
            NodeInstance(id="model", type="model", config={"family": "sam", "version": "v1", "size": "vit-b", "weight_id": sam_weight.weight_id, "device": "cpu", "precision": "fp32", "prompt_mode": "box", "box_prompt": "2,3,20,30"}),
            NodeInstance(id="preview", type="segmentation_preview"),
        ],
        edges=[
            Edge(id="one", source="source", source_handle="image", target="model", target_handle="input"),
            Edge(id="two", source="model", source_handle="output", target="preview", target_handle="result"),
        ],
    )
    result = WorkflowValidator(NodeRegistry(), model_registry, weights, assets).validate(workflow)
    assert result.valid, result.issues


def test_weightless_ocr_model_validates_and_connects_to_ocr_preview(model_registry, stores, registered_files):
    weights, assets = stores
    _, asset = registered_files
    workflow = Workflow(
        nodes=[
            NodeInstance(id="source", type="image_source", config={"asset_id": asset.asset_id}),
            NodeInstance(id="model", type="model", config={"family": "tesseract", "version": "5", "size": "system", "device": "cpu", "precision": "fp32", "ocr_language": "en"}),
            NodeInstance(id="preview", type="ocr_preview"),
        ],
        edges=[
            Edge(id="one", source="source", source_handle="image", target="model", target_handle="input"),
            Edge(id="two", source="model", source_handle="output", target="preview", target_handle="result"),
        ],
    )
    result = WorkflowValidator(NodeRegistry(), model_registry, weights, assets).validate(workflow)
    assert result.valid, result.issues
