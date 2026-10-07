from __future__ import annotations

from inference_studio.adapters import ModelConfig
from inference_studio.assets import AssetError, AssetStore
from inference_studio.models.registry import ModelRegistry, ModelRegistryError
from inference_studio.models.weights import WeightError, WeightStore
from inference_studio.nodes import NodeRegistry

from .planner import CycleError, topological_sort
from .schema import ValidationIssue, ValidationResult, Workflow


class WorkflowValidator:
    def __init__(self, nodes: NodeRegistry, models: ModelRegistry, weights: WeightStore, assets: AssetStore) -> None:
        self.nodes, self.models, self.weights, self.assets = nodes, models, weights, assets

    def validate(self, workflow: Workflow) -> ValidationResult:
        issues: list[ValidationIssue] = []
        node_map = {}
        for node in workflow.nodes:
            if node.id in node_map:
                issues.append(ValidationIssue(code="duplicate_node", message=f"Duplicate node id: {node.id}", node_id=node.id))
            node_map[node.id] = node
            try:
                definition = self.nodes.get(node.type)
            except ValueError as exc:
                issues.append(ValidationIssue(code="unknown_node", message=str(exc), node_id=node.id))
                continue
            for field in definition.config:
                value = node.config.get(field.id, field.default)
                if field.required and (value is None or value == ""):
                    issues.append(ValidationIssue(code="required_config", message=f"{field.label} is required", node_id=node.id))
                if isinstance(value, (int, float)):
                    if field.minimum is not None and value < field.minimum:
                        issues.append(ValidationIssue(code="config_range", message=f"{field.label} must be at least {field.minimum}", node_id=node.id))
                    if field.maximum is not None and value > field.maximum:
                        issues.append(ValidationIssue(code="config_range", message=f"{field.label} must be at most {field.maximum}", node_id=node.id))
            self._validate_special(node, issues)

        incoming: dict[tuple[str, str], list] = {}
        for edge in workflow.edges:
            source, target = node_map.get(edge.source), node_map.get(edge.target)
            if source is None or target is None:
                issues.append(ValidationIssue(code="missing_node", message="Edge refers to a missing node", edge_id=edge.id))
                continue
            try:
                source_definition, target_definition = self.nodes.get(source.type), self.nodes.get(target.type)
                source_port = next(port for port in source_definition.outputs if port.id == edge.source_handle)
                target_port = next(port for port in target_definition.inputs if port.id == edge.target_handle)
            except (ValueError, StopIteration):
                issues.append(ValidationIssue(code="invalid_port", message="Edge refers to an invalid port", edge_id=edge.id))
                continue
            incoming.setdefault((target.id, target_port.id), []).append(edge)
            source_types = self._output_types(source, source_port.data_types, workflow, node_map, set())
            if not set(source_types).intersection(target_port.data_types):
                issues.append(ValidationIssue(code="type_mismatch", message=f"Cannot connect {source_types} to {target_port.data_types}", edge_id=edge.id))

        for node in workflow.nodes:
            try:
                definition = self.nodes.get(node.type)
            except ValueError:
                continue
            for port in definition.inputs:
                count = len(incoming.get((node.id, port.id), []))
                if port.required and count == 0:
                    issues.append(ValidationIssue(code="missing_input", message=f"Input {port.label} is required", node_id=node.id))
                if count > 1:
                    issues.append(ValidationIssue(code="multiple_inputs", message=f"Input {port.label} accepts one connection", node_id=node.id))
        try:
            order = topological_sort(workflow)
        except CycleError as exc:
            issues.append(ValidationIssue(code="cycle", message=str(exc)))
            order = []
        return ValidationResult(valid=not issues, issues=issues, order=order)

    def _validate_special(self, node, issues: list[ValidationIssue]) -> None:
        if node.type in {"image_source", "video_source"} and node.config.get("asset_id"):
            try:
                asset = self.assets.get(str(node.config["asset_id"]))
                expected = "image" if node.type == "image_source" else "video"
                if asset.kind != expected:
                    raise AssetError(f"Expected a {expected} asset")
            except AssetError as exc:
                issues.append(ValidationIssue(code="invalid_asset", message=str(exc), node_id=node.id))
        if node.type == "model" and all(node.config.get(key) not in (None, "") for key in ("family", "version", "size", "weight_id")):
            try:
                config = ModelConfig.model_validate(node.config)
                self.models.validate_selection(config.family, config.version, config.size)
                self.weights.get(config.weight_id, config.family)
                if config.precision == "fp16" and config.device == "cpu":
                    raise ValueError("fp16 requires a CUDA device")
            except (ValueError, ModelRegistryError, WeightError) as exc:
                issues.append(ValidationIssue(code="invalid_model", message=str(exc), node_id=node.id))

    def _output_types(self, node, declared, workflow, node_map, visiting: set[str]):
        if node.type != "model":
            return declared
        if node.id in visiting:
            return []
        visiting.add(node.id)
        input_edge = next((edge for edge in workflow.edges if edge.target == node.id and edge.target_handle == "input"), None)
        if input_edge and input_edge.source in node_map:
            upstream = node_map[input_edge.source]
            try:
                definition = self.nodes.get(upstream.type)
                port = next(port for port in definition.outputs if port.id == input_edge.source_handle)
                if "video" in self._output_types(upstream, port.data_types, workflow, node_map, visiting):
                    return ["video"]
            except (ValueError, StopIteration):
                pass
        try:
            return [self.models.get(str(node.config.get("family"))).result_type]
        except ModelRegistryError:
            return declared

