from __future__ import annotations

from inference_studio.adapters import ModelConfig
from inference_studio.assets import AssetError, AssetStore
from inference_studio.models.registry import ModelRegistry, ModelRegistryError
from inference_studio.models.weights import WeightError, WeightStore
from inference_studio.nodes import NodeRegistry
from inference_studio.adapters.huggingface import parse_coordinates

from .planner import CycleError, build_execution_plan, topological_sort
from .schema import ValidationIssue, ValidationResult, Workflow


class WorkflowValidator:
    def __init__(self, nodes: NodeRegistry, models: ModelRegistry, weights: WeightStore, assets: AssetStore) -> None:
        self.nodes, self.models, self.weights, self.assets = nodes, models, weights, assets

    def validate(self, workflow: Workflow, injected_source_nodes: set[str] | None = None) -> ValidationResult:
        injected_source_nodes = injected_source_nodes or set()
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
            if not definition.available:
                issues.append(ValidationIssue(code="node_unavailable", message=definition.availability_reason or f"{definition.label} is unavailable", node_id=node.id))
                continue
            for field in definition.config:
                value = node.config.get(field.id, field.default)
                if node.id in injected_source_nodes and node.type == "image_source" and field.id == "asset_id":
                    continue
                if node.type == "model" and field.id == "weight_id":
                    try:
                        if not self.models.get(str(node.config.get("family"))).requires_weights:
                            continue
                    except ModelRegistryError:
                        pass
                if field.required and (value is None or value == ""):
                    issues.append(ValidationIssue(code="required_config", message=f"{field.label} is required", node_id=node.id))
                if isinstance(value, (int, float)):
                    if field.minimum is not None and value < field.minimum:
                        issues.append(ValidationIssue(code="config_range", message=f"{field.label} must be at least {field.minimum}", node_id=node.id))
                    if field.maximum is not None and value > field.maximum:
                        issues.append(ValidationIssue(code="config_range", message=f"{field.label} must be at most {field.maximum}", node_id=node.id))
            if node.id not in injected_source_nodes:
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
        if order:
            plan = build_execution_plan(workflow, self.nodes)
            for (source_id, source_handle), count in plan.consumers.items():
                if source_id in plan.stream_nodes and count > 1:
                    issues.append(ValidationIssue(code="non_replayable_stream_fanout", message=f"Stream output {source_handle} has {count} consumers; insert a collector/materialization node", node_id=source_id))
        self._validate_feature_providers(workflow, node_map, issues)
        return ValidationResult(valid=not issues, issues=issues, order=order)

    # Feature operations that only exist for one feature representation.
    _REQUIRED_PROVIDER = {"rfdetr_activation_xor": "rfdetr-v2.2", "mahalanobis_difference": "rfdetr-arch-b"}

    def _feature_providers(self, node_id: str, workflow: Workflow, node_map, seen: set[str]) -> set[str]:
        """Providers of the Feature Extract nodes feeding ``node_id`` (through normalize/difference nodes)."""
        if node_id in seen:
            return set()
        seen.add(node_id)
        node = node_map.get(node_id)
        if node is None:
            return set()
        if node.type == "feature_extract":
            return {str(node.config.get("provider", ""))}
        providers: set[str] = set()
        for edge in workflow.edges:
            if edge.target == node_id:
                providers |= self._feature_providers(edge.source, workflow, node_map, seen)
        return providers

    def _validate_feature_providers(self, workflow: Workflow, node_map, issues: list[ValidationIssue]) -> None:
        for node in workflow.nodes:
            required = self._REQUIRED_PROVIDER.get(node.type)
            if required is None:
                continue
            providers = self._feature_providers(node.id, workflow, node_map, set()) - {required}
            if providers:
                definition = self.nodes.get(node.type)
                issues.append(ValidationIssue(
                    code="incompatible_feature_provider",
                    message=f"{definition.label} accepts only {required} features; got {', '.join(sorted(providers))}",
                    node_id=node.id,
                ))
            if node.type == "mahalanobis_difference" and node.config.get("geometry_id") not in (None, ""):
                from inference_studio.features.geometry import GEOMETRY_ID
                if not GEOMETRY_ID.fullmatch(str(node.config["geometry_id"])):
                    issues.append(ValidationIssue(code="invalid_geometry", message="Invalid frozen geometry id", node_id=node.id))

    def _validate_special(self, node, issues: list[ValidationIssue]) -> None:
        if node.type in {"image_source", "video_source"} and node.config.get("asset_id"):
            try:
                asset = self.assets.get(str(node.config["asset_id"]))
                expected = "image" if node.type == "image_source" else "video"
                if asset.kind != expected:
                    raise AssetError(f"Expected a {expected} asset")
            except AssetError as exc:
                issues.append(ValidationIssue(code="invalid_asset", message=str(exc), node_id=node.id))
        if node.type == "model" and all(node.config.get(key) not in (None, "") for key in ("family", "version", "size")):
            try:
                config = ModelConfig.model_validate(node.config)
                family = self.models.validate_selection(config.family, config.version, config.size)
                allowed_patch_sizes = family.patch_sizes.get(config.size, [])
                if config.patch_size and allowed_patch_sizes and config.patch_size not in allowed_patch_sizes:
                    raise ValueError(
                        f"{config.family} {config.size} does not support patch size {config.patch_size}; "
                        f"choose {', '.join(map(str, allowed_patch_sizes))}"
                    )
                if family.requires_weights:
                    if not config.weight_id:
                        raise ValueError("Weights are required")
                    self.weights.get(config.weight_id, config.family)
                if "ocr_language" in family.runtime_fields and not config.ocr_language.strip():
                    raise ValueError("OCR language is required")
                if "endpoint_url" in family.runtime_fields:
                    if not config.endpoint_url.startswith(("http://", "https://")):
                        raise ValueError("Endpoint URL must start with http:// or https://")
                if "endpoint_model" in family.runtime_fields and not config.endpoint_model.strip():
                    raise ValueError("Endpoint model is required")
                if config.precision == "fp16" and config.device == "cpu":
                    raise ValueError("fp16 requires a CUDA device")
                if family.prompt_modes:
                    if config.prompt_mode not in family.prompt_modes:
                        raise ValueError(f"{family.display_name} requires one of these prompt modes: {', '.join(family.prompt_modes)}")
                    if config.prompt_mode == "text" and not config.text_prompt.strip():
                        raise ValueError("Text prompt is required")
                    if config.prompt_mode == "point":
                        parse_coordinates(config.point_prompt, 2, "Point prompt")
                    if config.prompt_mode == "box":
                        box = parse_coordinates(config.box_prompt, 4, "Box prompt")
                        if box[2] <= box[0] or box[3] <= box[1]:
                            raise ValueError("Box prompt must have x2 > x1 and y2 > y1")
            except (ValueError, ModelRegistryError, WeightError) as exc:
                issues.append(ValidationIssue(code="invalid_model", message=str(exc), node_id=node.id))
        if node.type == "feature_extract" and all(node.config.get(key) not in (None, "") for key in ("provider", "weight_id", "stage")):
            try:
                provider = str(node.config["provider"])
                try:
                    definition = self.models.manifest.feature_extractors[provider]
                except KeyError as exc:
                    raise ValueError(f"Unknown feature provider: {provider}") from exc
                stage = str(node.config["stage"])
                if stage not in definition.stages:
                    raise ValueError(
                        f"{definition.display_name} does not expose {stage}; "
                        f"choose one of {', '.join(definition.stages)}"
                    )
                self.weights.get(str(node.config["weight_id"]), definition.family)
                if node.config.get("precision", "fp32") == "fp16" and node.config.get("device", "cpu") == "cpu":
                    raise ValueError("fp16 feature extraction requires a CUDA device")
            except (ValueError, WeightError) as exc:
                issues.append(ValidationIssue(code="invalid_feature_extractor", message=str(exc), node_id=node.id))

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
