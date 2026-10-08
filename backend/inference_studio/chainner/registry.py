from __future__ import annotations

import ast
import importlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from inference_studio.nodes.base import ConfigField, NodeDefinition, PortDefinition

from .types import ChainnerTypeAdapter


PACKAGE_NAMES = (
    "chaiNNer_standard",
    "chaiNNer_pytorch",
    "chaiNNer_onnx",
    "chaiNNer_ncnn",
    "chaiNNer_tensorrt",
    "chaiNNer_external",
)


def _json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "value"):
        return _json(value.value)
    if hasattr(value, "to_dict"):
        return _json(value.to_dict())
    return str(value)


def _conditions(layout: list[Any]) -> dict[int, Any]:
    result: dict[int, Any] = {}

    def visit(item: Any, inherited: Any = None) -> None:
        if isinstance(item, int):
            if inherited is not None:
                result[item] = inherited
            return
        if not hasattr(item, "items"):
            return
        raw = item.to_dict()
        condition = raw.get("options", {}).get("condition", inherited)
        for child in item.items:
            visit(child, condition)

    for entry in layout:
        visit(entry)
    return result


class ChainnerRegistryAdapter:
    def __init__(self, source_dir: Path, report_path: Path) -> None:
        self.source_dir = source_dir.resolve()
        self.report_path = report_path.resolve()
        self.registry: Any = None
        self.definitions: list[NodeDefinition] = []
        self.load_errors: list[Any] = []
        self.package_status: dict[str, str] = {}
        self.report: dict[str, Any] = {}

    def discover_packages(self) -> None:
        if not self.source_dir.is_dir():
            raise FileNotFoundError(f"chaiNNer source directory does not exist: {self.source_dir}")
        # The supplied chaiNNer snapshot targets NumPy 1.x, where bool8 was an
        # alias. NumPy 2.x removed only the alias, not the underlying dtype.
        import numpy as np
        if not hasattr(np, "bool8"):
            setattr(np, "bool8", np.bool_)
        source = str(self.source_dir)
        if source not in sys.path:
            sys.path.insert(0, source)
        for name in PACKAGE_NAMES:
            try:
                importlib.import_module(f"packages.{name}")
                self.package_status[name] = "loaded"
            except Exception as exc:  # optional package failures must be isolated
                self.package_status[name] = f"{type(exc).__name__}: {exc}"
        api = importlib.import_module("api")
        self.registry = api.registry

    def discover_nodes(self) -> list[NodeDefinition]:
        self.discover_packages()
        self.load_errors = self.registry.load_nodes(str(self.source_dir / "server.py"))
        available = [self.adapt_node(schema_id, node, group) for schema_id, (node, group) in self.registry.nodes.items()]
        runtime_ids = {item.type for item in available}
        unavailable = [item for item in self._static_unavailable(runtime_ids)]
        self.definitions = sorted([*available, *unavailable], key=lambda item: (item.package.lower(), item.category.lower(), item.group.lower(), item.label.lower()))
        self._write_report()
        return self.definitions

    def get_node(self, schema_id: str) -> Any:
        if self.registry is None or schema_id not in self.registry.nodes:
            raise KeyError(schema_id)
        return self.registry.nodes[schema_id][0]

    @staticmethod
    def _field(input_value: Any, condition: Any) -> ConfigField | None:
        raw = _json(input_value.to_dict())
        kind = raw.get("kind")
        mapped: str | None = None
        if kind in {"number", "slider"}:
            mapped = "number"
        elif kind == "dropdown":
            mapped = "boolean" if getattr(input_value, "associated_type", None) is bool else "select"
        elif kind == "text":
            mapped = "textarea" if raw.get("multiline") else "string"
        elif kind in {"file", "directory", "color"}:
            mapped = kind
        if mapped is None:
            return None
        options = [{"label": option.get("option", str(option.get("value"))), "value": option.get("value")} for option in raw.get("options", [])]
        return ConfigField(
            id=f"input_{input_value.id}", label=input_value.label, kind=mapped, required=not input_value.optional,
            default=raw.get("def"), options=options, minimum=raw.get("min"), maximum=raw.get("max"),
            step=raw.get("controlsStep") or raw.get("sliderStep"), placeholder=raw.get("placeholder"),
            condition=_json(condition), chainner_metadata=raw,
        )

    def adapt_node(self, schema_id: str, node: Any, group: Any) -> NodeDefinition:
        conditions = _conditions(node.group_layout)
        iterable_inputs = {int(value) for info in node.iterable_inputs for value in info.inputs}
        iterable_outputs = {int(value) for info in node.iterable_outputs for value in info.outputs}
        fields: list[ConfigField] = []
        ports: list[PortDefinition] = []
        for input_value in node.inputs:
            field = self._field(input_value, conditions.get(int(input_value.id)))
            if field is not None:
                fields.append(field)
            if input_value.has_handle:
                data_types = ChainnerTypeAdapter.adapt(input_value.input_type, input_value.associated_type)
                if int(input_value.id) in iterable_inputs:
                    data_types = ["sequence", *data_types]
                ports.append(PortDefinition(
                    id=f"input_{input_value.id}", label=input_value.label,
                    data_types=data_types,
                    required=field is None and not input_value.optional, chainner_type=_json(input_value.input_type),
                ))
        outputs = []
        for output in node.outputs:
            if not output.has_handle:
                continue
            data_types = ChainnerTypeAdapter.adapt(output.output_type, output.associated_type)
            if int(output.id) in iterable_outputs:
                data_types = ["sequence", *data_types]
            outputs.append(PortDefinition(id=f"output_{output.id}", label=output.label, data_types=data_types, required=False, chainner_type=_json(output.output_type)))
        package = group.category.package
        metadata = {
            "icon": node.icon, "see_also": node.see_also, "features": list(node.features),
            "group_layout": [_json(item) for item in node.group_layout],
            "iterator_inputs": [_json(item.to_dict()) for item in node.iterable_inputs],
            "iterator_outputs": [_json(item.to_dict()) for item in node.iterable_outputs],
            "inputs": [_json(item.to_dict()) for item in node.inputs],
            "outputs": [_json(item.to_dict()) for item in node.outputs],
        }
        return NodeDefinition(
            type=schema_id, schema_id=schema_id, label=node.name, category=group.category.name,
            description=node.description, inputs=ports, outputs=outputs, config=fields,
            provider="chainner", package=package.id, group=group.name, kind=node.kind,
            available=True, availability="available", side_effects=node.side_effects,
            deprecated=node.deprecated, metadata=metadata,
        )

    def _static_unavailable(self, runtime_ids: set[str]) -> list[NodeDefinition]:
        errors_by_file = {Path(error.file).resolve(): f"{type(error.error).__name__}: {error.error}" for error in self.load_errors}
        result: list[NodeDefinition] = []
        for path in sorted((self.source_dir / "packages").rglob("*.py")):
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            except (OSError, SyntaxError):
                continue
            for fn in [node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]:
                for decorator in fn.decorator_list:
                    if not isinstance(decorator, ast.Call) or not isinstance(decorator.func, ast.Attribute) or decorator.func.attr != "register":
                        continue
                    values = {kw.arg: kw.value for kw in decorator.keywords if kw.arg}
                    schema_node = values.get("schema_id") or (decorator.args[0] if decorator.args else None)
                    name_node = values.get("name")
                    if not isinstance(schema_node, ast.Constant) or not isinstance(schema_node.value, str):
                        continue
                    schema_id = schema_node.value
                    if schema_id in runtime_ids:
                        continue
                    label = name_node.value if isinstance(name_node, ast.Constant) and isinstance(name_node.value, str) else fn.name.replace("_", " ").title()
                    relative = path.relative_to(self.source_dir / "packages")
                    package = relative.parts[0]
                    category = relative.parts[1].replace("_", " ").title() if len(relative.parts) > 2 else "Unavailable"
                    group = relative.parts[-2].replace("_", " ").title() if len(relative.parts) > 2 else "Unavailable"
                    reason = errors_by_file.get(path.resolve(), self.package_status.get(package, "Registration did not complete"))
                    result.append(NodeDefinition(
                        type=schema_id, schema_id=schema_id, label=label, category=category,
                        description=f"Unavailable — {reason}", provider="chainner", package=package,
                        group=group, available=False, availability="dependency_missing" if "ModuleNotFoundError" in reason else "import_failed",
                        availability_reason=reason, metadata={"source_file": str(path)},
                    ))
        return result

    def _write_report(self) -> None:
        static_files = list((self.source_dir / "packages").rglob("*.py"))
        static_count = sum(len(re.findall(r"@[^\n]+\.register\(", path.read_text(encoding="utf-8", errors="ignore"))) for path in static_files)
        counts = Counter(item.package for item in self.definitions if item.available)
        self.report = {
            "source_directory": str(self.source_dir), "python_files_scanned": len(static_files),
            "static_registrations": static_count, "packages": self.package_status,
            "runtime_discovered": sum(item.available for item in self.definitions),
            "vision_imported": len(self.definitions), "available": sum(item.available for item in self.definitions),
            "unavailable": sum(not item.available for item in self.definitions), "load_errors": len(self.load_errors),
            "kinds": dict(Counter(item.kind for item in self.definitions if item.available)), "available_by_package": dict(counts),
            "nodes": [{"schema_id": item.type, "name": item.label, "package": item.package, "category": item.category, "group": item.group, "kind": item.kind, "available": item.available, "dependency_status": item.availability, "error": item.availability_reason} for item in self.definitions],
        }
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.report_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.report, indent=2), encoding="utf-8")
        temporary.replace(self.report_path)
