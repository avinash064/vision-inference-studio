from __future__ import annotations

import uuid
from pathlib import Path

from inference_studio.durable import JsonDocumentStore
from inference_studio.models import WeightStore

from .benchmarks import BenchmarkRegistry, stable_hash
from .schemas import CanvasRecipePromoteRequest, InferenceExperiment, InferenceRecipe, PromoteRequest, RecipeLifecycle


class RecipeRegistry:
    def __init__(self, root: Path, benchmarks: BenchmarkRegistry, weights: WeightStore) -> None:
        self.store, self.benchmarks, self.weights = JsonDocumentStore(root, "rcp_"), benchmarks, weights

    def promote(self, experiment: InferenceExperiment, request: PromoteRequest) -> InferenceRecipe:
        variant = next((value for value in experiment.variants if value.variant_id == request.variant_id), None)
        if variant is None:
            raise ValueError("Unknown experiment variant")
        benchmark = self.benchmarks.get(experiment.benchmark_id)
        return self._freeze(
            workflow=variant.workflow,
            name=request.name or variant.name,
            recipe_id=request.recipe_id,
            model_artifact_id=experiment.model_artifact_id,
            benchmark_id=benchmark.benchmark_id,
            benchmark_hash=benchmark.benchmark_hash,
            source_experiment_id=experiment.experiment_id,
            promotion_source="EXPERIMENT",
            inference_configuration=variant.inference_configuration,
        )

    def promote_canvas(self, request: CanvasRecipePromoteRequest) -> InferenceRecipe:
        """Freeze a user-selected Canvas snapshot without requiring an experiment."""
        return self._freeze(
            workflow=request.workflow,
            name=request.name,
            recipe_id=request.recipe_id,
            promotion_source="CANVAS",
        )

    def _freeze(
        self,
        *,
        workflow: dict,
        name: str,
        recipe_id: str | None,
        promotion_source: str,
        model_artifact_id: str | None = None,
        benchmark_id: str | None = None,
        benchmark_hash: str | None = None,
        source_experiment_id: str | None = None,
        inference_configuration: dict | None = None,
    ) -> InferenceRecipe:
        model_nodes = [node for node in workflow.get("nodes", []) if node.get("type") == "model"]
        if len(model_nodes) != 1:
            raise ValueError("A promoted recipe must contain exactly one model node")
        model = model_nodes[0].get("config", {})
        weight_id, weight_sha = model.get("weight_id"), None
        if weight_id:
            weight_sha = self.weights.get(str(weight_id), model.get("family")).sha256
        nodes, edges = workflow.get("nodes", []), workflow.get("edges", [])
        model_id = model_nodes[0]["id"]
        incoming = {node["id"]: [] for node in nodes}; outgoing = {node["id"]: [] for node in nodes}
        for edge in edges:
            incoming.setdefault(edge["target"], []).append(edge["source"])
            outgoing.setdefault(edge["source"], []).append(edge["target"])

        def closure(seed: str, graph: dict[str, list[str]]) -> set[str]:
            found, pending = set(), list(graph.get(seed, []))
            while pending:
                value = pending.pop()
                if value not in found:
                    found.add(value); pending.extend(graph.get(value, []))
            return found

        before_ids, after_ids = closure(model_id, incoming), closure(model_id, outgoing)
        before = {"nodes": [node for node in nodes if node["id"] in before_ids], "edges": [edge for edge in edges if edge["source"] in before_ids and edge["target"] in before_ids]}
        after = {"nodes": [node for node in nodes if node["id"] in after_ids], "edges": [edge for edge in edges if edge["source"] in after_ids and edge["target"] in after_ids]}
        workflow_hash = stable_hash(workflow)
        recipe_id = recipe_id or f"rcp_{uuid.uuid4().hex}"
        existing_versions = [int(raw["recipe_version"]) for raw in self.store.list() if raw.get("recipe_id") == recipe_id]
        recipe_version = max(existing_versions, default=0) + 1
        inference_configuration = inference_configuration or {}
        recipe = InferenceRecipe(
            recipe_id=recipe_id, recipe_version=recipe_version, name=name,
            lifecycle=RecipeLifecycle.PROMOTED, model_artifact_id=model_artifact_id,
            model_family=model.get("family"), weight_id=weight_id, weight_sha256=weight_sha,
            preprocessing_graph=before, preprocessing_hash=stable_hash(before),
            inference_parameters={**model, **inference_configuration}, inference_configuration_hash=stable_hash({**model, **inference_configuration}),
            postprocessing_graph=after, postprocessing_hash=stable_hash(after), workflow=workflow,
            workflow_id=f"wf_{workflow_hash[:24]}", workflow_version=str(workflow.get("version", "0.2")), workflow_hash=workflow_hash,
            benchmark_id=benchmark_id, benchmark_hash=benchmark_hash,
            source_experiment_id=source_experiment_id, promotion_source=promotion_source,
            runtime_capabilities=[str(model.get("device", "cpu")), str(model.get("precision", "fp32"))],
        )
        self.store.put(f"{recipe.recipe_id}_v{recipe.recipe_version}", recipe.model_dump(mode="json"), create_only=True)
        return recipe

    def get(self, identity: str, version: int | None = None) -> InferenceRecipe:
        values = [InferenceRecipe.model_validate(raw) for raw in self.store.list() if raw.get("recipe_id") == identity]
        if version is not None:
            values = [recipe for recipe in values if recipe.recipe_version == version]
        if not values:
            raise KeyError(f"{identity} v{version}" if version is not None else identity)
        return max(values, key=lambda recipe: recipe.recipe_version)

    @staticmethod
    def public(recipe: InferenceRecipe, include_workflow: bool = False) -> dict:
        value = recipe.model_dump(mode="json", exclude=set() if include_workflow else {"workflow"})
        model = next((node.get("config", {}) for node in recipe.workflow.get("nodes", []) if node.get("type") == "model"), {})
        value["auto_annotation_recipe"] = {
            "recipe_id": recipe.recipe_id, "recipe_version": recipe.recipe_version,
            "model_family": recipe.model_family, "model_variant": f"{model.get('version', 'custom')}:{model.get('size', 'custom')}",
            "weight_id": recipe.weight_id, "weight_sha256": recipe.weight_sha256,
            "preprocessing": recipe.preprocessing_graph, "inference": recipe.inference_parameters,
            "postprocessing": recipe.postprocessing_graph, "workflow_id": recipe.workflow_id,
            "workflow_version": recipe.recipe_version, "workflow_hash": recipe.workflow_hash,
            "runtime_requirements": {"capabilities": recipe.runtime_capabilities},
        }
        return value

    def list(self, promoted_only: bool = True) -> list[dict]:
        values = [InferenceRecipe.model_validate(raw) for raw in self.store.list()]
        if promoted_only:
            values = [recipe for recipe in values if recipe.lifecycle == RecipeLifecycle.PROMOTED]
        return [self.public(recipe) for recipe in values]
