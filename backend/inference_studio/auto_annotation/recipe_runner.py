from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
from typing import Any

from inference_studio.binaries import resolve_binary
from inference_studio.graph import Workflow
from inference_studio.graph.validator import WorkflowValidator
from inference_studio.models import ModelRegistry, WeightStore
from inference_studio.public_hub.benchmarks import stable_hash
from inference_studio.public_hub.experiments import ExperimentRegistry
from inference_studio.public_hub.recipes import RecipeRegistry
from inference_studio.public_hub.schemas import InferenceRecipe, RecipeLifecycle

from .schemas import PreflightIssue, RecipePin, RecipePreflight


_OPTIONAL_RUNTIME = {
    "yolo": "ultralytics",
    "yolo-seg": "ultralytics",
    "rtdetr": "ultralytics",
    "rtdetrv2": "torch",
    "rfdetr": "rfdetr",
    "dinov3": "transformers",
    "grounding-dino": "groundingdino",
    "sam": "segment_anything",
    "sam2": "ultralytics",
    "sam3": "ultralytics",
    "paddleocr": "paddleocr",
    "tesseract": "pytesseract",
    "onnx": "onnxruntime",
    "vllm": "httpx",
}


class RecipeExecutor:
    """Preflight and execute an exact promoted recipe through the shared engine."""

    def __init__(
        self,
        recipes: RecipeRegistry,
        experiments: ExperimentRegistry,
        validator: WorkflowValidator,
        models: ModelRegistry,
        weights: WeightStore,
    ) -> None:
        self.recipes = recipes
        self.experiments = experiments
        self.validator = validator
        self.models = models
        self.weights = weights

    @staticmethod
    def recipe_hash(recipe: InferenceRecipe) -> str:
        return stable_hash(recipe.model_dump(mode="json"))

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
        return digest.hexdigest()

    def preflight(self, recipe_id: str, recipe_version: int) -> RecipePreflight:
        errors: list[PreflightIssue] = []
        try:
            recipe = self.recipes.get(recipe_id, recipe_version)
        except KeyError:
            return RecipePreflight(
                status="NOT_READY",
                errors=[
                    PreflightIssue(
                        code="recipe_not_found",
                        field="recipe_version",
                        message=f"Promoted recipe {recipe_id} version {recipe_version} was not found",
                    )
                ],
            )
        if recipe.lifecycle != RecipeLifecycle.PROMOTED:
            errors.append(
                PreflightIssue(
                    code="recipe_not_promoted",
                    field="recipe_id",
                    message="Only PROMOTED recipes can run Auto Annotation",
                )
            )
        if stable_hash(recipe.workflow) != recipe.workflow_hash:
            errors.append(
                PreflightIssue(
                    code="workflow_hash_mismatch",
                    message="The stored workflow no longer matches its immutable recipe hash",
                )
            )

        model_nodes = [
            node
            for node in recipe.workflow.get("nodes", [])
            if node.get("type") == "model"
        ]
        model = model_nodes[0].get("config", {}) if len(model_nodes) == 1 else {}
        if len(model_nodes) != 1:
            errors.append(
                PreflightIssue(
                    code="model_count",
                    message="A recipe must contain exactly one model node",
                )
            )
        family_name = str(model.get("family") or recipe.model_family or "")
        try:
            family = self.models.get(family_name)
            self.models.validate_selection(
                family_name, str(model.get("version", "")), str(model.get("size", ""))
            )
        except Exception as exc:
            family = None
            errors.append(
                PreflightIssue(
                    code="model_unavailable", field="model_family", message=str(exc)
                )
            )

        if family is not None:
            module = _OPTIONAL_RUNTIME.get(family.adapter)
            if module and importlib.util.find_spec(module) is None:
                errors.append(
                    PreflightIssue(
                        code="runtime_dependency_missing",
                        message=f"{family.display_name} requires Python package {module}",
                    )
                )
            if family.adapter == "tesseract" and resolve_binary("tesseract") is None:
                errors.append(
                    PreflightIssue(
                        code="runtime_dependency_missing",
                        message="Tesseract executable is not installed",
                    )
                )

        weight_id = str(model.get("weight_id") or recipe.weight_id or "") or None
        weight_sha = None
        if family is not None and family.requires_weights:
            try:
                weight = self.weights.get(str(weight_id), family_name)
                weight_sha = self._sha256(Path(weight.path))
                expected = recipe.weight_sha256
                if expected and weight_sha != expected:
                    errors.append(
                        PreflightIssue(
                            code="weight_hash_mismatch",
                            field="weight_id",
                            message="Checkpoint SHA-256 differs from the promoted recipe",
                        )
                    )
            except Exception as exc:
                errors.append(
                    PreflightIssue(
                        code="weight_unavailable", field="weight_id", message=str(exc)
                    )
                )

        device = str(model.get("device", "cpu"))
        precision = str(model.get("precision", "fp32"))
        if device.startswith("cuda"):
            try:
                import torch

                if not torch.cuda.is_available():
                    errors.append(
                        PreflightIssue(
                            code="device_unavailable",
                            field="device",
                            message=f"Recipe requires {device}, but CUDA is unavailable",
                        )
                    )
            except ImportError:
                errors.append(
                    PreflightIssue(
                        code="device_unavailable",
                        field="device",
                        message="Recipe requires CUDA, but PyTorch is unavailable",
                    )
                )
        if precision == "fp16" and device == "cpu":
            errors.append(
                PreflightIssue(
                    code="precision_unavailable",
                    field="precision",
                    message="fp16 cannot run on CPU",
                )
            )

        try:
            workflow = Workflow.model_validate(recipe.workflow)
            sources = {
                node.id for node in workflow.nodes if node.type == "image_source"
            }
            if not sources:
                errors.append(
                    PreflightIssue(
                        code="image_source_missing",
                        message="Recipe needs an image_source node for Auto Annotation",
                    )
                )
            validation = self.validator.validate(workflow, sources)
            errors.extend(
                PreflightIssue(code=issue.code, message=issue.message)
                for issue in validation.issues
            )
        except Exception as exc:
            errors.append(PreflightIssue(code="workflow_invalid", message=str(exc)))

        pin = RecipePin(
            recipe_id=recipe.recipe_id,
            recipe_version=recipe.recipe_version,
            recipe_hash=self.recipe_hash(recipe),
            workflow_id=recipe.workflow_id,
            workflow_version=recipe.workflow_version,
            workflow_hash=recipe.workflow_hash,
            model_family=family_name or None,
            model_variant=f"{model.get('version', 'custom')}:{model.get('size', 'custom')}",
            weight_id=weight_id,
            weight_sha256=weight_sha or recipe.weight_sha256,
            preprocessing_hash=recipe.preprocessing_hash,
            inference_hash=recipe.inference_configuration_hash,
            postprocessing_hash=recipe.postprocessing_hash,
            device=device,
            precision=precision,
            patch_size=int(model["patch_size"]) if model.get("patch_size") else None,
        )
        return RecipePreflight(
            status="NOT_READY" if errors else "READY",
            recipe=self.recipes.public(recipe),
            pin=pin,
            errors=errors,
        )

    async def execute(
        self,
        recipe: InferenceRecipe,
        image: Any,
        source_name: str,
        source_id: str,
        *,
        persist_preview: bool = True,
    ) -> dict[str, Any]:
        return await self.experiments.execute_workflow_for_image(
            recipe.workflow,
            image,
            source_name,
            source_id,
            persist_preview=persist_preview,
        )
