from __future__ import annotations

import uuid
from typing import Any

from .schemas import (
    CanonicalGeometry,
    CanonicalPrediction,
    GeometryType,
    PredictionProvenance,
    RecipePin,
    SourceProvenance,
)


def _prediction_id(item_id: str, index: int) -> str:
    return f"pred_{uuid.uuid5(uuid.NAMESPACE_URL, f'{item_id}:{index}').hex}"


def canonical_predictions(
    result: dict[str, Any], item_id: str, recipe: RecipePin, source: SourceProvenance
) -> list[CanonicalPrediction]:
    provenance = PredictionProvenance(
        recipe_id=recipe.recipe_id,
        recipe_version=recipe.recipe_version,
        recipe_hash=recipe.recipe_hash,
        workflow_id=recipe.workflow_id,
        workflow_version=recipe.workflow_version,
        workflow_hash=recipe.workflow_hash,
        model_family=recipe.model_family,
        model_variant=recipe.model_variant,
        weight_id=recipe.weight_id,
        weight_sha256=recipe.weight_sha256,
    )
    result_type = result.get("result_type")
    values: list[CanonicalPrediction] = []
    if result_type == "detections":
        rows = result.get("detections", [])
        for index, row in enumerate(rows):
            values.append(
                CanonicalPrediction(
                    prediction_id=_prediction_id(item_id, index),
                    source_item_id=item_id,
                    category_id=int(row.get("class_id", 0)),
                    category_name=str(
                        row.get("label", row.get("class_name", "unknown"))
                    ),
                    confidence=float(row.get("confidence", 0)),
                    geometry=CanonicalGeometry(
                        type=GeometryType.BBOX,
                        bbox=[float(value) for value in row["bbox"]],
                    ),
                    provenance=provenance,
                    source_provenance=source,
                )
            )
        return values
    if result_type == "segmentation":
        rows = result.get("masks", [])
        for index, row in enumerate(rows):
            polygons = [
                [float(value) for value in polygon]
                for polygon in row.get("polygons", [])
            ]
            values.append(
                CanonicalPrediction(
                    prediction_id=_prediction_id(item_id, index),
                    source_item_id=item_id,
                    category_id=int(row.get("class_id", 0)),
                    category_name=str(row.get("label", "unknown")),
                    confidence=float(row.get("confidence", 0)),
                    geometry=CanonicalGeometry(
                        type=GeometryType.POLYGON,
                        bbox=[float(value) for value in row["bbox"]],
                        polygons=polygons,
                    ),
                    attributes={"area": int(row.get("area", 0))},
                    provenance=provenance,
                    source_provenance=source,
                )
            )
        return values
    if result_type == "ocr":
        for index, row in enumerate(result.get("spans", [])):
            if not row.get("bbox"):
                continue
            values.append(
                CanonicalPrediction(
                    prediction_id=_prediction_id(item_id, index),
                    source_item_id=item_id,
                    category_id=0,
                    category_name="text",
                    confidence=float(row.get("confidence", 1)),
                    geometry=CanonicalGeometry(
                        type=GeometryType.BBOX,
                        bbox=[float(value) for value in row["bbox"]],
                    ),
                    attributes={
                        "text": str(row.get("text", "")),
                        "language": result.get("language"),
                    },
                    provenance=provenance,
                    source_provenance=source,
                )
            )
        return values
    if result_type in {"feature_map", "embedding", "tensor"}:
        return []
    raise ValueError(
        f"Auto Annotation does not support Vision result type: {result_type or 'unknown'}"
    )
