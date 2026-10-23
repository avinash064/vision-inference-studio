from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import cv2

from ..portable_paths import is_absolute_any_platform
from .repository import AutoAnnotationRunRepository
from .schemas import (
    AutoAnnotationItem,
    AutoAnnotationItemState,
    AutoAnnotationRun,
    ExportFormat,
    GeometryType,
)


def _json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _jsonl(path: Path, values: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for value in values:
            stream.write(
                json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
            )
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _categories(items: list[AutoAnnotationItem]) -> list[str]:
    return sorted(
        {prediction.category_name for item in items for prediction in item.predictions}
    )


class PackageBuilder:
    """Create and validate a portable, deterministic Auto Annotation package."""

    def __init__(self, repository: AutoAnnotationRunRepository) -> None:
        self.repository = repository

    @staticmethod
    def exported(items: list[AutoAnnotationItem]) -> list[AutoAnnotationItem]:
        return sorted(
            (item for item in items if item.state == AutoAnnotationItemState.EXPORTED),
            key=lambda item: item.output_image_relative_path or item.item_id,
        )

    def export_annotations(self, run: AutoAnnotationRun) -> dict[str, str]:
        root = Path(run.output_root)
        annotations = root / "annotations"
        annotations.mkdir(parents=True, exist_ok=True)
        items = self.exported(self.repository.items_for_run(run.run_id))
        outputs: dict[str, str] = {}
        if ExportFormat.CANONICAL in run.requested_exports:
            rows = []
            for item in items:
                rows.append(
                    {
                        "schema_version": 1,
                        "item_id": item.item_id,
                        "image": item.output_image_relative_path,
                        "width": item.width,
                        "height": item.height,
                        "source": self._portable_source(item),
                        "predictions": [
                            value.model_dump(mode="json") for value in item.predictions
                        ],
                    }
                )
            path = annotations / "canonical.jsonl"
            _jsonl(path, rows)
            outputs[ExportFormat.CANONICAL.value] = path.relative_to(root).as_posix()
        if ExportFormat.COCO in run.requested_exports:
            path = annotations / "coco.json"
            _json(path, self._coco(items))
            outputs[ExportFormat.COCO.value] = path.relative_to(root).as_posix()
        if ExportFormat.YOLO in run.requested_exports:
            yolo = annotations / "yolo"
            self._yolo(items, yolo)
            outputs[ExportFormat.YOLO.value] = yolo.relative_to(root).as_posix()
        return outputs

    @staticmethod
    def _portable_source(item: AutoAnnotationItem) -> dict:
        return {
            "source_type": item.source_type.value,
            "relative_source_path": item.relative_source_path,
            "source_sha256": item.source_sha256,
            "source_video_id": item.source_video_id,
            "source_video_name": item.source_video_name,
            "frame_index": item.frame_index,
            "pts": item.pts,
            "time_base": item.time_base,
            "timestamp_seconds": item.timestamp_seconds,
        }

    @staticmethod
    def _coco(items: list[AutoAnnotationItem]) -> dict:
        names = _categories(items)
        class_ids = {name: index + 1 for index, name in enumerate(names)}
        images, annotations = [], []
        annotation_id = 1
        for image_id, item in enumerate(items, 1):
            images.append(
                {
                    "id": image_id,
                    "file_name": item.output_image_relative_path,
                    "width": item.width,
                    "height": item.height,
                }
            )
            for prediction in item.predictions:
                bbox = prediction.geometry.bbox
                if not bbox:
                    continue
                x1, y1, x2, y2 = bbox
                x1, x2 = (
                    max(0.0, min(float(item.width), x1)),
                    max(0.0, min(float(item.width), x2)),
                )
                y1, y2 = (
                    max(0.0, min(float(item.height), y1)),
                    max(0.0, min(float(item.height), y2)),
                )
                width, height = max(0.0, x2 - x1), max(0.0, y2 - y1)
                row = {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": class_ids[prediction.category_name],
                    "bbox": [x1, y1, width, height],
                    "area": width * height,
                    "iscrowd": 0,
                    "score": prediction.confidence,
                }
                if (
                    prediction.geometry.type == GeometryType.POLYGON
                    and prediction.geometry.polygons
                ):
                    row["segmentation"] = prediction.geometry.polygons
                    row["area"] = float(
                        prediction.attributes.get("area") or row["area"]
                    )
                annotations.append(row)
                annotation_id += 1
        return {
            "info": {"description": "Vision Inference Studio Auto Annotation v0.1"},
            "licenses": [],
            "images": images,
            "annotations": annotations,
            "categories": [
                {"id": value, "name": name, "supercategory": "object"}
                for name, value in class_ids.items()
            ],
        }

    @staticmethod
    def _yolo(items: list[AutoAnnotationItem], root: Path) -> None:
        labels = root / "labels"
        labels.mkdir(parents=True, exist_ok=True)
        names = _categories(items)
        class_ids = {name: index for index, name in enumerate(names)}
        _json(
            root / "dataset.json",
            {
                "names": {str(index): name for name, index in class_ids.items()},
                "images": "../../images",
                "labels": "labels",
            },
        )
        for item in items:
            rows = []
            for prediction in item.predictions:
                class_id = class_ids[prediction.category_name]
                if (
                    prediction.geometry.type == GeometryType.POLYGON
                    and prediction.geometry.polygons
                ):
                    for polygon in prediction.geometry.polygons:
                        if len(polygon) >= 6:
                            points = [
                                f"{max(0.0, min(1.0, value / (item.width if index % 2 == 0 else item.height))):.8f}"
                                for index, value in enumerate(polygon)
                            ]
                            rows.append(" ".join([str(class_id), *points]))
                    continue
                if prediction.geometry.bbox:
                    x1, y1, x2, y2 = prediction.geometry.bbox
                    x1, x2 = (
                        max(0.0, min(float(item.width), x1)),
                        max(0.0, min(float(item.width), x2)),
                    )
                    y1, y2 = (
                        max(0.0, min(float(item.height), y1)),
                        max(0.0, min(float(item.height), y2)),
                    )
                    xc, yc = (x1 + x2) / (2 * item.width), (y1 + y2) / (2 * item.height)
                    width, height = (x2 - x1) / item.width, (y2 - y1) / item.height
                    rows.append(
                        f"{class_id} {xc:.8f} {yc:.8f} {width:.8f} {height:.8f}"
                    )
            relative = Path(
                item.output_image_relative_path or item.item_id
            ).with_suffix(".txt")
            target = labels / relative.name
            target.write_text(
                "\n".join(rows) + ("\n" if rows else ""), encoding="utf-8"
            )

    def build_manifest(self, run: AutoAnnotationRun) -> str:
        root = Path(run.output_root)
        items = self.exported(self.repository.items_for_run(run.run_id))
        rows = [{"type": "manifest", "schema_version": 1, "run_id": run.run_id}]
        rows.extend(
            {
                "type": "item",
                "item_id": item.item_id,
                "image": item.output_image_relative_path,
                "width": item.width,
                "height": item.height,
                "source": self._portable_source(item),
                "prediction_count": item.prediction_count,
            }
            for item in items
        )
        path = root / "manifest.jsonl"
        _jsonl(path, rows)
        return _sha256(path)

    def write_metadata(
        self, run: AutoAnnotationRun, exports: dict[str, str], manifest_sha: str
    ) -> None:
        root = Path(run.output_root)
        pin = run.recipe_pin.model_dump(mode="json") if run.recipe_pin else None
        _json(root / "recipe.json", pin)
        _json(
            root / "source.json",
            {"source_type": run.source_type.value, "metadata": run.source_metadata},
        )
        _jsonl(root / "failures.jsonl", self.repository.failures(run.run_id))
        _json(
            root / "run.json",
            {
                **run.model_dump(mode="json", exclude={"source_config", "output_root"}),
                "package_name": root.name,
                "manifest_sha256": manifest_sha,
                "exports": exports,
                "source": {
                    "source_type": run.source_type.value,
                    "recursive": run.source_config.recursive,
                },
            },
        )
        files = sorted(
            path
            for path in root.rglob("*")
            if path.is_file() and path.name != "checksums.json"
        )
        _json(
            root / "checksums.json",
            {path.relative_to(root).as_posix(): _sha256(path) for path in files},
        )

    def validate(self, run: AutoAnnotationRun) -> dict:
        root = Path(run.output_root).resolve()
        errors: list[str] = []
        required = [
            "run.json",
            "recipe.json",
            "source.json",
            "manifest.jsonl",
            "checksums.json",
            "failures.jsonl",
        ]
        for name in required:
            if not (root / name).is_file():
                errors.append(f"Missing {name}")
        try:
            metadata = json.loads((root / "run.json").read_text(encoding="utf-8"))
            if metadata.get("manifest_sha256") != _sha256(root / "manifest.jsonl"):
                errors.append("Run metadata has an invalid manifest SHA-256")
            for requested in run.requested_exports:
                if requested.value not in metadata.get("exports", {}):
                    errors.append(f"Requested {requested.value} export is missing")
            stored_recipe = json.loads(
                (root / "recipe.json").read_text(encoding="utf-8")
            )
            expected_recipe = (
                run.recipe_pin.model_dump(mode="json") if run.recipe_pin else None
            )
            if stored_recipe != expected_recipe:
                errors.append("Recipe metadata differs from the durable recipe pin")
            source_metadata = json.loads(
                (root / "source.json").read_text(encoding="utf-8")
            )
            if self._contains_absolute_path(source_metadata):
                errors.append("Portable source metadata contains an absolute path")
        except Exception as exc:
            errors.append(f"Invalid package metadata: {exc}")
        try:
            lines = [
                json.loads(line)
                for line in (root / "manifest.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
                if line
            ]
            entries = [line for line in lines if line.get("type") == "item"]
        except Exception as exc:
            entries = []
            errors.append(f"Invalid manifest: {exc}")
        exported = self.exported(self.repository.items_for_run(run.run_id))
        if len(entries) != len(exported):
            errors.append("Manifest item count does not match exported items")
        expected_images = {item.output_image_relative_path for item in exported}
        manifest_images = {entry.get("image") for entry in entries}
        if manifest_images != expected_images:
            errors.append("Manifest media does not exactly match exported item state")
        for entry in entries:
            relative = str(entry.get("image", ""))
            if is_absolute_any_platform(relative) or "\\" in relative:
                errors.append(f"Manifest path is not a portable relative POSIX path: {relative}")
                continue
            path = (root / relative).resolve()
            if root not in path.parents or not path.is_file():
                errors.append(f"Invalid manifest image path: {relative}")
                continue
            image = cv2.imread(str(path))
            if image is None:
                errors.append(f"Unreadable image: {relative}")
            elif [image.shape[1], image.shape[0]] != [
                entry.get("width"),
                entry.get("height"),
            ]:
                errors.append(f"Image dimensions differ: {relative}")
        try:
            checksums = json.loads(
                (root / "checksums.json").read_text(encoding="utf-8")
            )
            actual_files = {
                path.relative_to(root).as_posix()
                for path in root.rglob("*")
                if path.is_file() and path.name != "checksums.json"
            }
            if set(checksums) != actual_files:
                errors.append("Checksum inventory does not exactly match package files")
            for relative, expected in checksums.items():
                path = (root / relative).resolve()
                if (
                    root not in path.parents
                    or not path.is_file()
                    or _sha256(path) != expected
                ):
                    errors.append(f"Checksum mismatch: {relative}")
        except Exception as exc:
            errors.append(f"Invalid checksums: {exc}")
        canonical = root / "annotations" / "canonical.jsonl"
        if canonical.is_file():
            try:
                rows = [
                    json.loads(line)
                    for line in canonical.read_text(encoding="utf-8").splitlines()
                    if line
                ]
                if any(self._contains_absolute_path(row.get("source", {})) for row in rows):
                    errors.append("Canonical source metadata contains an absolute path")
                for row in rows:
                    for prediction in row.get("predictions", []):
                        if not 0 <= float(prediction["confidence"]) <= 1:
                            errors.append("Canonical confidence is outside 0..1")
                        if not prediction.get("category_name") or not prediction.get(
                            "provenance"
                        ):
                            errors.append(
                                "Canonical prediction lacks category or provenance"
                            )
            except Exception as exc:
                errors.append(f"Invalid canonical export: {exc}")
        coco = root / "annotations" / "coco.json"
        if coco.is_file():
            try:
                value = json.loads(coco.read_text(encoding="utf-8"))
                image_ids = {row["id"] for row in value["images"]}
                if any(is_absolute_any_platform(str(row.get("file_name", ""))) or "\\" in str(row.get("file_name", "")) for row in value["images"]):
                    errors.append("COCO file_name is not a portable relative POSIX path")
                category_ids = {row["id"] for row in value["categories"]}
                for row in value["annotations"]:
                    if (
                        row["image_id"] not in image_ids
                        or row["category_id"] not in category_ids
                    ):
                        errors.append("COCO annotation has an invalid reference")
                    if len(row.get("bbox", [])) != 4 or any(
                        number < 0 for number in row["bbox"][2:]
                    ):
                        errors.append("COCO annotation has an invalid bbox")
                    if "segmentation" in row and any(
                        len(polygon) < 6 or len(polygon) % 2
                        for polygon in row["segmentation"]
                    ):
                        errors.append("COCO annotation has an invalid polygon")
            except Exception as exc:
                errors.append(f"Invalid COCO export: {exc}")
        yolo = root / "annotations" / "yolo" / "labels"
        if yolo.is_dir():
            try:
                names = json.loads(
                    (yolo.parent / "dataset.json").read_text(encoding="utf-8")
                )["names"]
                class_count = len(names)
            except Exception:
                class_count = 0
                errors.append("Invalid YOLO dataset metadata")
            for path in yolo.glob("*.txt"):
                for line in path.read_text(encoding="utf-8").splitlines():
                    try:
                        fields = line.split()
                        class_id = int(fields[0])
                        values = [float(value) for value in fields[1:]]
                        valid_shape = len(values) == 4 or (
                            len(values) >= 6 and len(values) % 2 == 0
                        )
                        if (
                            class_id < 0
                            or class_id >= class_count
                            or not valid_shape
                            or any(value < 0 or value > 1 for value in values)
                        ):
                            errors.append(f"Invalid YOLO coordinates: {path.name}")
                    except Exception:
                        errors.append(f"Invalid YOLO line: {path.name}")
        return {"valid": not errors, "errors": errors, "exported_items": len(exported)}

    @staticmethod
    def _contains_absolute_path(value: Any) -> bool:
        if isinstance(value, dict):
            return any(
                PackageBuilder._contains_absolute_path(child)
                for child in value.values()
            )
        if isinstance(value, list):
            return any(PackageBuilder._contains_absolute_path(child) for child in value)
        if isinstance(value, str):
            # Recognise both OS styles: a package written on Windows is often validated on Linux.
            return is_absolute_any_platform(value)
        return False
