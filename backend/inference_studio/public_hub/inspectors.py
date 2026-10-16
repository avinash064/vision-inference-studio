from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from xml.etree import ElementTree

import cv2
import yaml

from .schemas import DatasetInspection, InspectionState, ModelInspection


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


class DatasetInspector:
    def inspect(self, root: Path) -> DatasetInspection:
        images = sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
        if not images:
            return DatasetInspection(state=InspectionState.INVALID, explanation="No supported image files were found")
        broken, hashes, items = [], Counter(), []
        for path in images:
            image = cv2.imread(str(path))
            relative = path.relative_to(root).as_posix()
            if image is None:
                broken.append(relative)
                continue
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            hashes[digest] += 1
            h, w = image.shape[:2]
            items.append({"item_id": digest[:24], "relative_path": relative, "sha256": digest, "width": w, "height": h, "classes": []})

        coco_files = []
        yolo_labels = list(root.rglob("*.txt"))
        voc_files = list(root.rglob("*.xml"))
        classes: list[str] = []
        annotation_count, task, fmt = 0, "unknown", "unknown"
        for json_path in root.rglob("*.json"):
            try:
                data = json.loads(json_path.read_text(encoding="utf-8"))
                if isinstance(data, dict) and {"images", "annotations", "categories"}.issubset(data):
                    coco_files.append((json_path, data))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                continue
        if coco_files:
            fmt = "COCO"
            categories = {int(c["id"]): str(c["name"]) for _, data in coco_files for c in data.get("categories", []) if "id" in c and "name" in c}
            classes = list(dict.fromkeys(categories.values()))
            annotation_count = sum(len(data.get("annotations", [])) for _, data in coco_files)
            task = "segmentation" if any(a.get("segmentation") for _, data in coco_files for a in data.get("annotations", [])) else "detection"
            names = {str(im.get("file_name", "")): im.get("id") for _, data in coco_files for im in data.get("images", [])}
            anns = {image_id: [] for image_id in names.values()}
            ground_truth = {image_id: [] for image_id in names.values()}
            for _, data in coco_files:
                for ann in data.get("annotations", []):
                    if ann.get("image_id") in anns and ann.get("category_id") in categories:
                        label = categories[ann["category_id"]]
                        anns[ann["image_id"]].append(label)
                        bbox = ann.get("bbox") or []
                        xyxy = [float(bbox[0]), float(bbox[1]), float(bbox[0] + bbox[2]), float(bbox[1] + bbox[3])] if len(bbox) == 4 else []
                        polygons = ann.get("segmentation") if isinstance(ann.get("segmentation"), list) else []
                        ground_truth[ann["image_id"]].append({"label": label, "class_id": int(ann["category_id"]), "bbox": xyxy, "polygons": polygons})
            for item in items:
                match = next((iid for name, iid in names.items() if item["relative_path"].endswith(name)), None)
                item["classes"] = sorted(set(anns.get(match, [])))
                item["ground_truth"] = ground_truth.get(match, [])
        elif yolo_labels:
            fmt, task = "YOLO", "detection"
            annotation_count = 0
            names: list[str] = []
            for config_name in ("data.yaml", "dataset.yaml"):
                config = next(iter(root.rglob(config_name)), None)
                if config:
                    try:
                        raw = yaml.safe_load(config.read_text(encoding="utf-8")) or {}
                        value = raw.get("names", [])
                        names = list(value.values()) if isinstance(value, dict) else list(value)
                    except Exception:
                        pass
            classes = [str(name) for name in names]
            label_by_stem = {}
            for label in yolo_labels:
                try:
                    rows = [line.split() for line in label.read_text(encoding="utf-8").splitlines() if line.strip()]
                    annotation_count += len(rows)
                    label_by_stem[label.stem] = [classes[int(row[0])] if classes and row and int(row[0]) < len(classes) else row[0] for row in rows if row]
                except Exception:
                    continue
            for item in items:
                item["classes"] = sorted(set(label_by_stem.get(Path(item["relative_path"]).stem, [])))
        elif voc_files:
            fmt, task = "VOC", "detection"
            found = []
            for path in voc_files:
                try:
                    objects = ElementTree.parse(path).getroot().findall("object")
                    annotation_count += len(objects)
                    found.extend(str(obj.findtext("name")) for obj in objects if obj.findtext("name"))
                except ElementTree.ParseError:
                    continue
            classes = sorted(set(found))
        else:
            candidate_classes = sorted({path.parent.name for path in images if path.parent != root and path.parent.name.lower() not in {"images", "train", "val", "test", "valid"}})
            if candidate_classes:
                fmt, task, classes, annotation_count = "classification-folders", "classification", candidate_classes, len(items)
                for item in items:
                    parent = Path(item["relative_path"]).parent.name
                    item["classes"] = [parent] if parent in classes else []

        splits = Counter()
        for item in items:
            parts = {part.lower() for part in Path(item["relative_path"]).parts}
            split = next((name for name in ("train", "val", "valid", "test") if name in parts), "unspecified")
            splits["val" if split == "valid" else split] += 1
        missing = sum(1 for item in items if not item["classes"]) if task != "unknown" else 0
        state = InspectionState.READY if fmt != "unknown" and not broken else (InspectionState.PARTIAL if items else InspectionState.INVALID)
        if fmt == "unknown":
            state = InspectionState.UNSUPPORTED
        return DatasetInspection(state=state, format=fmt, task=task, image_count=len(images), valid_image_count=len(items), annotation_count=annotation_count, classes=classes, splits=dict(splits), missing_labels=missing, broken_files=broken[:100], duplicate_images=sum(count - 1 for count in hashes.values()), items=items, explanation=f"Detected {fmt} {task} dataset" if fmt != "unknown" else "Images are readable but no supported annotation layout was detected")


class ModelCompatibilityInspector:
    def inspect(self, root: Path) -> ModelInspection:
        files = sorted(path for path in ([root] if root.is_file() else root.rglob("*")) if path.is_file())
        python_files = [path.name for path in files if path.suffix.lower() == ".py"]
        candidates = [path for path in files if path.suffix.lower() in {".onnx", ".pt", ".pth", ".safetensors", ".bin"}]
        config = {}
        config_path = next((path for path in files if path.name == "config.json"), None)
        if config_path:
            try:
                config = json.loads(config_path.read_text(encoding="utf-8"))
            except Exception:
                config = {}
        architecture = ", ".join(config.get("architectures", [])) or config.get("model_type")
        if not candidates:
            return ModelInspection(state=InspectionState.UNSUPPORTED, architecture=architecture, explanation="No recognized model artifact was found", ignored_python_files=python_files)
        if len(candidates) > 1 and len({path.suffix.lower() for path in candidates}) > 1:
            return ModelInspection(state=InspectionState.AMBIGUOUS, architecture=architecture, explanation="Multiple incompatible artifact formats were found; select an artifact explicitly", ignored_python_files=python_files)
        artifact = candidates[0]
        suffix, name, hint = artifact.suffix.lower(), artifact.name.lower(), str(architecture or "").lower()
        if suffix == ".onnx":
            return ModelInspection(state=InspectionState.READY, framework="ONNX", architecture=architecture, candidate_adapter="onnx", artifact_file=artifact.relative_to(root).as_posix() if root.is_dir() else artifact.name, explanation="ONNX artifact can use the built-in ONNX adapter", ignored_python_files=python_files)
        if suffix == ".pt" and any(token in name + hint for token in ("yolo", "ultralytics")):
            return ModelInspection(state=InspectionState.READY, framework="PyTorch", architecture=architecture or "Ultralytics", candidate_adapter="yolo", artifact_file=artifact.relative_to(root).as_posix() if root.is_dir() else artifact.name, explanation="Artifact metadata identifies a supported Ultralytics family", ignored_python_files=python_files)
        if suffix == ".pth" and "rfdetr" in name + hint.replace("-", ""):
            return ModelInspection(state=InspectionState.READY, framework="PyTorch", architecture=architecture or "RF-DETR", candidate_adapter="rfdetr", artifact_file=artifact.relative_to(root).as_posix() if root.is_dir() else artifact.name, explanation="Artifact identity matches the built-in RF-DETR adapter", ignored_python_files=python_files)
        if suffix == ".safetensors":
            adapter = "sam" if "sam" in hint and "sam2" not in hint and "sam3" not in hint else None
            return ModelInspection(state=InspectionState.READY if adapter else InspectionState.ADAPTER_REQUIRED, framework="safetensors", architecture=architecture, candidate_adapter=adapter, artifact_file=artifact.relative_to(root).as_posix() if root.is_dir() else artifact.name, explanation="Safetensors is non-pickle data; a known architecture is required for execution", ignored_python_files=python_files)
        return ModelInspection(state=InspectionState.ADAPTER_REQUIRED, framework="PyTorch" if suffix in {".pt", ".pth", ".bin"} else None, architecture=architecture, artifact_file=artifact.relative_to(root).as_posix() if root.is_dir() else artifact.name, explanation="Opaque checkpoints are stored but never loaded until a safe explicit adapter is configured", ignored_python_files=python_files)
