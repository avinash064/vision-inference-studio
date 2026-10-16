from __future__ import annotations

import asyncio
import copy
import statistics
import uuid
from pathlib import Path
from time import perf_counter
from typing import Any

import psutil
import cv2
import numpy as np

from inference_studio.assets import AssetStore
from inference_studio.durable import JsonDocumentStore
from inference_studio.graph import Workflow
from inference_studio.graph.executor import ExecutionEngine

from .assets import ImportedAssetRegistry
from .benchmarks import BenchmarkRegistry, stable_hash
from .schemas import ExperimentCreate, ExperimentState, InferenceExperiment, RecipeVariant, utc_now


class ExperimentRegistry:
    def __init__(self, root: Path, benchmarks: BenchmarkRegistry, imported: ImportedAssetRegistry,
                 runtime_assets: AssetStore, engine: ExecutionEngine) -> None:
        self.store = JsonDocumentStore(root / "experiments", "exp_")
        self.results = JsonDocumentStore(root / "results", "res_")
        self.units = JsonDocumentStore(root / "unit-cache", "unit_")
        self.benchmarks, self.imported = benchmarks, imported
        self.runtime_assets, self.engine = runtime_assets, engine
        self._tasks: dict[str, asyncio.Task] = {}

    @staticmethod
    def public(experiment: InferenceExperiment) -> dict:
        return experiment.model_dump(mode="json")

    def create(self, request: ExperimentCreate) -> InferenceExperiment:
        self.benchmarks.get(request.benchmark_id)
        if request.model_artifact_id:
            self.imported.get_model(request.model_artifact_id)
        variants = []
        model_identities = []
        for variant in request.variants:
            Workflow.model_validate(variant.workflow)
            model_nodes = [node for node in variant.workflow.get("nodes", []) if node.get("type") == "model"]
            if len(model_nodes) != 1:
                raise ValueError("Each experiment variant must contain exactly one model node")
            config = model_nodes[0].get("config", {})
            model_identities.append((config.get("family"), config.get("version"), config.get("size"), config.get("weight_id")))
            variants.append(variant)
        if len(set(model_identities)) != 1:
            raise ValueError("v0.6 preprocessing comparisons require the same model and weight in every variant")
        experiment = InferenceExperiment(experiment_id=f"exp_{uuid.uuid4().hex}", name=request.name, benchmark_id=request.benchmark_id, model_artifact_id=request.model_artifact_id, variants=variants)
        self._save(experiment, create_only=True)
        return experiment

    def _save(self, experiment: InferenceExperiment, create_only: bool = False) -> None:
        self.store.put(experiment.experiment_id, experiment.model_dump(mode="json"), create_only=create_only)

    def get(self, identity: str) -> InferenceExperiment:
        return InferenceExperiment.model_validate(self.store.get(identity))

    def list(self) -> list[dict]:
        return [self.public(InferenceExperiment.model_validate(raw)) for raw in sorted(self.store.list(), key=lambda x: x.get("created_at", ""), reverse=True)]

    def start(self, identity: str, force: bool = False) -> InferenceExperiment:
        experiment = self.get(identity)
        if experiment.status in {ExperimentState.QUEUED, ExperimentState.RUNNING}:
            return experiment
        experiment.status, experiment.cancel_requested = ExperimentState.QUEUED, False
        experiment.result_summary = {"force_rerun": force}
        experiment.failures = []
        self._save(experiment)
        self._schedule(identity, force)
        return experiment

    def cancel(self, identity: str) -> InferenceExperiment:
        experiment = self.get(identity)
        experiment.cancel_requested = True
        self._save(experiment)
        return experiment

    async def recover(self) -> None:
        for raw in self.store.list():
            experiment = InferenceExperiment.model_validate(raw)
            if experiment.status in {ExperimentState.QUEUED, ExperimentState.RUNNING}:
                experiment.status = ExperimentState.QUEUED
                self._save(experiment)
                self._schedule(experiment.experiment_id, False)

    def _schedule(self, identity: str, force: bool) -> None:
        if identity not in self._tasks or self._tasks[identity].done():
            self._tasks[identity] = asyncio.create_task(self._run(identity, force), name=f"experiment-{identity}")

    def experiment_results(self, identity: str) -> dict:
        experiment = self.get(identity)
        values = [raw for raw in self.results.list() if raw.get("experiment_id") == identity]
        return {"experiment": self.public(experiment), "summary": experiment.result_summary, "units": values}

    def item_results(self, identity: str, item_id: str) -> dict:
        self.get(identity)
        values = [raw for raw in self.results.list() if raw.get("experiment_id") == identity and raw.get("item_id") == item_id]
        if not values:
            raise KeyError(item_id)
        return {"experiment_id": identity, "item_id": item_id, "variants": values}

    async def _run(self, identity: str, force: bool) -> None:
        experiment = self.get(identity)
        experiment.status, experiment.started_at = ExperimentState.RUNNING, utc_now()
        self._save(experiment)
        benchmark = self.benchmarks.get(experiment.benchmark_id)
        dataset = self.imported.get_dataset(benchmark.dataset_id)
        failures, all_results = [], []
        try:
            for item in benchmark.selected_items:
                source = (Path(dataset.local_root) / item["relative_path"]).resolve()
                if Path(dataset.local_root).resolve() not in source.parents:
                    raise ValueError("Benchmark item escapes its dataset root")
                for variant in experiment.variants:
                    if self.get(identity).cancel_requested:
                        experiment = self.get(identity)
                        experiment.status, experiment.completed_at = ExperimentState.CANCELLED, utc_now()
                        self._save(experiment)
                        return
                    unit_key, hashes = self._unit_identity(benchmark.benchmark_hash, item, experiment.model_artifact_id, variant)
                    cached = None
                    if not force:
                        try:
                            cached = self.units.get(unit_key)
                        except KeyError:
                            pass
                    if cached and cached.get("status") == "COMPLETED":
                        result = {**cached, "result_id": f"res_{uuid.uuid4().hex}", "experiment_id": identity, "reused": True}
                    else:
                        try:
                            result = await self._execute_unit(identity, item, source, variant, unit_key, hashes)
                            self.units.put(unit_key, {key: value for key, value in result.items() if key not in {"result_id", "experiment_id"}})
                        except Exception as exc:
                            result = {"result_id": f"res_{uuid.uuid4().hex}", "experiment_id": identity, "item_id": item["item_id"], "variant_id": variant.variant_id, "unit_identity": unit_key, "status": "ERROR", "error": str(exc), **hashes}
                            failures.append({"item_id": item["item_id"], "variant_id": variant.variant_id, "error": str(exc)})
                    self.results.put(result["result_id"], result)
                    all_results.append(result)
            experiment = self.get(identity)
            experiment.status = ExperimentState.COMPLETED if any(r["status"] == "COMPLETED" for r in all_results) else ExperimentState.ERROR
            experiment.completed_at, experiment.failures = utc_now(), failures
            experiment.result_summary = self._summarize(experiment.variants, all_results, benchmark)
            self._save(experiment)
        except Exception as exc:
            experiment = self.get(identity)
            experiment.status, experiment.completed_at = ExperimentState.ERROR, utc_now()
            experiment.failures.append({"error": str(exc)})
            self._save(experiment)

    @staticmethod
    def _unit_identity(benchmark_hash: str, item: dict, model_artifact_id: str | None, variant: RecipeVariant) -> tuple[str, dict]:
        nodes = variant.workflow.get("nodes", [])
        model_index = next((i for i, node in enumerate(nodes) if node.get("type") == "model"), len(nodes))
        preprocessing_hash = stable_hash(nodes[:model_index])
        inference_hash = stable_hash({"model": nodes[model_index] if model_index < len(nodes) else None, "configuration": variant.inference_configuration})
        postprocessing_hash = stable_hash(nodes[model_index + 1:])
        model_sha = stable_hash(model_artifact_id or (nodes[model_index].get("config", {}).get("weight_id") if model_index < len(nodes) else "none"))
        hashes = {"benchmark_hash": benchmark_hash, "image_sha256": item["sha256"], "model_sha": model_sha, "preprocessing_hash": preprocessing_hash, "inference_configuration_hash": inference_hash, "postprocessing_hash": postprocessing_hash}
        return f"unit_{stable_hash(hashes)[:40]}", hashes

    async def _execute_unit(self, experiment_id: str, item: dict, source: Path, variant: RecipeVariant, unit_key: str, hashes: dict) -> dict:
        record = await asyncio.to_thread(self.runtime_assets.import_path, source, "image")
        raw = copy.deepcopy(variant.workflow)
        image_sources = [node for node in raw.get("nodes", []) if node.get("type") == "image_source"]
        if not image_sources:
            raise ValueError("Experiment workflow must have an image_source node")
        for node in image_sources:
            node.setdefault("config", {})["asset_id"] = record.asset_id
        workflow = Workflow.model_validate(raw)
        process = psutil.Process()
        memory_before = process.memory_info().rss
        started = perf_counter()
        execution = self.engine.start(workflow)
        events = []
        async for event in self.engine.subscribe(execution.run_id):
            events.append(event)
        total_ms = (perf_counter() - started) * 1000
        terminal = next((event for event in reversed(events) if event.get("type") == "run_state"), {})
        if terminal.get("state") != "COMPLETED":
            raise RuntimeError(terminal.get("error", f"Vision runtime ended in {terminal.get('state', 'unknown')}"))
        inference = next((event for event in reversed(events) if event.get("type") == "inference_result"), {})
        preview = next((event for event in reversed(events) if event.get("type") == "preview"), {})
        node_times = [event for event in events if event.get("type") == "node_state" and event.get("event") == "node_completed"]
        result = inference.get("result", preview.get("result", {})) or {}
        metrics = result.get("metrics", {}) or {}
        objects = result.get("detections", result.get("masks", result.get("predictions", []))) or []
        operational = {
            "preprocessing_ms": metrics.get("preprocess_ms"), "inference_ms": metrics.get("inference_ms"),
            "postprocessing_ms": metrics.get("postprocess_ms"), "total_ms": total_ms,
            "fps": 1000 / total_ms if total_ms else None,
            "peak_vram_mb": metrics.get("gpu_peak_memory_mb"),
            "cpu_memory_delta_mb": (process.memory_info().rss - memory_before) / 1024**2,
            "prediction_count": len(objects),
            "node_timings": {event["node_id"]: event.get("duration_ms") for event in node_times},
        }
        evaluation = self._evaluation(item, result)
        return {"result_id": f"res_{uuid.uuid4().hex}", "experiment_id": experiment_id, "item_id": item["item_id"], "variant_id": variant.variant_id, "unit_identity": unit_key, "status": "COMPLETED", "reused": False, "run_id": execution.run_id, "prediction": result, "preview_url": preview.get("prediction_url", preview.get("url")), "original_url": preview.get("original_url"), "operational_metrics": operational, "evaluation_metrics": evaluation, **hashes}

    @staticmethod
    def _evaluation(item: dict, result: dict) -> dict:
        ground_truth = item.get("classes", [])
        if not ground_truth:
            return {}
        annotations = item.get("ground_truth", [])
        if result.get("result_type") == "detections" and annotations:
            predictions = result.get("detections", [])
            thresholds = {}
            for threshold in [value / 100 for value in range(50, 100, 5)]:
                unmatched = set(range(len(annotations))); scored = []
                for prediction in sorted(predictions, key=lambda value: float(value.get("confidence", 0)), reverse=True):
                    best_index, best_iou = None, 0.0
                    for index in unmatched:
                        truth = annotations[index]
                        if prediction.get("label") != truth.get("label") or len(truth.get("bbox", [])) != 4:
                            continue
                        iou = ExperimentRegistry._bbox_iou(prediction.get("bbox", []), truth["bbox"])
                        if iou > best_iou:
                            best_index, best_iou = index, iou
                    matched = best_index is not None and best_iou >= threshold
                    if matched:
                        unmatched.remove(best_index)
                    scored.append({"confidence": float(prediction.get("confidence", 0)), "tp": 1 if matched else 0})
                tp = sum(row["tp"] for row in scored)
                thresholds[f"{threshold:.2f}"] = {"tp": tp, "fp": len(scored) - tp, "fn": len(unmatched), "gt": len(annotations), "scored": scored}
            return {"task": "detection", "thresholds": thresholds}
        if result.get("result_type") == "segmentation" and any(value.get("polygons") for value in annotations):
            height, width = int(item.get("height", 0)), int(item.get("width", 0))
            pairs, remaining = [], set(range(len(annotations)))
            for prediction in sorted(result.get("masks", []), key=lambda value: float(value.get("confidence", 0)), reverse=True):
                best, best_iou, best_dice = None, 0.0, 0.0
                predicted_mask = ExperimentRegistry._polygon_mask(prediction.get("polygons", []), width, height)
                for index in remaining:
                    truth = annotations[index]
                    if prediction.get("label") != truth.get("label"):
                        continue
                    truth_mask = ExperimentRegistry._polygon_mask(truth.get("polygons", []), width, height)
                    intersection = int(np.logical_and(predicted_mask, truth_mask).sum()); union = int(np.logical_or(predicted_mask, truth_mask).sum())
                    iou = intersection / union if union else 0.0
                    dice = 2 * intersection / (int(predicted_mask.sum()) + int(truth_mask.sum())) if predicted_mask.any() or truth_mask.any() else 0.0
                    if iou > best_iou: best, best_iou, best_dice = index, iou, dice
                if best is not None:
                    remaining.remove(best); pairs.append({"iou": best_iou, "dice": best_dice, "confidence": float(prediction.get("confidence", 0))})
            return {"task": "segmentation", "pairs": pairs, "gt": len(annotations), "predictions": len(result.get("masks", []))}
        predictions = result.get("predictions", [])
        if result.get("result_type") == "classification" and predictions:
            predicted = predictions[0].get("class_name", predictions[0].get("label"))
            return {"task": "classification", "truth": ground_truth[0], "predicted": predicted, "correct": 1 if predicted in ground_truth else 0}
        return {}

    @staticmethod
    def _bbox_iou(first: list, second: list) -> float:
        if len(first) != 4 or len(second) != 4: return 0.0
        left, top = max(float(first[0]), float(second[0])), max(float(first[1]), float(second[1]))
        right, bottom = min(float(first[2]), float(second[2])), min(float(first[3]), float(second[3]))
        intersection = max(0.0, right - left) * max(0.0, bottom - top)
        first_area = max(0.0, float(first[2]) - float(first[0])) * max(0.0, float(first[3]) - float(first[1]))
        second_area = max(0.0, float(second[2]) - float(second[0])) * max(0.0, float(second[3]) - float(second[1]))
        union = first_area + second_area - intersection
        return intersection / union if union else 0.0

    @staticmethod
    def _polygon_mask(polygons: list, width: int, height: int) -> np.ndarray:
        mask = np.zeros((max(1, height), max(1, width)), dtype=np.uint8)
        values = [polygons] if polygons and isinstance(polygons[0], (int, float)) else polygons
        for polygon in values or []:
            if len(polygon) >= 6:
                points = np.asarray(polygon, dtype=np.float32).reshape(-1, 2).round().astype(np.int32)
                cv2.fillPoly(mask, [points], 1)
        return mask.astype(bool)

    @staticmethod
    def _summarize(variants: list[RecipeVariant], results: list[dict], benchmark) -> dict:
        summaries = {}
        for variant in variants:
            rows = [row for row in results if row.get("variant_id") == variant.variant_id and row.get("status") == "COMPLETED"]
            totals = [row["operational_metrics"]["total_ms"] for row in rows]
            eval_rows = [row["evaluation_metrics"] for row in rows if row.get("evaluation_metrics")]
            evaluation = ExperimentRegistry._summarize_evaluation(eval_rows)
            summaries[variant.variant_id] = {
                "completed": len(rows), "failed": len([row for row in results if row.get("variant_id") == variant.variant_id and row.get("status") == "ERROR"]),
                "reused": len([row for row in rows if row.get("reused")]),
                "mean_total_ms": statistics.fmean(totals) if totals else None,
                "fps": 1000 / statistics.fmean(totals) if totals and statistics.fmean(totals) else None,
                "mean_predictions": statistics.fmean([row["operational_metrics"]["prediction_count"] for row in rows]) if rows else None,
                "evaluation": evaluation,
            }
        return {"benchmark_id": benchmark.benchmark_id, "benchmark_hash": benchmark.benchmark_hash, "variants": summaries}

    @staticmethod
    def _summarize_evaluation(rows: list[dict]) -> dict:
        if not rows: return {}
        if rows[0].get("task") == "detection":
            aps = {}
            precision50 = recall50 = 0.0
            for threshold in [f"{value / 100:.2f}" for value in range(50, 100, 5)]:
                values = [row["thresholds"][threshold] for row in rows]
                scored = sorted([score for value in values for score in value["scored"]], key=lambda item: item["confidence"], reverse=True)
                total_gt = sum(value["gt"] for value in values)
                cumulative_tp = cumulative_fp = 0; curve = []
                for score in scored:
                    cumulative_tp += score["tp"]; cumulative_fp += 1 - score["tp"]
                    curve.append((cumulative_tp / total_gt if total_gt else 0.0, cumulative_tp / max(cumulative_tp + cumulative_fp, 1)))
                ap = statistics.fmean([max((precision for recall, precision in curve if recall >= target), default=0.0) for target in np.linspace(0, 1, 101)])
                aps[threshold] = ap
                if threshold == "0.50":
                    tp, fp, fn = sum(v["tp"] for v in values), sum(v["fp"] for v in values), sum(v["fn"] for v in values)
                    precision50, recall50 = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
            return {"precision": precision50, "recall": recall50, "mAP50": aps["0.50"], "mAP50-95": statistics.fmean(aps.values())}
        if rows[0].get("task") == "segmentation":
            pairs = [pair for row in rows for pair in row.get("pairs", [])]
            return {"mask_IoU": statistics.fmean(pair["iou"] for pair in pairs) if pairs else 0.0, "Dice": statistics.fmean(pair["dice"] for pair in pairs) if pairs else 0.0}
        if rows[0].get("task") == "classification":
            classes = sorted({str(row.get("truth")) for row in rows} | {str(row.get("predicted")) for row in rows})
            precisions, recalls, f1s = [], [], []
            for label in classes:
                tp = sum(row.get("truth") == label and row.get("predicted") == label for row in rows)
                fp = sum(row.get("truth") != label and row.get("predicted") == label for row in rows)
                fn = sum(row.get("truth") == label and row.get("predicted") != label for row in rows)
                precision, recall = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
                precisions.append(precision); recalls.append(recall); f1s.append(2 * precision * recall / max(precision + recall, 1e-12))
            return {"accuracy": sum(row.get("correct", 0) for row in rows) / len(rows), "precision": statistics.fmean(precisions), "recall": statistics.fmean(recalls), "F1": statistics.fmean(f1s)}
        numeric = {key for row in rows for key, value in row.items() if isinstance(value, (int, float))}
        return {key: statistics.fmean(row[key] for row in rows if key in row) for key in numeric}

    async def execute_workflow_for_asset(self, workflow: dict[str, Any], asset_id: str) -> dict:
        raw = copy.deepcopy(workflow)
        for node in raw.get("nodes", []):
            if node.get("type") == "image_source":
                node.setdefault("config", {})["asset_id"] = asset_id
        execution = self.engine.start(Workflow.model_validate(raw))
        events = []
        async for event in self.engine.subscribe(execution.run_id):
            events.append(event)
        terminal = next((event for event in reversed(events) if event.get("type") == "run_state"), {})
        if terminal.get("state") != "COMPLETED":
            raise RuntimeError(terminal.get("error", "Recipe execution failed"))
        event = next((event for event in reversed(events) if event.get("type") in {"preview", "inference_result"}), {})
        return {"run_id": execution.run_id, "result": event.get("result", {}), "preview_url": event.get("prediction_url", event.get("url"))}
