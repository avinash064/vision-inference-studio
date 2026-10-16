from __future__ import annotations

import hashlib
import json
import random
import uuid
from collections import Counter, defaultdict
from pathlib import Path

from inference_studio.durable import JsonDocumentStore

from .assets import ImportedAssetRegistry
from .schemas import BenchmarkCreate, BenchmarkDataset


def stable_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


class BenchmarkRegistry:
    def __init__(self, root: Path, assets: ImportedAssetRegistry) -> None:
        self.store, self.assets = JsonDocumentStore(root, "bmk_"), assets

    def create(self, request: BenchmarkCreate) -> BenchmarkDataset:
        dataset = self.assets.get_dataset(request.dataset_id)
        items = list(dataset.inspection.items)
        if request.sampling_strategy == "manual":
            selected_ids = set(request.selected_item_ids)
            selected = [item for item in items if item["item_id"] in selected_ids]
            if len(selected) != len(selected_ids):
                raise ValueError("Manual selection contains unknown dataset items")
        elif request.sampling_strategy == "stratified" and any(item.get("classes") for item in items):
            groups: dict[str, list[dict]] = defaultdict(list)
            for item in items:
                groups[(item.get("classes") or ["unlabelled"])[0]].append(item)
            rng = random.Random(request.seed)
            for values in groups.values():
                rng.shuffle(values)
            selected = []
            while len(selected) < min(request.requested_size, len(items)) and groups:
                for key in sorted(list(groups)):
                    if groups[key] and len(selected) < request.requested_size:
                        selected.append(groups[key].pop())
                    if not groups[key]:
                        groups.pop(key)
        else:
            selected = list(items)
            random.Random(request.seed).shuffle(selected)
            selected = selected[:request.requested_size]
        if not selected:
            raise ValueError("Benchmark selection is empty")
        class_distribution = Counter(label for item in selected for label in item.get("classes", []))
        payload = {"dataset_hash": dataset.checksum, "items": [{"item_id": item["item_id"], "sha256": item["sha256"]} for item in selected], "seed": request.seed, "strategy": request.sampling_strategy}
        benchmark = BenchmarkDataset(benchmark_id=f"bmk_{uuid.uuid4().hex}", name=request.name, dataset_id=dataset.dataset_id, dataset_hash=dataset.checksum, selected_items=selected, seed=request.seed, sampling_strategy=request.sampling_strategy, requested_size=request.requested_size, actual_size=len(selected), class_distribution=dict(class_distribution), benchmark_hash=stable_hash(payload))
        self.store.put(benchmark.benchmark_id, benchmark.model_dump(mode="json"), create_only=True)
        return benchmark

    def get(self, identity: str) -> BenchmarkDataset:
        return BenchmarkDataset.model_validate(self.store.get(identity))

    def list(self) -> list[dict]:
        return self.store.list()
