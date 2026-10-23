from __future__ import annotations

from pathlib import Path

from inference_studio.durable import JsonDocumentStore
from inference_studio.public_hub.schemas import utc_now

from .schemas import AutoAnnotationItem, AutoAnnotationRun


class AutoAnnotationRunRepository:
    """Atomic local persistence for runs and per-item checkpoints."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.runs = JsonDocumentStore(self.root / "runs", "aa_")
        self.items = JsonDocumentStore(self.root / "items", "aai_")

    def create(self, run: AutoAnnotationRun) -> AutoAnnotationRun:
        self.runs.put(run.run_id, run.model_dump(mode="json"), create_only=True)
        return run

    def save(self, run: AutoAnnotationRun) -> AutoAnnotationRun:
        run.updated_at = utc_now()
        self.runs.put(run.run_id, run.model_dump(mode="json"))
        return run

    def get(self, run_id: str) -> AutoAnnotationRun:
        return AutoAnnotationRun.model_validate(self.runs.get(run_id))

    def list(self) -> list[AutoAnnotationRun]:
        values = [AutoAnnotationRun.model_validate(value) for value in self.runs.list()]
        return sorted(values, key=lambda value: value.created_at, reverse=True)

    def record_item(self, item: AutoAnnotationItem) -> AutoAnnotationItem:
        item.updated_at = utc_now()
        self.items.put(item.item_id, item.model_dump(mode="json"))
        return item

    def get_item(self, item_id: str) -> AutoAnnotationItem:
        return AutoAnnotationItem.model_validate(self.items.get(item_id))

    def items_for_run(self, run_id: str) -> list[AutoAnnotationItem]:
        values = [
            AutoAnnotationItem.model_validate(value)
            for value in self.items.list()
            if value.get("run_id") == run_id
        ]
        return sorted(
            values,
            key=lambda value: (
                value.frame_index if value.frame_index is not None else -1,
                value.relative_source_path or value.item_id,
            ),
        )

    def failures(self, run_id: str) -> list[dict]:
        return [
            {
                "item_id": item.item_id,
                "source": item.relative_source_path
                or item.source_video_name
                or item.source_path_or_ref,
                "frame_index": item.frame_index,
                "timestamp_seconds": item.timestamp_seconds,
                "stage": item.error_stage or "UNKNOWN",
                "error_type": "ItemError",
                "message": item.error or "Unknown item failure",
            }
            for item in self.items_for_run(run_id)
            if item.error
        ]
