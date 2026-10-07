from __future__ import annotations

import hashlib
import json
import uuid
from pathlib import Path
from typing import BinaryIO

from .manifest import WeightRecord
from .registry import ModelRegistry


class WeightError(ValueError):
    pass


class WeightStore:
    def __init__(self, root: Path, registry: ModelRegistry, max_bytes: int) -> None:
        self.root = root.resolve()
        self.registry = registry
        self.max_bytes = max_bytes
        self.root.mkdir(parents=True, exist_ok=True)

    def save(self, source: BinaryIO, filename: str, family: str) -> WeightRecord:
        definition = self.registry.get(family)
        extension = Path(filename).suffix.lower()
        if extension not in definition.extensions:
            raise WeightError(f"{extension or '(no extension)'} is not valid for {family}")
        family_dir = (self.root / family).resolve()
        family_dir.mkdir(parents=True, exist_ok=True)
        if family_dir.parent != self.root:
            raise WeightError("Invalid family path")
        weight_id = f"wgt_{uuid.uuid4().hex}"
        target = family_dir / f"{weight_id}{extension}"
        digest, total = hashlib.sha256(), 0
        try:
            with target.open("xb") as output:
                while chunk := source.read(1024 * 1024):
                    total += len(chunk)
                    if total > self.max_bytes:
                        raise WeightError("Weight exceeds configured size limit")
                    digest.update(chunk)
                    output.write(chunk)
            if total == 0:
                raise WeightError("Weight file is empty")
            record = WeightRecord(weight_id=weight_id, family=family, filename=Path(filename).name, sha256=digest.hexdigest(), size_bytes=total, path=str(target))
            (family_dir / f"{weight_id}.json").write_text(record.model_dump_json(indent=2), encoding="utf-8")
            return record
        except Exception:
            target.unlink(missing_ok=True)
            raise

    def get(self, weight_id: str, family: str | None = None) -> WeightRecord:
        if not weight_id.startswith("wgt_") or not weight_id[4:].isalnum():
            raise WeightError("Invalid weight id")
        families = [family] if family else list(self.registry.manifest.families)
        for candidate in families:
            if candidate not in self.registry.manifest.families:
                continue
            metadata = self.root / candidate / f"{weight_id}.json"
            if metadata.is_file():
                data = json.loads(metadata.read_text(encoding="utf-8"))
                data["path"] = str(self.root / candidate / f"{weight_id}{Path(data['filename']).suffix.lower()}")
                record = WeightRecord.model_validate(data)
                path = Path(record.path).resolve()
                if path.parent != (self.root / candidate).resolve() or not path.is_file():
                    raise WeightError("Weight data is missing")
                if family and record.family != family:
                    raise WeightError("Weight is incompatible with the selected family")
                return record
        raise WeightError(f"Unknown weight: {weight_id}")

    def list(self) -> list[WeightRecord]:
        records: list[WeightRecord] = []
        for family in self.registry.manifest.families:
            for metadata in sorted((self.root / family).glob("wgt_*.json")):
                try:
                    records.append(self.get(metadata.stem, family))
                except WeightError:
                    continue
        return records
