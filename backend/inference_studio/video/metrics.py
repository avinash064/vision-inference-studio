from __future__ import annotations

import os
from pathlib import Path
from statistics import mean, median
from time import perf_counter
from typing import Any


def _percentile(values: list[float], percentage: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentage / 100
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {key: 0.0 for key in ("count", "mean", "median", "p50", "p90", "p95", "p99", "min", "max")}
    return {
        "count": len(values), "mean": mean(values), "median": median(values),
        "p50": _percentile(values, 50), "p90": _percentile(values, 90),
        "p95": _percentile(values, 95), "p99": _percentile(values, 99),
        "min": min(values), "max": max(values),
    }


def current_rss_mb() -> float:
    try:
        for line in Path(f"/proc/{os.getpid()}/status").read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                return float(line.split()[1]) / 1024
    except OSError:
        pass
    return 0.0


class VideoMetricsCollector:
    PHASES = ("decode_ms", "preprocess_ms", "inference_ms", "postprocess_ms", "render_ms", "publish_ms", "total_frame_ms")

    def __init__(self, source: dict[str, Any], model: dict[str, Any], warmup_frames: int) -> None:
        self.source, self.model, self.warmup_frames = source, model, warmup_frames
        self.started = perf_counter()
        self.values = {phase: [] for phase in self.PHASES}
        self.frames = {"decoded": 0, "inferred": 0, "skipped": 0, "rendered": 0, "warmup": 0}
        self.rss_start_mb = current_rss_mb()
        self.rss_peak_mb = self.rss_start_mb
        self.gpu: dict[str, Any] = {}

    def decoded(self) -> None:
        self.frames["decoded"] += 1

    def skipped(self) -> None:
        self.frames["skipped"] += 1

    def add(self, metrics: dict[str, float], *, warmup: bool, rendered: bool, gpu: dict[str, Any]) -> None:
        self.frames["inferred"] += 1
        if rendered:
            self.frames["rendered"] += 1
        if warmup:
            self.frames["warmup"] += 1
        else:
            for phase in self.PHASES:
                self.values[phase].append(float(metrics.get(phase, 0)))
        self.gpu = {
            "device": gpu.get("device") or self.gpu.get("device"),
            "allocated_mb": max(float(gpu.get("allocated_mb") or 0), float(self.gpu.get("allocated_mb") or 0)),
            "reserved_mb": max(float(gpu.get("reserved_mb") or 0), float(self.gpu.get("reserved_mb") or 0)),
            "peak_allocated_mb": max(float(gpu.get("peak_allocated_mb") or 0), float(self.gpu.get("peak_allocated_mb") or 0)),
        }
        self.rss_peak_mb = max(self.rss_peak_mb, current_rss_mb())

    def report(self, state: str) -> dict[str, Any]:
        elapsed = max(perf_counter() - self.started, 1e-9)
        measured = len(self.values["total_frame_ms"])
        inference_mean = mean(self.values["inference_ms"]) if measured else 0
        total_mean = mean(self.values["total_frame_ms"]) if measured else 0
        rss_end = current_rss_mb()
        return {
            "version": "0.2", "state": state, "model": self.model, "source": self.source,
            "frames": dict(self.frames), "latency_ms": {phase: _summary(values) for phase, values in self.values.items()},
            "fps": {
                "source": float(self.source.get("fps") or 0),
                "model": 1000 / inference_mean if inference_mean else 0,
                "end_to_end": 1000 / total_mean if total_mean else 0,
                "wall_clock": self.frames["inferred"] / elapsed,
            },
            "gpu": self.gpu,
            "memory": {"rss_start_mb": self.rss_start_mb, "rss_end_mb": rss_end, "rss_peak_mb": max(self.rss_peak_mb, rss_end), "rss_growth_mb": rss_end - self.rss_start_mb},
            "elapsed_seconds": elapsed,
        }
