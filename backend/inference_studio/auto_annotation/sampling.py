from __future__ import annotations

import math

from .schemas import FrameSamplingPolicy, SamplingMode


class FrameSampler:
    """Deterministic sequential selector with timestamp-aware interval modes."""

    def __init__(self, policy: FrameSamplingPolicy, nominal_fps: float) -> None:
        if nominal_fps <= 0 or not math.isfinite(nominal_fps):
            raise ValueError("nominal_fps must be positive and finite")
        self.policy = policy
        self.nominal_fps = nominal_fps
        self.start = policy.start_seconds or 0.0
        self.last_bucket = -1
        self.start_frame = max(0, math.ceil(self.start * nominal_fps - 1e-9))

    def select(self, frame_index: int, timestamp_seconds: float) -> bool:
        if frame_index < 0 or not math.isfinite(timestamp_seconds):
            return False
        if timestamp_seconds + 1e-9 < self.start:
            return False
        if (
            self.policy.end_seconds is not None
            and timestamp_seconds > self.policy.end_seconds + 1e-9
        ):
            return False
        if self.policy.mode == SamplingMode.EVERY_N_FRAMES:
            return (
                frame_index >= self.start_frame
                and (frame_index - self.start_frame)
                % int(self.policy.every_n_frames or 1)
                == 0
            )
        interval = (
            1.0 / float(self.policy.target_fps or 1)
            if self.policy.mode == SamplingMode.TARGET_FPS
            else float(self.policy.interval_seconds or 1)
        )
        bucket = max(0, math.floor((timestamp_seconds - self.start + 1e-9) / interval))
        if bucket <= self.last_bucket:
            return False
        self.last_bucket = bucket
        return True

    @staticmethod
    def estimated_count(
        policy: FrameSamplingPolicy,
        nominal_fps: float,
        frame_count: int,
        duration_seconds: float,
    ) -> int:
        start = policy.start_seconds or 0.0
        end = min(
            policy.end_seconds if policy.end_seconds is not None else duration_seconds,
            duration_seconds,
        )
        if end < start:
            return 0
        if policy.mode == SamplingMode.EVERY_N_FRAMES:
            first = math.ceil(start * nominal_fps - 1e-9)
            last = min(frame_count - 1, math.floor(end * nominal_fps + 1e-9))
            return max(0, ((last - first) // int(policy.every_n_frames or 1)) + 1)
        rate = (
            float(policy.target_fps or 1)
            if policy.mode == SamplingMode.TARGET_FPS
            else 1.0 / float(policy.interval_seconds or 1)
        )
        return min(frame_count, max(0, math.floor((end - start) * rate + 1e-9) + 1))
