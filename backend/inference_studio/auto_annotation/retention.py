from __future__ import annotations

from dataclasses import dataclass

from .schemas import CanonicalPrediction, FrameRetentionPolicy, RetentionMode


@dataclass(frozen=True)
class RetentionDecision:
    retained: bool
    reason: str
    qualifying_predictions: list[CanonicalPrediction]
    prediction_rule_passed: bool
    temporal_spacing_passed: bool


class RetentionEvaluator:
    def __init__(self, policy: FrameRetentionPolicy) -> None:
        self.policy = policy
        self.last_retained_timestamp: float | None = None

    @staticmethod
    def _area(prediction: CanonicalPrediction) -> float:
        geometry = prediction.geometry
        if geometry.bbox:
            x1, y1, x2, y2 = geometry.bbox
            return max(0.0, x2 - x1) * max(0.0, y2 - y1)
        return 0.0

    def evaluate(
        self,
        predictions: list[CanonicalPrediction],
        timestamp_seconds: float | None = None,
        *,
        commit: bool = True,
    ) -> RetentionDecision:
        values = list(predictions)
        if self.policy.min_confidence is not None:
            values = [
                value
                for value in values
                if value.confidence >= self.policy.min_confidence
            ]
        if self.policy.min_area_px is not None:
            values = [
                value
                for value in values
                if self._area(value) >= self.policy.min_area_px
            ]

        class_values = values
        if self.policy.mode in {RetentionMode.CLASS_PRESENT, RetentionMode.CLASS_COUNT}:
            expected = (self.policy.class_name or "").casefold()
            class_values = [
                value for value in values if value.category_name.casefold() == expected
            ]

        if self.policy.mode == RetentionMode.ALL_SAMPLED:
            passed, detail = True, "All successfully inferred samples are retained"
        elif self.policy.mode == RetentionMode.ANY_PREDICTION:
            passed, detail = (
                bool(values),
                f"{len(values)} qualifying prediction(s); at least one required",
            )
        elif self.policy.mode == RetentionMode.TOTAL_OBJECT_COUNT:
            required = int(self.policy.min_total_objects or 1)
            passed, detail = (
                len(values) >= required,
                f"{len(values)} qualifying prediction(s); {required} required",
            )
        elif self.policy.mode == RetentionMode.CLASS_PRESENT:
            passed, detail = (
                bool(class_values),
                f"{len(class_values)} {self.policy.class_name} prediction(s); at least one required",
            )
        else:
            required = int(self.policy.min_class_count or 1)
            passed, detail = (
                len(class_values) >= required,
                f"{len(class_values)} {self.policy.class_name} prediction(s); {required} required",
            )

        spacing_passed = True
        if (
            passed
            and self.policy.min_save_interval_seconds
            and timestamp_seconds is not None
            and self.last_retained_timestamp is not None
        ):
            elapsed = timestamp_seconds - self.last_retained_timestamp
            spacing_passed = elapsed + 1e-9 >= self.policy.min_save_interval_seconds
            if not spacing_passed:
                detail += f"; temporal spacing failed ({elapsed:.3f}s < {self.policy.min_save_interval_seconds:.3f}s)"
        retained = passed and spacing_passed
        if retained and commit and timestamp_seconds is not None:
            self.last_retained_timestamp = timestamp_seconds
        return RetentionDecision(
            retained,
            detail,
            class_values
            if self.policy.mode
            in {RetentionMode.CLASS_PRESENT, RetentionMode.CLASS_COUNT}
            else values,
            passed,
            spacing_passed,
        )
