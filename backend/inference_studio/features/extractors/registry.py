from __future__ import annotations

from .base import FeatureExtractor


class FeatureExtractorRegistry:
    def __init__(self, extractors: list[FeatureExtractor] | None = None) -> None:
        self._extractors: dict[str, FeatureExtractor] = {}
        if extractors is None:
            from .dinov3 import DinoV3FeatureExtractor
            from .rfdetr_arch_b import RFDetrArchBFeatureExtractor
            from .rfdetr_v22 import RFDetrV22FeatureExtractor
            extractors = [DinoV3FeatureExtractor(), RFDetrV22FeatureExtractor(), RFDetrArchBFeatureExtractor()]
        for extractor in extractors:
            self.register(extractor)

    def register(self, extractor: FeatureExtractor) -> None:
        self._extractors[extractor.name] = extractor

    def get(self, name: str) -> FeatureExtractor:
        try:
            return self._extractors[name]
        except KeyError as exc:
            raise ValueError(f"Unknown feature extractor: {name}") from exc

    def list(self) -> list[FeatureExtractor]:
        return list(self._extractors.values())
