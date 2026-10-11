"""Generic backend-only feature runtime.

Imports are lazy so schema types can be used by runtime artifacts without a
package-initialization cycle.
"""

_EXPORTS = {
    "FeatureArtifact": ("inference_studio.runtime.artifacts", "FeatureArtifact"),
    "FeatureMetadata": ("inference_studio.features.schema", "FeatureMetadata"),
    "FeatureSpaceSignature": ("inference_studio.features.schema", "FeatureSpaceSignature"),
    "HeatmapMetadata": ("inference_studio.features.schema", "HeatmapMetadata"),
    "FeatureStore": ("inference_studio.features.store", "FeatureStore"),
    "FeatureStoreCapacityExceeded": ("inference_studio.features.store", "FeatureStoreCapacityExceeded"),
    "FeatureSpaceMismatch": ("inference_studio.features.ops", "FeatureSpaceMismatch"),
    "MahGeometryMismatch": ("inference_studio.features.ops", "MahGeometryMismatch"),
    "MahalanobisGeometry": ("inference_studio.features.ops", "MahalanobisGeometry"),
    "activation_xor": ("inference_studio.features.ops", "activation_xor"),
    "difference": ("inference_studio.features.ops", "difference"),
    "mahalanobis_difference": ("inference_studio.features.ops", "mahalanobis_difference"),
    "normalize": ("inference_studio.features.ops", "normalize"),
    "summary": ("inference_studio.features.ops", "tensor_summary"),
    "feature_heatmap": ("inference_studio.features.visualization", "feature_visual"),
    "render_heatmap": ("inference_studio.features.visualization", "render_heatmap"),
}


def __getattr__(name: str):
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    from importlib import import_module
    return getattr(import_module(module_name), attribute)


__all__ = list(_EXPORTS)
