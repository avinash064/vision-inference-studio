"""Verbatim source-of-truth feature code for the RF-DETR providers. Do not edit these files.

| Module | Source (/home/avinash/Skip_car/experiments/...) | SHA-256 (prefix) |
|---|---|---|
| rfdetr_xor_runtime.py | v2_2_deployable_package_20260929/candidate_attention/rfdetr_xor_runtime.py | fbbf76a671e8b303 |
| rfdetr_feature_xor.py | v2_2_deployable_package_20260929/candidate_attention/rfdetr_feature_xor.py | 1dba57dd61e9cfd5 |
| rsd_v1_extract_rfdetr_features.py | rfdetr_rsd_v1/extract_rfdetr_features.py (Architecture B backbone tap) | 9ff492861bc44b1f |
| b1_mahalanobis.py | ``mahalanobis_map`` from rfdetr_arch_b1_nldiff_obstruction/covariance.py | function copy |

The studio extractors in ``..rfdetr_v22`` / ``..rfdetr_arch_b`` call these functions instead of re-implementing
RF-DETR preprocessing, hooks or level discovery. ``tests/test_feature_space.py`` checks the whole-file hashes.
"""

VENDORED_SHA256_PREFIX = {
    "rfdetr_xor_runtime.py": "fbbf76a671e8b303",
    "rfdetr_feature_xor.py": "1dba57dd61e9cfd5",
    "rsd_v1_extract_rfdetr_features.py": "9ff492861bc44b1f",
}
