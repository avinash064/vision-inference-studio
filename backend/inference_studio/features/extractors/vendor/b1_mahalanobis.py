"""Verbatim copy of B1's ``mahalanobis_map`` (rfdetr_arch_b1_nldiff_obstruction/covariance.py).

Only this function is vendored: the source module imports its sibling ``nldiff_common``. It is the
numerical reference for the GPU implementation in ``inference_studio.features.ops``.
"""

from __future__ import annotations

import numpy as np
import scipy.linalg


def mahalanobis_map(delta_42x256: np.ndarray, fitted: FittedCovariance) -> np.ndarray:
    """delta_42x256: (42,42,256). Returns S_MAH, shape (42,42), via a stable triangular solve
    against the Cholesky factor -- never an explicit matrix inverse."""
    h, w, c = delta_42x256.shape
    centered = (delta_42x256.reshape(-1, c) - fitted.mu).T  # (256, N)
    # check_finite=False: soft-correspondence deltas carry NaN at unconditionally-masked border
    # tokens (Section 12's own no-partial-neighborhood rule) -- NaN must propagate through to
    # the output (NaN stays NaN, matching this codebase's established convention), not raise.
    y = scipy.linalg.solve_triangular(fitted.cholesky_lower, centered, lower=True, check_finite=False)
    sq = np.sum(y ** 2, axis=0)  # (N,) == centered^T Sigma^-1 centered
    return sq.reshape(h, w).astype(np.float32)
