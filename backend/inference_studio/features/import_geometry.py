"""Import a frozen Architecture B Mahalanobis geometry into the studio's geometry store.

    PYTHONPATH=backend python -m inference_studio.features.import_geometry \\
        --id geo_archb_b2_frozen --weight-id wgt_... \\
        --mu reproduced_mu_delta.npy --sigma reproduced_sigma_lambda.npy \\
        --corpus "B1 CLEAR_TRAIN ..." --method "LedoitWolf shrinkage (B1 fit_shrinkage_covariance)" \\
        --tau-preset 589.3056640625

The checkpoint hash comes from the registered weight record, so the geometry's FeatureSpaceSignature is exactly the
one the Architecture B extractor emits for that checkpoint. With ``--sigma`` the Cholesky factor is computed as B2's
frozen_mah.py does (``numpy.linalg.cholesky``); ``--cholesky`` imports a precomputed factor.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from inference_studio.config import Settings
from inference_studio.features.extractors.rfdetr_arch_b import architecture_b_signature
from inference_studio.features.geometry import MahalanobisGeometryStore
from inference_studio.models import ModelRegistry, WeightStore


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--id", required=True, help="geometry id, e.g. geo_archb_b2_frozen")
    parser.add_argument("--weight-id", required=True, help="registered rfdetr weight id of the Architecture B checkpoint")
    parser.add_argument("--mu", required=True, type=Path, help="mu_clear / mu_delta .npy [256]")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--sigma", type=Path, help="clear-delta covariance .npy [256,256]")
    source.add_argument("--cholesky", type=Path, help="precomputed lower Cholesky factor .npy [256,256]")
    parser.add_argument("--corpus", required=True, help="calibration corpus identity")
    parser.add_argument("--method", required=True, help="fitting method")
    parser.add_argument("--display-name", default="")
    parser.add_argument("--tau-preset", type=float, default=None, help="frozen raw threshold, recorded as metadata only")
    parser.add_argument("--note", action="append", default=[], help="extra provenance note (repeatable)")
    args = parser.parse_args(argv)

    settings = Settings.load()
    registry = ModelRegistry.load(settings.models_file)
    weight = WeightStore(settings.weights_dir, registry, settings.max_upload_bytes).get(args.weight_id, "rfdetr")
    signature = architecture_b_signature(weight.sha256)

    mu = np.load(args.mu)
    matrix_path = args.sigma or args.cholesky
    matrix = np.load(matrix_path)
    cholesky = np.linalg.cholesky(matrix.astype(np.float64)) if args.sigma else matrix
    metadata = {
        "display_name": args.display_name or args.id,
        "calibration_corpus": args.corpus,
        "fitting_method": args.method,
        "checkpoint_sha256": weight.sha256,
        "feature_signature": signature.display(),
        "tau_mah_preset": args.tau_preset,
        "tau_mah_applied": False,
        "cholesky_derivation": "numpy.linalg.cholesky(sigma) as frozen_mah.py" if args.sigma else "imported",
        "sources": {str(args.mu): _sha256(args.mu), str(matrix_path): _sha256(matrix_path)},
        "notes": list(args.note),
    }
    path = MahalanobisGeometryStore(settings.geometries_dir).save(args.id, mu, cholesky, signature, metadata)
    print(json.dumps({"geometry_id": args.id, "path": str(path), "feature_space": signature.display(), "checkpoint_sha256": weight.sha256}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
