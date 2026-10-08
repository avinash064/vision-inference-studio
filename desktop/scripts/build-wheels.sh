#!/usr/bin/env bash
# Build the pure-Python groundingdino and clip wheels the desktop runtime installs offline, from the commits
# pinned in requirements-prompt-models.txt. Needs a Python with torch installed (groundingdino's setup.py
# imports it): PYTHON=/home/avinash/venv/bin/python desktop/scripts/build-wheels.sh
set -Eeuo pipefail
ROOT="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../.." && pwd)"
PYTHON="${PYTHON:-python3}"
OUT="$ROOT/desktop/wheels"
rm -rf "$OUT"; mkdir -p "$OUT"
mapfile -t SOURCES < <(grep -E '^(groundingdino|clip) @ git\+' "$ROOT/requirements-prompt-models.txt")
"$PYTHON" -m pip wheel --disable-pip-version-check --no-deps --no-build-isolation -w "$OUT" "${SOURCES[@]}"
ls -l "$OUT"
