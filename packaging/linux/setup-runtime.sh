#!/usr/bin/env bash
set -Eeuo pipefail

APP_VERSION="${VIS_APP_VERSION:-0.4.0}"
SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" && pwd)"
APP_ROOT="${VIS_APP_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
DATA_HOME="${XDG_DATA_HOME:-$HOME/.local/share}"
CONFIG_HOME="${XDG_CONFIG_HOME:-$HOME/.config}"
DATA_ROOT="$DATA_HOME/vision-inference-studio"
CONFIG_ROOT="$CONFIG_HOME/vision-inference-studio"
RUNTIME_DIR="$DATA_ROOT/runtime"
VENV_DIR="$RUNTIME_DIR/venv"
INSTALLING_DIR="$RUNTIME_DIR/venv.installing"
METADATA_FILE="$RUNTIME_DIR/runtime-version.json"
LOCK_DIR="$RUNTIME_DIR/setup.lock"
LOG_DIR="$DATA_ROOT/logs"
LOG_FILE="$LOG_DIR/runtime-setup.log"
MODE="install"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode) MODE="${2:-}"; shift 2 ;;
    --print-requirements-hash) MODE="hash"; shift ;;
    --check) MODE="check"; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

requirements_hash() {
  local name
  {
    for name in requirements.txt requirements-runtimes.txt requirements-chainner-standard.txt requirements-desktop.txt requirements-lock.txt pyproject.toml; do
      printf '%s\n' "$name"
      cat "$APP_ROOT/$name"
      printf '\n'
    done
  } | sha256sum | awk '{print $1}'
}

if [[ "$MODE" == "hash" ]]; then
  requirements_hash
  exit 0
fi

mkdir -p "$RUNTIME_DIR" "$LOG_DIR" "$CONFIG_ROOT" "$DATA_ROOT/weights" "$DATA_ROOT/workflows" "$DATA_ROOT/workspace/assets" "$DATA_ROOT/workspace/previews" "$DATA_ROOT/benchmarks"
touch "$LOG_FILE"
chmod 600 "$LOG_FILE"
exec > >(tee -a "$LOG_FILE") 2>&1

if [[ "$MODE" == "check" ]]; then
  exec "$VENV_DIR/bin/python" -m inference_studio.runtime_check --json
fi

if [[ ! "$MODE" =~ ^(install|repair|reinstall)$ ]]; then
  echo "Invalid runtime mode: $MODE" >&2
  exit 2
fi

if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  if [[ -f "$LOCK_DIR/pid" ]] && ! kill -0 "$(cat "$LOCK_DIR/pid")" 2>/dev/null; then
    rm -f "$LOCK_DIR/pid"
    rmdir "$LOCK_DIR" 2>/dev/null || true
    mkdir "$LOCK_DIR"
  else
    echo "Another runtime setup is already running." >&2
    exit 3
  fi
fi
printf '%s\n' "$$" > "$LOCK_DIR/pid"

BUILD_SRC=""
cleanup() {
  rm -f "$LOCK_DIR/pid"
  rmdir "$LOCK_DIR" 2>/dev/null || true
  [[ -n "$BUILD_SRC" ]] && rm -rf -- "$BUILD_SRC"
  return 0
}
trap cleanup EXIT
trap 'echo "Runtime setup failed at line $LINENO. See $LOG_FILE" >&2' ERR

progress() { printf 'VIS_PROGRESS %s 10 %s\n' "$1" "$2"; }
select_python() {
  local candidate version_ok
  # Packaged builds ship their own relocatable CPython; the host Python is only a dev fallback.
  for candidate in "$APP_ROOT"/python/bin/python3.[0-9] "$APP_ROOT"/python/bin/python3.[0-9][0-9]; do
    if [[ -x "$candidate" ]]; then
      echo "$candidate"
      return 0
    fi
  done
  for candidate in python3.12 python3.13 python3.11 python3.10 python3; do
    command -v "$candidate" >/dev/null 2>&1 || continue
    version_ok="$($candidate -c 'import sys; print(int(sys.version_info >= (3, 10)))')"
    if [[ "$version_ok" == "1" ]] && "$candidate" -c 'import ensurepip, venv' >/dev/null 2>&1; then
      command -v "$candidate"
      return 0
    fi
  done
  return 1
}

START_SECONDS="$SECONDS"
progress 1 "Preparing private application directories"
for file in requirements.txt requirements-runtimes.txt requirements-chainner-standard.txt requirements-desktop.txt requirements-lock.txt pyproject.toml; do
  [[ -f "$APP_ROOT/$file" ]] || { echo "Missing packaged file: $APP_ROOT/$file" >&2; exit 4; }
done

progress 2 "Selecting Python"
SYSTEM_PYTHON="$(select_python)" || { echo "Python 3.10+ with the venv module is required (Debian/Ubuntu: sudo apt install python3-venv)." >&2; exit 5; }
PYTHON_VERSION="$($SYSTEM_PYTHON -c 'import platform; print(platform.python_version())')"
echo "Selected $SYSTEM_PYTHON ($PYTHON_VERSION)"

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "NVIDIA driver detected; PyTorch CUDA support will be validated after installation."
else
  echo "No NVIDIA driver detected; the runtime will use CPU fallback."
fi

progress 3 "Preparing isolated virtual environment"
if [[ "$MODE" == "reinstall" ]]; then
  case "$VENV_DIR" in "$RUNTIME_DIR/venv") rm -rf -- "$VENV_DIR" ;; *) echo "Refusing unsafe runtime path" >&2; exit 6 ;; esac
fi
# A runtime built on a different interpreter (e.g. an older release using the host Python) cannot be reused.
if [[ -e "$VENV_DIR" ]] && [[ "$(readlink -f "$VENV_DIR/bin/python" || true)" != "$(readlink -f "$SYSTEM_PYTHON")" ]]; then
  echo "Existing runtime was built with a different Python; recreating it."
  case "$VENV_DIR" in "$RUNTIME_DIR/venv") rm -rf -- "$VENV_DIR" ;; *) echo "Refusing unsafe runtime path" >&2; exit 6 ;; esac
fi

NEW_ENV=0
TARGET_VENV="$VENV_DIR"
if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  case "$INSTALLING_DIR" in "$RUNTIME_DIR/venv.installing") rm -rf -- "$INSTALLING_DIR" ;; *) exit 6 ;; esac
  VENV_START="$SECONDS"
  "$SYSTEM_PYTHON" -m venv "$INSTALLING_DIR"
  TARGET_VENV="$INSTALLING_DIR"
  NEW_ENV=1
  VENV_SECONDS=$((SECONDS - VENV_START))
else
  VENV_SECONDS=0
fi
VENV_PYTHON="$TARGET_VENV/bin/python"

progress 4 "Updating Python packaging tools"
"$VENV_PYTHON" -m pip install --disable-pip-version-check --upgrade pip setuptools wheel

INSTALL_START="$SECONDS"
# Prebuilt wheels only (end-user machines usually have no compiler), pinned to the tested lock.
progress 5 "Installing backend dependencies"
"$VENV_PYTHON" -m pip install --disable-pip-version-check --only-binary=:all: -c "$APP_ROOT/requirements-lock.txt" -r "$APP_ROOT/requirements.txt"

progress 6 "Installing PyTorch, inference runtimes and chaiNNer support"
# The lock pins CUDA 12.1 builds of torch/torchvision, which are published on the PyTorch index.
"$VENV_PYTHON" -m pip install --disable-pip-version-check --only-binary=:all: --extra-index-url https://download.pytorch.org/whl/cu121 -c "$APP_ROOT/requirements-lock.txt" -r "$APP_ROOT/requirements-desktop.txt"

progress 7 "Installing bundled Grounding DINO and CLIP"
# Built from pinned source commits at packaging time (pure-Python wheels); offline, no compiler needed.
shopt -s nullglob
BUNDLED_WHEELS=("$APP_ROOT"/wheels/*.whl)
shopt -u nullglob
if [[ ${#BUNDLED_WHEELS[@]} -gt 0 ]]; then
  "$VENV_PYTHON" -m pip install --disable-pip-version-check --no-index --no-deps "${BUNDLED_WHEELS[@]}"
fi

progress 8 "Installing the Vision Inference Studio backend"
# Build from a private copy: the packaged app root is read-only under /usr/lib.
BUILD_SRC="$(mktemp -d)"
cp -r "$APP_ROOT/pyproject.toml" "$APP_ROOT/backend" "$BUILD_SRC/"
chmod -R u+w "$BUILD_SRC"
"$VENV_PYTHON" -m pip install --disable-pip-version-check --no-deps "$BUILD_SRC"
INSTALL_SECONDS=$((SECONDS - INSTALL_START))

progress 9 "Validating required imports and compute support"
VALIDATE_START="$SECONDS"
"$VENV_PYTHON" -m inference_studio.runtime_check --json
VALIDATE_SECONDS=$((SECONDS - VALIDATE_START))

if [[ "$NEW_ENV" == "1" ]]; then
  mv "$INSTALLING_DIR" "$VENV_DIR"
  VENV_PYTHON="$VENV_DIR/bin/python"
fi

progress 10 "Writing deterministic runtime metadata"
REQ_HASH="$(requirements_hash)"
TOTAL_SECONDS=$((SECONDS - START_SECONDS))
"$VENV_PYTHON" - "$METADATA_FILE" "$APP_VERSION" "$REQ_HASH" "$VENV_SECONDS" "$INSTALL_SECONDS" "$VALIDATE_SECONDS" "$TOTAL_SECONDS" <<'PY'
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from inference_studio.runtime_check import inspect_runtime

target, app_version, requirements_hash, venv_seconds, install_seconds, validation_seconds, total_seconds = sys.argv[1:]
check = inspect_runtime()
if not check["ok"]:
    raise SystemExit("Runtime validation failed: " + ", ".join(check["failures"]))
metadata = {
    "app_version": app_version,
    "requirements_hash": requirements_hash,
    "python_version": check["python"]["version"],
    "python_executable": check["python"]["executable"],
    "cuda": check["cuda"],
    "packages": check["packages"],
    "created_at": datetime.now(timezone.utc).isoformat(),
    "durations_seconds": {
        "venv_creation": int(venv_seconds),
        "dependency_install": int(install_seconds),
        "validation": int(validation_seconds),
        "total": int(total_seconds),
    },
}
directory = os.path.dirname(target)
fd, temporary = tempfile.mkstemp(prefix="runtime-version.", suffix=".tmp", dir=directory)
try:
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.chmod(temporary, 0o600)
    os.replace(temporary, target)
finally:
    if os.path.exists(temporary):
        os.unlink(temporary)
PY
echo "RUNTIME READY ($TOTAL_SECONDS seconds)"
