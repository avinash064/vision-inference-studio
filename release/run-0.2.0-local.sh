#!/usr/bin/env bash
# Launch the packaged 0.2.0 app on this dev machine, using /home/avinash/venv as its runtime and the
# project's weights/workspace/workflows/benchmarks. Data and settings are isolated from a normal install.
set -Eeuo pipefail

HERE="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
PROJECT="$(cd "$HERE/.." && pwd)"
VENV=/home/avinash/venv
APP_DIR="$HOME/.local/opt/vision-inference-studio-0.2.0"
DATA_HOME="$HOME/.local/share/vis-0.2.0-local"
CONFIG_HOME="$HOME/.config/vis-0.2.0-local"
DATA="$DATA_HOME/vision-inference-studio"

if [[ ! -x "$APP_DIR/vision-inference-studio" ]]; then
  echo "Unpacking 0.2.0 app to $APP_DIR"
  mkdir -p "$APP_DIR"
  tar -xzf "$HERE/vision-inference-studio-0.2.0-linux-x64.tar.gz" -C "$APP_DIR" --strip-components=2 \
    vision-inference-studio-0.2.0-linux-x64/app
fi

mkdir -p "$DATA/runtime" "$DATA/logs" "$CONFIG_HOME"
ln -sfn "$VENV" "$DATA/runtime/venv"
for name in weights workspace workflows benchmarks; do
  [[ -e "$DATA/$name" && ! -L "$DATA/$name" ]] || ln -sfn "$PROJECT/$name" "$DATA/$name"
done

# The app only starts the backend when the runtime metadata matches its packaged requirements.
HASH="$(bash "$APP_DIR/resources/packaging/linux/setup-runtime.sh" --print-requirements-hash)"
"$VENV/bin/python" - "$DATA/runtime/runtime-version.json" "$HASH" <<'PY'
import json, platform, sys
import torch
target, requirements_hash = sys.argv[1:]
json.dump({
    "app_version": "0.2.0",
    "requirements_hash": requirements_hash,
    "python_version": platform.python_version(),
    "python_executable": sys.executable,
    "created_at": "linked to developer venv",
    "cuda": {"available": torch.cuda.is_available(), "version": torch.version.cuda,
             "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]},
}, open(target, "w"), indent=2)
PY

SANDBOX_FLAG=()
[[ "$(stat -c '%u %a' "$APP_DIR/chrome-sandbox")" == "0 4755" ]] || SANDBOX_FLAG=(--no-sandbox)

echo "Starting Vision Inference Studio 0.2.0 (runtime: $VENV)"
env -u ELECTRON_RUN_AS_NODE XDG_DATA_HOME="$DATA_HOME" XDG_CONFIG_HOME="$CONFIG_HOME" \
  setsid nohup "$APP_DIR/vision-inference-studio" "${SANDBOX_FLAG[@]}" "$@" > "$DATA/logs/launch.out" 2>&1 &
echo "Logs: $DATA/logs"
