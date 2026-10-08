#!/usr/bin/env bash
# Per-user install of the portable Vision Inference Studio build (no .deb needed).
#   ./install.sh [--data <bundle data dir>] [--weights <weights.tar|weights-dir>] [--setup-runtime] [--prefix <dir>]
set -Eeuo pipefail

HERE="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
PREFIX="$HOME/.local/opt/vision-inference-studio"
DATA_ROOT="${XDG_DATA_HOME:-$HOME/.local/share}/vision-inference-studio"
BIN_DIR="$HOME/.local/bin"
APPS_DIR="${XDG_DATA_HOME:-$HOME/.local/share}/applications"
WEIGHTS=""
DATA=""
SETUP_RUNTIME=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data) DATA="${2:?--data needs a path}"; shift 2 ;;
    --weights) WEIGHTS="${2:?--weights needs a path}"; shift 2 ;;
    --setup-runtime) SETUP_RUNTIME=1; shift ;;
    --prefix) PREFIX="${2:?--prefix needs a path}"; shift 2 ;;
    -h|--help) sed -n 2,3p "$0"; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

say() { printf '\n==> %s\n' "$*"; }
warn() { printf 'WARNING: %s\n' "$*" >&2; }

say "Checking this machine"
[[ "$(uname -m)" == "x86_64" ]] || { echo "This build is for x86_64 Linux only (found $(uname -m))." >&2; exit 1; }
PY_OK=0
for bundled in "$HERE"/app/resources/python/bin/python3.[0-9] "$HERE"/app/resources/python/bin/python3.[0-9][0-9]; do
  [[ -x "$bundled" ]] || continue
  echo "Python: bundled $("$bundled" --version 2>&1 | cut -d' ' -f2) (system Python not needed)"; PY_OK=1; break
done
for candidate in python3.12 python3.13 python3.11 python3.10 python3; do
  [[ "$PY_OK" == 1 ]] && break
  command -v "$candidate" >/dev/null 2>&1 || continue
  if "$candidate" -c 'import sys, ensurepip, venv; sys.exit(0 if sys.version_info >= (3, 10) else 1)' >/dev/null 2>&1; then
    echo "Python: $($candidate --version) ($(command -v "$candidate"))"; PY_OK=1; break
  fi
done
if [[ "$PY_OK" != 1 ]]; then
  warn "Python 3.10+ with venv was not found. The app's first-run setup will fail until you install it:"
  warn "  Ubuntu/Debian: sudo apt install python3 python3-venv"
fi
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "GPU: $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null | head -1)"
else
  warn "nvidia-smi not found; inference will run on CPU."
fi

say "Installing application to $PREFIX"
if [[ -e "$PREFIX" ]]; then
  [[ -x "$PREFIX/vision-inference-studio" ]] || { echo "Refusing to replace $PREFIX: it is not a previous install." >&2; exit 1; }
  rm -rf -- "$PREFIX" 2>/dev/null || sudo rm -rf -- "$PREFIX"
fi
mkdir -p "$(dirname "$PREFIX")"
cp -a "$HERE/app" "$PREFIX"

# Chromium's sandbox needs either unprivileged user namespaces or a setuid-root
# chrome-sandbox helper. Ubuntu 23.10+ restricts user namespaces via AppArmor.
SANDBOX="$PREFIX/chrome-sandbox"
userns_restricted=0
[[ "$(cat /proc/sys/kernel/apparmor_restrict_unprivileged_userns 2>/dev/null || echo 0)" == 1 ]] && userns_restricted=1
[[ "$(cat /proc/sys/kernel/unprivileged_userns_clone 2>/dev/null || echo 1)" == 0 ]] && userns_restricted=1
if [[ "$userns_restricted" == 1 ]]; then
  say "Configuring the Chromium sandbox helper (needs sudo once)"
  if sudo chown root:root "$SANDBOX" && sudo chmod 4755 "$SANDBOX"; then
    echo "chrome-sandbox is now setuid root."
  else
    warn "Could not configure chrome-sandbox; the launcher will start the app with --no-sandbox."
  fi
fi

say "Creating launcher and menu entry"
mkdir -p "$BIN_DIR" "$APPS_DIR"
cat > "$BIN_DIR/vision-inference-studio" <<EOF
#!/usr/bin/env bash
# Set by VS Code and other Electron hosts; it would make the app start as plain Node.
unset ELECTRON_RUN_AS_NODE
APP="$PREFIX/vision-inference-studio"
SANDBOX="$SANDBOX"
if [[ "\$(stat -c '%u %a' "\$SANDBOX" 2>/dev/null)" == "0 4755" ]] || [[ "\$(cat /proc/sys/kernel/apparmor_restrict_unprivileged_userns 2>/dev/null || echo 0)" != 1 ]]; then
  exec "\$APP" "\$@"
fi
echo "chrome-sandbox is not setuid root; starting with --no-sandbox" >&2
exec "\$APP" --no-sandbox "\$@"
EOF
chmod 755 "$BIN_DIR/vision-inference-studio"
sed -e "s|^Exec=.*|Exec=$BIN_DIR/vision-inference-studio %U|" \
    -e "s|^Icon=.*|Icon=$PREFIX/resources/assets/icon.png|" \
    "$PREFIX/resources/packaging/linux/vision-inference-studio.desktop" > "$APPS_DIR/vision-inference-studio.desktop"
command -v update-desktop-database >/dev/null 2>&1 && update-desktop-database "$APPS_DIR" >/dev/null 2>&1 || true

if [[ -n "$DATA" ]]; then
  # Bundle data/: weights/ (every model, already registered) and huggingface/ (offline SAM / Grounding DINO files).
  [[ -d "$DATA/weights" ]] || { echo "Not a bundle data directory (no weights/): $DATA" >&2; exit 1; }
  say "Importing model weights and Hugging Face files into $DATA_ROOT"
  mkdir -p "$DATA_ROOT"
  cp -a "$DATA"/. "$DATA_ROOT/"
  find "$DATA_ROOT/weights" -name 'wgt_*.json' -printf '  %P\n'
fi

if [[ -n "$WEIGHTS" ]]; then
  say "Importing model weights into $DATA_ROOT/weights"
  mkdir -p "$DATA_ROOT"
  if [[ -f "$WEIGHTS" ]]; then
    tar -xf "$WEIGHTS" -C "$DATA_ROOT"
  elif [[ -d "$WEIGHTS" ]]; then
    mkdir -p "$DATA_ROOT/weights"
    cp -a "$WEIGHTS"/. "$DATA_ROOT/weights/"
  else
    echo "Weights path not found: $WEIGHTS" >&2; exit 1
  fi
  find "$DATA_ROOT/weights" -name 'wgt_*.json' -printf '  %P\n'
fi

if [[ "$SETUP_RUNTIME" == 1 ]]; then
  say "Creating the inference runtime (about 4 GB download: PyTorch, Ultralytics, RF-DETR, ONNX Runtime, chaiNNer support)"
  bash "$PREFIX/resources/packaging/linux/setup-runtime.sh" --mode install
fi

say "Done"
echo "Start the app from your application menu or run: $BIN_DIR/vision-inference-studio"
[[ "$SETUP_RUNTIME" == 1 ]] || echo "On first launch the app installs its inference runtime (needs internet: about 4 GB download, 10 GB on disk)."
echo "Logs: $DATA_ROOT/logs"
