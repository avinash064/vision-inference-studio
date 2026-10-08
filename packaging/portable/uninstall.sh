#!/usr/bin/env bash
# Removes the per-user install made by install.sh.
#   ./uninstall.sh [--prefix <dir>] [--purge]   (--purge also deletes runtime, weights, workflows, logs)
set -Eeuo pipefail

PREFIX="$HOME/.local/opt/vision-inference-studio"
DATA_ROOT="${XDG_DATA_HOME:-$HOME/.local/share}/vision-inference-studio"
CONFIG_ROOT="${XDG_CONFIG_HOME:-$HOME/.config}/vision-inference-studio"
PURGE=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --prefix) PREFIX="${2:?}"; shift 2 ;;
    --purge) PURGE=1; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

# chrome-sandbox may be root-owned after install.sh configured it.
if [[ -e "$PREFIX" ]]; then
  [[ -x "$PREFIX/vision-inference-studio" ]] || { echo "Refusing to delete $PREFIX: it is not an install." >&2; exit 1; }
  rm -rf -- "$PREFIX" 2>/dev/null || sudo rm -rf -- "$PREFIX"
fi
rm -f "$HOME/.local/bin/vision-inference-studio" "${XDG_DATA_HOME:-$HOME/.local/share}/applications/vision-inference-studio.desktop"
if [[ "$PURGE" == 1 ]]; then
  rm -rf -- "$DATA_ROOT" "$CONFIG_ROOT"
  echo "Removed application and all user data."
else
  echo "Removed application. User data kept in $DATA_ROOT (use --purge to delete)."
fi
