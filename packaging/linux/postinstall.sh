#!/bin/sh
set -eu

APP_DIR=/usr/lib/vision-inference-studio
RESOURCES="$APP_DIR/resources"

if [ -d "$RESOURCES" ]; then
  install -Dm644 "$RESOURCES/packaging/linux/vision-inference-studio.desktop" /usr/share/applications/vision-inference-studio.desktop
  install -Dm644 "$RESOURCES/assets/icon.png" /usr/share/icons/hicolor/512x512/apps/vision-inference-studio.png
  ln -sfn "$RESOURCES/packaging/linux/setup-runtime.sh" /usr/bin/vision-inference-studio-setup
fi
command -v update-desktop-database >/dev/null 2>&1 && update-desktop-database /usr/share/applications || true
command -v gtk-update-icon-cache >/dev/null 2>&1 && gtk-update-icon-cache -q /usr/share/icons/hicolor || true
exit 0
