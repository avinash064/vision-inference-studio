#!/bin/sh
set -eu

rm -f /usr/bin/vision-inference-studio-setup
rm -f /usr/share/applications/vision-inference-studio.desktop
rm -f /usr/share/icons/hicolor/512x512/apps/vision-inference-studio.png
command -v update-desktop-database >/dev/null 2>&1 && update-desktop-database /usr/share/applications || true
command -v gtk-update-icon-cache >/dev/null 2>&1 && gtk-update-icon-cache -q /usr/share/icons/hicolor || true
echo "Vision Inference Studio user data was preserved under each user's XDG data directory."
exit 0
