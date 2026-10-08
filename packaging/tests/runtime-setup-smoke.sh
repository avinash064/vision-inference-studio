#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TEST_ROOT="$(mktemp -d)"
APP_ROOT="$TEST_ROOT/app"
export XDG_DATA_HOME="$TEST_ROOT/data"
export XDG_CONFIG_HOME="$TEST_ROOT/config"
export VIS_APP_ROOT="$APP_ROOT"
export VIS_APP_VERSION="0.2.0-test"

cleanup() {
  case "$TEST_ROOT" in /tmp/*) rm -rf -- "$TEST_ROOT" ;; esac
}
trap cleanup EXIT

mkdir -p "$APP_ROOT/backend/inference_studio"
printf '%s\n' '# smoke fixture' > "$APP_ROOT/requirements.txt"
printf '%s\n' '-r requirements.txt' > "$APP_ROOT/requirements-runtimes.txt"
printf '%s\n' '# smoke lock' > "$APP_ROOT/requirements-lock.txt"
cat > "$APP_ROOT/pyproject.toml" <<'EOF'
[build-system]
requires = ["setuptools"]
build-backend = "setuptools.build_meta"
[project]
name = "vision-inference-studio-runtime-smoke"
version = "0.0.1"
[tool.setuptools]
package-dir = {"" = "backend"}
[tool.setuptools.packages.find]
where = ["backend"]
EOF
printf '%s\n' '__all__ = []' > "$APP_ROOT/backend/inference_studio/__init__.py"
cat > "$APP_ROOT/backend/inference_studio/runtime_check.py" <<'PY'
import json
import platform
import sys

def inspect_runtime():
    return {
        "ok": True,
        "python": {"version": platform.python_version(), "executable": sys.executable},
        "packages": {"smoke": {"ok": True, "version": "1"}},
        "cuda": {"available": False, "version": None, "devices": []},
        "failures": [],
    }

if __name__ == "__main__":
    print(json.dumps(inspect_runtime(), sort_keys=True))
PY

SYSTEM_BEFORE="$(python3 -m pip freeze 2>/dev/null || true)"
"$PROJECT_ROOT/packaging/linux/setup-runtime.sh" --mode install
RUNTIME_ROOT="$XDG_DATA_HOME/vision-inference-studio/runtime"
test -x "$RUNTIME_ROOT/venv/bin/python"
test -f "$RUNTIME_ROOT/runtime-version.json"
test ! -e "$RUNTIME_ROOT/venv.installing"
test ! -e "$RUNTIME_ROOT/setup.lock"
mkdir -p "$XDG_DATA_HOME/vision-inference-studio/weights"
touch "$XDG_DATA_HOME/vision-inference-studio/weights/preserve-me"
"$PROJECT_ROOT/packaging/linux/setup-runtime.sh" --mode reinstall
test -f "$XDG_DATA_HOME/vision-inference-studio/weights/preserve-me"
"$PROJECT_ROOT/packaging/linux/setup-runtime.sh" --check >/dev/null
SYSTEM_AFTER="$(python3 -m pip freeze 2>/dev/null || true)"
test "$SYSTEM_BEFORE" = "$SYSTEM_AFTER"
echo "RUNTIME_SETUP_SMOKE_PASS"
