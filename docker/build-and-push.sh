#!/usr/bin/env bash
# Build the self-contained backend image (all dependencies + every model weight in weights/, the chaiNNer
# source, and the Hugging Face caches the SAM v1 / Grounding DINO adapters load) plus the frontend image,
# smoke-test them, and optionally push.
#
#   docker/build-and-push.sh            # build + smoke test
#   docker/build-and-push.sh --push     # ... then push (run `docker login` first)
#
# Environment: VIS_IMAGE_REPO (default avinash061/vision_studio), VIS_CHAINNER_SRC_DIR, HF_HOME.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)"
REPO="${VIS_IMAGE_REPO:-avinash061/vision_studio}"
VERSION="$(sed -n 's/^__version__ = "\(.*\)"/\1/p' "$ROOT/backend/inference_studio/__init__.py")"
BACKEND="$REPO:backend-$VERSION-cu121"
FRONTEND="$REPO:frontend-$VERSION"
CHAINNER_SRC="${VIS_CHAINNER_SRC_DIR:-$ROOT/../src}"
HF_HUB="${HF_HOME:-$HOME/.cache/huggingface}/hub"
HF_MODELS=(models--facebook--sam-vit-base models--bert-base-uncased)
NEED_GB=25
PUSH=0
[[ "${1:-}" == "--push" ]] && PUSH=1

say() { printf '\n==> %s\n' "$*"; }
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

say "Preflight"
[[ -n "$VERSION" ]] || die "could not read the backend version"
DOCKER_ROOT="$(docker info --format '{{.DockerRootDir}}')"
FREE_GB="$(df -BG --output=avail "$DOCKER_ROOT" | tail -1 | tr -dc 0-9)"
echo "Images: $BACKEND, $FRONTEND"
echo "Free space for Docker ($DOCKER_ROOT): ${FREE_GB} GB (need about $NEED_GB GB while building)"
(( FREE_GB >= NEED_GB )) || die "free at least $NEED_GB GB on the disk holding $DOCKER_ROOT first"
[[ -d "$CHAINNER_SRC" ]] || die "chaiNNer source not found at $CHAINNER_SRC (set VIS_CHAINNER_SRC_DIR)"
for model in "${HF_MODELS[@]}"; do [[ -d "$HF_HUB/$model" ]] || die "Hugging Face cache missing $HF_HUB/$model"; done
WEIGHTS=$(find "$ROOT/weights" -name 'wgt_*.json' | wc -l)
(( WEIGHTS > 0 )) || die "no registered weights in $ROOT/weights"
echo "Registered weights to bake in: $WEIGHTS ($(du -sh "$ROOT/weights" | cut -f1))"
if (( PUSH )); then
  grep -q '"https://index.docker.io/v1/"' "$HOME/.docker/config.json" 2>/dev/null || die "not logged in: run 'docker login -u ${REPO%%/*}' first"
fi

say "Building dependency layer (CUDA 12.1 PyTorch and all runtimes)"
docker build -f "$ROOT/docker/backend-full.Dockerfile" -t "$BACKEND-deps" "$ROOT"

say "Adding weights, chaiNNer source and Hugging Face caches"
CID="$(docker create "$BACKEND-deps")"
trap 'docker rm -f "$CID" >/dev/null 2>&1 || true' EXIT
docker cp "$ROOT/weights/." "$CID:/app/weights/"
docker cp "$CHAINNER_SRC/." "$CID:/app/chainner/src/"
docker cp "$ROOT/.env.example" "$CID:/app/.env.example"
for model in "${HF_MODELS[@]}"; do
  docker cp "$HF_HUB/$model" "$CID:/opt/huggingface/hub/$model"
done
docker commit -m "Vision Inference Studio $VERSION weights" "$CID" "$BACKEND" >/dev/null
docker rm "$CID" >/dev/null
trap - EXIT
docker rmi "$BACKEND-deps" >/dev/null

say "Building frontend"
docker build -f "$ROOT/frontend/Dockerfile" -t "$FRONTEND" "$ROOT"

say "Smoke test"
docker run --rm "$BACKEND" sh -c 'env | grep -c "^HF_TOKEN=" || true' | grep -qx 0 || die "HF_TOKEN found in the image environment"
docker run --rm "$BACKEND" sh -c 'test ! -e /app/.env' || die ".env was baked into the image"
SMOKE="vis-smoke-$$"
docker run -d --rm --name "$SMOKE" -p 127.0.0.1:18000:8000 "$BACKEND" >/dev/null
trap 'docker stop "$SMOKE" >/dev/null 2>&1 || true' EXIT
for _ in $(seq 60); do curl -fs http://127.0.0.1:18000/health >/dev/null && break; sleep 2; done
curl -fs http://127.0.0.1:18000/health
LISTED=$(curl -fs http://127.0.0.1:18000/api/v1/weights | python3 -c 'import json,sys; print(len(json.load(sys.stdin)))')
NODES=$(curl -fs http://127.0.0.1:18000/api/v1/nodes | python3 -c 'import json,sys; d=json.load(sys.stdin); n=d if isinstance(d,list) else d.get("nodes",d); print(len(n))')
echo "weights listed: $LISTED / $WEIGHTS, nodes: $NODES"
(( LISTED == WEIGHTS )) || die "image lists $LISTED weights, expected $WEIGHTS"
docker exec "$SMOKE" python -c "import torch, ultralytics, rfdetr, transformers, groundingdino, clip, onnxruntime, chainner_ext; print('imports ok, torch', torch.__version__)"
docker stop "$SMOKE" >/dev/null
trap - EXIT
docker image ls "$REPO"

if (( PUSH )); then
  say "Pushing"
  docker push "$BACKEND"
  docker push "$FRONTEND"
fi
say "Done"
