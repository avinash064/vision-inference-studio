from __future__ import annotations

import asyncio
import base64
import json
import os
import tempfile
from pathlib import Path
from urllib.parse import urlparse

import httpx
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from fastapi.responses import Response
import cv2

from inference_studio.adapters import ModelConfig
from inference_studio.graph import Workflow
from inference_studio.public_hub.schemas import (
    BenchmarkCreate, ExperimentCreate, ExternalResourceRef, ModelArtifactWeightRequest, PromoteRequest, RegisteredWeight,
    ProviderCredentialUpdate, ResourceDownloadRequest, ResourceKind,
)
from inference_studio.models.registry import ModelRegistryError
from inference_studio.public_hub.security import redact_error


router = APIRouter(prefix="/api/v1")


def services(request: Request):
    return request.app.state.services


def _first_dataset_image(value):
    if isinstance(value, dict):
        source = value.get("src")
        if isinstance(source, str) and (source.startswith("https://") or source.startswith("data:image/")):
            return source
        for child in value.values():
            found = _first_dataset_image(child)
            if found:
                return found
    elif isinstance(value, list):
        for child in value:
            found = _first_dataset_image(child)
            if found:
                return found
    return None


def _huggingface_dataset_preview(resource_id: str, revision: str | None) -> bytes:
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    with httpx.Client(timeout=20, follow_redirects=True, headers=headers) as client:
        split_response = client.get("https://datasets-server.huggingface.co/splits", params={"dataset": resource_id})
        split_response.raise_for_status()
        splits = split_response.json().get("splits", [])
        if not splits:
            raise ValueError("Dataset viewer has no available split")
        selected = splits[0]
        params = {"dataset": resource_id, "config": selected["config"], "split": selected["split"]}
        if revision:
            params["revision"] = revision
        rows_response = client.get("https://datasets-server.huggingface.co/first-rows", params=params)
        if rows_response.status_code == 422 and revision:
            params.pop("revision")
            rows_response = client.get("https://datasets-server.huggingface.co/first-rows", params=params)
        rows_response.raise_for_status()
        source = _first_dataset_image(rows_response.json().get("rows", []))
        if not source:
            raise ValueError("Dataset viewer has no image sample")
        if source.startswith("data:image/"):
            _, encoded = source.split(",", 1)
            raw = base64.b64decode(encoded, validate=True)
        else:
            host = (urlparse(source).hostname or "").lower()
            if host != "huggingface.co" and not host.endswith(".huggingface.co"):
                raise ValueError("Dataset viewer returned an unsupported image host")
            with client.stream("GET", source) as image_response:
                image_response.raise_for_status()
                length = int(image_response.headers.get("content-length", 0) or 0)
                if length > 8 * 1024 * 1024:
                    raise ValueError("Dataset preview image is too large")
                chunks, size = [], 0
                for chunk in image_response.iter_bytes():
                    size += len(chunk)
                    if size > 8 * 1024 * 1024:
                        raise ValueError("Dataset preview image is too large")
                    chunks.append(chunk)
                raw = b"".join(chunks)
    image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Dataset preview image cannot be decoded")
    height, width = image.shape[:2]
    if width > 480:
        image = cv2.resize(image, (480, max(1, round(height * 480 / width))))
    success, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not success:
        raise ValueError("Dataset preview image cannot be encoded")
    return encoded.tobytes()


@router.get("/nodes")
def list_nodes(request: Request):
    nodes = [node.model_dump(mode="json") for node in services(request).nodes.list()]
    geometries = services(request).geometries
    if geometries is not None:
        options = [{"label": f"{row['display_name']} · {row['checkpoint_sha256'][:12]}", "value": row["geometry_id"]} for row in geometries.list()]
        for node in nodes:
            for field in node.get("config", []):
                if node.get("metadata", {}).get("options_source", {}).get(field["id"]) == "feature_geometries":
                    field["options"] = options
    return nodes


@router.get("/features/geometries")
def list_feature_geometries(request: Request):
    geometries = services(request).geometries
    return geometries.list() if geometries is not None else []


@router.get("/chainner/report")
def chainner_report(request: Request):
    adapter = services(request).chainner
    if adapter is None:
        raise HTTPException(404, "chaiNNer source is not configured or discovery failed")
    return adapter.report


@router.get("/chainner/settings")
def chainner_settings(request: Request):
    store = services(request).chainner_settings
    return {"values": store.snapshot(), "definitions": store.definitions()}


@router.put("/chainner/settings")
def update_chainner_settings(values: dict, request: Request):
    try:
        return services(request).chainner_settings.update(values)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.get("/models")
def list_models(request: Request):
    value = services(request)
    return {**value.models.public_data(), "loaded": [item.model_dump(mode="json") for item in value.manager.status()]}


@router.get("/models/loaded")
def loaded_models(request: Request):
    return [item.model_dump(mode="json") for item in services(request).manager.status()]


@router.get("/devices")
def list_devices():
    devices = [{"id": "cpu", "label": "CPU", "kind": "cpu"}]
    try:
        import torch
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            devices.append({"id": f"cuda:{index}", "label": props.name, "kind": "cuda", "memory_mb": round(props.total_memory / 1024**2)})
    except ImportError:
        pass
    return devices


@router.get("/system/metrics")
def system_metrics():
    import psutil

    memory = psutil.virtual_memory()
    result = {
        "cpu": {
            "percent": psutil.cpu_percent(interval=None),
            "memory_percent": memory.percent,
            "memory_used_gb": round(memory.used / 1024**3, 2),
            "memory_total_gb": round(memory.total / 1024**3, 2),
            "load": list(os.getloadavg()),
        },
        "gpu": None,
    }
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
        gpu_memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
        name = pynvml.nvmlDeviceGetName(handle)
        result["gpu"] = {
            "name": name.decode() if isinstance(name, bytes) else str(name),
            "percent": float(utilization.gpu),
            "memory_percent": round(gpu_memory.used / gpu_memory.total * 100, 1),
            "memory_used_mb": round(gpu_memory.used / 1024**2),
            "memory_total_mb": round(gpu_memory.total / 1024**2),
            "temperature_c": pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU),
        }
    except Exception:
        pass
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass
    return result


@router.get("/features/stats")
def feature_stats(request: Request):
    return services(request).engine.feature_store.stats()


@router.get("/benchmarks")
def list_benchmarks(request: Request):
    return services(request).benchmark_registry.list()


@router.get("/runtime-benchmarks")
def list_runtime_benchmarks(request: Request):
    root = services(request).settings.benchmarks_dir
    reports = []
    for path in sorted(root.glob("run_*.json"), key=lambda item: item.stat().st_mtime, reverse=True):
        try:
            reports.append({"run_id": path.stem, **json.loads(path.read_text(encoding="utf-8"))})
        except (OSError, json.JSONDecodeError):
            continue
    return reports


@router.get("/benchmarks/{run_id}")
def get_benchmark(run_id: str, request: Request):
    if run_id.startswith("bmk_"):
        try:
            return services(request).benchmark_registry.get(run_id).model_dump(mode="json")
        except (KeyError, ValueError) as exc:
            raise HTTPException(404, "Unknown benchmark") from exc
    if not run_id.startswith("run_") or not run_id[4:].isalnum():
        raise HTTPException(404, "Unknown benchmark")
    path = services(request).settings.benchmarks_dir / f"{run_id}.json"
    if not path.is_file():
        raise HTTPException(404, "Unknown benchmark")
    return json.loads(path.read_text(encoding="utf-8"))


@router.post("/benchmarks", status_code=201)
def create_benchmark(payload: BenchmarkCreate, request: Request):
    try:
        return services(request).benchmark_registry.create(payload).model_dump(mode="json")
    except (KeyError, ValueError) as exc:
        raise HTTPException(422, redact_error(exc)) from exc


@router.get("/public-hub/providers")
def public_hub_providers(request: Request):
    return services(request).public_providers.list()


@router.post("/public-hub/providers/{provider}/test")
async def test_public_hub_provider(provider: str, request: Request):
    try:
        implementation = services(request).public_providers.get(provider)
        state = await asyncio.to_thread(implementation.test_auth)
        return {"provider": provider, "auth_state": state.value}
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc


@router.get("/public-hub/credentials")
def provider_credentials(request: Request):
    return services(request).credentials.status()


@router.put("/public-hub/credentials/{provider}")
async def save_provider_credentials(provider: str, payload: ProviderCredentialUpdate, request: Request):
    try:
        return await asyncio.to_thread(services(request).credentials.update, provider, payload)
    except ValueError as exc:
        raise HTTPException(422, redact_error(exc)) from exc


@router.delete("/public-hub/credentials/{provider}")
async def remove_provider_credentials(provider: str, request: Request):
    try:
        return await asyncio.to_thread(services(request).credentials.remove, provider)
    except ValueError as exc:
        raise HTTPException(404, redact_error(exc)) from exc


@router.get("/public-hub/datasets/search")
def search_public_datasets(request: Request, q: str = "", provider: str | None = None, page: int = 1, page_size: int = 20):
    try:
        return services(request).public_providers.search(ResourceKind.DATASET, q, provider, max(1, page), min(max(1, page_size), 100))
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc


@router.get("/public-hub/models/search")
def search_public_models(request: Request, q: str = "", provider: str | None = None, page: int = 1, page_size: int = 20):
    try:
        return services(request).public_providers.search(ResourceKind.MODEL, q, provider, max(1, page), min(max(1, page_size), 100))
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc


@router.get("/public-hub/dataset-preview")
async def public_dataset_preview(provider: str, resource_id: str, revision: str | None = None):
    if provider != "huggingface":
        raise HTTPException(404, "A generated preview is not available for this provider")
    if not resource_id or len(resource_id) > 512 or len(revision or "") > 256:
        raise HTTPException(422, "Invalid dataset preview identity")
    try:
        content = await asyncio.to_thread(_huggingface_dataset_preview, resource_id, revision)
    except Exception as exc:
        raise HTTPException(404, "Dataset sample preview is unavailable") from exc
    return Response(content=content, media_type="image/jpeg", headers={"Cache-Control": "private, max-age=3600", "X-Content-Type-Options": "nosniff"})


async def _resource_details(request: Request, provider: str, resource: str, kind: ResourceKind, version: str | None, revision: str | None):
    try:
        implementation = services(request).public_providers.get(provider)
        details = await asyncio.to_thread(implementation.details, ExternalResourceRef(provider=provider, kind=kind, resource_id=resource, version=version, revision=revision))
        return details.model_dump(mode="json")
    except ValueError as exc:
        raise HTTPException(422, redact_error(exc)) from exc


@router.get("/public-hub/datasets/{provider}/{resource:path}")
async def public_dataset_details(provider: str, resource: str, request: Request, version: str | None = None, revision: str | None = None):
    return await _resource_details(request, provider, resource, ResourceKind.DATASET, version, revision)


@router.get("/public-hub/models/{provider}/{resource:path}")
async def public_model_details(provider: str, resource: str, request: Request, version: str | None = None, revision: str | None = None):
    return await _resource_details(request, provider, resource, ResourceKind.MODEL, version, revision)


@router.post("/public-hub/downloads", status_code=202)
async def create_download(payload: ResourceDownloadRequest, request: Request):
    try:
        job = services(request).downloads.create(payload)
        return services(request).downloads.public(job)
    except ValueError as exc:
        raise HTTPException(422, redact_error(exc)) from exc


@router.get("/public-hub/downloads")
def list_downloads(request: Request):
    return services(request).downloads.list()


@router.get("/public-hub/downloads/events")
async def download_events(request: Request):
    async def stream():
        previous = ""
        while not await request.is_disconnected():
            value = json.dumps(services(request).downloads.list(), sort_keys=True)
            if value != previous:
                yield f"event: downloads\ndata: {value}\n\n"
                previous = value
            await asyncio.sleep(1)
    return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@router.get("/public-hub/downloads/{download_id}")
def get_download(download_id: str, request: Request):
    try:
        return services(request).downloads.public(services(request).downloads.get(download_id))
    except (KeyError, ValueError) as exc:
        raise HTTPException(404, "Unknown download") from exc


@router.post("/public-hub/downloads/{download_id}/cancel")
def cancel_download(download_id: str, request: Request):
    try:
        return services(request).downloads.public(services(request).downloads.cancel(download_id))
    except (KeyError, ValueError) as exc:
        raise HTTPException(404, "Unknown download") from exc


@router.post("/public-hub/downloads/{download_id}/retry", status_code=202)
def retry_download(download_id: str, request: Request):
    try:
        return services(request).downloads.public(services(request).downloads.retry(download_id))
    except KeyError as exc:
        raise HTTPException(404, "Unknown download") from exc
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc


@router.get("/assets/datasets")
def list_dataset_assets(request: Request):
    return [_public_dataset(raw) for raw in services(request).imported_assets.list_datasets()]


def _public_dataset(value: dict) -> dict:
    result = {key: item for key, item in value.items() if key != "local_root"}
    if isinstance(result.get("inspection"), dict):
        result["inspection"] = {key: item for key, item in result["inspection"].items() if key != "items"}
    return result


@router.get("/assets/datasets/{dataset_id}")
def get_dataset_asset(dataset_id: str, request: Request):
    try:
        return _public_dataset(services(request).imported_assets.get_dataset(dataset_id).model_dump(mode="json"))
    except (KeyError, ValueError) as exc:
        raise HTTPException(404, "Unknown imported dataset") from exc


@router.get("/assets/datasets/{dataset_id}/samples")
def dataset_samples(dataset_id: str, request: Request, limit: int = 4):
    try:
        asset = services(request).imported_assets.get_dataset(dataset_id)
    except (KeyError, ValueError) as exc:
        raise HTTPException(404, "Unknown imported dataset") from exc
    values = []
    for item in asset.inspection.items[:min(max(limit, 1), 12)]:
        values.append({"item_id": item["item_id"], "filename": Path(item["relative_path"]).name, "width": item.get("width"), "height": item.get("height"), "classes": item.get("classes", []), "url": f"/api/v1/assets/datasets/{dataset_id}/samples/{item['item_id']}"})
    return values


def _dataset_sample_path(request: Request, dataset_id: str, item_id: str) -> Path:
    try:
        asset = services(request).imported_assets.get_dataset(dataset_id)
    except (KeyError, ValueError) as exc:
        raise HTTPException(404, "Unknown imported dataset") from exc
    item = next((value for value in asset.inspection.items if value.get("item_id") == item_id), None)
    if item is None:
        raise HTTPException(404, "Unknown dataset sample")
    root = Path(asset.local_root).resolve(); source = (root / item["relative_path"]).resolve()
    if root not in source.parents or not source.is_file():
        raise HTTPException(404, "Dataset sample is unavailable")
    return source


@router.post("/assets/datasets/{dataset_id}/samples/{item_id}/import", status_code=201)
async def import_dataset_sample(dataset_id: str, item_id: str, request: Request):
    """Register one imported-dataset image as an ordinary image asset so a canvas Image node can use it."""
    source = _dataset_sample_path(request, dataset_id, item_id)
    try:
        record = await asyncio.to_thread(services(request).assets.import_path, source, "image")
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    return record.model_dump(mode="json")


@router.get("/assets/datasets/{dataset_id}/samples/{item_id}")
async def dataset_sample_image(dataset_id: str, item_id: str, request: Request):
    source = _dataset_sample_path(request, dataset_id, item_id)
    image = await asyncio.to_thread(cv2.imread, str(source))
    if image is None:
        raise HTTPException(422, "Dataset sample cannot be decoded")
    height, width = image.shape[:2]
    if width > 480:
        image = await asyncio.to_thread(cv2.resize, image, (480, max(1, round(height * 480 / width))))
    success, encoded = await asyncio.to_thread(cv2.imencode, ".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 82])
    if not success:
        raise HTTPException(500, "Dataset sample could not be rendered")
    return Response(content=encoded.tobytes(), media_type="image/jpeg", headers={"Cache-Control": "private, max-age=3600", "X-Content-Type-Options": "nosniff"})


@router.get("/assets/models")
def list_model_assets(request: Request):
    return [{key: value for key, value in raw.items() if key != "local_root"} for raw in services(request).imported_assets.list_models()]


@router.get("/assets/models/{model_artifact_id}")
def get_model_asset(model_artifact_id: str, request: Request):
    try:
        return services(request).imported_assets.get_model(model_artifact_id).model_dump(mode="json", exclude={"local_root"})
    except (KeyError, ValueError) as exc:
        raise HTTPException(404, "Unknown imported model") from exc


_ARTIFACT_LISTING_LIMIT = 200


def _model_root(request: Request, model_artifact_id: str):
    try:
        asset = services(request).imported_assets.get_model(model_artifact_id)
    except (KeyError, ValueError) as exc:
        raise HTTPException(404, "Unknown imported model") from exc
    return asset, Path(asset.local_root).resolve()


def _suggest_family(relative_path: str, families: list[str]) -> str | None:
    """Best family whose normalized name appears in the file's path (e.g. weights/YOLOv12-S/best.pt -> yolo)."""
    text = "".join(ch for ch in relative_path.lower() if ch.isalnum())
    matches = [family for family in families if "".join(ch for ch in family.lower() if ch.isalnum()) in text]
    return max(matches, key=len) if matches else None


@router.get("/assets/models/{model_artifact_id}/artifacts")
def list_model_artifact_files(model_artifact_id: str, request: Request):
    """Checkpoint files inside an imported model, with the built-in families that accept each one."""
    asset, root = _model_root(request, model_artifact_id)
    families = services(request).models.manifest.families
    extensions = {extension for family in families.values() for extension in family.extensions}
    registered = {item.relative_path: item for item in asset.registered_weights}
    files = []
    for path in sorted(root.rglob("*")) if root.is_dir() else [root]:
        relative = path.relative_to(root).as_posix() if root.is_dir() else path.name
        if not path.is_file() or path.suffix.lower() not in extensions or relative.startswith(".cache/") or "/." in f"/{relative}":
            continue
        compatible = sorted(name for name, family in families.items() if path.suffix.lower() in family.extensions)
        entry = registered.get(relative)
        files.append({"relative_path": relative, "size_bytes": path.stat().st_size, "families": compatible,
                      "suggested_family": entry.family if entry else _suggest_family(relative, compatible),
                      "weight_id": entry.weight_id if entry else None})
        if len(files) >= _ARTIFACT_LISTING_LIMIT:
            break
    return {"model_artifact_id": model_artifact_id, "files": files, "truncated": len(files) >= _ARTIFACT_LISTING_LIMIT}


@router.post("/assets/models/{model_artifact_id}/artifacts/register", status_code=201)
async def register_model_artifact_file(model_artifact_id: str, payload: ModelArtifactWeightRequest, request: Request):
    """Copy one explicitly chosen checkpoint into the weight store for a built-in adapter (no code is imported)."""
    asset, root = _model_root(request, model_artifact_id)
    source = (root / payload.relative_path).resolve() if root.is_dir() else root
    if (root.is_dir() and root not in source.parents) or not source.is_file():
        raise HTTPException(404, "Unknown artifact file")
    existing = next((item for item in asset.registered_weights if (item.relative_path, item.family) == (payload.relative_path, payload.family)), None)
    if existing is not None:
        return existing.model_dump(mode="json")
    try:
        def save():
            with source.open("rb") as handle:
                return services(request).weights.save(handle, source.name, payload.family)
        record = await asyncio.to_thread(save)
    except (KeyError, ValueError, ModelRegistryError) as exc:  # unknown family, or an extension the family does not accept
        raise HTTPException(422, str(exc)) from exc
    entry = RegisteredWeight(relative_path=payload.relative_path, family=payload.family, weight_id=record.weight_id)
    services(request).imported_assets.add_registered_weight(model_artifact_id, entry)
    return entry.model_dump(mode="json")


@router.post("/experiments", status_code=201)
def create_experiment(payload: ExperimentCreate, request: Request):
    try:
        return services(request).experiments.public(services(request).experiments.create(payload))
    except (KeyError, ValueError) as exc:
        raise HTTPException(422, str(exc)) from exc


@router.get("/experiments")
def list_experiments(request: Request):
    return services(request).experiments.list()


@router.get("/experiments/{experiment_id}")
def get_experiment(experiment_id: str, request: Request):
    try:
        return services(request).experiments.public(services(request).experiments.get(experiment_id))
    except (KeyError, ValueError) as exc:
        raise HTTPException(404, "Unknown experiment") from exc


@router.post("/experiments/{experiment_id}/start", status_code=202)
def start_experiment(experiment_id: str, request: Request, force: bool = False):
    try:
        return services(request).experiments.public(services(request).experiments.start(experiment_id, force))
    except (KeyError, ValueError) as exc:
        raise HTTPException(404, str(exc)) from exc


@router.post("/experiments/{experiment_id}/cancel")
def cancel_experiment(experiment_id: str, request: Request):
    try:
        return services(request).experiments.public(services(request).experiments.cancel(experiment_id))
    except (KeyError, ValueError) as exc:
        raise HTTPException(404, str(exc)) from exc


@router.get("/experiments/{experiment_id}/results")
def experiment_results(experiment_id: str, request: Request):
    try:
        return services(request).experiments.experiment_results(experiment_id)
    except (KeyError, ValueError) as exc:
        raise HTTPException(404, str(exc)) from exc


@router.get("/experiments/{experiment_id}/items/{item_id}")
def experiment_item_results(experiment_id: str, item_id: str, request: Request):
    try:
        return services(request).experiments.item_results(experiment_id, item_id)
    except (KeyError, ValueError) as exc:
        raise HTTPException(404, str(exc)) from exc


@router.post("/experiments/{experiment_id}/promote", status_code=201)
def promote_experiment(experiment_id: str, payload: PromoteRequest, request: Request):
    try:
        experiment = services(request).experiments.get(experiment_id)
        if experiment.status.value != "COMPLETED":
            raise ValueError("Only a completed experiment can be promoted")
        recipe = services(request).recipes.promote(experiment, payload)
        return services(request).recipes.public(recipe, include_workflow=True)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@router.get("/recipes")
def list_recipes(request: Request, include_non_promoted: bool = False):
    return services(request).recipes.list(promoted_only=not include_non_promoted)


@router.get("/recipes/{recipe_id}")
def get_recipe(recipe_id: str, request: Request, version: int | None = None):
    try:
        return services(request).recipes.public(services(request).recipes.get(recipe_id, version), include_workflow=True)
    except (KeyError, ValueError) as exc:
        raise HTTPException(404, "Unknown recipe version") from exc


def _allowed_recipe_path(path: Path, request: Request) -> bool:
    value = path.resolve()
    configured = services(request).settings.recipe_input_roots
    if any(value == root or root in value.parents for root in configured):
        return True
    temporary = Path(tempfile.gettempdir()).resolve()
    return temporary in value.parents and any(part.startswith("aa-preview-") for part in value.relative_to(temporary).parts[:-1])


@router.post("/recipes/execute")
async def execute_recipe(payload: dict, request: Request):
    recipe_value = payload.get("recipe", payload)
    recipe_id = recipe_value.get("recipe_id")
    version = recipe_value.get("recipe_version")
    try:
        recipe = services(request).recipes.get(str(recipe_id), int(version) if version is not None else None)
    except (KeyError, ValueError, TypeError) as exc:
        raise HTTPException(404, "Unknown promoted recipe version") from exc
    results = []
    for index, item in enumerate(payload.get("items", [])):
        asset_id = item.get("asset_id")
        local_path = item.get("local_path")
        if not asset_id and local_path:
            path = Path(local_path)
            if not _allowed_recipe_path(path, request) or not path.is_file():
                raise HTTPException(403, "Recipe input path is outside configured roots")
            try:
                asset_id = (await asyncio.to_thread(services(request).assets.import_path, path, "image")).asset_id
            except ValueError as exc:
                raise HTTPException(422, str(exc)) from exc
        if not asset_id:
            raise HTTPException(422, f"Item {index} requires asset_id or an allowed local_path")
        output = await services(request).experiments.execute_workflow_for_asset(recipe.workflow, str(asset_id))
        results.append({"item_id": item.get("item_id", str(index)), "image_ref": item.get("image_ref"), **output})
    return results


@router.get("/weights")
def list_weights(request: Request):
    return [item.model_dump(mode="json") for item in services(request).weights.list()]


@router.post("/weights", status_code=201)
def upload_weight(request: Request, family: str = Form(...), file: UploadFile = File(...)):
    try:
        record = services(request).weights.save(file.file, file.filename or "weight", family)
        return record.model_dump(mode="json")
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(400, str(exc)) from exc


@router.post("/models/load")
async def load_model(config: ModelConfig, request: Request):
    try:
        await services(request).manager.load(config)
        return {"model_id": services(request).manager.model_id(config), "state": "LOADED"}
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(400, str(exc)) from exc


@router.post("/models/unload")
async def unload_model(config: ModelConfig, request: Request):
    try:
        await services(request).manager.unload(config)
        return {"model_id": services(request).manager.model_id(config), "state": "UNLOADED"}
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(400, str(exc)) from exc


async def _upload_asset(request: Request, file: UploadFile, kind: str):
    try:
        record = services(request).assets.save(file.file, file.filename or kind, kind)
        return record.model_dump(mode="json")
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.post("/assets/image", status_code=201)
async def upload_image(request: Request, file: UploadFile = File(...)):
    return await _upload_asset(request, file, "image")


@router.post("/assets/video", status_code=201)
async def upload_video(request: Request, file: UploadFile = File(...)):
    return await _upload_asset(request, file, "video")


@router.post("/workflows/validate")
def validate_workflow(workflow: Workflow, request: Request):
    return services(request).validator.validate(workflow)


@router.post("/workflows/run", status_code=202)
async def run_workflow(workflow: Workflow, request: Request):
    try:
        execution = services(request).engine.start(workflow)
        return {"run_id": execution.run_id, "state": execution.state}
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@router.get("/workflows")
def list_workflows(request: Request):
    return services(request).workflow_store.list()


@router.post("/workflows", status_code=201)
def save_workflow(workflow: Workflow, request: Request):
    workflow_id = services(request).workflow_store.save(workflow)
    return {"workflow_id": workflow_id}


@router.get("/workflows/{workflow_id}")
def get_workflow(workflow_id: str, request: Request):
    try:
        return services(request).workflow_store.get(workflow_id)
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc


@router.post("/executions/{run_id}/pause")
async def pause_execution(run_id: str, request: Request):
    try:
        await services(request).engine.pause(run_id)
        return {"state": "PAUSED"}
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc


@router.post("/executions/{run_id}/resume")
async def resume_execution(run_id: str, request: Request):
    try:
        await services(request).engine.resume(run_id)
        return {"state": "RUNNING"}
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc


@router.post("/executions/{run_id}/stop")
async def stop_execution(run_id: str, request: Request):
    try:
        await services(request).engine.stop(run_id)
        return {"state": "STOPPING"}
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc


async def execution_socket(websocket: WebSocket, run_id: str, protocol: str | None = None) -> None:
    await websocket.accept(subprotocol=protocol)
    engine = websocket.app.state.services.engine
    try:
        async for event in engine.subscribe(run_id):
            await websocket.send_json(event)
    except (ValueError, WebSocketDisconnect) as exc:
        if isinstance(exc, ValueError):
            await websocket.send_json({"type": "error", "error": str(exc)})
    finally:
        try:
            await websocket.close()
        except RuntimeError:
            pass
