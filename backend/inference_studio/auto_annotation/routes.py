from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .schemas import (
    AutoAnnotationRunCreate,
    InferencePreviewRequest,
    SamplingPreviewRequest,
    SourceInspectRequest,
)


router = APIRouter(prefix="/auto-annotation", tags=["auto-annotation"])


def manager(request: Request):
    return request.app.state.services.auto_annotation


def fail(exc: Exception, status: int = 422):
    raise HTTPException(status, str(exc)) from exc


@router.get("/recipes")
def recipes(request: Request):
    return request.app.state.services.recipes.list(promoted_only=True)


@router.post("/recipes/{recipe_id}/versions/{recipe_version}/preflight")
def preflight(recipe_id: str, recipe_version: int, request: Request):
    return (
        manager(request)
        .executor.preflight(recipe_id, recipe_version)
        .model_dump(mode="json")
    )


@router.post("/sources/inspect")
def inspect(payload: SourceInspectRequest, request: Request):
    try:
        return manager(request).inspect(payload)
    except (KeyError, ValueError) as exc:
        fail(exc)


@router.post("/previews/sampling")
def sampling_preview(payload: SamplingPreviewRequest, request: Request):
    try:
        return manager(request).sampling_preview(payload)
    except (KeyError, ValueError) as exc:
        fail(exc)


@router.post("/previews/inference")
async def inference_preview(payload: InferencePreviewRequest, request: Request):
    try:
        return await manager(request).inference_preview(payload)
    except (KeyError, ValueError, RuntimeError) as exc:
        fail(exc)


@router.post("/runs", status_code=202)
async def create_run(payload: AutoAnnotationRunCreate, request: Request):
    try:
        return manager(request).public(manager(request).create(payload))
    except (KeyError, ValueError, OSError) as exc:
        fail(exc)


@router.get("/runs")
def list_runs(request: Request):
    return manager(request).list()


@router.get("/runs/{run_id}")
def get_run(run_id: str, request: Request):
    try:
        return manager(request).detail(run_id)
    except (KeyError, ValueError) as exc:
        fail(exc, 404)


@router.get("/runs/{run_id}/review-items")
def review_items(
    run_id: str,
    request: Request,
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=100, ge=1, le=200),
    prediction_filter: str = Query(
        default="ALL", pattern="^(ALL|WITH_PREDICTIONS|WITHOUT_PREDICTIONS)$"
    ),
):
    try:
        return manager(request).review_items(
            run_id, offset, limit, prediction_filter
        )
    except KeyError as exc:
        fail(exc, 404)
    except ValueError as exc:
        fail(exc)


@router.get("/runs/{run_id}/items/{item_id}")
def review_item(run_id: str, item_id: str, request: Request):
    try:
        return manager(request).review_item(run_id, item_id)
    except KeyError as exc:
        fail(exc, 404)


@router.get("/runs/{run_id}/items/{item_id}/image")
def review_image(run_id: str, item_id: str, request: Request):
    try:
        return FileResponse(manager(request).review_image(run_id, item_id))
    except KeyError as exc:
        fail(exc, 404)
    except ValueError as exc:
        fail(exc, 404)


@router.post("/runs/{run_id}/cancel")
def cancel_run(run_id: str, request: Request):
    try:
        return manager(request).public(manager(request).cancel(run_id))
    except (KeyError, ValueError) as exc:
        fail(exc, 404)


@router.post("/runs/{run_id}/resume", status_code=202)
async def resume_run(run_id: str, request: Request):
    try:
        return manager(request).public(manager(request).resume(run_id))
    except KeyError as exc:
        fail(exc, 404)
    except ValueError as exc:
        fail(exc)


class UploadCommandRequest(BaseModel):
    provider: str = Field(pattern="^(gcs|s3)$")
    destination: str = Field(min_length=6, max_length=2048)


@router.post("/runs/{run_id}/upload-command")
def upload_command(run_id: str, payload: UploadCommandRequest, request: Request):
    try:
        return manager(request).upload_command(
            run_id, payload.provider, payload.destination
        )
    except KeyError as exc:
        fail(exc, 404)
    except ValueError as exc:
        fail(exc)
