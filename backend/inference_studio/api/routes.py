from __future__ import annotations

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect

from inference_studio.adapters import ModelConfig
from inference_studio.graph import Workflow


router = APIRouter(prefix="/api/v1")


def services(request: Request):
    return request.app.state.services


@router.get("/nodes")
def list_nodes(request: Request):
    return [node.model_dump(mode="json") for node in services(request).nodes.list()]


@router.get("/models")
def list_models(request: Request):
    value = services(request)
    return {**value.models.public_data(), "loaded": [item.model_dump(mode="json") for item in value.manager.status()]}


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


async def execution_socket(websocket: WebSocket, run_id: str) -> None:
    await websocket.accept()
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

