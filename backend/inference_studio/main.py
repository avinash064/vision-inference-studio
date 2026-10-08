from __future__ import annotations

from dataclasses import dataclass

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from inference_studio.adapters import AdapterRegistry
from inference_studio.api import execution_socket, router
from inference_studio.assets import AssetStore
from inference_studio.config import Settings
from inference_studio.graph.executor import ExecutionEngine
from inference_studio.graph.store import WorkflowStore
from inference_studio.graph.validator import WorkflowValidator
from inference_studio.models import ModelManager, ModelRegistry, WeightStore
from inference_studio.nodes import NodeRegistry
from inference_studio.security import DesktopSessionMiddleware, websocket_protocol


@dataclass
class Services:
    settings: Settings
    nodes: NodeRegistry
    models: ModelRegistry
    weights: WeightStore
    assets: AssetStore
    manager: ModelManager
    validator: WorkflowValidator
    engine: ExecutionEngine
    workflow_store: WorkflowStore


def create_app(settings: Settings | None = None) -> FastAPI:
    config = settings or Settings.load()
    config.prepare()
    models = ModelRegistry.load(config.models_file)
    weights = WeightStore(config.weights_dir, models, config.max_upload_bytes)
    assets = AssetStore(config.workspace_dir / "assets", config.max_upload_bytes)
    nodes = NodeRegistry()
    manager = ModelManager(models, weights, AdapterRegistry())
    validator = WorkflowValidator(nodes, models, weights, assets)
    engine = ExecutionEngine(validator, assets, manager, config.workspace_dir / "previews", config.benchmarks_dir)
    app = FastAPI(title="Vision Inference Studio", version="0.3.0")
    app.state.services = Services(config, nodes, models, weights, assets, manager, validator, engine, WorkflowStore(config.workflows_dir))
    app.add_middleware(CORSMiddleware, allow_origins=["null", "http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:5174", "http://127.0.0.1:5174"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
    app.add_middleware(DesktopSessionMiddleware)
    app.include_router(router)
    app.mount("/previews", StaticFiles(directory=config.workspace_dir / "previews"), name="previews")

    @app.get("/health")
    def health():
        return {"status": "ok", "version": "0.3.0"}

    @app.websocket("/api/v1/executions/{run_id}")
    async def websocket_route(websocket: WebSocket, run_id: str):
        protocol = websocket_protocol(websocket)
        if protocol == "":
            await websocket.close(code=1008, reason="Invalid desktop session")
            return
        await execution_socket(websocket, run_id, protocol)

    return app


app = create_app()
