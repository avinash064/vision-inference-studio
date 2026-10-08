from __future__ import annotations

import os
import signal
import threading
import time
from dataclasses import dataclass

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from inference_studio.adapters import AdapterRegistry
from inference_studio.api import execution_socket, router
from inference_studio.assets import AssetStore
from inference_studio.chainner import ChainnerNodeExecutor, ChainnerRegistryAdapter
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
    chainner: ChainnerRegistryAdapter | None = None


def watch_parent_process(parent_pid: int, interval: float = 2.0) -> threading.Thread:
    """Shut down when the desktop shell that started this backend is gone.

    The shell stops the backend on a clean quit, but a crash, SIGKILL or a signal handled by Chromium skips
    that, leaving an orphaned backend holding loaded models (GBs of GPU memory)."""

    # When the shell spawned us directly, being re-parented also means it died (immune to zombies/PID reuse).
    direct_child = os.getppid() == parent_pid

    def run() -> None:
        while True:
            time.sleep(interval)
            try:
                os.kill(parent_pid, 0)
                gone = direct_child and os.getppid() != parent_pid
            except ProcessLookupError:
                gone = True
            except PermissionError:
                gone = False
            if gone:
                os.kill(os.getpid(), signal.SIGTERM)  # graceful uvicorn shutdown releases the GPU
                return

    thread = threading.Thread(target=run, name="parent-watchdog", daemon=True)
    thread.start()
    return thread


def create_app(settings: Settings | None = None) -> FastAPI:
    parent_pid = os.getenv("VIS_PARENT_PID")
    if parent_pid and parent_pid.isdigit():
        watch_parent_process(int(parent_pid))
    config = settings or Settings.load()
    config.prepare()
    models = ModelRegistry.load(config.models_file)
    weights = WeightStore(config.weights_dir, models, config.max_upload_bytes)
    assets = AssetStore(config.workspace_dir / "assets", config.max_upload_bytes)
    chainner = None
    imported_nodes = []
    if config.chainner_src_dir is not None:
        try:
            chainner = ChainnerRegistryAdapter(config.chainner_src_dir, (config.reports_dir or config.workspace_dir / "reports") / "chainner-node-import.json")
            imported_nodes = chainner.discover_nodes()
        except Exception as exc:
            import logging
            logging.getLogger(__name__).exception("chaiNNer discovery failed: %s", exc)
            chainner = None
    nodes = NodeRegistry(imported=imported_nodes)
    manager = ModelManager(models, weights, AdapterRegistry())
    validator = WorkflowValidator(nodes, models, weights, assets)
    chainner_executor = ChainnerNodeExecutor(chainner, config.workspace_dir / "chainner-storage") if chainner is not None else None
    engine = ExecutionEngine(validator, assets, manager, config.workspace_dir / "previews", config.benchmarks_dir, chainner=chainner_executor)
    app = FastAPI(title="Vision Inference Studio", version="0.4.0")
    app.state.services = Services(config, nodes, models, weights, assets, manager, validator, engine, WorkflowStore(config.workflows_dir), chainner)
    app.add_middleware(CORSMiddleware, allow_origins=["null", "http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:5174", "http://127.0.0.1:5174"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
    app.add_middleware(DesktopSessionMiddleware)
    app.include_router(router)
    app.mount("/previews", StaticFiles(directory=config.workspace_dir / "previews"), name="previews")

    @app.get("/health")
    def health():
        return {"status": "ok", "version": "0.4.0"}

    @app.websocket("/api/v1/executions/{run_id}")
    async def websocket_route(websocket: WebSocket, run_id: str):
        protocol = websocket_protocol(websocket)
        if protocol == "":
            await websocket.close(code=1008, reason="Invalid desktop session")
            return
        await execution_socket(websocket, run_id, protocol)

    return app


app = create_app()
