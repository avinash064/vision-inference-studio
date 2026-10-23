from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from inference_studio import __version__
from inference_studio.desktop_lifecycle import watch_parent_process
from inference_studio.adapters import AdapterRegistry
from inference_studio.api import execution_socket, router
from inference_studio.assets import AssetStore
from inference_studio.chainner import ChainnerNodeExecutor, ChainnerRegistryAdapter, ChainnerSettingsStore
from inference_studio.config import Settings
from inference_studio.features.geometry import MahalanobisGeometryStore
from inference_studio.graph.executor import ExecutionEngine
from inference_studio.graph.store import WorkflowStore
from inference_studio.graph.validator import WorkflowValidator
from inference_studio.models import ModelManager, ModelRegistry, WeightStore
from inference_studio.nodes import NodeRegistry
from inference_studio.security import DesktopSessionMiddleware, websocket_protocol
from inference_studio.public_hub.assets import ImportedAssetRegistry
from inference_studio.public_hub.benchmarks import BenchmarkRegistry
from inference_studio.public_hub.downloads import DownloadManager
from inference_studio.public_hub.credentials import ProviderCredentialStore
from inference_studio.public_hub.experiments import ExperimentRegistry
from inference_studio.public_hub.providers import HuggingFaceProvider, KaggleProvider, RoboflowProvider
from inference_studio.public_hub.recipes import RecipeRegistry
from inference_studio.public_hub.registry import ExternalProviderRegistry
from inference_studio.auto_annotation import AutoAnnotationManager
from inference_studio.auto_annotation.recipe_runner import RecipeExecutor
from inference_studio.auto_annotation.routes import router as auto_annotation_router
from inference_studio.auto_annotation.sources import SourceService


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
    chainner_settings: ChainnerSettingsStore | None = None
    geometries: MahalanobisGeometryStore | None = None
    public_providers: ExternalProviderRegistry | None = None
    imported_assets: ImportedAssetRegistry | None = None
    downloads: DownloadManager | None = None
    benchmark_registry: BenchmarkRegistry | None = None
    experiments: ExperimentRegistry | None = None
    recipes: RecipeRegistry | None = None
    credentials: ProviderCredentialStore | None = None
    auto_annotation: AutoAnnotationManager | None = None


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
    settings_path = (config.config_dir or config.models_file.parent) / "chainner-settings.json"
    chainner_settings = ChainnerSettingsStore(settings_path, chainner.registry if chainner is not None else None)
    chainner_executor = ChainnerNodeExecutor(chainner, config.workspace_dir / "chainner-storage") if chainner is not None else None
    geometries = MahalanobisGeometryStore(config.geometries_dir)
    engine = ExecutionEngine(validator, assets, manager, config.workspace_dir / "previews", config.benchmarks_dir, chainner=chainner_executor, settings_snapshot=chainner_settings.snapshot, feature_store_max_entries=config.feature_store_max_entries, feature_store_max_bytes=config.feature_store_max_bytes, geometries=geometries)
    providers = ExternalProviderRegistry([HuggingFaceProvider(), RoboflowProvider(), KaggleProvider()])
    imported_assets = ImportedAssetRegistry(config.public_hub_root / "metadata" / "assets")
    downloads = DownloadManager(config, providers, imported_assets, weights)
    benchmark_registry = BenchmarkRegistry(config.public_hub_root / "metadata" / "benchmarks", imported_assets)
    experiments = ExperimentRegistry(config.public_hub_root / "metadata", benchmark_registry, imported_assets, assets, engine)
    recipes = RecipeRegistry(config.public_hub_root / "metadata" / "recipes", benchmark_registry, weights)
    credentials = ProviderCredentialStore(Path(os.getenv("VIS_ENV_FILE", config.project_root / ".env")).expanduser(), providers)
    annotation_sources = SourceService(
        assets, config.auto_annotation_input_roots, config.workspace_dir / "auto-annotation-previews",
        # Desktop mode remembers files/folders the user picked in the native dialog (see /desktop/approve-path).
        approvals_file=(config.config_dir / "approved-input-roots.json") if os.getenv("VIS_DESKTOP_MODE") == "1" and config.config_dir else None,
    )
    recipe_executor = RecipeExecutor(recipes, experiments, validator, models, weights)
    auto_annotation = AutoAnnotationManager(config.auto_annotation_root, config.auto_annotation_output_root, annotation_sources, recipes, recipe_executor)
    app = FastAPI(title="Vision Inference Studio", version=__version__)
    app.state.services = Services(config, nodes, models, weights, assets, manager, validator, engine, WorkflowStore(config.workflows_dir), chainner, chainner_settings, geometries, providers, imported_assets, downloads, benchmark_registry, experiments, recipes, credentials, auto_annotation)
    app.add_middleware(CORSMiddleware, allow_origins=["null", "http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:5174", "http://127.0.0.1:5174"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
    app.add_middleware(DesktopSessionMiddleware)
    app.include_router(router)
    app.include_router(auto_annotation_router, prefix="/api/v1")
    app.mount("/previews", StaticFiles(directory=config.workspace_dir / "previews"), name="previews")
    app.mount("/auto-annotation-previews", StaticFiles(directory=config.workspace_dir / "auto-annotation-previews"), name="auto-annotation-previews")

    @app.get("/health")
    def health():
        return {"status": "ok", "version": __version__}

    @app.get("/")
    def root():
        return {"name": "Vision Inference Studio API", "version": __version__, "health": "/health", "docs": "/docs", "frontend": "http://127.0.0.1:5174"}

    async def recover_background_work():
        await downloads.recover()
        await experiments.recover()
        await auto_annotation.recover()

    app.add_event_handler("startup", recover_background_work)

    async def release_models():
        await manager.unload_all()

    app.add_event_handler("shutdown", release_models)

    @app.websocket("/api/v1/executions/{run_id}")
    async def websocket_route(websocket: WebSocket, run_id: str):
        protocol = websocket_protocol(websocket)
        if protocol == "":
            await websocket.close(code=1008, reason="Invalid desktop session")
            return
        await execution_socket(websocket, run_id, protocol)

    return app


app = create_app()
