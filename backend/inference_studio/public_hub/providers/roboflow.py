from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import httpx

from ..base import DownloadedResource, ExternalProvider
from ..schemas import AuthState, ExternalResource, ExternalResourceRef, ProviderCapabilities, ResourceKind, SearchPage


class RoboflowProvider(ExternalProvider):
    name = "roboflow"
    display_name = "Roboflow Universe"
    capabilities = ProviderCapabilities(search_datasets=True, download_datasets=True, search_models=False, download_models=False, revisions=True, formats=True)

    @property
    def token(self) -> str | None:
        return os.getenv("ROBOFLOW_API_KEY")

    def auth_state(self) -> AuthState:
        return AuthState.CONFIGURED if self.token else AuthState.NOT_CONFIGURED

    def test_auth(self) -> AuthState:
        if not self.token:
            return AuthState.NOT_CONFIGURED
        try:
            from roboflow import rfapi
            rfapi.search_universe("test", api_key=self.token, limit=1)
            return AuthState.CONFIGURED
        except Exception:
            return AuthState.AUTH_FAILED

    def search(self, kind: ResourceKind, query: str, page: int, page_size: int) -> SearchPage:
        if kind != ResourceKind.DATASET:
            return SearchPage(items=[], page=page, page_size=page_size, has_more=False)
        from roboflow import rfapi
        raw = rfapi.search_universe(query, api_key=self.token, limit=page_size, page=page)
        values = raw.get("results", raw.get("projects", [])) if isinstance(raw, dict) else raw
        items = []
        for item in (values or [])[:page_size]:
            workspace_value = item.get("workspace", item.get("workspace_url", ""))
            workspace = str(workspace_value.get("url", "") if isinstance(workspace_value, dict) else workspace_value)
            source_url = str(item.get("url", ""))
            project = str(item.get("project", item.get("project_url", source_url.rstrip("/").split("/")[-1] if source_url else item.get("id", ""))))
            version = item.get("latestVersion", item.get("version", item.get("version_number")))
            rid = f"{workspace}/{project}".strip("/")
            preview = str(item.get("annotationThumbnail") or item.get("thumbnail") or "")
            if not preview.startswith("https://source.roboflow.com/"):
                preview = ""
            items.append(ExternalResource(
                ref=ExternalResourceRef(provider=self.name, kind=kind, resource_id=rid, version=str(version) if version else None),
                title=str(item.get("name", project)), owner=workspace or None,
                task=item.get("type", item.get("project_type")), classes=list(item.get("classes", []) or []),
                image_count=item.get("images", item.get("image_count")), license=item.get("license"), tags=list(item.get("tags", []) or []),
                formats=["coco", "yolov8", "voc"], source_url=source_url or f"https://universe.roboflow.com/{rid}",
                preview_url=preview or None,
            ))
        count = raw.get("count") if isinstance(raw, dict) else None
        return SearchPage(items=items, page=page, page_size=page_size, has_more=(page * page_size < count) if isinstance(count, int) else len(items) >= page_size)

    def details(self, ref: ExternalResourceRef) -> ExternalResource:
        if ref.kind != ResourceKind.DATASET:
            raise ValueError("Roboflow public model artifacts are not supported")
        parts = ref.resource_id.strip("/").split("/")
        if len(parts) != 2 or not ref.version:
            raise ValueError("Roboflow identity must be workspace/project plus a version")
        from roboflow import rfapi
        raw = rfapi.get_version(self.token or "", parts[0], parts[1], ref.version)
        data = raw if isinstance(raw, dict) else {}
        project = data.get("project", {}) or {}
        version_data = data.get("version", {}) or {}
        class_value = project.get("classes", version_data.get("classes", []))
        classes = list(class_value) if isinstance(class_value, dict) else list(class_value or [])
        formats = list(version_data.get("exports", []) or [])
        if "coco" not in formats:
            formats.insert(0, "coco")
        return ExternalResource(
            ref=ref, title=str(project.get("name", parts[1])), owner=parts[0], task=project.get("type"),
            classes=classes, image_count=version_data.get("images", project.get("images")),
            license=project.get("license"), formats=formats,
            source_url=f"https://universe.roboflow.com/{ref.resource_id}/dataset/{ref.version}", resolved_revision=ref.version,
            metadata={"version": ref.version, "splits": version_data.get("splits", {}), "preprocessing": version_data.get("preprocessing", {})},
        )

    def download(self, ref: ExternalResourceRef, destination: Path, requested_format: str | None, progress, cancelled) -> DownloadedResource:
        details = self.details(ref)
        parts = ref.resource_id.split("/")
        from roboflow import rfapi
        response, url = {}, None
        for _ in range(60):
            if cancelled():
                raise InterruptedError("Download cancelled")
            response = rfapi.get_version_export(self.token or "", parts[0], parts[1], ref.version or "", requested_format or "coco")
            url = response.get("export", {}).get("link") if isinstance(response, dict) else None
            url = url or (response.get("link") if isinstance(response, dict) else None)
            if url:
                break
            time.sleep(2)
        if not url:
            raise RuntimeError("Roboflow export did not become ready within 120 seconds")
        destination.mkdir(parents=True, exist_ok=True)
        target = destination / "roboflow-export.zip"
        total = 0
        with httpx.stream("GET", url, follow_redirects=True, timeout=120) as remote:
            remote.raise_for_status()
            expected = int(remote.headers.get("content-length", 0)) or None
            with target.open("xb") as output:
                for chunk in remote.iter_bytes(1024 * 1024):
                    if cancelled():
                        raise InterruptedError("Download cancelled")
                    output.write(chunk)
                    total += len(chunk)
                    progress(total, expected)
        return DownloadedResource(target, details.resolved_revision)
