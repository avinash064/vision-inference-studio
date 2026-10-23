"""Cross-platform behaviour required by the Windows desktop build, testable on any OS.

Windows-only branches are exercised by flipping module-level IS_WINDOWS flags (never ``os.name``, which would
make pathlib build WindowsPath objects on Linux).
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import signal
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from inference_studio import desktop_lifecycle, runtime_check, runtime_diagnostics
from inference_studio.auto_annotation import manager as aa_manager
from inference_studio.auto_annotation.package import PackageBuilder
from inference_studio.auto_annotation.sources import SourceAccessError, SourceService
from inference_studio.config import Settings
from inference_studio.main import create_app
from inference_studio.portable_paths import is_absolute_any_platform, is_windows_safe_name, safe_filename, to_portable_relative, windows_unsafe_member
from inference_studio.public_hub import integrity
from inference_studio.public_hub.integrity import IntegrityError, safe_extract_archive

ROOT = Path(__file__).parents[2]


# -- portable paths ------------------------------------------------------------------------------------------

@pytest.mark.parametrize("value", ["/home/user/video.mp4", r"D:\AutoAnnotation\AA-1\images\a.jpg", "D:/AutoAnnotation/x.jpg", "C:relative.jpg", r"\\server\share\x.jpg", "//server/share/x.jpg"])
def test_absolute_paths_of_both_platforms_are_detected_everywhere(value):
    assert is_absolute_any_platform(value)


@pytest.mark.parametrize("value", ["images/frame_000001.jpg", "annotations/coco/coco.json", "gs://bucket/aa/AA-1", "s3://bucket/key", "image/jpeg", "1/30", "person", ""])
def test_portable_values_are_not_flagged(value):
    assert not is_absolute_any_platform(value)


def test_portable_relative_paths_use_posix_separators_with_spaces_and_unicode(tmp_path):
    root = tmp_path / "Vision Data" / "Camera 05"
    image = root / "images" / "frame ü 000001.jpg"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"x")
    assert to_portable_relative(image, root) == "images/frame ü 000001.jpg"
    with pytest.raises(ValueError):
        to_portable_relative(tmp_path / "outside.jpg", root)


@pytest.mark.parametrize("name,expected", [("a:b?.jpg", "a_b_.jpg"), ("CON", "CON_"), ("con.txt", "con_.txt"), ("COM1.log", "COM1_.log"), ("LPT9", "LPT9_"), ("trailing. ", "trailing"), ('x<y>|"z*.jpg', "x_y___z_.jpg"), ("Camera 05.mp4", "Camera 05.mp4"), ("ünïcödé.jpg", "ünïcödé.jpg")])
def test_safe_filename_replaces_windows_invalid_and_reserved_names(name, expected):
    assert safe_filename(name) == expected and is_windows_safe_name(safe_filename(name))


def test_safe_filename_bounds_length_and_keeps_extension():
    value = safe_filename("a" * 300 + ".jpg")
    assert len(value) <= 120 and value.endswith(".jpg")


def test_windows_unsafe_archive_members_are_identified():
    assert windows_unsafe_member("train/images/a:b.jpg") == "a:b.jpg"
    assert windows_unsafe_member("train/aux/x.jpg") == "aux"
    assert windows_unsafe_member("train/images/ok file.jpg") is None


# -- Auto Annotation package portability ---------------------------------------------------------------------

def test_package_validation_rejects_windows_absolute_paths_even_on_linux():
    assert PackageBuilder._contains_absolute_path({"source": {"relative_source_path": r"D:\CameraVideos\cam05.mp4"}})
    assert PackageBuilder._contains_absolute_path([{"file_name": "C:/x/images/a.jpg"}])
    assert not PackageBuilder._contains_absolute_path({"source": {"relative_source_path": "cam05.mp4", "frame_index": 12}})


def test_upload_commands_offer_powershell_and_bash_with_safe_quoting():
    root = r"D:\AutoAnnotation\it's here\AA-aa_123"
    value = aa_manager.upload_commands(root, "gcs", "gs://bucket/auto-annotation/", windows=True)
    assert value["shell"] == "powershell" and value["executes_upload"] is False
    assert value["command"] == "gcloud storage rsync -r `\n  'D:\\AutoAnnotation\\it''s here\\AA-aa_123' `\n  'gs://bucket/auto-annotation/AA-aa_123'"
    assert value["commands"]["bash"].startswith("gcloud storage rsync -r 'D:\\AutoAnnotation\\it'\"'\"'s here\\AA-aa_123'")
    linux = aa_manager.upload_commands("/data/Auto Annotation/AA-aa_1", "s3", "s3://b/prefix", windows=False)
    assert linux["shell"] == "bash" and linux["command"] == "aws s3 sync '/data/Auto Annotation/AA-aa_1' s3://b/prefix/AA-aa_1"


@pytest.mark.parametrize("destination", ["gs://bucket\nrm -rf /", "gs://bucket\rx", "gs://bu\x00cket", "s3://bucket"])
def test_upload_commands_reject_multiline_or_mismatched_destinations(destination):
    with pytest.raises(ValueError):
        aa_manager.upload_commands("/data/AA-1", "gcs", destination)


# -- desktop lifecycle ---------------------------------------------------------------------------------------

def test_windows_parent_check_never_calls_os_kill(monkeypatch):
    """os.kill(pid, 0) is TerminateProcess on Windows: it would kill the Electron shell."""
    monkeypatch.setattr(desktop_lifecycle, "IS_WINDOWS", True)
    monkeypatch.setattr(desktop_lifecycle.os, "kill", lambda *args: pytest.fail("os.kill must not be used on Windows"))
    psutil = pytest.importorskip("psutil")
    monkeypatch.setattr(psutil, "pid_exists", lambda pid: pid == 4242)
    monkeypatch.setattr(desktop_lifecycle, "process_create_time", lambda pid: 100.0)
    assert desktop_lifecycle.parent_alive(4242, 100.0)
    assert not desktop_lifecycle.parent_alive(4242, 250.0)  # PID reused by another process
    assert not desktop_lifecycle.parent_alive(9999, 100.0)


def test_windows_graceful_shutdown_raises_sigterm_in_process(monkeypatch):
    raised = []
    monkeypatch.setattr(desktop_lifecycle, "IS_WINDOWS", True)
    monkeypatch.setattr(desktop_lifecycle.signal, "raise_signal", raised.append)
    monkeypatch.setattr(desktop_lifecycle.os, "kill", lambda *args: pytest.fail("os.kill must not be used on Windows"))
    desktop_lifecycle.request_graceful_shutdown()
    assert raised == [signal.SIGTERM]


def test_watchdog_shuts_down_once_parent_is_gone():
    calls = []
    states = iter([True, False])
    thread = desktop_lifecycle.watch_parent_process(os.getpid() + 10_000, interval=0.01, alive=lambda pid, created: next(states), shutdown=lambda: calls.append("stop"))
    thread.join(timeout=2)
    assert calls == ["stop"]


# -- runtime diagnostics -------------------------------------------------------------------------------------

def test_dll_load_failures_get_actionable_messages():
    message = runtime_diagnostics.classify_import_error(ImportError("DLL load failed while importing _C: The specified module could not be found."))
    assert "Native library failed to load" in message and "Visual C++" in message
    assert "not installed" in runtime_diagnostics.classify_import_error(ModuleNotFoundError("No module named 'paddle'", name="paddle"))


def test_family_status_reports_missing_dependencies_and_tesseract_configuration(monkeypatch):
    monkeypatch.setattr(runtime_diagnostics, "probe_module", lambda name: {"ok": name != "groundingdino", "error": "Python package not installed (groundingdino)"})
    assert runtime_diagnostics.family_status("grounding-dino")["status"] == runtime_diagnostics.DEPENDENCY_MISSING
    assert runtime_diagnostics.family_status("yolo")["status"] == runtime_diagnostics.AVAILABLE
    monkeypatch.setattr(runtime_diagnostics, "resolve_binary", lambda name: None)
    assert runtime_diagnostics.family_status("tesseract")["status"] == runtime_diagnostics.CONFIGURATION_REQUIRED


def test_runtime_check_core_profile_and_optional_failures(monkeypatch):
    real_import = runtime_check.importlib.import_module
    monkeypatch.setattr(runtime_check.importlib, "import_module", lambda name: (_ for _ in ()).throw(OSError("DLL load failed")) if name == "paddle" else real_import(name))
    full = runtime_check.inspect_runtime("full")
    assert "paddle" in full["optional_failures"] and "paddle" not in full["failures"]
    assert full["packages"]["paddle"]["optional"] is True and "Native library" in full["packages"]["paddle"]["error"]
    core = runtime_check.inspect_runtime("core")
    assert set(core["packages"]) == set(runtime_check.CORE_IMPORTS) and core["ok"]


def _desktop_client(monkeypatch, tmp_path: Path, shell_token: str = "shell-secret"):
    monkeypatch.setenv("VIS_DESKTOP_MODE", "1")
    monkeypatch.setenv("VIS_API_TOKEN", "session-token")
    monkeypatch.setenv("VIS_SHELL_TOKEN", shell_token)
    monkeypatch.setenv("VIS_RUNTIME_SOURCE", "managed")
    settings = Settings(project_root=ROOT, workspace_dir=tmp_path / "workspace", weights_dir=tmp_path / "weights", workflows_dir=tmp_path / "workflows", benchmarks_dir=tmp_path / "benchmarks", models_file=ROOT / "config" / "models.yaml", config_dir=tmp_path / "config")
    return TestClient(create_app(settings))


def test_diagnostics_endpoint_hides_secrets_and_managed_paths(monkeypatch, tmp_path):
    monkeypatch.setattr(runtime_diagnostics, "probe_module", lambda name: {"ok": False, "error": "Python package not installed (torch)"})
    with _desktop_client(monkeypatch, tmp_path) as client:
        value = client.get("/api/v1/runtime/diagnostics", headers={"X-Vision-Session": "session-token"})
    assert value.status_code == 200
    body = value.json()
    assert body["runtime_source"] == "managed" and body["python_executable"] is None
    assert body["cuda_available"] is False and body["cuda_status"] == "CUDA UNAVAILABLE" and "PyTorch unavailable" in body["cuda_detail"]
    assert body["families"]["yolo"]["status"] == "DEPENDENCY_MISSING"
    assert "session-token" not in value.text and "shell-secret" not in value.text


def test_desktop_shutdown_requires_session_and_desktop_mode(monkeypatch, tmp_path):
    triggered = []
    monkeypatch.setattr(desktop_lifecycle, "request_graceful_shutdown", lambda: triggered.append(True))
    with _desktop_client(monkeypatch, tmp_path) as client:
        assert client.post("/api/v1/desktop/shutdown").status_code == 401
        assert client.post("/api/v1/desktop/shutdown", headers={"X-Vision-Session": "session-token"}).json() == {"state": "shutting_down"}


def test_only_the_desktop_shell_can_approve_source_paths(monkeypatch, tmp_path):
    video = tmp_path / "Vision Data" / "Camera 05" / "cam05 ü.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"not a real video")
    session = {"X-Vision-Session": "session-token"}
    with _desktop_client(monkeypatch, tmp_path) as client:
        assert client.post("/api/v1/desktop/approve-path", json={"path": str(video)}, headers=session).status_code == 403
        assert client.post("/api/v1/desktop/approve-path", json={"path": str(video)}, headers={**session, "X-Vision-Shell": "wrong"}).status_code == 403
        approved = client.post("/api/v1/desktop/approve-path", json={"path": str(video)}, headers={**session, "X-Vision-Shell": "shell-secret"})
    assert approved.status_code == 200 and approved.json()["approved"] == str(video.resolve())
    assert json.loads((tmp_path / "config" / "approved-input-roots.json").read_text(encoding="utf-8")) == [str(video.resolve())]


def test_source_approvals_persist_and_extend_the_allowlist(tmp_path):
    folder = tmp_path / "D Drive" / "Camera 05"
    folder.mkdir(parents=True)
    approvals = tmp_path / "config" / "approved-input-roots.json"
    service = SourceService(SimpleNamespace(), [tmp_path / "server-root"], tmp_path / "previews", approvals_file=approvals)
    with pytest.raises(SourceAccessError):
        service._allowed_path(str(folder), directory=True)
    service.approve(str(folder))
    reloaded = SourceService(SimpleNamespace(), [tmp_path / "server-root"], tmp_path / "previews", approvals_file=approvals)
    assert reloaded._allowed_path(str(folder), directory=True) == folder.resolve()
    server_only = SourceService(SimpleNamespace(), [tmp_path / "server-root"], tmp_path / "previews")
    with pytest.raises(SourceAccessError):
        server_only.approve(str(folder))  # approvals exist only in desktop mode


# -- archive extraction ----------------------------------------------------------------------------------------

def _zip(path: Path, member: str) -> Path:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(member, b"x")
    return path


def test_archive_members_with_drive_letters_are_rejected_on_every_platform(tmp_path):
    for member in ("C:/Windows/evil.dll", "C:\\Windows\\evil.dll", "D:relative.txt"):
        with pytest.raises(IntegrityError):
            safe_extract_archive(_zip(tmp_path / f"{abs(hash(member))}.zip", member), tmp_path / f"out{abs(hash(member))}", 10_000, 10_000)


def test_windows_invalid_archive_names_fail_clearly_on_windows_only(tmp_path, monkeypatch):
    archive = _zip(tmp_path / "names.zip", "train/images/frame:01.jpg")
    safe_extract_archive(archive, tmp_path / "linux-out", 10_000, 10_000)  # legal on Linux: behaviour unchanged
    monkeypatch.setattr(integrity, "IS_WINDOWS", True)
    with pytest.raises(IntegrityError, match="cannot be created on Windows"):
        safe_extract_archive(archive, tmp_path / "windows-out", 10_000, 10_000)


# -- managed runtime installer ---------------------------------------------------------------------------------

def _setup_runtime():
    spec = importlib.util.spec_from_file_location("setup_runtime", ROOT / "packaging" / "runtime" / "setup_runtime.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["setup_runtime"] = module
    spec.loader.exec_module(module)
    return module


def test_runtime_installer_hash_matches_the_desktop_shell_formula():
    setup = _setup_runtime()
    digest = hashlib.sha256()
    for name in ("requirements.txt", "requirements-runtimes.txt", "requirements-chainner-standard.txt", "requirements-desktop.txt", "requirements-lock.txt", "requirements-public-hub.txt", "pyproject.toml"):
        digest.update(f"{name}\n".encode() + (ROOT / name).read_bytes() + b"\n")
    assert setup.requirements_hash(ROOT) == digest.hexdigest()


def test_runtime_installer_finds_interpreters_in_both_layouts(tmp_path, monkeypatch):
    setup = _setup_runtime()
    (tmp_path / "linux" / "bin").mkdir(parents=True)
    (tmp_path / "linux" / "bin" / "python3.12").write_text("")
    (tmp_path / "linux" / "bin" / "python3.12-config").write_text("")
    assert setup.python_in(tmp_path / "linux").name == "python3.12"
    monkeypatch.setattr(setup, "IS_WINDOWS", True)
    assert setup.python_in(tmp_path / "win") == tmp_path / "win" / "python.exe"
    assert setup.runtime_python(tmp_path / "runtime") == tmp_path / "runtime" / "python" / "python.exe"


def test_runtime_installer_recovers_a_stale_lock(tmp_path):
    setup = _setup_runtime()
    args = setup.parse_args(["--mode", "install", "--data-root", str(tmp_path), "--app-root", str(ROOT)])
    installer = setup.Setup(args)
    installer.runtime_root.mkdir(parents=True)
    installer.lock.write_text("999999999")  # a PID that is not running
    installer.acquire_lock()
    assert installer.lock.read_text() == str(os.getpid())
    with pytest.raises(SystemExit, match="already running"):
        installer.acquire_lock()
