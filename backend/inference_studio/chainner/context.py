from __future__ import annotations

from pathlib import Path
from typing import Any, Callable


def build_context(execution: Any, node_id: str, package_id: str, storage_dir: Path, progress: Callable[[float], None]):
    """Build chaiNNer's NodeContext after its local API has been imported."""

    from api import NodeContext, SettingsParser

    scope = execution.scope
    package_settings = scope.settings.get(package_id, {}) if scope is not None else {}

    class ChainnerContext(NodeContext):
        @property
        def aborted(self) -> bool:
            return execution.stop_requested

        @property
        def paused(self) -> bool:
            return execution.pause_requested

        @property
        def settings(self):
            return SettingsParser(package_settings)

        @property
        def storage_dir(self) -> Path:
            return storage_dir

        def set_progress(self, value: float) -> None:
            self.check_aborted()
            progress(max(0.0, min(float(value), 1.0)))

        def add_cleanup(self, fn: Callable[[], None], after: str = "chain") -> None:
            if scope is None:
                raise RuntimeError("chaiNNer cleanup requires an execution scope")
            if after == "node":
                scope.add_node_cleanup(node_id, fn)
            else:
                scope.add_chain_cleanup(fn)

    return ChainnerContext()
