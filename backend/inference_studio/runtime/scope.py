from __future__ import annotations

import inspect
from collections import defaultdict
from copy import deepcopy
from typing import Any, Callable

from inference_studio.features.store import FeatureStore


class ExecutionScope:
    def __init__(self, run_id: str, feature_store: FeatureStore, settings: dict[str, Any] | None = None) -> None:
        self.run_id, self.feature_store = run_id, feature_store
        self.settings = deepcopy(settings or {})
        self._node_cleanup: dict[str, list[Callable[[], Any]]] = defaultdict(list)
        self._chain_cleanup: list[Callable[[], Any]] = []
        self.closed = False

    def add_cleanup(self, fn: Callable[[], Any], *, after: str = "chain", node_id: str | None = None) -> None:
        if self.closed:
            raise RuntimeError("Execution scope is closed")
        if after == "node":
            if node_id is None:
                raise ValueError("node_id is required for node cleanup")
            self._node_cleanup[node_id].append(fn)
        else:
            self._chain_cleanup.append(fn)

    def add_node_cleanup(self, node_id: str, fn: Callable[[], Any]) -> None:
        self.add_cleanup(fn, after="node", node_id=node_id)

    def add_chain_cleanup(self, fn: Callable[[], Any]) -> None:
        self.add_cleanup(fn, after="chain")

    async def close_node(self, node_id: str) -> None:
        await self._run_lifo(self._node_cleanup.pop(node_id, []))

    async def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        first_error: Exception | None = None
        for node_id in list(self._node_cleanup):
            try:
                await self.close_node(node_id)
            except Exception as exc:
                first_error = first_error or exc
        try:
            await self._run_lifo(self._chain_cleanup)
        except Exception as exc:
            first_error = first_error or exc
        finally:
            self._chain_cleanup.clear()
            self.feature_store.release_run(self.run_id)
        if first_error is not None:
            raise first_error

    @staticmethod
    async def _run_lifo(callbacks: list[Callable[[], Any]]) -> None:
        first_error: Exception | None = None
        for callback in reversed(callbacks):
            try:
                result = callback()
                if inspect.isawaitable(result):
                    await result
            except Exception as exc:  # cleanup continues; original execution error wins
                first_error = first_error or exc
        if first_error is not None:
            raise first_error
