from __future__ import annotations

import asyncio
import sys
from types import SimpleNamespace

import pytest

from inference_studio.chainner.settings import ChainnerSettingsStore
from inference_studio.chainner.context import build_context
from inference_studio.runtime import ExecutionScope, FeatureStore, StreamValue


async def test_execution_scope_runs_node_and_chain_cleanup_lifo_on_close():
    calls = []
    scope = ExecutionScope("run_test", FeatureStore())
    scope.add_cleanup(lambda: calls.append("chain-1"))
    scope.add_cleanup(lambda: calls.append("node-1"), after="node", node_id="node")
    scope.add_cleanup(lambda: calls.append("node-2"), after="node", node_id="node")
    scope.add_cleanup(lambda: calls.append("chain-2"))
    await scope.close_node("node")
    await scope.close()
    await scope.close()
    assert calls == ["node-2", "node-1", "chain-2", "chain-1"]
    assert scope.closed


async def test_execution_scope_continues_cleanup_after_failure_and_cancel_path():
    calls = []
    scope = ExecutionScope("run_cancel", FeatureStore())
    scope.add_chain_cleanup(lambda: calls.append("last"))
    def failing():
        calls.append("fails")
        raise RuntimeError("cleanup failure")
    scope.add_chain_cleanup(failing)
    scope.add_chain_cleanup(lambda: calls.append("first"))
    with pytest.raises(RuntimeError, match="cleanup failure"):
        await scope.close()
    assert calls == ["first", "fails", "last"]
    assert scope.closed


def test_stream_is_lazy_single_pass_and_closable():
    produced, closed = [], []

    def supplier():
        for value in range(3):
            produced.append(value)
            yield value

    stream = StreamValue(supplier, length=3, close=lambda: closed.append(True))
    assert produced == []
    iterator = iter(stream)
    assert next(iterator) == 0 and produced == [0]
    assert list(iterator) == [1, 2]
    with pytest.raises(RuntimeError, match="already been consumed"):
        list(stream)
    stream.close()
    stream.close()
    assert closed == [True]


def test_chainner_settings_are_persisted_and_snapshots_are_isolated(tmp_path):
    store = ChainnerSettingsStore(tmp_path / "chainner.json")
    assert store.snapshot() == {}
    assert store.update({"use_cpu": True, "device": 2}) == {"use_cpu": True, "device": 2}
    snapshot = store.snapshot()
    snapshot["device"] = 9
    assert ChainnerSettingsStore(tmp_path / "chainner.json").snapshot()["device"] == 2
    scope = ExecutionScope("run", FeatureStore(), store.snapshot())
    store.update({"use_cpu": False, "device": 7})
    assert scope.settings["device"] == 2


def test_chainner_context_reads_its_package_snapshot(monkeypatch, tmp_path):
    class NodeContext:
        def check_aborted(self):
            if self.aborted:
                raise InterruptedError

    class SettingsParser:
        def __init__(self, raw): self.raw = raw
        def get_bool(self, key, default): return self.raw.get(key, default)

    monkeypatch.setitem(sys.modules, "api", SimpleNamespace(NodeContext=NodeContext, SettingsParser=SettingsParser))
    scope = ExecutionScope("run", FeatureStore(), {"chaiNNer_pytorch": {"use_cpu": True}})
    execution = SimpleNamespace(scope=scope, stop_requested=False, pause_requested=False)
    context = build_context(execution, "node", "chaiNNer_pytorch", tmp_path, lambda _: None)
    scope.settings["chaiNNer_pytorch"] = {"use_cpu": False}
    assert context.settings.get_bool("use_cpu", False) is True
