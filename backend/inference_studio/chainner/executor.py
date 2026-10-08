from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from inference_studio.nodes.runtime import ImageValue

from .image import ChainnerImageAdapter


@dataclass
class ChainnerOutputs:
    values: dict[str, Any]


@dataclass
class SequenceValue:
    values: list[Any]


class ChainnerNodeExecutor:
    def __init__(self, registry: Any, storage_dir: Path) -> None:
        self.registry = registry
        self.storage_dir = storage_dir
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _raw(value: Any) -> Any:
        if isinstance(value, SequenceValue):
            return [ChainnerNodeExecutor._raw(item) for item in value.values]
        return ChainnerImageAdapter.to_chainner(value.image) if isinstance(value, ImageValue) else value

    @staticmethod
    def _wrap(value: Any) -> Any:
        return ImageValue(value) if isinstance(value, np.ndarray) and value.ndim in (2, 3) else value

    @classmethod
    def _wrap_sequence(cls, values: list[Any]) -> SequenceValue:
        return SequenceValue([cls._wrap(value) for value in values])

    @staticmethod
    def _select_value(input_value: Any, value: Any) -> Any:
        options = getattr(input_value, "options", [])
        if options:
            for option in options:
                if str(option.get("value")) == str(value):
                    return option.get("value")
        return value

    def _context(self, execution: Any, node_id: str, progress: Callable[[float], None]):
        from api import NodeContext, SettingsParser

        storage = self.storage_dir

        class Context(NodeContext):
            def __init__(self) -> None:
                self._node_cleanup: list[Callable[[], None]] = []
                self._chain_cleanup: list[Callable[[], None]] = []

            @property
            def aborted(self) -> bool:
                return execution.stop_requested

            @property
            def paused(self) -> bool:
                return execution.pause_requested

            @property
            def settings(self):
                return SettingsParser({})

            @property
            def storage_dir(self) -> Path:
                return storage

            def set_progress(self, value: float) -> None:
                self.check_aborted()
                progress(max(0.0, min(float(value), 1.0)))

            def add_cleanup(self, fn: Callable[[], None], after: str = "chain") -> None:
                (self._node_cleanup if after == "node" else self._chain_cleanup).append(fn)

        return Context()

    async def execute(self, schema_id: str, config: dict[str, Any], incoming: dict[str, Any], execution: Any, node_id: str, publish: Callable) -> Any:
        node = self.registry.get_node(schema_id)
        loop = asyncio.get_running_loop()

        def progress(value: float) -> None:
            loop.call_soon_threadsafe(asyncio.create_task, publish(execution, {"type": "node_progress", "event": "chainner_progress", "node_id": node_id, "progress": value}))

        context = self._context(execution, node_id, progress)

        def run() -> Any:
            raw_values: list[Any] = []
            for input_value in node.inputs:
                key = f"input_{input_value.id}"
                if key in incoming:
                    value = incoming[key]
                elif key in config:
                    value = self._select_value(input_value, config[key])
                elif hasattr(input_value, "default"):
                    value = input_value.default
                elif input_value.optional:
                    value = None
                else:
                    raise ValueError(f"{node.name}: input '{input_value.label}' is required")
                raw_values.append(value)

            def finish(outputs: list[Any], sequence: bool = False) -> Any:
                if len(outputs) != len(node.outputs):
                    raise RuntimeError(f"{node.name} returned {len(outputs)} outputs; expected {len(node.outputs)}")
                mapped = {}
                for output, value in zip(node.outputs, outputs, strict=True):
                    if sequence:
                        mapped[f"output_{output.id}"] = self._wrap_sequence([output.enforce(item) for item in value])
                    else:
                        mapped[f"output_{output.id}"] = self._wrap(output.enforce(value))
                return next(iter(mapped.values())) if len(mapped) == 1 else ChainnerOutputs(mapped)

            def call_regular(values: list[Any]) -> list[Any]:
                enforced = [input_value.enforce_(self._raw(value)) for input_value, value in zip(node.inputs, values, strict=True)]
                result = node.run(context, *enforced) if node.node_context else node.run(*enforced)
                return [] if not node.outputs else [result] if len(node.outputs) == 1 else list(result)

            try:
                if node.kind == "regularNode":
                    sequences = [value for value in raw_values if isinstance(value, SequenceValue)]
                    if not sequences:
                        return finish(call_regular(raw_values))
                    length = min(len(value.values) for value in sequences)
                    collected = [[] for _ in node.outputs]
                    for index in range(length):
                        context.check_aborted()
                        iteration = [value.values[index] if isinstance(value, SequenceValue) else value for value in raw_values]
                        for output_index, value in enumerate(call_regular(iteration)):
                            collected[output_index].append(value)
                        progress((index + 1) / max(length, 1))
                    return finish(collected, sequence=True)

                iterable_input_ids = {int(value) for info in node.iterable_inputs for value in info.inputs}
                values = []
                sequence_inputs: list[list[Any]] = []
                for input_value, value in zip(node.inputs, raw_values, strict=True):
                    if int(input_value.id) in iterable_input_ids:
                        if not isinstance(value, SequenceValue):
                            raise ValueError(f"{node.name}: '{input_value.label}' requires a sequence")
                        enforced_sequence = [input_value.enforce_(self._raw(item)) for item in value.values]
                        sequence_inputs.append(enforced_sequence)
                        values.append(None if node.kind == "collector" else enforced_sequence)
                    else:
                        values.append(input_value.enforce_(self._raw(value)))

                raw = node.run(context, *values) if node.node_context else node.run(*values)
                if node.kind == "collector":
                    length = min((len(value) for value in sequence_inputs), default=0)
                    for index in range(length):
                        context.check_aborted()
                        item = sequence_inputs[0][index] if len(sequence_inputs) == 1 else tuple(value[index] for value in sequence_inputs)
                        raw.on_iterate(item)
                        progress((index + 1) / max(length, 1))
                    completed = raw.on_complete()
                    outputs = [] if not node.outputs else [completed] if len(node.outputs) == 1 else list(completed)
                    return finish(outputs)

                if node.kind == "generator":
                    iterable_order = [int(value) for info in node.iterable_outputs for value in info.outputs]
                    iterable_ids = set(iterable_order)
                    if len(iterable_ids) == len(node.outputs):
                        generator, partial = raw, {}
                    else:
                        generator, *constants = raw
                        partial = {int(output.id): constants.pop(0) for output in node.outputs if int(output.id) not in iterable_ids}
                    collected = {output_id: [] for output_id in iterable_ids}
                    items = list(generator.supplier())
                    for index, item in enumerate(items):
                        if isinstance(item, Exception):
                            raise item
                        values_out = [item] if len(iterable_ids) == 1 else list(item)
                        for output_id, value in zip(iterable_order, values_out, strict=True):
                            collected[output_id].append(value)
                        progress((index + 1) / max(len(items), 1))
                    mapped = {f"output_{output.id}": self._wrap(output.enforce(partial[int(output.id)])) if int(output.id) in partial else self._wrap_sequence([output.enforce(value) for value in collected[int(output.id)]]) for output in node.outputs}
                    return next(iter(mapped.values())) if len(mapped) == 1 else ChainnerOutputs(mapped)

                if node.kind == "transformer":
                    items = list(raw.supplier())
                    collected = [[] for _ in node.outputs]
                    for index, item in enumerate(items):
                        if isinstance(item, Exception):
                            raise item
                        values_out = [item] if len(node.outputs) == 1 else list(item)
                        for output_index, value in enumerate(values_out):
                            collected[output_index].append(value)
                        progress((index + 1) / max(len(items), 1))
                    return finish(collected, sequence=True)

                raise RuntimeError(f"Unsupported chaiNNer node kind: {node.kind}")
            finally:
                for cleanup in context._node_cleanup:
                    cleanup()
                for cleanup in context._chain_cleanup:
                    cleanup()

        return await asyncio.to_thread(run)
