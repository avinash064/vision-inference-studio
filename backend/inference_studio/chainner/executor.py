from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from inference_studio.runtime import StreamValue

from .context import build_context
from .values import ChainnerValueAdapter


@dataclass
class ChainnerOutputs:
    values: dict[str, Any]


SequenceValue = StreamValue


class ChainnerNodeExecutor:
    def __init__(self, registry: Any, storage_dir: Path, settings: dict[str, Any] | None = None) -> None:
        self.registry, self.storage_dir = registry, storage_dir
        self.settings = dict(settings or {})
        self.values = ChainnerValueAdapter()
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _stream(execution: Any, supplier: Callable, length: int | None = None) -> StreamValue:
        stream = StreamValue(supplier, length=length)
        if execution.scope is not None:
            execution.scope.add_chain_cleanup(stream.close)
        return stream

    @staticmethod
    def _select_value(input_value: Any, value: Any) -> Any:
        for option in getattr(input_value, "options", []):
            if str(option.get("value")) == str(value):
                return option.get("value")
        return value

    def _context(self, execution: Any, node_id: str, schema_id: str, progress: Callable[[float], None]):
        package_id = self.registry.registry.get_package(schema_id).id
        return build_context(execution, node_id, package_id, self.storage_dir, progress)

    async def execute(self, schema_id: str, config: dict[str, Any], incoming: dict[str, Any], execution: Any, node_id: str, publish: Callable) -> Any:
        node = self.registry.get_node(schema_id)
        loop = asyncio.get_running_loop()

        def progress(value: float) -> None:
            loop.call_soon_threadsafe(asyncio.create_task, publish(execution, {"type": "node_progress", "event": "chainner_progress", "node_id": node_id, "progress": value}))

        def progress_count(processed: int, length: int | None) -> None:
            interval = max(1, length // 100) if length is not None else 100
            if processed % interval and processed != length:
                return
            event = {"type": "node_progress", "event": "chainner_progress", "node_id": node_id, "processed": processed}
            if length is not None:
                event["progress"] = processed / max(length, 1)
                event["length"] = length
            loop.call_soon_threadsafe(asyncio.create_task, publish(execution, event))

        context = self._context(execution, node_id, schema_id, progress)

        def run() -> Any:
            def check_control() -> None:
                if execution.pause_requested and getattr(execution.state, "value", execution.state) != "PAUSED":
                    state_type = type(execution.state)
                    execution.state = state_type("PAUSED")
                    loop.call_soon_threadsafe(asyncio.create_task, publish(execution, {"type": "run_state", "event": "workflow_paused", "state": execution.state}))
                while execution.pause_requested and not execution.stop_requested:
                    time.sleep(0.02)
                context.check_aborted()

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

            def enforce(input_value: Any, value: Any) -> Any:
                return input_value.enforce_(self.values.to_chainner(input_value, value))

            def mapped(outputs: list[Any]) -> Any:
                if len(outputs) != len(node.outputs):
                    raise RuntimeError(f"{node.name} returned {len(outputs)} outputs; expected {len(node.outputs)}")
                values = {f"output_{port.id}": self.values.to_vision(port, port.enforce(value)) for port, value in zip(node.outputs, outputs, strict=True)}
                return next(iter(values.values())) if len(values) == 1 else ChainnerOutputs(values)

            def call_regular(values: list[Any]) -> list[Any]:
                args = [enforce(port, value) for port, value in zip(node.inputs, values, strict=True)]
                result = node.run(context, *args) if node.node_context else node.run(*args)
                return [] if not node.outputs else [result] if len(node.outputs) == 1 else list(result)

            if node.kind == "regularNode":
                if not any(isinstance(value, StreamValue) for value in raw_values):
                    return mapped(call_regular(raw_values))

                stream_values = [value for value in raw_values if isinstance(value, StreamValue)]
                known_lengths = [value.length for value in stream_values]
                stream_length = min(known_lengths) if known_lengths and all(value is not None for value in known_lengths) else None

                def regular_supplier():
                    streams = [iter(value) for value in raw_values if isinstance(value, StreamValue)]
                    index = 0
                    while True:
                        check_control()
                        try:
                            items = [next(stream) for stream in streams]
                        except StopIteration:
                            return
                        item_iterator = iter(items)
                        values = [next(item_iterator) if isinstance(value, StreamValue) else value for value in raw_values]
                        outputs = call_regular(values)
                        yield outputs[0] if len(outputs) == 1 else tuple(outputs)
                        index += 1
                        progress_count(index, stream_length)

                result = {}
                for index, output in enumerate(node.outputs):
                    def supplier(i=index, port=output):
                        for item in regular_supplier():
                            value = item if len(node.outputs) == 1 else item[i]
                            yield self.values.to_vision(port, port.enforce(value))
                    result[f"output_{output.id}"] = self._stream(execution, supplier, stream_length)
                return next(iter(result.values())) if len(result) == 1 else ChainnerOutputs(result)

            iterable_input_ids = {int(value) for info in node.iterable_inputs for value in info.inputs}
            values, streams = [], []
            for input_value, value in zip(node.inputs, raw_values, strict=True):
                if int(input_value.id) in iterable_input_ids:
                    if not isinstance(value, StreamValue):
                        raise ValueError(f"{node.name}: '{input_value.label}' requires a stream")
                    stream = (enforce(input_value, item) for item in value)
                    streams.append(stream)
                    values.append(None if node.kind == "collector" else stream)
                else:
                    values.append(enforce(input_value, value))

            raw = node.run(context, *values) if node.node_context else node.run(*values)
            if node.kind == "collector":
                index = 0
                known_lengths = [value.length for value in raw_values if isinstance(value, StreamValue)]
                stream_length = min(known_lengths) if known_lengths and all(value is not None for value in known_lengths) else None
                for item in zip(*streams, strict=False):
                    check_control()
                    raw.on_iterate(item[0] if len(item) == 1 else item)
                    index += 1
                    progress_count(index, stream_length)
                completed = raw.on_complete()
                return mapped([] if not node.outputs else [completed] if len(node.outputs) == 1 else list(completed))

            if node.kind == "generator":
                iterable_order = [int(value) for info in node.iterable_outputs for value in info.outputs]
                iterable_ids = set(iterable_order)
                if len(iterable_ids) == len(node.outputs):
                    generator, constants = raw, []
                else:
                    generator, *constants = raw
                constant_iter = iter(constants)
                result: dict[str, Any] = {}
                for output in node.outputs:
                    output_id = int(output.id)
                    if output_id not in iterable_ids:
                        result[f"output_{output.id}"] = self.values.to_vision(output, output.enforce(next(constant_iter)))
                        continue
                    position = iterable_order.index(output_id)

                    def supplier(p=position, port=output):
                        for item in generator.supplier():
                            check_control()
                            if isinstance(item, Exception):
                                raise item
                            value = item if len(iterable_ids) == 1 else item[p]
                            yield self.values.to_vision(port, port.enforce(value))

                    result[f"output_{output.id}"] = self._stream(execution, supplier, getattr(generator, "expected_length", None))
                return next(iter(result.values())) if len(result) == 1 else ChainnerOutputs(result)

            if node.kind == "transformer":
                def transform_supplier():
                    for item in raw.supplier():
                        check_control()
                        if isinstance(item, Exception):
                            raise item
                        yield item

                result = {}
                for index, output in enumerate(node.outputs):
                    def supplier(i=index, port=output):
                        for item in transform_supplier():
                            value = item if len(node.outputs) == 1 else item[i]
                            yield self.values.to_vision(port, port.enforce(value))
                    result[f"output_{output.id}"] = self._stream(execution, supplier, getattr(raw, "expected_length", None))
                return next(iter(result.values())) if len(result) == 1 else ChainnerOutputs(result)

            raise RuntimeError(f"Unsupported chaiNNer node kind: {node.kind}")

        return await asyncio.to_thread(run)
