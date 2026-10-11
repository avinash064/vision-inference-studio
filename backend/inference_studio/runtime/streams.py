from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any
import uuid


class StreamValue:
    """A lazy, cancellation-aware, single-pass runtime stream."""

    def __init__(self, supplier: Callable[[], Iterator[Any]], *, stream_id: str | None = None, item_type: Any = None, length: int | None = None, replayable: bool = False, close: Callable[[], None] | None = None) -> None:
        self.supplier, self.length, self.replayable = supplier, length, replayable
        self.stream_id, self.item_type = stream_id or f"stream_{uuid.uuid4().hex}", item_type
        self._close, self._consumed, self._closed = close, False, False
        self._iterator: Iterator[Any] | None = None

    def __iter__(self) -> Iterator[Any]:
        if self._closed:
            raise RuntimeError("stream_closed: stream is closed")
        if self._consumed and not self.replayable:
            raise RuntimeError("stream_closed: non-replayable stream has already been consumed")
        self._consumed = True
        self._iterator = iter(self.supplier())
        yield from self._iterator

    def close(self) -> None:
        if not self._closed and self._close is not None:
            self._close()
        if self._iterator is not None and hasattr(self._iterator, "close"):
            self._iterator.close()
        self._closed = True
