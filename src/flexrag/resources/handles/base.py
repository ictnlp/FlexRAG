from __future__ import annotations

from typing import Any

from ..runtime import RuntimeTarget


class TypedHandle:
    """Base class for typed resource handles.

    A handle exposes a narrow resource interface over a ``RuntimeTarget``. It
    never holds the raw resource directly and deliberately does not expose
    lifecycle methods such as ``close`` or ``async_close``.
    """

    def __init__(
        self,
        target: RuntimeTarget,
        *,
        batching: bool = True,
    ) -> None:
        """Create a typed handle.

        :param target: Runtime target that executes raw resource calls.
        :param batching: Whether this interface supports batched public calls.
        """
        self._target = target
        self._batching = batching
        return

    @property
    def _effective_batch_size(self) -> int:
        if not self._batching:
            return 1
        return self._target.batch_size

    def _batches(self, items: list[Any]) -> list[list[Any]]:
        batch_size = self._effective_batch_size
        return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]
