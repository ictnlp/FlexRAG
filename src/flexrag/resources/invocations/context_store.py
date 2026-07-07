from __future__ import annotations

from collections.abc import AsyncIterator, Iterable
from typing import Any

from flexrag.common.dataclasses import Context


class ContextStoreInvocation:
    """Invocation semantics for managed context stores."""

    def __init__(self, runtime: Any) -> None:
        """Create a context-store invocation.

        :param runtime: Runtime adapter used to execute context-store calls.
        """
        self.runtime = runtime
        return

    def set_many(self, contexts: Iterable[Context]) -> None:
        """Store or replace multiple contexts."""
        self.runtime.call("set_many", contexts)
        return

    async def async_set_many(self, contexts: Iterable[Context]) -> None:
        """Asynchronously store or replace multiple contexts."""
        await self.runtime.acall("async_set_many", contexts)
        return

    def get(self, context_id: str) -> Context:
        """Return a context by id."""
        return self.runtime.call("get", context_id)

    async def async_get(self, context_id: str) -> Context:
        """Asynchronously return a context by id."""
        return await self.runtime.acall("async_get", context_id)

    def get_many(self, context_ids: Iterable[str]) -> list[Context]:
        """Return contexts for the requested ids."""
        return self.runtime.call("get_many", context_ids)

    async def async_get_many(self, context_ids: Iterable[str]) -> list[Context]:
        """Asynchronously return contexts for the requested ids."""
        return await self.runtime.acall("async_get_many", context_ids)

    def iter_contexts(self) -> Iterable[Context]:
        """Iterate over all stored contexts."""
        return self.runtime.call("iter_contexts")

    async def async_iter_contexts(self) -> AsyncIterator[Context]:
        """Asynchronously iterate over all stored contexts."""
        contexts = await self.runtime.acall("async_iter_contexts")
        async for context in contexts:
            yield context
        return

    @property
    def ids(self) -> list[str]:
        """Return all stored context ids."""
        return self.runtime.getattr("ids")

    async def async_ids(self) -> list[str]:
        """Asynchronously return all stored context ids."""
        return await self.runtime.acall("async_ids")

    def count(self) -> int:
        """Return the number of stored contexts."""
        return self.runtime.call("count")

    async def async_count(self) -> int:
        """Asynchronously return the number of stored contexts."""
        return await self.runtime.acall("async_count")

    def clear(self) -> None:
        """Delete all stored contexts without deleting artifacts."""
        self.runtime.call("clear")
        return

    async def async_clear(self) -> None:
        """Asynchronously delete all stored contexts."""
        await self.runtime.acall("async_clear")
        return
