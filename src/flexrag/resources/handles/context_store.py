from __future__ import annotations

from collections.abc import AsyncIterator, Iterable

from flexrag.common.dataclasses import Context

from .base import TypedHandle


class ContextStoreHandle(TypedHandle):
    """Typed proxy for context store resources.

    The handle forwards the formal context store read/write contract. It does
    not expose store close methods; lifecycle remains with the manager target.
    """

    def set_many(self, contexts: Iterable[Context]) -> None:
        """Synchronously write multiple contexts."""
        self._target.call("set_many", list(contexts))
        return

    async def async_set_many(self, contexts: Iterable[Context]) -> None:
        """Asynchronously write multiple contexts."""
        await self._target.async_call("async_set_many", list(contexts))
        return

    def get(self, context_id: str) -> Context:
        """Synchronously read one context by id."""
        return self._target.call("get", context_id)

    async def async_get(self, context_id: str) -> Context:
        """Asynchronously read one context by id."""
        return await self._target.async_call("async_get", context_id)

    def get_many(self, context_ids: Iterable[str]) -> list[Context]:
        """Synchronously read contexts by id in request order."""
        return self._target.call("get_many", list(context_ids))

    async def async_get_many(self, context_ids: Iterable[str]) -> list[Context]:
        """Asynchronously read contexts by id in request order."""
        return await self._target.async_call("async_get_many", list(context_ids))

    def get_all(self) -> list[Context]:
        """Synchronously return a materialized context snapshot."""
        return self._target.call("get_all")

    async def async_get_all(self) -> list[Context]:
        """Asynchronously return a materialized context snapshot."""
        return await self._target.async_call("async_get_all")

    def iter_contexts(self) -> Iterable[Context]:
        """Synchronously iterate over a target-materialized snapshot."""
        return iter(self.get_all())

    async def async_iter_contexts(self) -> AsyncIterator[Context]:
        """Asynchronously iterate over a target-materialized snapshot."""
        for context in await self.async_get_all():
            yield context
        return

    @property
    def ids(self) -> list[str]:
        """Return stored context ids."""
        return self._target.getattr("ids")

    async def async_ids(self) -> list[str]:
        """Asynchronously return stored context ids."""
        return await self._target.async_call("async_ids")

    def count(self) -> int:
        """Synchronously count stored contexts."""
        return self._target.call("count")

    async def async_count(self) -> int:
        """Asynchronously count stored contexts."""
        return await self._target.async_call("async_count")

    def clear(self) -> None:
        """Synchronously clear all stored contexts."""
        self._target.call("clear")
        return

    async def async_clear(self) -> None:
        """Asynchronously clear all stored contexts."""
        await self._target.async_call("async_clear")
        return
