from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from flexrag.common.dataclasses import Context
from flexrag.retrievers.backends import Hit

from .base import TypedHandle


class CollectionBackendHandle(TypedHandle):
    """Typed proxy for collection backend resources.

    The handle forwards the formal collection backend contract to the target.
    It does not expose fake-only convenience search methods and does not own
    backend lifecycle.
    """

    @property
    def is_addable(self) -> bool:
        """Return whether the backend supports append-style additions."""
        return bool(self._target.getattr("is_addable"))

    @property
    def requires_context_store(self) -> bool:
        """Return whether the backend requires an external context store."""
        return bool(self._target.getattr("requires_context_store"))

    @property
    def view(self) -> Any:
        """Return the backend retrieval view."""
        return self._target.getattr("view")

    def add_contexts(self, contexts: Iterable[Context]) -> None:
        """Synchronously add contexts to the backend."""
        self._target.call("add_contexts", list(contexts))
        return

    async def async_add_contexts(self, contexts: Iterable[Context]) -> None:
        """Asynchronously add contexts to the backend."""
        await self._target.async_call("async_add_contexts", list(contexts))
        return

    def rebuild(self, contexts: Iterable[Context]) -> None:
        """Synchronously rebuild the backend from contexts."""
        self._target.call("rebuild", list(contexts))
        return

    async def async_rebuild(self, contexts: Iterable[Context]) -> None:
        """Asynchronously rebuild the backend from contexts."""
        await self._target.async_call("async_rebuild", list(contexts))
        return

    def search_hits(
        self,
        queries: list[Any],
        top_k: int,
        *,
        search_options: dict[str, Any] | None = None,
    ) -> list[list[Hit]]:
        """Synchronously search and return hit lists for queries."""
        return self._target.call(
            "search_hits",
            queries,
            top_k,
            search_options=search_options,
        )

    async def async_search_hits(
        self,
        queries: list[Any],
        top_k: int,
        *,
        search_options: dict[str, Any] | None = None,
    ) -> list[list[Hit]]:
        """Asynchronously search and return hit lists for queries."""
        return await self._target.async_call(
            "async_search_hits",
            queries,
            top_k,
            search_options=search_options,
        )

    def get_context(self, context_id: str) -> Context:
        """Synchronously hydrate one context by id."""
        return self._target.call("get_context", context_id)

    async def async_get_context(self, context_id: str) -> Context:
        """Asynchronously hydrate one context by id."""
        return await self._target.async_call("async_get_context", context_id)

    def count(self) -> int:
        """Synchronously count contexts represented by the backend."""
        return self._target.call("count")

    async def async_count(self) -> int:
        """Asynchronously count contexts represented by the backend."""
        return await self._target.async_call("async_count")

    def clear(self) -> None:
        """Synchronously clear backend state."""
        self._target.call("clear")
        return

    async def async_clear(self) -> None:
        """Asynchronously clear backend state."""
        await self._target.async_call("async_clear")
        return
