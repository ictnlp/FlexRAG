from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Coroutine, Protocol, TypeVar

from flexrag.common.dataclasses import Context

from ..view import RetrievalView

T = TypeVar("T")


@dataclass(frozen=True)
class Hit:
    """Lightweight backend search result before optional context hydration.

    :param context_id: Identifier of the matched context.
    :param score: Backend-local or merged score. Higher is better.
    :param backend: Retriever-owned backend name that produced this hit.
    :param view: Retrieval view name used by the backend.
    :param context: Optional native payload for backends that can hydrate hits.
    """

    context_id: str
    score: float
    backend: str
    view: str
    context: Context | None = field(default=None, compare=False, repr=False)


class CollectionBackend(Protocol):
    """Structural contract implemented by retriever collection backends.

    A backend stores and searches one persisted collection projection. It may
    own native payloads, or it may require an external context store for
    hydration and rebuilds.
    """

    requires_context_store: bool
    view: RetrievalView | None

    @property
    def is_addable(self) -> bool:
        """Return whether this backend can append contexts incrementally."""
        ...

    def add_contexts(self, contexts: Iterable[Context]) -> None:
        """Append contexts to an existing backend state.

        :param contexts: Context objects to append.
        :raises NotImplementedError: If the backend requires full-corpus rebuilds.
        """
        ...

    async def async_add_contexts(self, contexts: Iterable[Context]) -> None:
        """Asynchronously append contexts to an existing backend state.

        :param contexts: Context objects to append.
        :raises NotImplementedError: If the backend requires full-corpus rebuilds.
        """
        ...

    def rebuild(self, contexts: Iterable[Context]) -> None:
        """Rebuild the backend from a complete corpus.

        :param contexts: Complete context corpus to index.
        """
        ...

    async def async_rebuild(self, contexts: Iterable[Context]) -> None:
        """Asynchronously rebuild the backend from a complete corpus.

        :param contexts: Complete context corpus to index.
        """
        ...

    def search_hits(
        self,
        queries: list[Any],
        top_k: int,
        *,
        search_options: dict[str, Any] | None = None,
    ) -> list[list[Hit]]:
        """Search the backend and return lightweight hits.

        :param queries: Query objects accepted by the concrete backend.
        :param top_k: Maximum number of hits per query.
        :param search_options: Optional backend-specific search-time overrides.
        :returns: One hit list per input query.
        """
        ...

    async def async_search_hits(
        self,
        queries: list[Any],
        top_k: int,
        *,
        search_options: dict[str, Any] | None = None,
    ) -> list[list[Hit]]:
        """Asynchronously search the backend and return lightweight hits.

        :param queries: Query objects accepted by the concrete backend.
        :param top_k: Maximum number of hits per query.
        :param search_options: Optional backend-specific search-time overrides.
        :returns: One hit list per input query.
        """
        ...

    def clear(self) -> None:
        """Clear all backend artifacts and in-memory state."""
        ...

    async def async_clear(self) -> None:
        """Asynchronously clear all backend artifacts and in-memory state."""
        ...

    def get_context(self, context_id: str) -> Context:
        """Return a stored context if the backend owns native payloads.

        :param context_id: Context identifier to fetch.
        :raises KeyError: If the backend cannot hydrate the requested context.
        """
        ...

    async def async_get_context(self, context_id: str) -> Context:
        """Asynchronously return a stored context if native payloads exist.

        :param context_id: Context identifier to fetch.
        :raises KeyError: If the backend cannot hydrate the requested context.
        """
        ...

    def count(self) -> int:
        """Return the number of unique contexts indexed by this backend.

        :returns: Unique ``context_id`` count, not physical row count.
        """
        ...

    async def async_count(self) -> int:
        """Asynchronously return the number of unique indexed contexts.

        :returns: Unique ``context_id`` count, not physical row count.
        """
        ...

    def close(self) -> None:
        """Release backend resources."""
        ...

    async def async_close(self) -> None:
        """Asynchronously release backend resources."""
        ...


class CollectionBackendBase:
    """Shared state helpers for collection backends.

    :param view: Retrieval view bound to this backend, or ``None`` when the
        concrete backend can recover the view from existing artifacts.
    """

    requires_context_store: bool
    is_addable: bool = False

    def __init__(self, view: RetrievalView | None) -> None:
        """Initialize shared backend state.

        :param view: Runtime or persisted retrieval view for projection.
        """
        self.view = view
        return

    def _require_view(self) -> RetrievalView:
        if self.view is None:
            raise ValueError(
                f"{self.__class__.__name__} requires a RetrievalView. "
                "Pass view when creating a new backend or load it from existing artifacts."
            )
        return self.view

    def _load_persisted_view(self, payload: dict[str, Any] | None) -> None:
        if payload is None:
            return
        persisted = RetrievalView.from_dict(payload)
        if self.view is None:
            self.view = persisted
            return
        if self.view != persisted:
            raise ValueError(
                "RetrievalView does not match the persisted backend artifact view."
            )
        return


class SyncCollectionBackendBase(CollectionBackendBase, ABC):
    """Base class for sync-native collection backends.

    Subclasses implement synchronous core methods. The async methods are
    thread-backed compatibility bridges and do not make the backend natively
    asynchronous.
    """

    def add_contexts(self, contexts: Iterable[Context]) -> None:
        """Append contexts to an existing backend state.

        The default implementation means the backend is not incrementally
        addable.

        :param contexts: Context objects to append.
        :raises NotImplementedError: Always raised by the default implementation.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support incremental add; "
            "use rebuild() with the complete corpus instead."
        )

    async def async_add_contexts(self, contexts: Iterable[Context]) -> None:
        """Asynchronously append contexts through the sync implementation.

        :param contexts: Context objects to append.
        """
        await asyncio.to_thread(self.add_contexts, contexts)
        return

    @abstractmethod
    def rebuild(self, contexts: Iterable[Context]) -> None:
        """Rebuild the backend from a complete corpus.

        :param contexts: Complete context corpus to index.
        """
        raise NotImplementedError

    async def async_rebuild(self, contexts: Iterable[Context]) -> None:
        """Asynchronously rebuild through the sync implementation.

        :param contexts: Complete context corpus to index.
        """
        await asyncio.to_thread(self.rebuild, contexts)
        return

    @abstractmethod
    def search_hits(
        self,
        queries: list[Any],
        top_k: int,
        *,
        search_options: dict[str, Any] | None = None,
    ) -> list[list[Hit]]:
        """Search the backend and return lightweight hits.

        :param queries: Query objects accepted by the concrete backend.
        :param top_k: Maximum number of hits per query.
        :param search_options: Optional backend-specific search-time overrides.
        :returns: One hit list per input query.
        """
        raise NotImplementedError

    async def async_search_hits(
        self,
        queries: list[Any],
        top_k: int,
        *,
        search_options: dict[str, Any] | None = None,
    ) -> list[list[Hit]]:
        """Asynchronously search through the sync implementation.

        :param queries: Query objects accepted by the concrete backend.
        :param top_k: Maximum number of hits per query.
        :param search_options: Optional backend-specific search-time overrides.
        :returns: One hit list per input query.
        """
        return await asyncio.to_thread(
            self.search_hits,
            queries,
            top_k,
            search_options=search_options,
        )

    @abstractmethod
    def clear(self) -> None:
        """Clear backend artifacts and in-memory state."""
        raise NotImplementedError

    async def async_clear(self) -> None:
        """Asynchronously clear through the sync implementation."""
        await asyncio.to_thread(self.clear)
        return

    def get_context(self, context_id: str) -> Context:
        """Return a native payload context if the backend stores it.

        :param context_id: Context identifier to fetch.
        :raises KeyError: Always raised by the default implementation.
        """
        raise KeyError(context_id)

    async def async_get_context(self, context_id: str) -> Context:
        """Asynchronously fetch a native payload context.

        :param context_id: Context identifier to fetch.
        :raises KeyError: If the backend cannot hydrate the context.
        """
        return await asyncio.to_thread(self.get_context, context_id)

    @abstractmethod
    def count(self) -> int:
        """Return the number of unique indexed contexts.

        :returns: Unique ``context_id`` count, not physical row count.
        """
        raise NotImplementedError

    async def async_count(self) -> int:
        """Asynchronously count through the sync implementation.

        :returns: Unique ``context_id`` count.
        """
        return await asyncio.to_thread(self.count)

    def close(self) -> None:
        """Release backend resources.

        The default implementation is a no-op for backends without explicit
        resources.
        """
        return

    async def async_close(self) -> None:
        """Asynchronously release backend resources through ``close``."""
        await asyncio.to_thread(self.close)
        return


class AsyncCollectionBackendBase(CollectionBackendBase, ABC):
    """Base class for async-native collection backends.

    Subclasses implement asynchronous core methods. Synchronous methods bridge
    with ``asyncio.run`` and raise ``RuntimeError`` when called inside a running
    event loop.
    """

    def add_contexts(self, contexts: Iterable[Context]) -> None:
        """Synchronously append contexts through ``async_add_contexts``.

        :param contexts: Context objects to append.
        :raises RuntimeError: If called inside a running event loop.
        """
        self._run_coroutine_sync(self.async_add_contexts(contexts))
        return

    async def async_add_contexts(self, contexts: Iterable[Context]) -> None:
        """Append contexts to an existing backend state.

        The default implementation means the backend is not incrementally
        addable.

        :param contexts: Context objects to append.
        :raises NotImplementedError: Always raised by the default implementation.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support incremental add; "
            "use rebuild() with the complete corpus instead."
        )

    def rebuild(self, contexts: Iterable[Context]) -> None:
        """Synchronously rebuild through ``async_rebuild``.

        :param contexts: Complete context corpus to index.
        :raises RuntimeError: If called inside a running event loop.
        """
        self._run_coroutine_sync(self.async_rebuild(contexts))
        return

    @abstractmethod
    async def async_rebuild(self, contexts: Iterable[Context]) -> None:
        """Asynchronously rebuild the backend from a complete corpus.

        :param contexts: Complete context corpus to index.
        """
        raise NotImplementedError

    def search_hits(
        self,
        queries: list[Any],
        top_k: int,
        *,
        search_options: dict[str, Any] | None = None,
    ) -> list[list[Hit]]:
        """Synchronously search through ``async_search_hits``.

        :param queries: Query objects accepted by the concrete backend.
        :param top_k: Maximum number of hits per query.
        :param search_options: Optional backend-specific search-time overrides.
        :returns: One hit list per input query.
        :raises RuntimeError: If called inside a running event loop.
        """
        return self._run_coroutine_sync(
            self.async_search_hits(
                queries,
                top_k,
                search_options=search_options,
            )
        )

    @abstractmethod
    async def async_search_hits(
        self,
        queries: list[Any],
        top_k: int,
        *,
        search_options: dict[str, Any] | None = None,
    ) -> list[list[Hit]]:
        """Asynchronously search the backend and return lightweight hits.

        :param queries: Query objects accepted by the concrete backend.
        :param top_k: Maximum number of hits per query.
        :param search_options: Optional backend-specific search-time overrides.
        :returns: One hit list per input query.
        """
        raise NotImplementedError

    def clear(self) -> None:
        """Synchronously clear through ``async_clear``.

        :raises RuntimeError: If called inside a running event loop.
        """
        self._run_coroutine_sync(self.async_clear())
        return

    @abstractmethod
    async def async_clear(self) -> None:
        """Asynchronously clear backend artifacts and in-memory state."""
        raise NotImplementedError

    def get_context(self, context_id: str) -> Context:
        """Synchronously fetch a native payload context.

        :param context_id: Context identifier to fetch.
        :raises KeyError: If the backend cannot hydrate the context.
        :raises RuntimeError: If called inside a running event loop.
        """
        return self._run_coroutine_sync(self.async_get_context(context_id))

    async def async_get_context(self, context_id: str) -> Context:
        """Return a native payload context if the backend stores it.

        :param context_id: Context identifier to fetch.
        :raises KeyError: Always raised by the default implementation.
        """
        raise KeyError(context_id)

    def count(self) -> int:
        """Synchronously return the number of unique indexed contexts.

        :returns: Unique ``context_id`` count.
        :raises RuntimeError: If called inside a running event loop.
        """
        return self._run_coroutine_sync(self.async_count())

    @abstractmethod
    async def async_count(self) -> int:
        """Asynchronously return the number of unique indexed contexts.

        :returns: Unique ``context_id`` count, not physical row count.
        """
        raise NotImplementedError

    def close(self) -> None:
        """Synchronously release backend resources.

        :raises RuntimeError: If called inside a running event loop.
        """
        self._run_coroutine_sync(self.async_close())
        return

    async def async_close(self) -> None:
        """Asynchronously release backend resources.

        The default implementation is a no-op for backends without explicit
        resources.
        """
        return

    def _run_coroutine_sync(self, coroutine: Coroutine[Any, Any, T]) -> T:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coroutine)
        coroutine.close()
        raise RuntimeError(
            "Cannot call a synchronous backend method inside a running event loop; "
            "use the corresponding async_* method instead."
        )
