from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from flexrag.common.dataclasses import Context, RetrievedContext
from flexrag.retrievers.backends import Hit
from flexrag.retrievers.merge import MergeMethod

from .base import TypedHandle


class RetrieverHandle(TypedHandle):
    """Typed proxy for managed retriever resources.

    The handle exposes retriever data and orchestration operations over a fixed
    resource graph. Backend attachment, dependency access, and lifecycle
    methods remain outside the managed interface.
    """

    def add_contexts(self, contexts: Iterable[Context]) -> None:
        """Add a materialized context snapshot to the retriever.

        :param contexts: Contexts to store and index.
        """
        self._target.call("add_contexts", list(contexts))
        return

    async def async_add_contexts(self, contexts: Iterable[Context]) -> None:
        """Asynchronously add a materialized context snapshot.

        :param contexts: Contexts to store and index.
        """
        await self._target.async_call("async_add_contexts", list(contexts))
        return

    def rebuild(self, backend_name: str | None = None) -> None:
        """Rebuild one or all configured backends.

        :param backend_name: Optional backend name. ``None`` rebuilds all.
        """
        self._target.call("rebuild", backend_name)
        return

    async def async_rebuild(self, backend_name: str | None = None) -> None:
        """Asynchronously rebuild one or all configured backends.

        :param backend_name: Optional backend name. ``None`` rebuilds all.
        """
        await self._target.async_call("async_rebuild", backend_name)
        return

    def search_hits(
        self,
        queries: Any | Iterable[Any],
        *,
        top_k: int | None = None,
        used_backends: list[str] | None = None,
        candidate_k: int | None = None,
        merge_method: MergeMethod | None = None,
        backend_weights: Mapping[str, float] | None = None,
        backend_search_options: Mapping[str, dict[str, Any]] | None = None,
    ) -> list[list[Hit]]:
        """Search configured backends and return lightweight hits.

        :param queries: One query or an iterable of queries.
        :param top_k: Maximum hits per query.
        :param used_backends: Optional backend name subset.
        :param candidate_k: Per-backend candidate count before merging.
        :param merge_method: Optional merge method override.
        :param backend_weights: Optional merge weights by backend name.
        :param backend_search_options: Backend-specific search options.
        :returns: One hit list per normalized query.
        """
        return self._target.call(
            "search_hits",
            self._normalize_queries(queries),
            top_k=top_k,
            used_backends=used_backends,
            candidate_k=candidate_k,
            merge_method=merge_method,
            backend_weights=backend_weights,
            backend_search_options=backend_search_options,
        )

    async def async_search_hits(
        self,
        queries: Any | Iterable[Any],
        *,
        top_k: int | None = None,
        used_backends: list[str] | None = None,
        candidate_k: int | None = None,
        merge_method: MergeMethod | None = None,
        backend_weights: Mapping[str, float] | None = None,
        backend_search_options: Mapping[str, dict[str, Any]] | None = None,
    ) -> list[list[Hit]]:
        """Asynchronously search configured backends for hits.

        :param queries: One query or an iterable of queries.
        :param top_k: Maximum hits per query.
        :param used_backends: Optional backend name subset.
        :param candidate_k: Per-backend candidate count before merging.
        :param merge_method: Optional merge method override.
        :param backend_weights: Optional merge weights by backend name.
        :param backend_search_options: Backend-specific search options.
        :returns: One hit list per normalized query.
        """
        return await self._target.async_call(
            "async_search_hits",
            self._normalize_queries(queries),
            top_k=top_k,
            used_backends=used_backends,
            candidate_k=candidate_k,
            merge_method=merge_method,
            backend_weights=backend_weights,
            backend_search_options=backend_search_options,
        )

    def search(
        self,
        queries: Any | Iterable[Any],
        *,
        top_k: int | None = None,
        used_backends: list[str] | None = None,
        candidate_k: int | None = None,
        merge_method: MergeMethod | None = None,
        backend_weights: Mapping[str, float] | None = None,
        backend_search_options: Mapping[str, dict[str, Any]] | None = None,
    ) -> list[list[RetrievedContext]]:
        """Search and hydrate retrieved contexts.

        :param queries: One query or an iterable of queries.
        :param top_k: Maximum results per query.
        :param used_backends: Optional backend name subset.
        :param candidate_k: Per-backend candidate count before merging.
        :param merge_method: Optional merge method override.
        :param backend_weights: Optional merge weights by backend name.
        :param backend_search_options: Backend-specific search options.
        :returns: One retrieved-context list per normalized query.
        """
        return self._target.call(
            "search",
            self._normalize_queries(queries),
            top_k=top_k,
            used_backends=used_backends,
            candidate_k=candidate_k,
            merge_method=merge_method,
            backend_weights=backend_weights,
            backend_search_options=backend_search_options,
        )

    async def async_search(
        self,
        queries: Any | Iterable[Any],
        *,
        top_k: int | None = None,
        used_backends: list[str] | None = None,
        candidate_k: int | None = None,
        merge_method: MergeMethod | None = None,
        backend_weights: Mapping[str, float] | None = None,
        backend_search_options: Mapping[str, dict[str, Any]] | None = None,
    ) -> list[list[RetrievedContext]]:
        """Asynchronously search and hydrate retrieved contexts.

        :param queries: One query or an iterable of queries.
        :param top_k: Maximum results per query.
        :param used_backends: Optional backend name subset.
        :param candidate_k: Per-backend candidate count before merging.
        :param merge_method: Optional merge method override.
        :param backend_weights: Optional merge weights by backend name.
        :param backend_search_options: Backend-specific search options.
        :returns: One retrieved-context list per normalized query.
        """
        return await self._target.async_call(
            "async_search",
            self._normalize_queries(queries),
            top_k=top_k,
            used_backends=used_backends,
            candidate_k=candidate_k,
            merge_method=merge_method,
            backend_weights=backend_weights,
            backend_search_options=backend_search_options,
        )

    def get(self, context_id: str) -> Context:
        """Return one complete context by id.

        :param context_id: Context identifier to fetch.
        :returns: Complete context.
        """
        return self._target.call("get", context_id)

    async def async_get(self, context_id: str) -> Context:
        """Asynchronously return one complete context by id.

        :param context_id: Context identifier to fetch.
        :returns: Complete context.
        """
        return await self._target.async_call("async_get", context_id)

    def clear(self) -> None:
        """Clear the context store and every configured backend."""
        self._target.call("clear")
        return

    async def async_clear(self) -> None:
        """Asynchronously clear all retriever-managed data."""
        await self._target.async_call("async_clear")
        return

    def count(self) -> int:
        """Return the authoritative context count."""
        return self._target.call("count")

    async def async_count(self) -> int:
        """Asynchronously return the authoritative context count."""
        return await self._target.async_call("async_count")

    def list_backends(self) -> list[str]:
        """Return configured backend names in retriever order."""
        return self._target.call("list_backends")

    async def async_list_backends(self) -> list[str]:
        """Asynchronously return configured backend names in retriever order."""
        return await self._target.async_call("list_backends")

    @staticmethod
    def _normalize_queries(queries: Any | Iterable[Any]) -> list[Any]:
        if isinstance(queries, str) or isinstance(queries, Mapping):
            return [queries]
        if isinstance(queries, Iterable):
            return list(queries)
        return [queries]
