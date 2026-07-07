from __future__ import annotations

import asyncio
from collections.abc import Iterable, Mapping
from typing import Annotated, Any

from flexrag.common import Choices, configure
from flexrag.common.dataclasses import Context, RetrievedContext

from .backends.base import CollectionBackend, Hit
from .context_store import ContextStoreProtocol
from .merge import (
    MergeMethod,
    merge_hits,
    normalize_backend_weights,
    normalize_merge_method,
)
from .utils import _iter_batches


@configure
class FlexRetrieverConfig:
    """Runtime configuration for ``FlexRetriever`` orchestration.

    :param batch_size: Orchestration batch size for adding contexts.
    :param default_top_k: Default number of results per query.
    :param default_merge_method: Default multi-backend hit merge method.
    :param rrf_base: Denominator base used by RRF merge.
    """

    batch_size: int = 32
    default_top_k: int = 10
    default_merge_method: Annotated[str, Choices("rrf", "linear")] = "rrf"
    rrf_base: int = 60


class FlexRetriever:
    """Runtime coordinator for collection backends and optional context store.

    The retriever owns backend names and search orchestration. It does not own a
    collection-level artifact manifest or the lifecycle of injected backends and
    context stores. When a context store is present, it is the authoritative
    source for hydration, counting, and rebuild backfill.
    """

    def __init__(
        self,
        *,
        backends: Mapping[str, CollectionBackend],
        config: FlexRetrieverConfig | None = None,
        context_store: ContextStoreProtocol | None = None,
    ) -> None:
        """Create a retriever from preconstructed backends.

        :param backends: Mapping from stable backend name to backend instance.
        :param config: Optional orchestration configuration.
        :param context_store: Optional complete context store.
        :raises ValueError: If a backend requires a context store and none is
            provided, or if backend names are invalid.
        """
        self.config = config or FlexRetrieverConfig()
        self.backends = self._normalize_backends(backends)
        self.context_store = context_store
        if self.context_store is None and any(
            backend.requires_context_store for backend in self.backends.values()
        ):
            raise ValueError(
                "At least one backend requires context_store; pass context_store "
                "explicitly."
            )
        for name, backend in self.backends.items():
            self._prepare_backend(name, backend)
        return

    @classmethod
    def from_backends(
        cls,
        backends: Mapping[str, CollectionBackend],
        *,
        context_store: ContextStoreProtocol | None = None,
        config: FlexRetrieverConfig | None = None,
    ) -> "FlexRetriever":
        """Construct a retriever from existing backend instances.

        :param backends: Mapping from backend name to backend instance.
        :param context_store: Optional complete context store.
        :param config: Optional orchestration configuration.
        :returns: Initialized retriever.
        """
        return cls(
            backends=backends,
            config=config,
            context_store=context_store,
        )

    def add_contexts(self, contexts: Iterable[Context]) -> None:
        """Add contexts and eagerly update all backends.

        Contexts are consumed in ``config.batch_size`` batches. Addable backends
        receive each new batch. Non-addable backends are rebuilt once from the
        context store after all input is consumed.

        :param contexts: Context objects to add.
        :raises ValueError: If a rebuild-only backend exists without a context
            store.
        """
        addable_backends, rebuild_backends = self._split_backends_by_addability()
        if rebuild_backends and self.context_store is None:
            names = ", ".join(name for name, _ in rebuild_backends)
            raise ValueError(f"Backends require context_store for rebuild: {names}")
        has_contexts = False
        for batch in _iter_batches(contexts, self.config.batch_size):
            has_contexts = True
            if self.context_store is not None:
                self.context_store.set_many(batch)
            for _, backend in addable_backends:
                backend.add_contexts(batch)
        if has_contexts:
            for name, backend in rebuild_backends:
                self._rebuild_backend_from_store(name, backend)
        return

    async def async_add_contexts(self, contexts: Iterable[Context]) -> None:
        """Asynchronously add contexts and eagerly update all backends.

        The operation has the same eager consistency semantics as
        ``add_contexts``. Failures do not roll back already-written store or
        backend changes.

        :param contexts: Context objects to add.
        :raises ValueError: If a rebuild-only backend exists without a context
            store.
        """
        addable_backends, rebuild_backends = self._split_backends_by_addability()
        if rebuild_backends and self.context_store is None:
            names = ", ".join(name for name, _ in rebuild_backends)
            raise ValueError(f"Backends require context_store for rebuild: {names}")
        has_contexts = False
        for batch in _iter_batches(contexts, self.config.batch_size):
            has_contexts = True
            if self.context_store is not None:
                await self.context_store.async_set_many(batch)
            for _, backend in addable_backends:
                await backend.async_add_contexts(batch)
        if has_contexts:
            for name, backend in rebuild_backends:
                await self._async_rebuild_backend_from_store(name, backend)
        return

    def add_backend(
        self,
        name: str,
        backend: CollectionBackend,
        *,
        rebuild: bool = True,
    ) -> None:
        """Attach a backend at runtime.

        :param name: Retriever-owned backend name.
        :param backend: Backend instance to attach.
        :param rebuild: Whether to backfill from the context store immediately.
        :raises ValueError: If the name already exists, is invalid, or rebuild is
            requested without a context store.
        """
        self._validate_backend_name(name)
        if name in self.backends:
            raise ValueError(f"Backend name already exists: {name!r}")
        self._prepare_backend(name, backend)
        if rebuild:
            self._rebuild_backend_from_store(name, backend)
        self.backends[name] = backend
        return

    async def async_add_backend(
        self,
        name: str,
        backend: CollectionBackend,
        *,
        rebuild: bool = True,
    ) -> None:
        """Asynchronously attach a backend at runtime.

        :param name: Retriever-owned backend name.
        :param backend: Backend instance to attach.
        :param rebuild: Whether to backfill from the context store immediately.
        :raises ValueError: If the name already exists, is invalid, or rebuild is
            requested without a context store.
        """
        self._validate_backend_name(name)
        if name in self.backends:
            raise ValueError(f"Backend name already exists: {name!r}")
        self._prepare_backend(name, backend)
        if rebuild:
            await self._async_rebuild_backend_from_store(name, backend)
        self.backends[name] = backend
        return

    def remove_backend(
        self,
        name: str,
        *,
        close: bool = False,
        clear: bool = False,
    ) -> CollectionBackend:
        """Detach a backend from the retriever.

        :param name: Backend name to detach.
        :param close: Whether to close the backend after detaching it.
        :param clear: Whether to clear backend artifacts after detaching it.
        :returns: Detached backend instance.
        :raises KeyError: If the backend name is unknown.
        """
        backend = self.backends.pop(name)
        if clear:
            backend.clear()
        if close:
            backend.close()
        return backend

    async def async_remove_backend(
        self,
        name: str,
        *,
        close: bool = False,
        clear: bool = False,
    ) -> CollectionBackend:
        """Asynchronously detach a backend from the retriever.

        :param name: Backend name to detach.
        :param close: Whether to close the backend after detaching it.
        :param clear: Whether to clear backend artifacts after detaching it.
        :returns: Detached backend instance.
        :raises KeyError: If the backend name is unknown.
        """
        backend = self.backends.pop(name)
        if clear:
            await backend.async_clear()
        if close:
            await backend.async_close()
        return backend

    def rebuild(self, backend_name: str | None = None) -> None:
        """Rebuild one or all backends from the context store.

        :param backend_name: Optional backend name. ``None`` rebuilds all
            backends.
        :raises ValueError: If no context store is available.
        :raises KeyError: If ``backend_name`` is unknown.
        """
        targets = (
            [(backend_name, self.backends[backend_name])]
            if backend_name is not None
            else list(self.backends.items())
        )
        for name, backend in targets:
            self._rebuild_backend_from_store(name, backend)
        return

    async def async_rebuild(self, backend_name: str | None = None) -> None:
        """Asynchronously rebuild one or all backends from the context store.

        :param backend_name: Optional backend name. ``None`` rebuilds all
            backends.
        :raises ValueError: If no context store is available.
        :raises KeyError: If ``backend_name`` is unknown.
        """
        targets = (
            [(backend_name, self.backends[backend_name])]
            if backend_name is not None
            else list(self.backends.items())
        )
        for name, backend in targets:
            await self._async_rebuild_backend_from_store(name, backend)
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
        """Search backends and return lightweight hits.

        Single-backend searches return backend hits directly. Multi-backend
        searches retrieve candidates from each selected backend and merge them.

        :param queries: One query object or an iterable of query objects.
        :param top_k: Maximum hits per query, or config default.
        :param used_backends: Optional backend name subset.
        :param candidate_k: Per-backend candidate count before merge.
        :param merge_method: Merge algorithm override.
        :param backend_weights: Optional per-backend merge weights.
        :param backend_search_options: Backend-specific search options by name.
        :returns: One hit list per normalized query.
        :raises KeyError: If a selected backend name is unknown.
        """
        normalized_queries = self._normalize_queries(queries)
        top_k = self.config.default_top_k if top_k is None else top_k
        if top_k <= 0:
            return [[] for _ in normalized_queries]
        backend_names = list(self.backends) if used_backends is None else used_backends
        selected = [(name, self.backends[name]) for name in backend_names]
        if not selected:
            return [[] for _ in normalized_queries]
        if len(selected) == 1:
            name, backend = selected[0]
            return self._search_backend(
                name,
                backend,
                normalized_queries,
                top_k,
                search_options=(backend_search_options or {}).get(name),
            )
        per_backend_top_k = max(top_k, candidate_k or top_k)
        per_backend = [
            self._search_backend(
                name,
                backend,
                normalized_queries,
                per_backend_top_k,
                search_options=(backend_search_options or {}).get(name),
            )
            for name, backend in selected
        ]
        method = normalize_merge_method(
            merge_method,
            self.config.default_merge_method,
        )
        weights = normalize_backend_weights(backend_names, backend_weights)
        return merge_hits(
            per_backend,
            backend_names=backend_names,
            weights=weights,
            top_k=top_k,
            merge_method=method,
            rrf_base=self.config.rrf_base,
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
        """Asynchronously search backends and return lightweight hits.

        Multi-backend searches run selected backend searches concurrently before
        merging hits.

        :param queries: One query object or an iterable of query objects.
        :param top_k: Maximum hits per query, or config default.
        :param used_backends: Optional backend name subset.
        :param candidate_k: Per-backend candidate count before merge.
        :param merge_method: Merge algorithm override.
        :param backend_weights: Optional per-backend merge weights.
        :param backend_search_options: Backend-specific search options by name.
        :returns: One hit list per normalized query.
        :raises KeyError: If a selected backend name is unknown.
        """
        normalized_queries = self._normalize_queries(queries)
        top_k = self.config.default_top_k if top_k is None else top_k
        if top_k <= 0:
            return [[] for _ in normalized_queries]
        backend_names = list(self.backends) if used_backends is None else used_backends
        selected = [(name, self.backends[name]) for name in backend_names]
        if not selected:
            return [[] for _ in normalized_queries]
        if len(selected) == 1:
            name, backend = selected[0]
            return await self._async_search_backend(
                name,
                backend,
                normalized_queries,
                top_k,
                search_options=(backend_search_options or {}).get(name),
            )
        per_backend_top_k = max(top_k, candidate_k or top_k)
        per_backend = await asyncio.gather(
            *[
                self._async_search_backend(
                    name,
                    backend,
                    normalized_queries,
                    per_backend_top_k,
                    search_options=(backend_search_options or {}).get(name),
                )
                for name, backend in selected
            ]
        )
        method = normalize_merge_method(
            merge_method,
            self.config.default_merge_method,
        )
        weights = normalize_backend_weights(backend_names, backend_weights)
        return merge_hits(
            per_backend,
            backend_names=backend_names,
            weights=weights,
            top_k=top_k,
            merge_method=method,
            rrf_base=self.config.rrf_base,
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
        """Search and hydrate hits into ``RetrievedContext`` objects.

        Hydration prefers hit-native payloads, then the context store, then
        backend-native payload lookup.

        :param queries: One query object or an iterable of query objects.
        :param top_k: Maximum results per query, or config default.
        :param used_backends: Optional backend name subset.
        :param candidate_k: Per-backend candidate count before merge.
        :param merge_method: Merge algorithm override.
        :param backend_weights: Optional per-backend merge weights.
        :param backend_search_options: Backend-specific search options by name.
        :returns: One hydrated result list per normalized query.
        """
        normalized_queries = self._normalize_queries(queries)
        hits = self.search_hits(
            normalized_queries,
            top_k=top_k,
            used_backends=used_backends,
            candidate_k=candidate_k,
            merge_method=merge_method,
            backend_weights=backend_weights,
            backend_search_options=backend_search_options,
        )
        return [
            [self._hit_to_retrieved_context(hit, query) for hit in query_hits]
            for query, query_hits in zip(normalized_queries, hits)
        ]

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
        """Asynchronously search and hydrate hits.

        :param queries: One query object or an iterable of query objects.
        :param top_k: Maximum results per query, or config default.
        :param used_backends: Optional backend name subset.
        :param candidate_k: Per-backend candidate count before merge.
        :param merge_method: Merge algorithm override.
        :param backend_weights: Optional per-backend merge weights.
        :param backend_search_options: Backend-specific search options by name.
        :returns: One hydrated result list per normalized query.
        """
        normalized_queries = self._normalize_queries(queries)
        hits = await self.async_search_hits(
            normalized_queries,
            top_k=top_k,
            used_backends=used_backends,
            candidate_k=candidate_k,
            merge_method=merge_method,
            backend_weights=backend_weights,
            backend_search_options=backend_search_options,
        )
        results: list[list[RetrievedContext]] = []
        for query, query_hits in zip(normalized_queries, hits):
            results.append(
                [
                    await self._async_hit_to_retrieved_context(hit, query)
                    for hit in query_hits
                ]
            )
        return results

    def get(self, context_id: str) -> Context:
        """Hydrate a context by id.

        :param context_id: Context identifier to fetch.
        :returns: Complete context.
        :raises KeyError: If the id cannot be hydrated.
        """
        if self.context_store is not None:
            return self.context_store.get(context_id)
        for backend in self.backends.values():
            try:
                return backend.get_context(context_id)
            except KeyError:
                continue
        raise KeyError(context_id)

    async def async_get(self, context_id: str) -> Context:
        """Asynchronously hydrate a context by id.

        :param context_id: Context identifier to fetch.
        :returns: Complete context.
        :raises KeyError: If the id cannot be hydrated.
        """
        if self.context_store is not None:
            return await self.context_store.async_get(context_id)
        for backend in self.backends.values():
            try:
                return await backend.async_get_context(context_id)
            except KeyError:
                continue
        raise KeyError(context_id)

    def clear(self) -> None:
        """Clear the context store and all attached backend artifacts."""
        if self.context_store is not None:
            self.context_store.clear()
        for backend in self.backends.values():
            backend.clear()
        return

    async def async_clear(self) -> None:
        """Asynchronously clear the context store and all backend artifacts."""
        if self.context_store is not None:
            await self.context_store.async_clear()
        for backend in self.backends.values():
            await backend.async_clear()
        return

    def count(self) -> int:
        """Return the authoritative unique context count.

        The context store is preferred. Without a store, all backends must agree
        on their unique context count.

        :returns: Unique context count.
        :raises RuntimeError: If backend counts differ and no store exists.
        """
        if self.context_store is not None:
            return self.context_store.count()
        return self._resolve_backend_count(
            [backend.count() for backend in self.backends.values()]
        )

    async def async_count(self) -> int:
        """Asynchronously return the authoritative unique context count.

        :returns: Unique context count.
        :raises RuntimeError: If backend counts differ and no store exists.
        """
        if self.context_store is not None:
            return await self.context_store.async_count()
        counts = []
        for backend in self.backends.values():
            counts.append(await backend.async_count())
        return self._resolve_backend_count(counts)

    def list_backends(self) -> list[str]:
        """Return backend names in retriever order."""
        return list(self.backends)

    def _prepare_backend(self, name: str, backend: CollectionBackend) -> None:
        if backend.requires_context_store and self.context_store is None:
            raise ValueError(f"Backend {name!r} requires context_store.")
        return

    @staticmethod
    def _resolve_backend_count(counts: list[int]) -> int:
        if not counts:
            return 0
        if len(set(counts)) != 1:
            raise RuntimeError(
                "Backend counts differ and no context_store is available as the "
                "authoritative corpus source."
            )
        return counts[0]

    def _split_backends_by_addability(
        self,
    ) -> tuple[list[tuple[str, CollectionBackend]], list[tuple[str, CollectionBackend]]]:
        addable_backends = []
        rebuild_backends = []
        for name, backend in self.backends.items():
            if backend.is_addable:
                addable_backends.append((name, backend))
            else:
                rebuild_backends.append((name, backend))
        return addable_backends, rebuild_backends

    def _rebuild_backend_from_store(
        self,
        name: str,
        backend: CollectionBackend,
    ) -> None:
        if self.context_store is None:
            raise ValueError(
                f"Backend {name!r} cannot be rebuilt without context_store."
            )
        backend.rebuild(self.context_store.iter_contexts())
        return

    async def _async_rebuild_backend_from_store(
        self,
        name: str,
        backend: CollectionBackend,
    ) -> None:
        if self.context_store is None:
            raise ValueError(
                f"Backend {name!r} cannot be rebuilt without context_store."
            )
        await backend.async_rebuild(self.context_store.iter_contexts())
        return

    def _search_backend(
        self,
        name: str,
        backend: CollectionBackend,
        queries: list[Any],
        top_k: int,
        *,
        search_options: dict[str, Any] | None = None,
    ) -> list[list[Hit]]:
        return [
            [
                Hit(
                    context_id=hit.context_id,
                    score=hit.score,
                    backend=name,
                    view=hit.view,
                    context=hit.context,
                )
                for hit in query_hits
            ]
            for query_hits in backend.search_hits(
                queries,
                top_k,
                search_options=search_options,
            )
        ]

    async def _async_search_backend(
        self,
        name: str,
        backend: CollectionBackend,
        queries: list[Any],
        top_k: int,
        *,
        search_options: dict[str, Any] | None = None,
    ) -> list[list[Hit]]:
        return [
            [
                Hit(
                    context_id=hit.context_id,
                    score=hit.score,
                    backend=name,
                    view=hit.view,
                    context=hit.context,
                )
                for hit in query_hits
            ]
            for query_hits in await backend.async_search_hits(
                queries,
                top_k,
                search_options=search_options,
            )
        ]

    def _hit_to_retrieved_context(self, hit: Hit, query: str) -> RetrievedContext:
        context = hit.context or self.get(hit.context_id)
        return RetrievedContext(
            context_id=context.context_id,
            data=dict(context.data),
            source=context.source,
            meta_data=dict(context.meta_data),
            retriever=hit.backend,
            query=query if isinstance(query, str) else str(query),
            score=hit.score,
        )

    async def _async_hit_to_retrieved_context(
        self,
        hit: Hit,
        query: str,
    ) -> RetrievedContext:
        context = hit.context or await self.async_get(hit.context_id)
        return RetrievedContext(
            context_id=context.context_id,
            data=dict(context.data),
            source=context.source,
            meta_data=dict(context.meta_data),
            retriever=hit.backend,
            query=query if isinstance(query, str) else str(query),
            score=hit.score,
        )

    @staticmethod
    def _normalize_queries(queries: Any | Iterable[Any]) -> list[Any]:
        if isinstance(queries, str) or isinstance(queries, Mapping):
            return [queries]
        if isinstance(queries, Iterable):
            return list(queries)
        return [queries]

    @classmethod
    def _normalize_backends(
        cls,
        backends: Mapping[str, CollectionBackend],
    ) -> dict[str, CollectionBackend]:
        normalized: dict[str, CollectionBackend] = {}
        for name, backend in backends.items():
            cls._validate_backend_name(name)
            if name in normalized:
                raise ValueError("Backend names must be unique.")
            normalized[name] = backend
        return normalized

    @staticmethod
    def _validate_backend_name(name: str) -> None:
        if not isinstance(name, str) or not name:
            raise ValueError("Backend name must be a non-empty string.")
        return
