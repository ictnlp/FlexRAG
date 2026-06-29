import asyncio
import inspect
import time
from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import asdict, is_dataclass
from typing import Any, cast

import numpy as np

from flexrag.common import (
    LOGGER_MANAGER,
    ProgressDisplay,
    Register,
    configure,
    warning_once,
)
from flexrag.common.dataclasses import Context, RetrievedContext
from flexrag.common.runtime_cache import get_runtime_cache, make_runtime_cache_key

logger = LOGGER_MANAGER.get_logger("flexrag.retrievers")
_RETRIEVAL_CACHE_NAMESPACE = "retrieval.search"
_RETRIEVAL_CACHE_SCHEMA_VERSION = 1
DEFAULT_TOP_K = 10
_RUNTIME_ONLY_CONFIG_FIELDS = {
    "batch_size",
}
_RUNTIME_ONLY_SEARCH_KWARGS = {
    "display",
    "log_interval",
}


@configure
class RetrieverBaseConfig:
    """Base configuration for collection-like retrievers.

    :param batch_size: Batch size used by direct retriever calls when adding
        passages, searching, or serializing implementation-specific state.
        Defaults to 32.
    """

    batch_size: int = 32


class RetrieverBase(ABC):
    """Base class for a collection-like retriever.

    Subclasses own one retrievable collection and must implement search,
    insertion, lookup, counting, clearing, and field inspection. The default
    asynchronous methods are direct-use conveniences backed by
    :func:`asyncio.to_thread`; they do not provide runtime isolation.
    """

    cfg: RetrieverBaseConfig

    def __init__(self, cfg: RetrieverBaseConfig):
        self.cfg = cfg
        return

    @staticmethod
    def _normalize_query(query: Iterable[Any] | Any) -> list[Any]:
        if isinstance(query, str):
            return [query]
        if isinstance(query, Iterable):
            return list(query)
        return [query]

    def _build_search_cache_keys(
        self,
        query: list[Any],
        search_kwargs: dict[str, Any],
    ) -> list[str]:
        if is_dataclass(self.cfg):
            retriever_config = asdict(self.cfg)
        else:
            retriever_config = dict(getattr(self.cfg, "__dict__", {}))
        for key in _RUNTIME_ONLY_CONFIG_FIELDS:
            retriever_config.pop(key, None)

        cache_search_kwargs = dict(search_kwargs)
        for key in _RUNTIME_ONLY_SEARCH_KWARGS:
            cache_search_kwargs.pop(key, None)

        retriever_name = f"{self.__class__.__module__}.{self.__class__.__qualname__}"
        retriever_fingerprint = self._cache_fingerprint()
        return [
            make_runtime_cache_key(
                {
                    "namespace": _RETRIEVAL_CACHE_NAMESPACE,
                    "schema_version": _RETRIEVAL_CACHE_SCHEMA_VERSION,
                    "retriever": retriever_name,
                    "retriever_config": retriever_config,
                    "retriever_fingerprint": retriever_fingerprint,
                    "query": q,
                    "search_kwargs": cache_search_kwargs,
                }
            )
            for q in query
        ]

    def _search_batches(
        self,
        query: list[Any],
        batch_size: int,
        search_kwargs: dict[str, Any],
    ) -> list[list[RetrievedContext]]:
        results: list[list[RetrievedContext]] = []
        for start in range(0, len(query), batch_size):
            batch = query[start : start + batch_size]
            results.extend(self._search(batch, **dict(search_kwargs)))
        return results

    async def _async_search_batches(
        self,
        query: list[Any],
        batch_size: int,
        search_kwargs: dict[str, Any],
    ) -> list[list[RetrievedContext]]:
        results: list[list[RetrievedContext]] = []
        for start in range(0, len(query), batch_size):
            batch = query[start : start + batch_size]
            results.extend(await self._async_search(batch, **dict(search_kwargs)))
        return results

    @staticmethod
    def _deserialize_cached_results(
        cached: list[Any | None],
    ) -> list[list[RetrievedContext] | None]:
        return [
            (
                [RetrievedContext(**item) for item in cache_item]
                if cache_item is not None
                else None
            )
            for cache_item in cached
        ]

    async def async_search(
        self,
        query: Iterable[Any] | Any,
        disable_cache: bool = False,
        top_k: int = DEFAULT_TOP_K,
        **search_kwargs,
    ) -> list[list[RetrievedContext]]:
        """Search queries asynchronously.

        :param query: A single query or a batch of queries.
        :param disable_cache: Whether to bypass the runtime search cache.
            Defaults to False.
        :param top_k: Number of contexts to return per query. Defaults to 10.
        :param search_kwargs: Additional implementation-specific search
            arguments.
        :return: Retrieved contexts grouped by input query.
        """
        search_kwargs = dict(search_kwargs)
        search_kwargs["top_k"] = top_k
        query = self._normalize_query(query)
        batch_size = max(1, self.cfg.batch_size)

        if disable_cache:
            return await self._async_search_batches(query, batch_size, search_kwargs)

        keys = self._build_search_cache_keys(query, search_kwargs)
        try:
            cache = get_runtime_cache(_RETRIEVAL_CACHE_NAMESPACE)
            results = self._deserialize_cached_results(cache.get_many(keys))
        except Exception as e:
            warning_once(logger, "Runtime cache read failed; bypassing cache: %s", e)
            return await self._async_search_batches(query, batch_size, search_kwargs)

        missing_indices = [i for i, item in enumerate(results) if item is None]
        if not missing_indices:
            return cast(list[list[RetrievedContext]], results)

        cache_items: dict[str, list[dict[str, Any]]] = {}
        for start in range(0, len(missing_indices), batch_size):
            indices = missing_indices[start : start + batch_size]
            batch = [query[i] for i in indices]
            batch_results = await self._async_search(batch, **dict(search_kwargs))
            if len(batch_results) != len(batch):
                raise ValueError(
                    f"{self.__class__.__qualname__} returned {len(batch_results)} "
                    f"results for {len(batch)} queries."
                )
            for idx, item in zip(indices, batch_results):
                results[idx] = item
                cache_items[keys[idx]] = [asdict(context) for context in item]

        try:
            cache.set_many(
                cache_items,
                metadata={
                    "schema_version": _RETRIEVAL_CACHE_SCHEMA_VERSION,
                    "retriever": self.__class__.__qualname__,
                },
            )
        except Exception as e:
            warning_once(logger, "Runtime cache write failed; ignoring cache: %s", e)
        return cast(list[list[RetrievedContext]], results)

    def search(
        self,
        query: Iterable[Any] | Any,
        disable_cache: bool = False,
        top_k: int = DEFAULT_TOP_K,
        **search_kwargs,
    ) -> list[list[RetrievedContext]]:
        """Search queries with batching and runtime result caching.

        :param query: A single query or a batch of queries.
        :param disable_cache: Whether to disable runtime result cache for this call.
            Defaults to False.
        :param top_k: The number of retrieved documents. Defaults to 10.
        :param search_kwargs: Additional implementation-specific search
            arguments.
        :return: Retrieved contexts grouped by input query.
        """
        search_kwargs = dict(search_kwargs)
        search_kwargs["top_k"] = top_k
        query = self._normalize_query(query)

        # search without cache
        batch_size = max(1, self.cfg.batch_size)
        if disable_cache:
            return self._search_batches(query, batch_size, search_kwargs)

        # prepare cache keys
        keys = self._build_search_cache_keys(query, search_kwargs)

        # try to get results from cache
        try:
            cache = get_runtime_cache(_RETRIEVAL_CACHE_NAMESPACE)
            results = self._deserialize_cached_results(cache.get_many(keys))
        except Exception as e:
            warning_once(logger, "Runtime cache read failed; bypassing cache: %s", e)
            return self._search_batches(query, batch_size, search_kwargs)

        missing_indices = [i for i, item in enumerate(results) if item is None]
        if not missing_indices:
            return cast(list[list[RetrievedContext]], results)

        # search missing items
        cache_items: dict[str, list[dict[str, Any]]] = {}
        for start in range(0, len(missing_indices), batch_size):
            indices = missing_indices[start : start + batch_size]
            batch = [query[i] for i in indices]
            batch_results = self._search(batch, **dict(search_kwargs))
            if len(batch_results) != len(batch):
                raise ValueError(
                    f"{self.__class__.__qualname__} returned {len(batch_results)} "
                    f"results for {len(batch)} queries."
                )
            for idx, item in zip(indices, batch_results):
                results[idx] = item
                cache_items[keys[idx]] = [asdict(context) for context in item]

        # write back to cache
        try:
            cache.set_many(
                cache_items,
                metadata={
                    "schema_version": _RETRIEVAL_CACHE_SCHEMA_VERSION,
                    "retriever": self.__class__.__qualname__,
                },
            )
        except Exception as e:
            warning_once(logger, "Runtime cache write failed; ignoring cache: %s", e)
        return cast(list[list[RetrievedContext]], results)

    def _cache_fingerprint(self) -> dict[str, Any]:
        """Return mutable runtime state that should participate in cache keys.

        Stateless retrievers can keep the default empty fingerprint. Mutable
        retrievers should override this to avoid sharing cache entries across
        collections or collection generations.

        :return: Cache fingerprint dictionary.
        """
        return {}

    @abstractmethod
    def _search(
        self,
        query: list[Any],
        **search_kwargs,
    ) -> list[list[RetrievedContext]]:
        """Search a batch of queries.

        :param query: Query batch to search.
        :param search_kwargs: Additional implementation-specific search
            arguments.
        :return: Retrieved contexts grouped by query.
        """
        return

    async def _async_search(
        self,
        query: list[Any],
        **search_kwargs,
    ) -> list[list[RetrievedContext]]:
        """Search a batch of queries asynchronously.

        Direct local retrievers keep the default thread-backed convenience
        wrapper. Remote retrievers should override this with native async I/O.

        :param query: Query batch to search.
        :param search_kwargs: Additional implementation-specific search
            arguments.
        :return: Retrieved contexts grouped by query.
        """
        return await asyncio.to_thread(self._search, query, **search_kwargs)

    @abstractmethod
    def add_passages(
        self,
        passages: Iterable[Context],
        log_interval: int = 10000,
        display: ProgressDisplay = "auto",
    ) -> None:
        """Add passages to the retriever collection.

        :param passages: Contexts to insert or upsert.
        :param log_interval: Progress logging interval. Defaults to 10000.
        :param display: Progress display mode. Defaults to ``"auto"``.
        :return: None.
        """
        return

    async def async_add_passages(
        self,
        passages: Iterable[Context],
        log_interval: int = 10000,
        display: ProgressDisplay = "auto",
    ) -> None:
        """Add passages asynchronously.

        :param passages: Contexts to insert or upsert.
        :param log_interval: Progress logging interval. Defaults to 10000.
        :param display: Progress display mode. Defaults to ``"auto"``.
        :return: None.
        """
        return await asyncio.to_thread(
            self.add_passages,
            passages=passages,
            log_interval=log_interval,
            display=display,
        )

    @abstractmethod
    def clear(self) -> None:
        """Clear all contexts from the retriever collection.

        :return: None.
        """
        return

    async def async_clear(self) -> None:
        """Clear the retriever collection asynchronously.

        :return: None.
        """
        return await asyncio.to_thread(self.clear)

    @abstractmethod
    def get(self, context_id: str) -> Context:
        """Get a context by id.

        :param context_id: Context id to fetch.
        :return: The fetched context.
        :raises KeyError: If the context does not exist.
        """
        return

    async def async_get(self, context_id: str) -> Context:
        """Get a context by id asynchronously.

        :param context_id: Context id to fetch.
        :return: The fetched context.
        :raises KeyError: If the context does not exist.
        """
        return await asyncio.to_thread(self.get, context_id=context_id)

    @abstractmethod
    def count(self) -> int:
        """Return the number of contexts in the retriever collection.

        :return: Number of contexts.
        """
        return

    async def async_count(self) -> int:
        """Return the number of contexts asynchronously.

        :return: Number of contexts.
        """
        return await asyncio.to_thread(self.count)

    def __getitem__(self, context_id: str) -> Context:
        """Get a context by id.

        :param context_id: Context id to fetch.
        :return: The fetched context.
        :raises KeyError: If the context does not exist.
        """
        return self.get(context_id)

    def __len__(self) -> int:
        """Return the number of contexts in the collection.

        :return: Number of contexts.
        """
        return self.count()

    @property
    @abstractmethod
    def fields(self) -> list[str]:
        """Return fields available in the retriever collection.

        :return: Field names.
        """
        return

    async def async_fields(self) -> list[str]:
        """Return fields available in the retriever collection asynchronously.

        :return: Field names.
        """
        return await asyncio.to_thread(lambda: self.fields)

    def test_speed(
        self,
        sample_num: int = 10000,
        test_times: int = 10,
        top_k: int = DEFAULT_TOP_K,
        **search_kwargs,
    ) -> float:
        """Test the speed of the retriever.

        :param sample_num: The number of samples to test.
        :param test_times: The number of times to test.
        :return: The time consumed for retrieval.
        """
        from nltk.corpus import brown

        total_times = []
        sents = [" ".join(i) for i in brown.sents()]
        for _ in range(test_times):
            query = [sents[i % len(sents)] for i in range(sample_num)]
            start_time = time.perf_counter()
            _ = self.search(query, top_k=top_k, **search_kwargs)
            end_time = time.perf_counter()
            total_times.append(end_time - start_time)
        avg_time = sum(total_times) / test_times
        std_time = np.std(total_times)
        logger.info(
            f"Retrieval {sample_num} items consume: {avg_time:.4f} ± {std_time:.4f} s"
        )
        return end_time - start_time


class RemoteRetrieverBase(RetrieverBase):
    """Thin base class for directly usable remote retrievers.

    Subclasses implement native asynchronous I/O methods. The synchronous
    public methods run the async methods with :func:`asyncio.run` and must not
    be called from an already running event loop. Runtime policies such as
    retry, rate limiting, and background-loop execution belong to runtime
    adapters, not this raw retriever base.
    """

    @staticmethod
    def _ensure_sync_bridge_allowed(
        method_name: str,
        async_method_name: str | None = None,
    ) -> None:
        """Raise when a synchronous bridge is called inside an event loop.

        :param method_name: Synchronous method being called.
        :param async_method_name: Async method users should call instead.
            Defaults to ``"async_{method_name}"``.
        :raises RuntimeError: If the current thread is already running an
            event loop.
        :return: None.
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return
        async_method_name = async_method_name or f"async_{method_name}"
        raise RuntimeError(
            f"{method_name} cannot be called from a running event loop. "
            f"Use {async_method_name} instead."
        )

    def search(
        self,
        query: Iterable[Any] | Any,
        disable_cache: bool = False,
        top_k: int = DEFAULT_TOP_K,
        **search_kwargs,
    ) -> list[list[RetrievedContext]]:
        """Search queries synchronously for direct raw-retriever use.

        :param query: A single query or a batch of queries.
        :param disable_cache: Whether to bypass the runtime search cache.
        :param top_k: Number of contexts to return per query.
        :param search_kwargs: Additional implementation-specific search
            arguments.
        :return: Retrieved contexts grouped by input query.
        :raises RuntimeError: If called from a running event loop.
        """
        self._ensure_sync_bridge_allowed("search")
        return asyncio.run(
            self.async_search(
                query=query,
                disable_cache=disable_cache,
                top_k=top_k,
                **search_kwargs,
            )
        )

    def _search(
        self,
        query: list[Any],
        **search_kwargs,
    ) -> list[list[RetrievedContext]]:
        """Search one canonical query batch synchronously.

        :param query: Query batch to search.
        :param search_kwargs: Additional implementation-specific search
            arguments.
        :return: Retrieved contexts grouped by query.
        :raises RuntimeError: If called from a running event loop.
        """
        self._ensure_sync_bridge_allowed("_search", "_async_search")
        return asyncio.run(self._async_search(query, **search_kwargs))

    @abstractmethod
    async def _async_search(
        self,
        query: list[Any],
        **search_kwargs,
    ) -> list[list[RetrievedContext]]:
        """Search one canonical query batch asynchronously.

        :param query: Query batch to search.
        :param search_kwargs: Additional implementation-specific search
            arguments.
        :return: Retrieved contexts grouped by query.
        """
        return

    def add_passages(
        self,
        passages: Iterable[Context],
        log_interval: int = 10000,
        display: ProgressDisplay = "auto",
    ) -> None:
        """Add passages synchronously for direct raw-retriever use.

        :param passages: Contexts to insert or upsert.
        :param log_interval: Progress logging interval.
        :param display: Progress display mode.
        :return: None.
        :raises RuntimeError: If called from a running event loop.
        """
        self._ensure_sync_bridge_allowed("add_passages")
        return asyncio.run(
            self.async_add_passages(
                passages=passages,
                log_interval=log_interval,
                display=display,
            )
        )

    @abstractmethod
    async def async_add_passages(
        self,
        passages: Iterable[Context],
        log_interval: int = 10000,
        display: ProgressDisplay = "auto",
    ) -> None:
        """Add passages asynchronously with native remote I/O.

        :param passages: Contexts to insert or upsert.
        :param log_interval: Progress logging interval.
        :param display: Progress display mode.
        :return: None.
        """
        return

    def clear(self) -> None:
        """Clear contexts synchronously for direct raw-retriever use.

        :return: None.
        :raises RuntimeError: If called from a running event loop.
        """
        self._ensure_sync_bridge_allowed("clear")
        return asyncio.run(self.async_clear())

    @abstractmethod
    async def async_clear(self) -> None:
        """Clear contexts asynchronously with native remote I/O.

        :return: None.
        """
        return

    def get(self, context_id: str) -> Context:
        """Get one context synchronously for direct raw-retriever use.

        :param context_id: Context id to fetch.
        :return: The fetched context.
        :raises KeyError: If the context does not exist.
        :raises RuntimeError: If called from a running event loop.
        """
        self._ensure_sync_bridge_allowed("get")
        return asyncio.run(self.async_get(context_id))

    @abstractmethod
    async def async_get(self, context_id: str) -> Context:
        """Get one context asynchronously with native remote I/O.

        :param context_id: Context id to fetch.
        :return: The fetched context.
        :raises KeyError: If the context does not exist.
        """
        return

    def count(self) -> int:
        """Return context count synchronously for direct raw-retriever use.

        :return: Number of contexts.
        :raises RuntimeError: If called from a running event loop.
        """
        self._ensure_sync_bridge_allowed("count")
        return asyncio.run(self.async_count())

    @abstractmethod
    async def async_count(self) -> int:
        """Return context count asynchronously with native remote I/O.

        :return: Number of contexts.
        """
        return

    @property
    def fields(self) -> list[str]:
        """Return fields synchronously for direct raw-retriever use.

        :return: Field names.
        :raises RuntimeError: If called from a running event loop.
        """
        self._ensure_sync_bridge_allowed("fields")
        return asyncio.run(self.async_fields())

    @abstractmethod
    async def async_fields(self) -> list[str]:
        """Return fields asynchronously with native remote I/O.

        :return: Field names.
        """
        return

    def close(self) -> None:
        """Close remote resources synchronously.

        :return: None.
        :raises RuntimeError: If called from a running event loop.
        """
        self._ensure_sync_bridge_allowed("close", "aclose")
        return asyncio.run(self.aclose())

    async def aclose(self) -> None:
        """Close remote resources asynchronously.

        :return: None.
        """
        return

    @staticmethod
    async def _maybe_await(result: Any) -> Any:
        if inspect.isawaitable(result):
            return await result
        return result


RETRIEVERS = Register[RetrieverBase]("retriever", True)
