import asyncio
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
        return await asyncio.to_thread(
            self.search,
            query=query,
            disable_cache=disable_cache,
            top_k=top_k,
            **search_kwargs,
        )

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

        # normalize query
        if isinstance(query, str):
            query = [query]
        elif isinstance(query, Iterable):
            query = list(query)
        else:
            query = [query]

        # search without cache
        batch_size = max(1, self.cfg.batch_size)
        if disable_cache:
            results: list[list[RetrievedContext]] = []
            for start in range(0, len(query), batch_size):
                batch = query[start : start + batch_size]
                results.extend(self._search(batch, **dict(search_kwargs)))
            return results

        # prepare cache keys
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
        keys = [
            make_runtime_cache_key(
                {
                    "namespace": _RETRIEVAL_CACHE_NAMESPACE,
                    "schema_version": _RETRIEVAL_CACHE_SCHEMA_VERSION,
                    "retriever": retriever_name,
                    "retriever_config": retriever_config,
                    "query": q,
                    "search_kwargs": cache_search_kwargs,
                }
            )
            for q in query
        ]

        # try to get results from cache
        try:
            cache = get_runtime_cache(_RETRIEVAL_CACHE_NAMESPACE)
            cached = cache.get_many(keys)
            results: list[list[RetrievedContext] | None] = [
                (
                    [RetrievedContext(**item) for item in cache_item]
                    if cache_item is not None
                    else None
                )
                for cache_item in cached
            ]
        except Exception as e:
            warning_once(logger, "Runtime cache read failed; bypassing cache: %s", e)
            final_results: list[list[RetrievedContext]] = []
            for start in range(0, len(query), batch_size):
                batch = query[start : start + batch_size]
                final_results.extend(self._search(batch, **dict(search_kwargs)))
            return final_results

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


RETRIEVERS = Register[RetrieverBase]("retriever", True)
