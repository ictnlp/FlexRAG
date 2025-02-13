import asyncio
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
from omegaconf import DictConfig, OmegaConf

from flexrag.cache import (
    LMDBBackendConfig,
    PersistentCacheBase,
    PersistentCacheConfig,
    LRUPersistentCache,
    LFUPersistentCache,
    FIFOPersistentCache,
)
from flexrag.common_dataclass import Context, RetrievedContext
from flexrag.text_process import TextProcessPipeline, TextProcessPipelineConfig
from flexrag.utils import LOGGER_MANAGER, Register, SimpleProgressLogger

logger = LOGGER_MANAGER.get_logger("flexrag.retrievers")


# load cache for retrieval
RETRIEVAL_CACHE: PersistentCacheBase | None
if os.environ.get("DISABLE_CACHE", "False") == "True":
    RETRIEVAL_CACHE = None
else:
    cache_config = PersistentCacheConfig(
        maxsize=10000000,
        storage_backend_type="lmdb",
        lmdb_config=LMDBBackendConfig(
            db_path=os.environ.get(
                "RETRIEVAL_CACHE_PATH",
                os.path.join(
                    os.path.expanduser("~"), ".cache", "flexrag", "cache.lmdb"
                ),
            )
        ),
    )
    match os.environ.get("RETRIEVAL_CACHE_TYPE", "FIFO"):
        case "LRU":
            RETRIEVAL_CACHE = LRUPersistentCache(cache_config)
        case "LFU":
            RETRIEVAL_CACHE = LFUPersistentCache(cache_config)
        case "FIFO":
            RETRIEVAL_CACHE = FIFOPersistentCache(cache_config)
        case _:
            logger.warning("Invalid cache type, cache is disabled.")
            RETRIEVAL_CACHE = None


def batched_cache(func):
    def dataclass_to_dict(data):
        if not isinstance(data, DictConfig):
            return OmegaConf.to_container(DictConfig(data))
        return OmegaConf.to_container(data)

    def retrieved_to_dict(data: list[RetrievedContext]) -> list[dict]:
        return [r.to_dict() for r in data]

    def dict_to_retrieved(data: list[dict] | None) -> list[RetrievedContext] | None:
        if data is None:
            return None
        return [RetrievedContext(**r) for r in data]

    def check(data: list):
        for d in data:
            assert isinstance(d, list)
            for r in d:
                assert isinstance(r, RetrievedContext)
        return

    def wrapper(
        self,
        query: list[str],
        **search_kwargs,
    ):
        # check query
        if isinstance(query, str):
            query = [query]

        # direct search
        if RETRIEVAL_CACHE is None:
            return func(self, query, **search_kwargs)

        # search from cache
        cfg = dataclass_to_dict(self.cfg)
        keys = [
            {
                "retriever_config": cfg,
                "query": q,
                "search_kwargs": search_kwargs,
            }
            for q in query
        ]
        results = [dict_to_retrieved(RETRIEVAL_CACHE.get(k, None)) for k in keys]

        # search from database
        new_query = [q for q, r in zip(query, results) if r is None]
        new_indices = [n for n, r in enumerate(results) if r is None]
        if new_query:
            new_results = func(self, new_query, **search_kwargs)
            # update cache
            for n, r in zip(new_indices, new_results):
                results[n] = r
                RETRIEVAL_CACHE[keys[n]] = retrieved_to_dict(r)
        # check results
        check(results)
        return results

    return wrapper


@dataclass
class RetrieverBaseConfig:
    """Base configuration class for all retrievers.

    :param log_interval: The interval of logging. Default: 100.
    :type log_interval: int
    :param top_k: The number of retrieved documents. Default: 10.
    :type top_k: int
    """

    log_interval: int = 100
    top_k: int = 10


@dataclass
class EditableRetrieverConfig(RetrieverBaseConfig):
    """Configuration class for LocalRetriever.

    :param batch_size: The batch size for retrieval. Default: 32.
    :type batch_size: int
    :param query_preprocess_pipeline: The text process pipeline for query. Default: TextProcessPipelineConfig.
    :type query_preprocess_pipeline: TextProcessPipelineConfig
    """

    batch_size: int = 32
    query_preprocess_pipeline: TextProcessPipelineConfig = field(default_factory=TextProcessPipelineConfig)  # type: ignore


class RetrieverBase(ABC):
    """The base class for all retrievers.
    The subclasses should implement the ``search`` method and the ``fields`` property.
    """

    def __init__(self, cfg: RetrieverBaseConfig):
        self.cfg = cfg
        self.log_interval = cfg.log_interval
        self.top_k = cfg.top_k
        return

    async def async_search(
        self,
        query: list[str],
        **search_kwargs,
    ) -> list[list[RetrievedContext]]:
        """Search queries asynchronously."""
        return await asyncio.to_thread(
            self.search,
            query=query,
            **search_kwargs,
        )

    @abstractmethod
    def search(
        self,
        query: list[str],
        **search_kwargs,
    ) -> list[list[RetrievedContext]]:
        """Search queries.

        :param query: Queries to search.
        :type query: list[str]
        :param search_kwargs: Keyword arguments, contains other search arguments.
        :type search_kwargs: Any
        :return: A batch of list that contains k RetrievedContext.
        :rtype: list[list[RetrievedContext]]
        """
        return

    @property
    @abstractmethod
    def fields(self) -> list[str]:
        """The fields of the retrieved data."""
        return

    def test_speed(
        self,
        sample_num: int = 10000,
        test_times: int = 10,
        **search_kwargs,
    ) -> float:
        """Test the speed of the retriever.

        :param sample_num: The number of samples to test.
        :type sample_num: int, optional
        :param test_times: The number of times to test.
        :type test_times: int, optional
        :return: The time consumed for retrieval.
        :rtype: float
        """
        from nltk.corpus import brown

        total_times = []
        sents = [" ".join(i) for i in brown.sents()]
        for _ in range(test_times):
            query = [sents[i % len(sents)] for i in range(sample_num)]
            start_time = time.perf_counter()
            _ = self.search(query, self.top_k, disable_cache=True, **search_kwargs)
            end_time = time.perf_counter()
            total_times.append(end_time - start_time)
        avg_time = sum(total_times) / test_times
        std_time = np.std(total_times)
        logger.info(
            f"Retrieval {sample_num} items consume: {avg_time:.4f} ± {std_time:.4f} s"
        )
        return end_time - start_time


class EditableRetriever(RetrieverBase):
    def __init__(self, cfg: EditableRetrieverConfig) -> None:
        super().__init__(cfg)
        # set args for process documents
        self.batch_size = cfg.batch_size
        self.query_preprocess_pipeline = TextProcessPipeline(
            cfg.query_preprocess_pipeline
        )
        return

    @abstractmethod
    def add_passages(self, passages: Iterable[Context]):
        """
        Add passages to the retriever database.

        :param passages: The passages to add.
        :type passages: Iterable[Context]
        :return: None
        """
        return

    @abstractmethod
    def search_batch(
        self,
        query: list[str],
        **search_kwargs,
    ) -> list[list[RetrievedContext]]:
        """Search queries using local retriever.

        :param query: Queries to search.
        :type query: list[str]
        :return: A batch of list that contains k RetrievedContext.
        :rtype: list[list[RetrievedContext]]
        """
        return

    @batched_cache
    def search(
        self,
        query: list[str] | str,
        no_preprocess: bool = False,
        **search_kwargs,
    ) -> list[list[RetrievedContext]]:
        # search for documents
        query = [query] if isinstance(query, str) else query
        if not no_preprocess:
            query = [self.query_preprocess_pipeline(q) for q in query]
        final_results = []
        p_logger = SimpleProgressLogger(logger, len(query), self.log_interval)
        for idx in range(0, len(query), self.batch_size):
            p_logger.update(1, "Retrieving")
            batch = query[idx : idx + self.batch_size]
            results_ = self.search_batch(batch, **search_kwargs)
            final_results.extend(results_)
        return final_results

    @abstractmethod
    def clean(self) -> None:
        """Clean the retriever database."""
        return

    @abstractmethod
    def __len__(self):
        """Return the number of documents in the retriever database."""
        return


RETRIEVERS = Register[RetrieverBase]("retriever")
