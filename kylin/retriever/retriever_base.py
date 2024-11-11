import asyncio
import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterable, Optional

import numpy as np

from kylin.text_process import Pipeline, PipelineConfig
from kylin.utils import SimpleProgressLogger, TimeMeter, Register

from .cache import PersistentLRUCache, hashkey


logger = logging.getLogger(__name__)


SEMANTIC_RETRIEVERS = Register("semantic_retriever")
SPARSE_RETRIEVERS = Register("sparse_retriever")
WEB_RETRIEVERS = Register("web_retriever")


@dataclass
class RetrieverConfig:
    cache_path: str = os.path.join(
        os.path.expanduser("~"), ".cache", "librarian", "cache"
    )
    cache_size: int = 10000000
    log_interval: int = 100


@dataclass
class LocalRetrieverConfig(RetrieverConfig):
    batch_size: int = 32
    add_title_to_passage: bool = False
    passage_preprocess_pipeline: PipelineConfig = field(default_factory=PipelineConfig)  # type: ignore
    query_preprocess_pipeline: PipelineConfig = field(default_factory=PipelineConfig)  # type: ignore


@dataclass
class RetrievedContext:
    retriever: str
    query: str
    text: str
    full_text: str
    source: Optional[str] = None
    chunk_id: Optional[str] = None
    score: float = 0.0
    title: Optional[str] = None
    section: Optional[str] = None

    def to_dict(self):
        return {
            "retriever": self.retriever,
            "query": self.query,
            "text": self.text,
            "full_text": self.full_text,
            "source": self.source,
            "chunk_id": self.chunk_id,
            "score": self.score,
            "title": self.title,
            "section": self.section,
        }


class Retriever(ABC):
    def __init__(self, cfg: RetrieverConfig):
        self._cache = PersistentLRUCache(cfg.cache_path, maxsize=cfg.cache_size)
        self.log_interval = cfg.log_interval
        return

    async def async_search(
        self,
        query: list[str],
        top_k: int = 10,
        disable_cache: bool = False,
        **search_kwargs,
    ) -> list[list[RetrievedContext]]:
        return await asyncio.to_thread(
            self.search,
            query=query,
            top_k=top_k,
            disable_cache=disable_cache,
            **search_kwargs,
        )

    @TimeMeter("retrieve")
    def search(
        self,
        query: list[str],
        top_k: int = 10,
        disable_cache: bool = False,
        **search_kwargs,
    ) -> list[list[RetrievedContext]]:
        """Search queries.

        Args:
            query (list[str]): Queries to search.
            top_k (int, optional): N documents to return. Defaults to 10.

        Returns:
            list[list[RetrievedContext]]: A batch of list that contains k RetrievedContext.
        """
        # check query
        if isinstance(query, str):
            query = [query]

        # direct search
        if disable_cache:
            return self._search(query, top_k, **search_kwargs)

        # search from cache
        cache_keys = [
            hashkey(fingerprint=self.fingerprint, query=q, top_k=top_k, **search_kwargs)
            for q in query
        ]
        results = [self._cache.get(k, None) for k in cache_keys]

        # search from database
        new_query = [q for q, r in zip(query, results) if r is None]
        new_indices = [n for n, r in enumerate(results) if r is None]
        if new_query:
            new_results = self._search(new_query, top_k, **search_kwargs)
            for n, r in zip(new_indices, new_results):
                results[n] = r
                self._cache[cache_keys[n]] = r
        assert all(r is not None for r in results)
        return results

    @abstractmethod
    def _search(
        self,
        query: list[str],
        top_k: int = 10,
        **search_kwargs,
    ) -> list[list[RetrievedContext]]:
        return

    def test_speed(
        self,
        sample_num: int = 10000,
        test_times: int = 10,
        top_k: int = 1,
        **search_kwargs,
    ) -> float:
        from nltk.corpus import brown

        total_times = []
        sents = [" ".join(i) for i in brown.sents()]
        for _ in range(test_times):
            query = [sents[i % len(sents)] for i in range(sample_num)]
            start_time = time.perf_counter()
            _ = self.search(query, top_k, disable_cache=True, **search_kwargs)
            end_time = time.perf_counter()
            total_times.append(end_time - start_time)
        avg_time = sum(total_times) / test_times
        std_time = np.std(total_times)
        logger.info(
            f"Retrieval {sample_num} items consume: {avg_time:.4f} ± {std_time:.4f} s"
        )
        return end_time - start_time

    def close(self):
        return

    @property
    @abstractmethod
    def fingerprint(self):
        return


class LocalRetriever(Retriever):
    def __init__(self, cfg: LocalRetrieverConfig) -> None:
        super().__init__(cfg)
        # set args for process documents
        self.batch_size = cfg.batch_size
        self.passage_preprocess_pipeline = Pipeline(cfg.passage_preprocess_pipeline)
        self.query_preprocess_pipeline = Pipeline(cfg.query_preprocess_pipeline)
        self.add_title_to_passage = cfg.add_title_to_passage
        return

    def add_passages(
        self,
        passages: Iterable[dict[str, str] | str],
        full_text_field: str = "text",
        title_field: str = "title",
        section_field: str = "section",
    ):
        """
        Add passages to the retriever database
        """

        def get_passage():
            for p in passages:
                if isinstance(p, str):
                    p = {"text": p}
                else:
                    p["text"] = p[full_text_field]
                p["title"] = p.get(title_field, "")
                p["section"] = p.get(section_field, "")
                if self.add_title_to_passage:
                    p["text"] = f"{p['title']} {p['text']}"
                text = self.passage_preprocess_pipeline(p["text"])
                if text is None:
                    continue
                p["text"] = text
                yield p

        return self._add_passages(get_passage())

    @abstractmethod
    def _add_passages(self, passages: Iterable[dict[str, str]]):
        return

    @abstractmethod
    def search_batch(
        self,
        query: list[str],
        top_k: int = 10,
        **search_kwargs,
    ) -> list[list[RetrievedContext]]:
        """Search queries using local retriever.

        Args:
            query (list[str]): Queries to search.
            top_k (int, optional): N documents to return. Defaults to 10.

        Returns:
            list[list[RetrievedContext]]: A batch of list that contains k RetrievedContext.
        """
        return

    def _search(
        self,
        query: list[str] | str,
        top_k: int = 10,
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
            results_ = self.search_batch(batch, top_k, **search_kwargs)
            final_results.extend(results_)
        return final_results

    @abstractmethod
    def close(self):
        return

    @abstractmethod
    def clean(self) -> None:
        return

    @abstractmethod
    def __len__(self):
        return

    def _clean_cache(self):
        if self._cache is not None:
            self._cache.clean()
        return
