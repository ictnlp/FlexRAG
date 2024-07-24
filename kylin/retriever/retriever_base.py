import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from kylin.text_process import normalize_token
from kylin.utils import SimpleProgressLogger

from .cache import PersistentLRUCache, hashkey


logger = logging.getLogger(__name__)


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
    max_query_length: int = 512
    max_passage_length: int = 512
    no_title: bool = False
    lowercase: bool = False
    normalize_text: bool = True


class Retriever(ABC):
    def __init__(self, cfg: RetrieverConfig):
        self._cache = PersistentLRUCache(cfg.cache_path, maxsize=cfg.cache_size)
        self.log_interval = cfg.log_interval
        return

    def search(
        self,
        query: list[str],
        top_k: int = 10,
        disable_cache: bool = False,
        **search_kwargs,
    ) -> list[list[dict[str, str]]]:
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
    ) -> list[list[dict[str, str]]]:
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
        self.max_query_length = cfg.max_query_length
        self.max_passage_length = cfg.max_passage_length
        self.no_title = cfg.no_title
        self.lowercase = cfg.lowercase
        self.normalize_text = cfg.normalize_text
        return

    @abstractmethod
    def add_passages(
        self,
        passages: Iterable[dict[str, str]] | Iterable[str],
        reinit: bool = False,
    ):
        """
        Add passages to the retriever database
        """
        return

    @abstractmethod
    def search_batch(
        self,
        query: list[str],
        top_k: int = 10,
        disable_cache: bool = False,
        **search_kwargs,
    ) -> list[list[dict[str, str]]]:
        """Search queries using local retriever.

        Args:
            query (list[str]): Queries to search.
            top_k (int, optional): N documents to return. Defaults to 10.

        Returns:
            list[list[dict[str, str]]]: A list of retrieved documents: [
                {
                    "retriever": str,
                    "query": str,
                    "source": str,
                    "chunk_id": int,
                    "score": float,
                    "title": str,
                    "section": str,
                    "text": str,
                    "full_text": str,
                }
            ]
        """
        return

    def _search(
        self,
        query: list[str] | str,
        top_k: int = 10,
        **search_kwargs,
    ) -> list[list[dict[str, str]]]:
        # search for documents
        query = [query] if isinstance(query, str) else query
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

    @property
    @abstractmethod
    def __len__(self):
        return

    def _clear_cache(self):
        if self._cache is not None:
            self._cache.clear()
        return

    def _prepare_text(self, document: dict[str, str] | str) -> str:
        # add title
        if isinstance(document, dict):
            if ("title" in document) and (not self.no_title):
                text = f"{document['title']} {document['text']}"
            else:
                text = document["text"]
        else:
            assert isinstance(document, str)
            text = document
        # lowercase
        if self.lowercase:
            text = text.lower()
        # normalize
        if self.normalize_text:
            text = normalize_token(text)
        return text
