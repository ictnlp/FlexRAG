import logging
import time
from abc import ABC, abstractmethod
from argparse import ArgumentParser, Namespace

import numpy as np
from cachetools import cached

from .cache import PersistentLRUCache


logger = logging.getLogger(__name__)


class Retriever(ABC):
    @staticmethod
    def add_args(parser: ArgumentParser) -> ArgumentParser:
        parser.add_argument(
            "--cache_size",
            default=0,
            type=int,
            help="The size of the cache",
        )
        parser.add_argument(
            "--database_path",
            type=str,
            required=True,
            help="The path to the Retriever database",
        )
        return parser

    def __init__(self, args: Namespace) -> None:
        if args.cache_size > 0:
            self._cache = PersistentLRUCache(
                args.database_path, maxsize=args.cache_size
            )
        else:
            self._cache = None
        return

    @abstractmethod
    def add_passages(
        self,
        passages: list[dict[str, str]] | list[str],
        source: str = None,
    ):
        """
        Add passages to the retriever database
        """
        return

    def search_batch(
        self,
        query: list[str],
        top_k: int = 10,
        **search_kwargs,
    ) -> list[dict[str, str | list]]:
        if self._cache is None:
            return self._search_batch(query, top_k, **search_kwargs)

        # search from cache
        results = [None] * len(query)
        new_query = []
        new_indices = []
        for n, q in enumerate(query):
            if q in self._cache:
                results[n] = self._cache[q]
            else:
                new_indices.append(n)
                new_query.append(q)

        # search from database
        if new_query:
            new_results = self._search_batch(new_query, top_k, **search_kwargs)
            for n, r in zip(new_indices, new_results):
                results[n] = r
                self._cache[new_query[n]] = r
        assert all(r is not None for r in results)
        return results

    @abstractmethod
    def _search_batch(
        self,
        query: list[str],
        top_k: int = 10,
        **search_kwargs,
    ) -> list[dict[str, str | list]]:
        return

    @cached(cache=PersistentLRUCache(""))
    def search(
        self,
        query: list[str],
        top_k: int = 10,
        **search_kwargs,
    ) -> list[dict[str, str | list]]:
        # search for documents
        final_results = []
        for n, idx in enumerate(range(0, len(query), self.batch_size)):
            if n % self.log_interval == 0:
                logger.info(f"Searching for batch {n}")
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

    @property
    @abstractmethod
    def embedding_size(self):
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
            _ = self.search(query, top_k, **search_kwargs)
            end_time = time.perf_counter()
            total_times.append(end_time - start_time)
        avg_time = sum(total_times) / test_times
        std_time = np.std(total_times)
        logger.info(
            f"Retrieval {sample_num} items consume: {avg_time:.4f} ± {std_time:.4f} s"
        )
        return end_time - start_time
