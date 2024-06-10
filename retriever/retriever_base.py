import logging
import time
import os
from abc import ABC, abstractmethod
from argparse import ArgumentParser, Namespace

import numpy as np

from .cache import PersistentLRUCache, hashkey
from .utils import normalize


logger = logging.getLogger(__name__)


class Retriever(ABC):
    @staticmethod
    def add_args(parser: ArgumentParser) -> ArgumentParser:
        parser.add_argument(
            "--database_path",
            type=str,
            required=True,
            help="The path to the Retriever database",
        )
        parser.add_argument(
            "--cache_size",
            default=0,
            type=int,
            help="The size of the cache",
        )
        return parser

    @abstractmethod
    def search(
        self,
        query: list[str],
        top_k: int = 10,
        **search_kwargs,
    ) -> list[dict[str, str | list]]:
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


class LocalRetriever(Retriever):
    @staticmethod
    def add_args(parser: ArgumentParser) -> ArgumentParser:
        parser.add_argument(
            "--batch_size",
            type=int,
            default=512,
            help="The batch size to use for encoding",
        )
        # encoding arguments
        parser.add_argument(
            "--max_query_length",
            type=int,
            default=256,
            help="The maximum length of the queries",
        )
        parser.add_argument(
            "--max_passage_length",
            type=int,
            default=2048,
            help="The maximum length of the passages",
        )
        parser.add_argument(
            "--no_title",
            action="store_true",
            default=False,
            help="Whether to include the title in the passage",
        )
        parser.add_argument(
            "--lowercase",
            action="store_true",
            default=False,
            help="Whether to lowercase the text",
        )
        parser.add_argument(
            "--normalize_text",
            action="store_true",
            default=False,
            help="Whether to normalize the text",
        )
        return parser

    def __init__(self, args: Namespace) -> None:
        # set args for process documents
        self.batch_size = args.batch_size
        self.max_query_length = args.max_query_length
        self.max_passage_length = args.max_passage_length
        self.no_title = args.no_title
        self.lowercase = args.lowercase
        self.normalize_text = args.normalize_text
        self.log_interval = args.log_interval

        # set cache for retrieve
        if args.cache_size > 0:
            self._cache = PersistentLRUCache(
                persistant_path=os.path.join(args.database_path, "cache"),
                maxsize=args.cache_size,
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
        disable_cache: bool = False,
        **search_kwargs,
    ) -> list[dict[str, str | list]]:
        if (self._cache is None) or disable_cache:
            return self._search_batch(query, top_k, **search_kwargs)

        # search from cache
        cache_keys = [hashkey(query=q, top_k=top_k, **search_kwargs) for q in query]
        results = [None] * len(query)
        new_query = []
        new_indices = []
        for n, (q, k) in enumerate(zip(query, cache_keys)):
            if k in self._cache:
                results[n] = self._cache[k]
            else:
                new_indices.append(n)
                new_query.append(q)

        # search from database
        if new_query:
            new_results = self._search_batch(new_query, top_k, **search_kwargs)
            for n, r in zip(new_indices, new_results):
                results[n] = r
                self._cache[cache_keys[n]] = r
        assert all(r is not None for r in results)
        return results

    @abstractmethod
    def _search_batch(
        self,
        query: list[str],
        top_k: int = 10,
        **search_kwargs,
    ) -> list[dict[str, str | list]]:
        """Search queries using local retriever.

        Args:
            query (list[str]): Queries to search.
            top_k (int, optional): N documents to return. Defaults to 10.

        Returns:
            list[dict[str, str | list]]: The retrieved documents: [
                {
                    "indices": list[int|str],
                    "scores": list[float],
                    "titles": list[str],
                    "sections": list[str],
                    "texts": list[str],
                }
            ]
        """
        return

    def search(
        self,
        query: list[str] | str,
        top_k: int = 10,
        disable_cache: bool = False,
        **search_kwargs,
    ) -> list[dict[str, str | list]]:
        # search for documents
        query = [query] if isinstance(query, str) else query
        final_results = []
        for n, idx in enumerate(range(0, len(query), self.batch_size)):
            if (n % self.log_interval == 0) and (len(query) > self.batch_size):
                logger.info(
                    f"Searching for batch {n} / {len(query) // self.batch_size}"
                )
            batch = query[idx : idx + self.batch_size]
            results_ = self.search_batch(batch, top_k, disable_cache, **search_kwargs)
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
            text = normalize(text)
        return text
