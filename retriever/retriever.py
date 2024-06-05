import logging
import time
from abc import ABC, abstractmethod

import numpy as np


logger = logging.getLogger(__name__)


class Retriever(ABC):
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

    @abstractmethod
    def search_batch(
        self,
        query: list[str],
        top_k: int = 10,
        **search_kwargs,
    ) -> list[dict[str, str | list]]:
        return

    def search(
        self,
        query: list[str],
        top_k: int = 10,
        **search_kwargs,
    ) -> list[dict[str, str | list]]:
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
        total_times = []
        for _ in range(test_times):
            query = np.random.randn(sample_num, self.embedding_size).astype(np.float32)
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
