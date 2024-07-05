import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np
from omegaconf import MISSING

from kylin.utils import Choices, SimpleProgressLogger


logger = logging.getLogger("DenseIndex")


@dataclass
class DenseIndexConfig:
    distance_function: Choices(["IP", "L2"]) = "IP"  # type: ignore
    embedding_size: Optional[int] = None
    index_train_num: int = 1000000
    index_path: str = MISSING
    log_interval: int = 10000
    batch_size: int = 512


class DenseIndex(ABC):
    def __init__(self, cfg: DenseIndexConfig):
        self.distance_function = cfg.distance_function
        self.index_train_num = cfg.index_train_num
        self.index_path = cfg.index_path
        self.batch_size = cfg.batch_size
        self.log_interval = cfg.log_interval
        return

    @abstractmethod
    def build_index(self, embeddings: np.ndarray, ids: np.ndarray | list[int] = None):
        return

    def add_embeddings(
        self, embeddings: np.ndarray, ids: np.ndarray | list[int] = None
    ) -> None:
        if ids is None:
            ids = list(range(len(self), len(self) + len(embeddings)))
        assert len(ids) == len(embeddings)

        p_logger = SimpleProgressLogger(
            logger, total=embeddings.shape[0], interval=self.log_interval
        )
        for idx in range(0, len(embeddings), self.batch_size):
            p_logger.update(step=self.batch_size, desc="Adding embeddings")
            embeds_to_add = embeddings[idx : idx + self.batch_size]
            if ids is not None:
                ids_to_add = ids[idx : idx + self.batch_size]
            else:
                ids_to_add = None
            self._add_embeddings_batch(embeds_to_add, ids_to_add)
        self.serialize()
        return

    @abstractmethod
    def _add_embeddings_batch(
        self, embeddings: np.ndarray, ids: np.ndarray | list[int]
    ) -> None:
        return

    def search(
        self,
        query: np.ndarray,
        top_k: int = 10,
        **search_kwargs,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Search for the top_k most similar embeddings to the query.

        Args:
            query (np.ndarray): query embeddings. [n, d]
            top_k (int, optional): Number of most similar embeddings to return. Defaults to 10.
            search_kwargs (dict, optional): Additional search arguments. Defaults to {}.

        Returns:
            tuple[np.ndarray, np.ndarray]: (ids [n, k], scores [n, k])
        """
        scores = []
        indices = []
        p_logger = SimpleProgressLogger(
            logger, total=query.shape[0], interval=self.log_interval
        )
        for idx in range(0, len(query), self.batch_size):
            p_logger.update(step=self.batch_size, desc="Searching")
            q = query[idx : idx + self.batch_size]
            r = self._search_batch(q, top_k, **search_kwargs)
            scores.append(r[1])
            indices.append(r[0])
        scores = np.concatenate(scores, axis=0)
        indices = np.concatenate(indices, axis=0)
        return indices, scores

    @abstractmethod
    def _search_batch(
        self, query: np.ndarray, top_k: int, **search_kwargs
    ) -> tuple[np.ndarray, np.ndarray]:
        return

    @abstractmethod
    def serialize(self):
        """Serialize the index to self.index_path."""
        return

    @abstractmethod
    def deserialize(self) -> None:
        """Deserialize the index from self.index_path."""
        return

    @abstractmethod
    def clear(self) -> None:
        """Clear the index."""
        return

    @property
    @abstractmethod
    def is_trained(self) -> bool:
        """Return True if the index is trained."""
        return

    @property
    @abstractmethod
    def embedding_size(self) -> int:
        """Return the embedding size of the index."""
        return

    @abstractmethod
    def __len__(self) -> int:
        """Return the number of embeddings in the index."""
        return
