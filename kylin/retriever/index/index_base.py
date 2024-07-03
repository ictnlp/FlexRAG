from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np
from omegaconf import MISSING

from kylin.utils import Choices


@dataclass
class DenseIndexConfig:
    distance_function: Choices(["IP", "L2"]) = "IP"  # type: ignore
    embedding_size: Optional[int] = None
    index_train_num: int = 1000000
    index_path: str = MISSING
    log_interval: int = 1000


class DenseIndex(ABC):
    def __init__(self, cfg: DenseIndexConfig):
        self.distance_function = cfg.distance_function
        self.index_train_num = cfg.index_train_num
        self.index_path = cfg.index_path
        return

    @abstractmethod
    def train_index(self, embedings: np.ndarray):
        return

    @abstractmethod
    def add_embeddings(self, embeddings: np.ndarray, ids: np.ndarray, batch_size: int):
        return

    @abstractmethod
    def add_embeddings_batch(self, embeddings: np.ndarray, ids: np.ndarray):
        return

    @abstractmethod
    def search(
        self,
        query: np.ndarray,
        top_k: int = 10,
        batch_size: int = 512,
        **search_kwargs,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Search for the top_k most similar embeddings to the query.

        Args:
            query (np.ndarray): query embeddings. [n, d]
            top_k (int, optional): Number of most similar embeddings to return. Defaults to 10.
            batch_size (int, optional): Batch size for searching. Defaults to 512.
            search_kwargs (dict, optional): Additional search arguments. Defaults to {}.

        Returns:
            tuple[np.ndarray, np.ndarray]: (ids [n, k], scores [n, k])
        """
        return

    @abstractmethod
    def search_batch(
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
