from abc import ABC, abstractmethod

import numpy as np


class DenseIndex(ABC):
    @abstractmethod
    def train_index(self, embedings: np.ndarray):
        return

    @abstractmethod
    def add_embeddings(self, embeddings: np.ndarray, ids: np.ndarray):
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

    @property
    @abstractmethod
    def is_trained(self) -> bool:
        """Return True if the index is trained."""
        return

    @abstractmethod
    def __len__(self) -> int:
        """Return the number of embeddings in the index."""
        return
