from abc import ABC, abstractmethod
from dataclasses import dataclass
from time import perf_counter

import numpy as np

from flexrag.utils import (
    Choices,
    SimpleProgressLogger,
    Register,
    TIME_METER,
    LOGGER_MANAGER,
)


logger = LOGGER_MANAGER.get_logger("flexrag.retrievers.index")


@dataclass
class DenseIndexBaseConfig:
    """The configuration for the `DenseIndexBase`.

    :param distance_function: The distance function to use. Defaults to "IP".
        available choices are "IP" and "L2".
    :type distance_function: str
    :param index_train_num: The number of embeddings to train the index. Defaults to 1000000.
    :type index_train_num: int
    :log_interval: The interval to log the progress. Defaults to 10000.
    :type log_interval: int
    :batch_size: The batch size to add embeddings. Defaults to 512.
    :type batch_size: int
    """

    distance_function: Choices(["IP", "L2"]) = "IP"  # type: ignore
    index_train_num: int = 1000000
    log_interval: int = 10000
    batch_size: int = 512


class DenseIndexBase(ABC):
    """The base class for all dense indexes."""

    def __init__(self, cfg: DenseIndexBaseConfig, index_path: str = None):

        self.distance_function = cfg.distance_function
        self.index_train_num = cfg.index_train_num
        self.index_path = index_path
        self.batch_size = cfg.batch_size
        self.log_interval = cfg.log_interval
        return

    @abstractmethod
    def build_index(self, embeddings: np.ndarray) -> None:
        """Build the index with embeddings.
        The index will be serialized automatically if the `index_path` is set.

        :param embeddings: The embeddings to build the index.
        :type embeddings: np.ndarray
        :return: None
        """
        return

    def add_embeddings(self, embeddings: np.ndarray) -> None:
        """Add embeddings to the index.
        This method will add embeddings to the index in batches and automatically perform the `serialize` method when the `index_path` is set.

        :param embeddings: The embeddings to add.
        :type embeddings: np.ndarray
        :return: None
        """
        assert self.is_addable, "Index is not trained, please build the index first."
        p_logger = SimpleProgressLogger(
            logger, total=embeddings.shape[0], interval=self.log_interval
        )
        for idx in range(0, len(embeddings), self.batch_size):
            p_logger.update(step=self.batch_size, desc="Adding embeddings")
            embeds_to_add = embeddings[idx : idx + self.batch_size]
            self.add_embeddings_batch(embeds_to_add)
        if self.index_path is not None:
            self.serialize()
        return

    @abstractmethod
    def add_embeddings_batch(self, embeddings: np.ndarray) -> None:
        """Add a batch of embeddings to the index.
        This method will not serialize the index automatically.
        Thus, you should call the `serialize` method after adding all embeddings.

        :param embeddings: The embeddings to add.
        :type embeddings: np.ndarray
        :return: None
        """
        return

    @TIME_METER("retrieve", "index")
    def search(
        self,
        query: np.ndarray,
        top_k: int = 10,
        **search_kwargs,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Search for the top_k most similar embeddings to the query.

        :param query: The query embeddings with shape [n, d].
        :type query: np.ndarray
        :param top_k: The number of most similar embeddings to return, defaults to 10.
        :type top_k: int, optional
        :param search_kwargs: Additional search arguments.
        :type search_kwargs: Any
        :return: The indices and scores of the top_k most similar embeddings with shape [n, k].
        :rtype: tuple[np.ndarray, np.ndarray]
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
    def serialize(self, index_path: str = None) -> None:
        """Serialize the index to self.index_path.
        If the `index_path` is given, the index will be serialized to the `index_path`.

        :param index_path: The path to serialize the index. Defaults to None.
        :type index_path: str, optional
        """
        return

    @abstractmethod
    def deserialize(self) -> None:
        """Deserialize the index from self.index_path."""
        return

    @abstractmethod
    def clean(self) -> None:
        """Clean the index."""
        return

    @property
    @abstractmethod
    def embedding_size(self) -> int:
        """Return the embedding size of the index."""
        return

    @property
    @abstractmethod
    def is_addable(self) -> bool:
        """Return `True` if the index supports adding embeddings."""
        return

    @abstractmethod
    def __len__(self) -> int:
        """Return the number of embeddings in the index."""
        return

    def test_accuracy(self, queries: np.ndarray, labels: np.ndarray, top_k: int = 10):
        """Test the top-k accuracy of the index."""
        # search
        start_time = perf_counter()
        retrieved, _ = self.search(queries, top_k)
        end_time = perf_counter()
        time_cost = end_time - start_time

        # compute accuracy
        acc_map = labels.reshape(-1, 1) == retrieved
        top_k_acc = [acc_map[:, : k + 1].sum() / len(queries) for k in range(top_k)]

        # log accuracy and search time
        acc_info_str = "\n".join(
            [f"Top {k + 1} accuracy: {acc*100:.2f}%" for k, acc in enumerate(top_k_acc)]
        )
        logger.info(f"Top k accuracy:\n{acc_info_str}")
        logger.info(f"Search time: {time_cost:.4f} s")
        return top_k_acc


DENSE_INDEX = Register[DenseIndexBase]("index")
