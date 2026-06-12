import os
from abc import ABC, abstractmethod
from typing import Annotated, Any, Generator, Iterable, Optional
from uuid import uuid4

import numpy as np

from flexrag.common import (
    FLEXRAG_CACHE_DIR,
    LOGGER_MANAGER,
    Choices,
    ProgressDisplay,
    Register,
    SimpleProgressLogger,
    configure,
    trace,
)
from flexrag.models import EncoderProtocol

logger = LOGGER_MANAGER.get_logger("flexrag.retrievers.index")


@configure
class RetrieverIndexBaseConfig:
    """The configuration for the `RetrieverIndexBase`.

    :batch_size: The batch size to add data to the index. Defaults to 512.
    :type batch_size: int
    :param index_path: The path to save the index.
        If not specified, the index will be kept in memory.
        Defaults to None.
    :type index_path: Optional[str]
    """

    batch_size: int = 512
    index_path: Optional[str] = None


class RetrieverIndexBase(ABC):
    """The base class for all retriever indexes.
    This class provides the basic interface for building, adding, and searching the index.

    The subclass should implement the following methods:
    - `build_index`: Build the index from the data.
    - `insert`: Add a batch of data to the index.
    - `search`: Search for the top_k most similar data indices to the query.
    - `serialize`: Serialize the index to the disk.
    - `clear`: Clear the index and remove the serialized index files.
    - `__len__`: Return the number of data in the index.
    - `is_addable`: Return whether the index is addable.
    """

    cfg: RetrieverIndexBaseConfig

    @abstractmethod
    def build_index(self, data: Iterable[Any]) -> None:
        """Build the index.

        This method only builds the in-memory index state. Callers that own
        additional metadata should persist the complete index explicitly after
        building.

        :param data: The data to build the index.
        :type data: Iterable[Any]
        :return: None
        """
        return

    def insert_batch(
        self,
        data: Iterable[Any],
        batch_size: Optional[int] = None,
        serialize: bool = True,
        log_interval: int = 10000,
        display: ProgressDisplay = "auto",
    ) -> None:
        """Add data to the index in batches.
        This method will automatically perform the `serialize` method if the `index_path` is set.

        :param data: The data to add.
        :type data: Iterable[Any]
        :param batch_size: The batch size to add data to the index. Defaults to self.batch_size.
        :type batch_size: Optional[int]
        :param serialize: Whether to serialize the index after adding data. Defaults to True.
        :type serialize: bool
        :param log_interval: The interval to log the progress. Defaults to 10000.
        :type log_interval: int
        :param display: The display mode for progress updates. Defaults to "auto".
        :type display: ProgressDisplay
        :return: None
        """
        assert self.is_addable, "Current index is not addable."
        batch_size = batch_size or self.cfg.batch_size

        def get_data_batch() -> Generator[list[Any], None, None]:
            """A helper function that yields data in batches."""
            batch = []
            for item in data:
                batch.append(item)
                if len(batch) == batch_size:
                    yield batch
                    batch = []
            if batch:
                yield batch

        # iterate over the data in batches
        with SimpleProgressLogger(
            logger, interval=log_interval, display=display
        ) as p_logger:
            for batch in get_data_batch(data):
                self.insert(batch, serialize=False)
                p_logger.update(step=len(batch), desc="Adding data")

        # serialize if the `index_path` is set
        if (self.cfg.index_path is not None) and serialize:
            self.save_to_local()
        return

    @abstractmethod
    def insert(
        self,
        data: list[Any],
        serialize: bool = True,
    ) -> None:
        """Add a batch of data to the index.

        :param data: The data to add.
        :type data: list[Any]
        :param serialize: Whether to serialize the index after adding data. Defaults to True.
        :type serialize: bool
        :return: None
        """
        return

    @trace("retriever.index.search")
    def search_batch(
        self,
        query: Iterable[Any],
        top_k: int = 10,
        batch_size: Optional[int] = None,
        log_interval: int = 10000,
        display: ProgressDisplay = "auto",
        **search_kwargs,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Search for the top_k most similar data indices to the query.
        This method will search the index in batches.

        :param query: The query data.
        :type query: list[Any]
        :param top_k: The number of most similar data indices to return, defaults to 10.
        :type top_k: int, optional
        :param batch_size: The batch size to search. Defaults to self.batch_size.
        :type batch_size: Optional[int]
        :param log_interval: The interval to log the progress. Defaults to 10000.
        :type log_interval: int
        :param display: The display mode for progress updates. Defaults to "auto".
        :type display: ProgressDisplay
        :param search_kwargs: Additional search arguments.
        :type search_kwargs: Any
        :return: The indices and scores of the top_k most similar data indices.
        :rtype: tuple[np.ndarray, np.ndarray]
        """

        def get_batch():
            """Yield data in batches."""
            batch = []
            for item in query:
                batch.append(item)
                if len(batch) == batch_size:
                    yield batch
                    batch = []
            if batch:
                yield batch

        scores = []
        indices = []
        batch_size = batch_size or self.cfg.batch_size
        total = len(query) if hasattr(query, "__len__") else None
        with SimpleProgressLogger(
            logger, total, interval=log_interval, display=display
        ) as p_logger:
            for q in get_batch():
                r = self.search(q, top_k, **search_kwargs)
                scores.append(r[1])
                indices.append(r[0])
                p_logger.update(step=len(q), desc="Searching")
        scores = np.concatenate(scores, axis=0)
        indices = np.concatenate(indices, axis=0)
        return indices, scores

    @abstractmethod
    def search(
        self,
        query: list[Any],
        top_k: int,
        **search_kwargs,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Search for the top_k most similar data indices to the query.

        :param query: The query data.
        :type query: list[Any]
        :param top_k: The number of most similar data indices to return, defaults to 10.
        :type top_k: int, optional
        :param search_kwargs: Additional search arguments.
        :type search_kwargs: Any
        :return: The indices and scores of the top_k most similar data indices.
        :rtype: tuple[np.ndarray, np.ndarray]
        """
        return

    @property
    @abstractmethod
    def is_addable(self) -> bool:
        return

    @abstractmethod
    def save_to_local(self, index_path: Optional[str] = None) -> None:
        """Serialize the index to self.index_path.
        If the `index_path` is given, the index will be serialized to the `index_path`.

        :param index_path: The path to serialize the index. Defaults to self.index_path.
        :type index_path: str, optional
        """
        return

    @staticmethod
    def load_from_local(index_path: str, **kwargs) -> "RetrieverIndexBase":
        """Load the index from the local path.

        :param index_path: The path to load the index.
        :type index_path: str
        :param kwargs: Additional arguments to pass to the index constructor.
            Dense indexes require explicit encoder dependencies here.
        :type kwargs: Any
        """
        assert os.path.exists(index_path), f"Index path {index_path} does not exist."

        # load cls_id
        id_path = os.path.join(index_path, "cls.id")
        assert os.path.exists(id_path), f"Index ID file {id_path} does not exist."
        index_name = open(id_path, "r").read().strip()
        index_cls = RETRIEVER_INDEX[index_name]["item"]

        # load configuration
        config_cls = RETRIEVER_INDEX[index_name]["config_class"]
        config_path = os.path.join(index_path, "config.yaml")
        assert os.path.exists(config_path), (
            f"Configuration file {config_path} does not exist."
        )
        cfg = config_cls.load(config_path)
        cfg.index_path = index_path

        # load the index
        index = index_cls(cfg, **kwargs)
        return index

    @abstractmethod
    def clear(self) -> None:
        """Reset the index and remove the serialized index files."""
        return

    @abstractmethod
    def __len__(self) -> int:
        """Return the number of data in the index."""
        return

    @property
    @abstractmethod
    def infimum(self) -> float:
        """Return the infimum of the similarity scores for the index."""
        return

    @property
    @abstractmethod
    def supremum(self) -> float:
        """Return the supremum of the similarity scores for the index."""
        return


@configure
class DenseIndexBaseConfig(RetrieverIndexBaseConfig):
    """The configuration for the `DenseIndexBase`.

    :param distance_function: The distance function to use. Defaults to "IP".
        available choices are "IP", "L2", and "COS.
    :type distance_function: str
    """

    distance_function: Annotated[str, Choices("IP", "L2", "COS")] = "IP"


class DenseIndexBase(RetrieverIndexBase):
    """The base class for all dense indexes."""

    def __init__(
        self,
        cfg: DenseIndexBaseConfig,
        query_encoder: EncoderProtocol,
        passage_encoder: EncoderProtocol | None = None,
    ):
        self.query_encoder = query_encoder
        self.passage_encoder = passage_encoder or query_encoder
        self.distance_function = cfg.distance_function
        return

    def encode_data_batch(
        self,
        data: Iterable[Any],
        is_query: bool = False,
        use_memmap: bool = True,
        log_interval: int = 10000,
        display: ProgressDisplay = "auto",
    ) -> np.ndarray:
        """A helper function that encodes all data into embeddings.

        :param data: The data to encode.
        :type data: Iterable[dict[str, Any]]
        :param is_query: Whether the data is query data.
            If True, the query encoder will be used.
            If False, the passage encoder will be used.
            Defaults to False.
        :type is_query: bool
        :param use_memmap: Whether to use memory mapping for the embeddings.
            If True, the embeddings will be saved to disk and loaded as a memory map.
            If False, the embeddings will be kept in memory.
            Note that you should remove the memory map file after use.
            Defaults to True.
        :type use_memmap: bool
        :param log_interval: The interval to log the progress. Defaults to 10000.
        :type log_interval: int
        :param display: The display mode for progress updates. Defaults to "auto".
        :type display: ProgressDisplay
        :return: The embeddings of the data.
        :rtype: np.ndarray
        """

        # prepare_mmap_path
        if use_memmap:
            if self.cfg.index_path is not None:
                mmap_path = os.path.join(self.cfg.index_path, "embeddings")
            else:
                mmap_path = os.path.join(FLEXRAG_CACHE_DIR, "embeddings")
            os.makedirs(mmap_path, exist_ok=True)
        else:
            mmap_path = None

        def get_batch() -> Generator[tuple[list[int], list[Any]], None, None]:
            """A helper function that yields data in batches."""
            batch = []
            for item in data:
                batch.append(item)
                if len(batch) == self.cfg.batch_size:
                    yield batch
                    batch = []
            if batch:
                yield batch

        embeddings = []
        n_embeddings = 0
        # encode the data
        with SimpleProgressLogger(
            logger, interval=log_interval, display=display
        ) as p_logger:
            for batch in get_batch():
                emb = self.encode_data(batch, is_query=is_query)
                if mmap_path is not None:
                    file_name = os.path.join(mmap_path, f"{uuid4()}.npy")
                    np.save(file_name, emb)
                    embeddings.append(file_name)
                else:
                    embeddings.append(emb)
                n_embeddings += emb.shape[0]
                p_logger.update(step=len(batch), desc="Encoding data")

        # concatenate the embeddings
        if isinstance(embeddings[0], str):
            logger.info("Copying embeddings to memory map")
            emb_path = embeddings[0]
            emb = np.load(emb_path)
            emb_map = np.memmap(
                os.path.join(mmap_path, f"embeddings.npy"),
                dtype=np.float32,
                mode="w+",
                shape=(n_embeddings, emb.shape[1]),
            )
            idx = 0
            for emb_path in embeddings:
                emb = np.load(emb_path)
                emb_map[idx : idx + emb.shape[0]] = emb
                idx += emb.shape[0]
                del emb
                os.remove(emb_path)
            embeddings = emb_map
        else:
            embeddings = np.concatenate(embeddings, axis=0)
        return embeddings

    def encode_data(self, data: list[Any], is_query: bool = False) -> np.ndarray:
        """A helper function that encodes the data using the encoder.

        :param data: The data to be encoded.
        :type data: list[Any]
        :param is_query: Whether the data is query data.
            If True, the query encoder will be used.
            If False, the passage encoder will be used.
            Defaults to False.
        :type is_query: bool
        :return: The encoded data.
        :rtype: np.ndarray
        """
        # set the encoder
        if is_query:
            assert self.query_encoder is not None, "Query encoder is not set."
            encoder = self.query_encoder
        else:
            assert self.passage_encoder is not None, "Passage encoder is not set."
            encoder = self.passage_encoder

        # encode the data
        embeds = encoder.encode(data).astype("float32")
        return self._normalize_embeddings_for_metric(embeds)

    def _normalize_embeddings_for_metric(self, embeds: np.ndarray) -> np.ndarray:
        if self.distance_function != "COS" or embeds.size == 0:
            return embeds
        norms = np.linalg.norm(embeds, axis=1, keepdims=True)
        norms = np.clip(norms, a_min=1e-12, a_max=None)
        return embeds / norms

    def _postprocess_scores(self, scores: np.ndarray) -> np.ndarray:
        if self.distance_function == "L2":
            return -scores
        return scores

    def add_embeddings_batch(
        self,
        embeds: np.ndarray,
        log_interval: int = 10000,
        display: ProgressDisplay = "auto",
    ) -> None:
        """A helper function that adds embeddings to the index in batches.
        This method will not serialize the index automatically.
        Thus, you should call the `serialize` method after adding all data.

        :param embeds: The embeddings to add.
        :type embeds: np.ndarray
        :param log_interval: The interval to log the progress. Defaults to 10000.
        :type log_interval: int
        :param display: The display mode for progress updates. Defaults to "auto".
        :type display: ProgressDisplay
        :return: None
        """
        with SimpleProgressLogger(
            logger, embeds.shape[0], log_interval, display=display
        ) as p_logger:
            for i in range(0, embeds.shape[0], self.cfg.batch_size):
                batch_embeds = embeds[i : i + self.cfg.batch_size]
                self.add_embeddings(batch_embeds)
                p_logger.update(step=batch_embeds.shape[0], desc="Adding embeddings")
        return

    @abstractmethod
    def add_embeddings(self, embeds: np.ndarray) -> None:
        """A helper function that adds embeddings to the index.

        :param embeds: The embeddings to add.
        :type embeds: np.ndarray
        :return: None
        """
        return

    def insert(self, data: list[Any]) -> None:
        embeddings = self.encode_data(data, is_query=False)
        self.add_embeddings(embeddings)
        return

    @property
    @abstractmethod
    def embedding_size(self) -> int:
        """Return the embedding size of the index."""
        return

    @property
    def infimum(self) -> float:
        # Dense index scores follow the convention "larger is more relevant".
        if self.distance_function == "L2":
            return float("-inf")
        if self.distance_function == "COS":
            return -1.0
        return float("-inf")

    @property
    def supremum(self) -> float:
        # Dense index scores follow the convention "larger is more relevant".
        if self.distance_function == "L2":
            return 0.0
        if self.distance_function == "COS":
            return 1.0
        return float("inf")


RETRIEVER_INDEX = Register[RetrieverIndexBase]("index")
