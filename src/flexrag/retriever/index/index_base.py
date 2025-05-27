import os
from abc import ABC, abstractmethod
from dataclasses import field
from typing import Annotated, Any, Generator, Iterable, Optional
from uuid import uuid4

import numpy as np
from huggingface_hub import HfApi
from omegaconf import OmegaConf

from flexrag.models import ENCODERS, EncoderConfig
from flexrag.utils import (
    FLEXRAG_CACHE_DIR,
    LOGGER_MANAGER,
    TIME_METER,
    Choices,
    Register,
    SimpleProgressLogger,
    configure,
)

logger = LOGGER_MANAGER.get_logger("flexrag.retrievers.index")


@configure
class RetrieverIndexBaseConfig:
    """The configuration for the `RetrieverIndexBase`.

    :log_interval: The interval to log the progress. Defaults to 10000.
    :type log_interval: int
    :batch_size: The batch size to add data to the index. Defaults to 512.
    :type batch_size: int
    :param index_path: The path to save the index.
        If not specified, the index will be kept in memory.
        Defaults to None.
    :type index_path: Optional[str]
    """

    log_interval: int = 10000
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
        The index will be serialized automatically if the `index_path` is set.

        :param data: The data to build the index.
        :type data: Iterable[Any]
        :return: None
        """
        return

    def insert_batch(
        self,
        data: Iterable[Any],
        batch_size: int = None,
        serialize: bool = True,
    ) -> None:
        """Add data to the index in batches.
        This method will automatically perform the `serialize` method if the `index_path` is set.

        :param data: The data to add.
        :type data: Iterable[Any]
        :param batch_size: The batch size to add data to the index. Defaults to self.batch_size.
        :type batch_size: int
        :param serialize: Whether to serialize the index after adding data. Defaults to True.
        :type serialize: bool
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
        p_logger = SimpleProgressLogger(logger, interval=self.cfg.log_interval)
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

    @TIME_METER("retrieve", "index", "search")
    def search_batch(
        self,
        query: Iterable[Any],
        top_k: int = 10,
        batch_size: Optional[int] = None,
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
        p_logger = SimpleProgressLogger(logger, total, interval=self.cfg.log_interval)
        for q in get_batch():
            r = self.search(q, top_k, **search_kwargs)
            scores.append(r[1])
            indices.append(r[0])
            p_logger.update(step=batch_size, desc="Searching")
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
    def save_to_local(self, index_path: str = None) -> None:
        """Serialize the index to self.index_path.
        If the `index_path` is given, the index will be serialized to the `index_path`.

        :param index_path: The path to serialize the index. Defaults to self.index_path.
        :type index_path: str, optional
        """
        return

    @staticmethod
    def load_from_local(index_path: str) -> None:
        """Load the index from the local path.

        :param index_path: The path to load the index.
        :type index_path: str
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
        assert os.path.exists(
            config_path
        ), f"Configuration file {config_path} does not exist."
        with open(config_path, "r", encoding="utf-8") as f:
            local_cfg = OmegaConf.load(f)
        cfg = OmegaConf.merge(config_cls, local_cfg)
        cfg.index_path = index_path

        # load the index
        index = index_cls(cfg)
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

    :param query_encoder_config: Configuration for the query encoder. Default: None.
    :type query_encoder_config: EncoderConfig
    :param passage_encoder_config: Configuration for the passage encoder. Default: None.
    :type passage_encoder_config: EncoderConfig
    :param distance_function: The distance function to use. Defaults to "IP".
        available choices are "IP", "L2", and "COS.
    :type distance_function: str
    """

    query_encoder_config: EncoderConfig = field(default_factory=EncoderConfig)  # type: ignore
    passage_encoder_config: EncoderConfig = field(default_factory=EncoderConfig)  # type: ignore
    distance_function: Annotated[str, Choices("IP", "L2", "COS")] = "IP"


class DenseIndexBase(RetrieverIndexBase):
    """The base class for all dense indexes."""

    def __init__(self, cfg: DenseIndexBaseConfig):
        # load the encoder
        self.query_encoder = ENCODERS.load(cfg.query_encoder_config)
        self.passage_encoder = ENCODERS.load(cfg.passage_encoder_config)

        # set basic args
        self.distance_function = cfg.distance_function
        return

    def encode_data_batch(
        self, data: Iterable[Any], is_query: bool = False, use_memmap: bool = True
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

        # encode the data
        p_logger = SimpleProgressLogger(logger, interval=self.cfg.log_interval)
        embeddings = []
        n_embeddings = 0
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
        embeds = encoder.encode(data)
        return embeds.astype("float32")

    def add_embeddings_batch(self, embeds: np.ndarray) -> None:
        """A helper function that adds embeddings to the index in batches.
        This method will not serialize the index automatically.
        Thus, you should call the `serialize` method after adding all data.

        :param embeds: The embeddings to add.
        :type embeds: np.ndarray
        :return: None
        """
        p_logger = SimpleProgressLogger(logger, embeds.shape[0], self.cfg.log_interval)
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

    @staticmethod
    def check_configuration(cfg: DenseIndexBaseConfig, token: str = None) -> None:
        """A helper function that checks the configuration of the index.
        This function is useful before saving the index to the HuggingFace hub.
        It checks if the encoder is available in the HuggingFace hub.
        If the encoder is not available, it will raise a warning.

        :param cfg: The configuration of the index.
        :type cfg: DenseIndexBaseConfig
        :param token: The token to access the HuggingFace hub. Defaults to None.
        :type token: str
        """
        client = HfApi(token=token)
        # check the query encoder
        if cfg.query_encoder_config.encoder_type is None:
            logger.warning(
                "Query encoder is not provided. "
                "Please make sure loading the appropriate encoder when loading the retriever."
            )
        elif cfg.query_encoder_config.encoder_type == "hf":
            if not client.repo_exists(cfg.query_encoder_config.hf_config.model_path):
                logger.warning(
                    "Query encoder model is not available in the HuggingFace model hub."
                    "Please make sure loading the appropriate encoder when loading the retriever."
                )
        else:
            logger.warning(
                "Query encoder is not a model hosted on the HuggingFace model hub."
                "Please make sure loading the appropriate encoder when loading the retriever."
            )
        # check the passage encoder
        if cfg.passage_encoder_config.encoder_type is None:
            logger.warning(
                "Passage encoder is not provided. "
                "Please make sure loading the appropriate encoder when loading the retriever."
            )
        elif cfg.passage_encoder_config.encoder_type == "hf":
            if not client.repo_exists(cfg.passage_encoder_config.hf_config.model_path):
                logger.warning(
                    "Passage encoder model is not available in the HuggingFace model hub."
                    "Please make sure loading the appropriate encoder when loading the retriever."
                )
        else:
            logger.warning(
                "Passage encoder is not a model hosted on the HuggingFace model hub."
                "Please make sure loading the appropriate encoder when loading the retriever."
            )
        return

    @property
    def infimum(self) -> float:
        # For L2 distance, the infimum is 0.0
        # For IP distance, the infimum is -infinity
        # For COS distance, the infimum is -1.0
        return 0.0

    @property
    def supremum(self) -> float:
        # For L2 distance, the supremum is infinity
        # For IP distance, the supremum is infinity
        # For COS distance, the supremum is 1.0
        if self.distance_function == "COS":
            return 1.0
        return float("inf")


RETRIEVER_INDEX = Register[RetrieverIndexBase]("index")
