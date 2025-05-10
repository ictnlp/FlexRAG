import asyncio
import os
import tempfile
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Generator, Iterable, Optional

import numpy as np
from huggingface_hub import HfApi
from omegaconf import DictConfig, OmegaConf

from flexrag.common_dataclass import Context, RetrievedContext
from flexrag.text_process import TextProcessPipeline, TextProcessPipelineConfig
from flexrag.utils import (
    __VERSION__,
    FLEXRAG_CACHE_DIR,
    LOGGER_MANAGER,
    ConfigureBase,
    LRUPersistentCache,
    Register,
    SimpleProgressLogger,
)

logger = LOGGER_MANAGER.get_logger("flexrag.retrievers")


# load cache for retrieval
RETRIEVAL_CACHE: LRUPersistentCache | None
if os.environ.get("DISABLE_CACHE", "False") == "True":
    RETRIEVAL_CACHE = None
else:
    cache_size = os.environ.get("CACHE_SIZE", None)
    cache_path = os.environ.get(
        "CACHE_PATH", os.path.join(FLEXRAG_CACHE_DIR, "retrieval_cache")
    )
    RETRIEVAL_CACHE = LRUPersistentCache(maxsize=cache_size, cache_path=cache_path)


# TODO: fix this
def batched_cache(func):
    """The helper function to cache the retrieval results in batch.
    You can use this function to decorate the `search` method of the retriever class to cache the retrieval results in batch.
    """

    def dataclass_to_dict(data):
        if not isinstance(data, DictConfig):
            return OmegaConf.to_container(DictConfig(data))
        return OmegaConf.to_container(data)

    def retrieved_to_dict(data: list[RetrievedContext]) -> list[dict]:
        return [r.to_dict() for r in data]

    def dict_to_retrieved(data: list[dict] | None) -> list[RetrievedContext] | None:
        if data is None:
            return None
        return [RetrievedContext(**r) for r in data]

    def check(data: list):
        for d in data:
            assert isinstance(d, list)
            for r in d:
                assert isinstance(r, RetrievedContext)
        return

    def wrapper(
        self,
        query: list[str],
        disable_cache: bool = False,
        **search_kwargs,
    ):
        # check query
        if isinstance(query, str):
            query = [query]

        # direct search
        if (RETRIEVAL_CACHE is None) or disable_cache:
            return func(self, query, **search_kwargs)

        # search from cache
        cfg = dataclass_to_dict(self.cfg)
        keys = [
            {
                "retriever_config": cfg,
                "query": q,
                "search_kwargs": search_kwargs,
            }
            for q in query
        ]
        results = [dict_to_retrieved(RETRIEVAL_CACHE.get(k, None)) for k in keys]

        # search from database
        new_query = [q for q, r in zip(query, results) if r is None]
        new_indices = [n for n, r in enumerate(results) if r is None]
        if new_query:
            new_results = func(self, new_query, **search_kwargs)
            # update cache
            for n, r in zip(new_indices, new_results):
                results[n] = r
                RETRIEVAL_CACHE[keys[n]] = retrieved_to_dict(r)
        # check results
        check(results)
        return results

    return wrapper


@dataclass
class RetrieverBaseConfig(ConfigureBase):
    """Base configuration class for all retrievers.

    :param log_interval: The interval of logging. Default: 100.
    :type log_interval: int
    :param top_k: The number of retrieved documents. Default: 10.
    :type top_k: int
    :param batch_size: The batch size for retrieval. Default: 32.
    :type batch_size: int
    :param query_preprocess_pipeline: The text process pipeline for query. Default: TextProcessPipelineConfig.
    :type query_preprocess_pipeline: TextProcessPipelineConfig
    """

    log_interval: int = 100
    top_k: int = 10
    batch_size: int = 32
    query_preprocess_pipeline: TextProcessPipelineConfig = field(  # type: ignore
        default_factory=TextProcessPipelineConfig
    )


class RetrieverBase(ABC):
    """The base class for all retrievers.
    The subclasses should implement the ``search`` method and the ``fields`` property.
    """

    cfg: RetrieverBaseConfig

    def __init__(self, cfg: RetrieverBaseConfig):
        # load preprocess pipeline
        self.query_preprocess_pipeline = TextProcessPipeline(
            cfg.query_preprocess_pipeline
        )
        return

    async def async_search(
        self,
        query: list[Any],
        **search_kwargs,
    ) -> list[list[RetrievedContext]]:
        """Search queries asynchronously."""
        return await asyncio.to_thread(
            self.search,
            query=query,
            **search_kwargs,
        )

    @batched_cache
    def search_batch(
        self,
        query: Iterable[Any],
        no_preprocess: bool = False,
        **search_kwargs,
    ) -> list[list[RetrievedContext]]:
        """Search queries in batches.

        :param query: Queries to search.
        :type query: list[Any]
        :param no_preprocess: Whether to preprocess the query. Default: False.
        :type no_preprocess: bool
        :param search_kwargs: Other search arguments.
        :type search_kwargs: Any
        :return: A batch of list that contains k RetrievedContext.
        :rtype: list[list[RetrievedContext]]
        """

        def get_batch() -> Generator[list[Any], None, None]:
            batch = []
            for q in query:
                if not no_preprocess:
                    batch.append(self.query_preprocess_pipeline(q))
                else:
                    batch.append(q)
                if len(batch) == self.cfg.batch_size:
                    yield batch
                    batch = []
            if batch:
                yield batch
            return

        final_results = []
        total = len(query) if hasattr(query, "__len__") else None
        p_logger = SimpleProgressLogger(logger, total, self.cfg.log_interval)
        for batch in get_batch():
            results_ = self.search(batch, **search_kwargs)
            final_results.extend(results_)
            p_logger.update(1, "Retrieving")
        return final_results

    @abstractmethod
    def search(
        self,
        query: list[Any] | Any,
        **search_kwargs,
    ) -> list[list[RetrievedContext]]:
        """Search a batch of queries.

        :param query: Queries to search.
        :type query: list[Any] | Any
        :param search_kwargs: Keyword arguments, contains other search arguments.
        :type search_kwargs: Any
        :return: A batch of list that contains k RetrievedContext.
        :rtype: list[list[RetrievedContext]]
        """
        return

    @property
    @abstractmethod
    def fields(self) -> list[str]:
        """The fields of the retrieved data."""
        return

    def test_speed(
        self,
        sample_num: int = 10000,
        test_times: int = 10,
        **search_kwargs,
    ) -> float:
        """Test the speed of the retriever.

        :param sample_num: The number of samples to test.
        :type sample_num: int, optional
        :param test_times: The number of times to test.
        :type test_times: int, optional
        :return: The time consumed for retrieval.
        :rtype: float
        """
        from nltk.corpus import brown

        total_times = []
        sents = [" ".join(i) for i in brown.sents()]
        for _ in range(test_times):
            query = [sents[i % len(sents)] for i in range(sample_num)]
            start_time = time.perf_counter()
            _ = self.search(query, self.cfg.top_k, **search_kwargs)
            end_time = time.perf_counter()
            total_times.append(end_time - start_time)
        avg_time = sum(total_times) / test_times
        std_time = np.std(total_times)
        logger.info(
            f"Retrieval {sample_num} items consume: {avg_time:.4f} ± {std_time:.4f} s"
        )
        return end_time - start_time


RETRIEVERS = Register[RetrieverBase]("retriever", True)


@dataclass
class EditableRetrieverConfig(RetrieverBaseConfig):
    """Configuration class for LocalRetriever."""


class EditableRetriever(RetrieverBase):
    """The base class for all `editable` retrievers.
    In FlexRAG, the ``EditableRetriever`` is a concept referring to a retriever that includes the ``add_passages`` and ``clear`` methods,
    allowing you to build the retriever using your own knowledge base.
    FlexRAG provides following editable retrievers: ``FlexRetriever``, ``ElasticRetriever``, ``TypesenseRetriever``, and ``HydeRetriever``.
    The subclasses should implement the ``add_passages``, ``clear``, and ``__len__`` methods.
    """

    @abstractmethod
    def add_passages(self, passages: Iterable[Context]):
        """
        Add passages to the retriever database.

        :param passages: The passages to add.
        :type passages: Iterable[Context]
        :return: None
        """
        return

    @abstractmethod
    def clear(self) -> None:
        """Clear the retriever database."""
        return

    @abstractmethod
    def __len__(self):
        """Return the number of documents in the retriever database."""
        return


@dataclass
class LocalRetrieverConfig(EditableRetrieverConfig):
    """The configuration class for LocalRetriever.

    :param retriever_path: The path to the local database. Default: None.
        If specified, all modifications to the retriever will be applied simultaneously on the disk.
        If not specified, the retriever will be kept in memory.
    :type retriever_path: Optional[str]
    """

    retriever_path: Optional[str] = None


class LocalRetriever(EditableRetriever):
    """The base class for all `local` retrievers.

    In FlexRAG, the ``LocalRetriever`` is a concept referring to a retriever that can be saved to the local disk.
    The subclasses provide the ``save_to_local`` and ``load_from_local`` methods to save and load the retriever from the local disk,
    and the ``save_to_hub`` and ``load_from_hub`` methods to save and load the retriever from the HuggingFace Hub.

    FlexRAG provides following local retrievers: ``FlexRetriever``, and ``HydeRetriever``.

    For example, to load a retriever hosted on the HuggingFace Hub, you can run the following code:

    .. code-block:: python

        from flexrag.retriever import LocalRetriever

        retriever = LocalRetriever.load_from_hub("flexrag/wiki2021_atlas_bm25s")

    To save a retriever to the HuggingFace Hub, you can run the following code:

    .. code-block:: python

        retriever.save_to_hub("<your-repo-id>", token="<your-token>")

    """

    cfg: LocalRetrieverConfig

    @staticmethod
    def load_from_hub(
        repo_id: str,
        revision: str = None,
        token: str = None,
        cache_dir: str = FLEXRAG_CACHE_DIR,
        retriever_config: LocalRetrieverConfig = None,
        **kwargs,
    ) -> "LocalRetriever":
        """Load a retriever from the HuggingFace Hub.

        :param repo_id: The repo id of the retriever on the HuggingFace Hub.
        :type repo_id: str
        :param revision: The revision of the retriever on the HuggingFace Hub. Default: None.
        :type revision: str
        :param token: The token to access the HuggingFace Hub. Default: None.
        :type token: str
        :param cache_dir: The cache directory to store the retriever. Default: FLEXRAG_CACHE_DIR.
        :type cache_dir: str
        :param retriever_config: Additional configuration for the retriever. Default: None.
        :type retriever_config: LocalRetrieverConfig
        :param kwargs: Additional arguments for the retriever.
        :type kwargs: Any
        :return: The loaded retriever.
        :rtype: LocalRetriever
        """
        # check if the retriever exists
        api = HfApi(token=token)
        repo_info = api.repo_info(repo_id)
        if repo_info is None:
            raise ValueError(f"Retriever {repo_id} not found on the HuggingFace Hub.")
        repo_id = repo_info.id
        dir_name = os.path.join(
            cache_dir, f"{repo_id.split('/')[0]}--{repo_id.split('/')[1]}"
        )
        # lancedb does not support loading the database from a symlink
        snapshot = api.snapshot_download(
            repo_id=repo_id,
            revision=revision,
            token=token,
            local_dir=dir_name,
        )
        if snapshot is None:
            raise RuntimeError(f"Retriever {repo_id} download failed.")

        # load the retriever
        return LocalRetriever.load_from_local(snapshot, retriever_config, **kwargs)

    def save_to_hub(
        self,
        repo_id: str,
        token: str = os.environ.get("HF_TOKEN", None),
        commit_message: str = "Update FlexRAG retriever",
        retriever_card: str = None,
        private: bool = False,
        **kwargs,
    ) -> str:
        """Save the retriever to the HuggingFace Hub.

        :param repo_id: The repo id of the retriever on the HuggingFace Hub.
        :type repo_id: str
        :param token: The token to access the HuggingFace Hub. Default: None.
        :type token: str
        :param commit_message: The commit message for the retriever. Default: "Update FlexRAG retriever".
        :type commit_message: str
        :param retriever_card: The markdown readme file for the retriever. Default: None.
        :type retriever_card: str
        :param private: Whether to create a private repo. Default: False.
        :type private: bool
        :param kwargs: Additional arguments for uploading the retriever.
        :type kwargs: Any
        :return: The repo url of the retriever.
        :rtype: str
        """
        # make a temporary directory if retriever_path is not specified
        if self.cfg.retriever_path is None:
            with tempfile.TemporaryDirectory(prefix="flexrag-retriever") as tmp_dir:
                logger.info(
                    (
                        "As the `retriever_path` is not set, "
                        f"the retriever will be saved temporarily at {tmp_dir}."
                    )
                )
                self.save_to_local(tmp_dir, update_config=True)
                self.save_to_hub(
                    token=token,
                    repo_id=repo_id,
                    commit_message=commit_message,
                    retriever_card=retriever_card,
                    private=private,
                    **kwargs,
                )
            self.cfg.retriever_path = None
            return

        # prepare the client
        api = HfApi(token=token)

        # create repo if not exists
        repo_url = api.create_repo(
            repo_id=repo_id,
            token=api.token,
            private=private,
            repo_type="model",
            exist_ok=True,
        )
        repo_id = repo_url.repo_id

        # push to hub
        api.upload_folder(
            repo_id=repo_id,
            commit_message=commit_message,
            folder_path=self.cfg.retriever_path,
            **kwargs,
        )
        return repo_url

    @staticmethod
    def load_from_local(
        repo_path: str = None, retriever_config: LocalRetrieverConfig = None
    ) -> "LocalRetriever":
        """Load a retriever from the local disk.

        :param repo_path: The path to the local database. Default: None.
        :type repo_path: str
        :param retriever_config: Additional configuration for the retriever. Default: None.
        :type retriever_config: LocalRetrieverConfig
        :return: The loaded retriever.
        :rtype: LocalRetriever
        """
        # prepare the cls
        id_path = os.path.join(repo_path, "cls.id")
        with open(id_path, "r", encoding="utf-8") as f:
            retriever_name = f.read()
        retriever_cls = RETRIEVERS[retriever_name]["item"]
        config_cls = RETRIEVERS[retriever_name]["config_class"]

        # prepare the configuration
        config_path = os.path.join(repo_path, "config.yaml")
        with open(config_path, "r", encoding="utf-8") as f:
            local_cfg = OmegaConf.load(f)
        local_cfg = OmegaConf.merge(config_cls(), local_cfg)
        cfg: LocalRetrieverConfig
        if retriever_config is None:
            cfg = local_cfg
        else:
            cfg = OmegaConf.merge(local_cfg, retriever_config)
        cfg.retriever_path = repo_path

        # load the retriever
        retriever = retriever_cls(cfg)
        return retriever

    @abstractmethod
    def save_to_local(self, retriever_path: str = None):
        """Save the retriever to the local disk.

        :param retriever_path: The path to the local database. Default: None.
        :type retriever_path: str
        :return: None
        :rtype: None
        """
        return

    @abstractmethod
    def detach(self):
        """Detach the retriever from the local database.
        After detaching, the retriever will be kept in memory and all modifications will not be applied to the disk.
        """
        return
