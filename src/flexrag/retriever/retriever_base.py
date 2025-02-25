import asyncio
import os
import tempfile
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterable, Optional

import numpy as np
from huggingface_hub import HfApi
from omegaconf import DictConfig, OmegaConf

from flexrag.utils import __VERSION__
from flexrag.cache import (
    FIFOPersistentCache,
    LFUPersistentCache,
    LMDBBackendConfig,
    LRUPersistentCache,
    PersistentCacheBase,
    PersistentCacheConfig,
)
from flexrag.common_dataclass import Context, RetrievedContext
from flexrag.text_process import TextProcessPipeline, TextProcessPipelineConfig
from flexrag.utils import (
    FLEXRAG_CACHE_DIR,
    LOGGER_MANAGER,
    Register,
    SimpleProgressLogger,
)

logger = LOGGER_MANAGER.get_logger("flexrag.retrievers")


# load cache for retrieval
RETRIEVAL_CACHE: PersistentCacheBase | None
if os.environ.get("DISABLE_CACHE", "False") == "True":
    RETRIEVAL_CACHE = None
else:
    cache_config = PersistentCacheConfig(
        maxsize=10000000,
        storage_backend_type=os.environ.get("CACHE_BACKEND", "dict"),
        lmdb_config=LMDBBackendConfig(
            db_path=os.path.join(FLEXRAG_CACHE_DIR, "cache.lmdb")
        ),
    )
    match os.environ.get("RETRIEVAL_CACHE_TYPE", "FIFO"):
        case "LRU":
            RETRIEVAL_CACHE = LRUPersistentCache(cache_config)
        case "LFU":
            RETRIEVAL_CACHE = LFUPersistentCache(cache_config)
        case "FIFO":
            RETRIEVAL_CACHE = FIFOPersistentCache(cache_config)
        case _:
            logger.warning("Invalid cache type, cache is disabled.")
            RETRIEVAL_CACHE = None


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
class RetrieverBaseConfig:
    """Base configuration class for all retrievers.

    :param log_interval: The interval of logging. Default: 100.
    :type log_interval: int
    :param top_k: The number of retrieved documents. Default: 10.
    :type top_k: int
    """

    log_interval: int = 100
    top_k: int = 10


class RetrieverBase(ABC):
    """The base class for all retrievers.
    The subclasses should implement the ``search`` method and the ``fields`` property.
    """

    def __init__(self, cfg: RetrieverBaseConfig):
        self.cfg = cfg
        self.log_interval = cfg.log_interval
        self.top_k = cfg.top_k
        return

    async def async_search(
        self,
        query: list[str],
        **search_kwargs,
    ) -> list[list[RetrievedContext]]:
        """Search queries asynchronously."""
        return await asyncio.to_thread(
            self.search,
            query=query,
            **search_kwargs,
        )

    @abstractmethod
    def search(
        self,
        query: list[str],
        **search_kwargs,
    ) -> list[list[RetrievedContext]]:
        """Search queries.

        :param query: Queries to search.
        :type query: list[str]
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
            _ = self.search(query, self.top_k, **search_kwargs)
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
    """Configuration class for LocalRetriever.

    :param batch_size: The batch size for retrieval. Default: 32.
    :type batch_size: int
    :param query_preprocess_pipeline: The text process pipeline for query. Default: TextProcessPipelineConfig.
    :type query_preprocess_pipeline: TextProcessPipelineConfig
    """

    batch_size: int = 32
    query_preprocess_pipeline: TextProcessPipelineConfig = field(default_factory=TextProcessPipelineConfig)  # type: ignore


class EditableRetriever(RetrieverBase):
    """The base class for all `editable` retrievers.
    In FlexRAG, the ``EditableRetriever`` is a concept referring to a retriever that includes the ``add_passages`` and ``clean`` methods,
    allowing you to build the retriever using your own knowledge base.
    FlexRAG provides following editable retrievers: ``BM25SRetriever``, ``DenseRetriever``, ``ElasticRetriever``, ``TypesenseRetriever``, and ``HydeRetriever``.
    """

    def __init__(self, cfg: EditableRetrieverConfig) -> None:
        super().__init__(cfg)
        # set args for process documents
        self.batch_size = cfg.batch_size
        self.query_preprocess_pipeline = TextProcessPipeline(
            cfg.query_preprocess_pipeline
        )
        return

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
    def search_batch(
        self,
        query: list[str],
        **search_kwargs,
    ) -> list[list[RetrievedContext]]:
        """Search queries using local retriever.

        :param query: Queries to search.
        :type query: list[str]
        :return: A batch of list that contains k RetrievedContext.
        :rtype: list[list[RetrievedContext]]
        """
        return

    @batched_cache
    def search(
        self,
        query: list[str] | str,
        no_preprocess: bool = False,
        **search_kwargs,
    ) -> list[list[RetrievedContext]]:
        # search for documents
        query = [query] if isinstance(query, str) else query
        if not no_preprocess:
            query = [self.query_preprocess_pipeline(q) for q in query]
        final_results = []
        p_logger = SimpleProgressLogger(logger, len(query), self.log_interval)
        for idx in range(0, len(query), self.batch_size):
            p_logger.update(1, "Retrieving")
            batch = query[idx : idx + self.batch_size]
            results_ = self.search_batch(batch, **search_kwargs)
            final_results.extend(results_)
        return final_results

    @abstractmethod
    def clean(self) -> None:
        """Clean the retriever database."""
        return

    @abstractmethod
    def __len__(self):
        """Return the number of documents in the retriever database."""
        return


RETRIEVER_CARD_TEMPLATE = """---
language: en
library_name: FlexRAG
tags:
- FlexRAG
- retrieval
- search
- lexical
- RAG
---

# FlexRAG Retriever

This is a {retriever_type} created with the [`FlexRAG`](https://github.com/ictnlp/flexrag) library (version `{version}`).

## Installation

You can install the `FlexRAG` library with `pip`:

```bash
pip install flexrag
```

## Loading a `FlexRAG` retriever

You can use this retriever for information retrieval tasks. Here is an example:

```python
from flexrag.retriever import LocalRetriever

# Load the retriever from the HuggingFace Hub
retriever = LocalRetriever.load_from_hub("{repo_id}")

# You can retrieve now
results = retriever.search("Who is Bruce Wayne?")
```

FlexRAG Related Links:
* 📚[Documentation](https://flexrag.readthedocs.io/en/latest/)
* 💻[GitHub Repository](https://github.com/ictnlp/flexrag)
"""


@dataclass
class LocalRetrieverConfig(EditableRetrieverConfig):
    """The configuration class for LocalRetriever.

    :param database_path: The path to the local database. Default: None.
        If specified, all modifications to the retriever will be applied simultaneously on the disk.
        If not specified, the retriever will be kept in memory.
    :type database_path: Optional[str]
    """

    database_path: Optional[str] = None


class LocalRetriever(EditableRetriever):
    """The base class for all `local` retrievers.

    In FlexRAG, the ``LocalRetriever`` is a concept referring to a retriever that can be saved to the local disk.
    The subclasses provide the ``save_to_local`` and ``load_from_local`` methods to save and load the retriever from the local disk,
    and the ``save_to_hub`` and ``load_from_hub`` methods to save and load the retriever from the HuggingFace Hub.

    FlexRAG provides following local retrievers: ``BM25SRetriever``, ``DenseRetriever``, and ``HydeRetriever``.

    For example, to load a retriever hosted on the HuggingFace Hub, you can run the following code:

    .. code-block:: python

        from flexrag.retriever import LocalRetriever

        retriever = LocalRetriever.load_from_hub("flexrag/wiki2021_atlas_bm25s")

    You can also override the configuration when loading the retriever:

    .. code-block:: python

        from flexrag.retriever import LocalRetriever, BM25SRetrieverConfig

        cfg = BM25SRetrieverConfig(top_k=20)
        retriever = LocalRetriever.load_from_hub("flexrag/wiki2021_atlas_bm25s", retriever_config=cfg)

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
        # make a temporary directory if database_path is not specified
        if self.cfg.database_path is None:
            with tempfile.TemporaryDirectory(prefix="flexrag-retriever") as tmp_dir:
                logger.info(
                    (
                        "As the `database_path` is not set, "
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
            self.cfg.database_path = None
            return
        else:
            # make sure the configuration file is saved
            if retriever_card is None:
                if not os.path.exists(
                    os.path.join(self.cfg.database_path, "README.md")
                ):
                    retriever_card = RETRIEVER_CARD_TEMPLATE.format(
                        retriever_type=self.__class__.__name__,
                        repo_id=repo_id,
                        version=__VERSION__,
                    )
            self._save_configures(self.cfg.database_path, retriever_card)

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
            folder_path=self.cfg.database_path,
            **kwargs,
        )
        return repo_url

    @staticmethod
    def load_from_local(
        repo_path: str = None, retriever_config: LocalRetrieverConfig = None
    ) -> "LocalRetriever":
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
        if retriever_config is None:
            cfg = local_cfg
        else:
            cfg = OmegaConf.merge(local_cfg, retriever_config)
        cfg.database_path = repo_path

        # load the retriever
        retriever = retriever_cls(cfg)
        return retriever

    def save_to_local(
        self,
        database_path: str = None,
        overwrite: bool = False,
        retriever_card: str = None,
        update_config: bool = False,
    ):
        # check if the database_path is available
        db_path = database_path or self.cfg.database_path
        if db_path is None:
            raise ValueError("The `database_path` is not specified.")
        if not os.path.exists(db_path):
            os.makedirs(db_path, exist_ok=True)
        if not len(os.listdir(db_path)) == 0:
            if not overwrite:
                raise ValueError(f"Database path {db_path} is not empty.")

        # save the configures
        self._save_configures(db_path, retriever_card)

        # save the retriever
        self._save_to_local(db_path)

        # update the configuration
        if update_config:
            self.cfg.database_path = db_path
        return

    def _save_configures(self, database_path: str, retriever_card: str = None):
        # save the retriever card
        if retriever_card is not None:
            card_path = os.path.join(database_path, "README.md")
            with open(card_path, "w", encoding="utf-8") as f:
                f.write(retriever_card)

        # save the configuration
        config_path = os.path.join(database_path, "config.yaml")
        with open(config_path, "w", encoding="utf-8") as f:
            OmegaConf.save(self.cfg, f)
        id_path = os.path.join(database_path, "cls.id")
        with open(id_path, "w", encoding="utf-8") as f:
            f.write(self.__class__.__name__)
        return

    @abstractmethod
    def _save_to_local(self, database_path: str):
        return

    def detach(self):
        """Detach the retriever from the local database.
        After detaching, the retriever will be kept in memory and all modifications will not be applied to the disk.
        """
        self.cfg.database_path = None
        return
