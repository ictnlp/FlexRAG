from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType

from datasets import Dataset, DatasetDict, get_dataset_config_names, load_dataset
from huggingface_hub import snapshot_download

from flexrag.common import FLEXRAG_CACHE_DIR, LOGGER_MANAGER, configure
from flexrag.common.dataclasses import Context

from ...core import DATASETS
from .retrieval_dataset_base import RetrievalDatasetBase

logger = LOGGER_MANAGER.get_logger("flexrag.datasets.mteb_dataset")


@configure
class MTEBDatasetConfig:
    """Configuration for loading `MTEB <https://huggingface.co/mteb>`_ Retrieval Dataset.
    The __getitem__ method will return `IREvalData` objects.

    :param subset: The dataset name of the MTEB datasets. Required.
    :type subset: str
    :param split: The split of the dataset to load. Required.
    :type split: str
    :param data_path: The local path to the dataset.
        If not provided, the dataset will be downloaded to the cache directory.
    :type data_path: str | None
    :param load_corpus: Whether to load the corpus of the dataset. Default: False.
        If set to False, the contexts in the `IREvalData` will not contain the actual data.
        If set to True, it will take more time to load the dataset.
    :type load_corpus: bool

    You can use the following code to load the dataset directly from the MTEB repository:

        >>> config = MTEBDatasetConfig(
        ...     subset="nq",
        ...     split="test",
        ... )
        >>> dataset = MTEBDataset(config)

    For more information about the MTEB datasets,
    please refer to the `MTEB repository <https://huggingface.co/mteb>`_.
    """

    subset: str
    split: str
    data_path: str | None = None
    load_corpus: bool = False


@DATASETS("mteb", MTEBDatasetConfig)
class MTEBDataset(RetrievalDatasetBase):
    """Dataset for loading MTEB Retrieval Dataset."""

    def __init__(self, config: MTEBDatasetConfig) -> None:
        repo_id = f"mteb/{config.subset}"
        if config.data_path is None:
            data_path = FLEXRAG_CACHE_DIR / "datasets" / "mteb" / config.subset
        else:
            data_path = Path(config.data_path)

        # download the dataset if needed
        if not data_path.exists():
            logger.info(f"Downloading MTEB dataset '{config.subset}' to {data_path}...")
            snapshot_download(
                repo_id=repo_id,
                local_dir=data_path.as_posix(),
                repo_type="dataset",
            )

        # load corpus if needed
        config_name = get_dataset_config_names(path=data_path.as_posix())
        self._context_data = {}
        if config.load_corpus:
            if "corpus" in config_name:
                corpus_subset = "corpus"
            else:
                raise ValueError(f"No corpus found for MTEB dataset '{config.subset}'.")
            corpus = load_dataset(path=data_path.as_posix(), name=corpus_subset)
            if isinstance(corpus, DatasetDict) and "corpus" in corpus:
                corpus = corpus["corpus"]
            elif isinstance(corpus, DatasetDict) and config.split in corpus:
                corpus = corpus[config.split]
            elif isinstance(corpus, Dataset):
                corpus = corpus
            else:
                raise ValueError(
                    f"Cannot find corpus for MTEB dataset '{config.subset}'."
                )

            for item in corpus:
                self._context_data[item["_id"]] = Context(
                    context_id=item["_id"],
                    data=item,
                    source=repo_id,
                )

        # load queries
        if "queries" in config_name:
            queries_subset = "queries"
        else:
            raise ValueError(f"No queries found for MTEB dataset '{config.subset}'.")
        queries = load_dataset(
            path=data_path.as_posix(),
            name=queries_subset,
        )
        if isinstance(queries, DatasetDict) and "queries" in queries:
            queries = queries["queries"]
        elif isinstance(queries, DatasetDict) and config.split in queries:
            queries = queries[config.split]
        elif isinstance(queries, Dataset):
            queries = queries
        else:
            raise ValueError(f"Cannot find queries for MTEB dataset '{config.subset}'.")

        self._queries_data = {query["_id"]: query["text"] for query in queries}

        # load qrels
        if "qrels" in config_name:
            qrels_subset = "qrels"
        elif "default" in config_name:
            qrels_subset = "default"
        else:
            raise ValueError(f"No qrels found for MTEB dataset '{config.subset}'.")
        qrels = load_dataset(
            path=data_path.as_posix(),
            name=qrels_subset,
        )
        if isinstance(qrels, DatasetDict) and "qrels" in qrels:
            qrels = qrels["qrels"]
        elif isinstance(qrels, DatasetDict) and config.split in qrels:
            qrels = qrels[config.split]
        elif isinstance(qrels, Dataset):
            qrels = qrels
        else:
            raise ValueError(f"Cannot find qrels for MTEB dataset '{config.subset}'.")
        self._qrels_data = defaultdict(dict)
        for qrel in qrels:
            self._qrels_data[qrel["query-id"]][qrel["corpus-id"]] = float(qrel["score"])
        return

    @property
    def contexts(self) -> Mapping[str, Context]:
        """Return a mapping from context_id to Context object."""
        return MappingProxyType(self._context_data)

    @property
    def queries(self) -> Mapping[str, str]:
        """Return a mapping from query_id to query text."""
        return MappingProxyType(self._queries_data)

    @property
    def qrels(self) -> Mapping[str, Mapping[str, float]]:
        """Return a mapping from query_id to a set of relevant context_ids."""
        return MappingProxyType(self._qrels_data)
