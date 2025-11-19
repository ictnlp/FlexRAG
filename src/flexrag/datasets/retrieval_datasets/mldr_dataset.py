from collections import defaultdict
from pathlib import Path
from typing import Mapping

from huggingface_hub import snapshot_download

from flexrag.common import FLEXRAG_CACHE_DIR, LOGGER_MANAGER, configure
from flexrag.common.dataclasses import Context

from ..reader import LineDelimitedReader
from .retrieval_dataset import RETRIEVAL_DATASETS, RetrievalDataset

logger = LOGGER_MANAGER.get_logger("flexrag.datasets.mldr_dataset")


@configure
class MultiLongDocRetrievalDatasetConfig:
    """Configuration for loading `MLDR <https://huggingface.co/datasets/Shitao/MLDR>`_ Retrieval Dataset.
    The __getitem__ method will return `IREvalData` objects.

    :param subset: The subset of the dataset to load. Required.
        Available choices are: `train`, `dev`, `test`.
    :type subset: str
    :param lang: The language of the dataset. Default is `en`.
        Available choices are:

        - `ar`: Arabic
        - `de`: German
        - `en`: English
        - `es`: Spanish
        - `fr`: French
        - `hi`: Hindi
        - `it`: Italian
        - `ja`: Japanese
        - `ko`: Korean
        - `pt`: Portuguese
        - `ru`: Russian
        - `th`: Thai
        - `zh`: Chinese

    :type lang: str
    :param data_path: data_path: The local path to the dataset.
        If not specified, it will download the dataset to the cache directory.
    :type data_path: str | None
    """

    subset: str
    lang: str = "en"
    data_path: str | None = None


@RETRIEVAL_DATASETS("mldr", config_class=MultiLongDocRetrievalDatasetConfig)
class MultiLongDocRetrievalDataset(RetrievalDataset):
    def __init__(self, config: MultiLongDocRetrievalDatasetConfig) -> None:
        # prepare dataset path
        if config.data_path is None:
            data_path = FLEXRAG_CACHE_DIR / "datasets" / "MLDR"
        else:
            data_path = Path(config.data_path)

        # download the dataset if not exists
        if not data_path.exists():
            logger.info(f"Downloading MLDR dataset to {data_path.as_posix()}...")
            snapshot_download(repo_id="Shitao/MLDR", local_dir=data_path.as_posix())

        # load the corpus
        corpus_path = data_path / f"mldr-v1.0-{config.lang}" / "corpus.jsonl.gz"
        corpus_reader = LineDelimitedReader(corpus_path)
        self._context_data = {}
        for item in corpus_reader:
            docid = str(item["docid"])
            self._context_data[docid] = Context(
                context_id=docid,
                data=item,
                source="mldr-v1.0",
            )

        # load the subset
        subset_path = (
            data_path / f"mldr-v1.0-{config.lang}" / f"{config.subset}.jsonl.gz"
        )
        self._subset = list(LineDelimitedReader(subset_path))

        # for training subset, we need to add documents from self._subset to the corpus
        for item in self._subset:
            for p in item["positive_passages"]:
                docid = str(p["docid"])
                if docid not in self._context_data:
                    self._context_data[docid] = Context(
                        context_id=docid,
                        data=p,
                        source="mldr_subset",
                    )
            for n in item["negative_passages"]:
                docid = str(n["docid"])
                if docid not in self._context_data:
                    self._context_data[docid] = Context(
                        context_id=docid,
                        data=n,
                        source="mldr_subset",
                    )

        # set up queries and qrels
        self._queries_data = {}
        for item in self._subset:
            self._queries_data[item["query_id"]] = item["query"]
        self._qrels_data = defaultdict(dict)
        for item in self._subset:
            query_id = item["query_id"]
            for p in item["positive_passages"]:
                self._qrels_data[query_id][p["docid"]] = 1.0
            for n in item["negative_passages"]:
                self._qrels_data[query_id][n["docid"]] = 0.0
        return

    @property
    def _contexts(self) -> Mapping[str, Context]:
        """Return a mapping from context_id to Context object."""
        return self._context_data

    @property
    def _queries(self) -> Mapping[str, str]:
        """Return a mapping from query_id to query text."""
        return self._queries_data

    @property
    def _qrels(self) -> Mapping[str, set[str]]:
        """Return a mapping from query_id to a set of relevant context_ids."""
        return self._qrels_data
