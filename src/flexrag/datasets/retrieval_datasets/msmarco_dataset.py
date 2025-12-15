from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path
from typing import Literal, Mapping

from flexrag.common import (
    FLEXRAG_CACHE_DIR,
    LOGGER_MANAGER,
    SimpleProgressLogger,
    configure,
    download,
    download_and_extract,
)
from flexrag.common.dataclasses import Context

from ..reader import LineDelimitedReader
from .retrieval_dataset import RETRIEVAL_DATASETS, RetrievalDatasetBase

logger = LOGGER_MANAGER.get_logger("flexrag.datasets.msmarco_dataset")


@configure
class MSMARCODatasetConfig:
    """Configuration for loading `MS MARCO <https://microsoft.github.io/msmarco>`_ Retrieval Dataset.
    The __getitem__ method will return `IREvalData` objects.

    :param data_name: The name of the dataset to load. Required.
        Supported values are:

            - msmarco_passage_ranking_v1
            - msmarco_passage_ranking_v2
            - msmarco_document_ranking_v1
            - msmarco_document_ranking_v2

    :type data_name: str
    :param split: The split of the dataset to load. Required.
    :type split: str
    :param data_path: The local path to the dataset.
        If not specified, it will download the dataset to the cache directory.
    :type data_path: str | None
    :param load_corpus: Whether to load the corpus of the dataset. Default: False.
        If set to False, the contexts in the `IREvalData` will not contain the actual data.
        If set to True, it will take more time to load the dataset.
    :type load_corpus: bool

    For example, you can use the following code to load the train split of the msmarco_passage_ranking_v1:

        >>> config = MSMARCOConfig(
        ...     data_name="msmarco_passage_ranking_v1",
        ...     split="train",
        ...     load_corpus=True,
        ... )
        >>> dataset = MSMARCODataset(config)

    For more information about the of the MS MARCO dataset,
    please refer to the `MS MARCO repository <https://github.com/microsoft/MSMARCO>`_.
    """

    data_name: str
    split: str
    data_path: str | None = None
    load_corpus: bool = False


@RETRIEVAL_DATASETS("msmarco", MSMARCODatasetConfig)
class MSMARCODataset(RetrievalDatasetBase):
    """Dataset for loading MSMARCO Retrieval Dataset."""

    def __init__(self, config: MSMARCODatasetConfig) -> None:
        match config.data_name:
            case "msmarco_passage_ranking_v1":
                loader = _MSMARCOPassageRankingV1Loader(
                    split=config.split, data_path=config.data_path
                )
            case "msmarco_document_ranking_v1":
                loader = _MSMARCODocumentRankingV1Loader(
                    split=config.split, data_path=config.data_path
                )
            case "msmarco_passage_ranking_v2":
                loader = _MSMARCOPassageRankingV2Loader(
                    split=config.split, data_path=config.data_path
                )
            case "msmarco_document_ranking_v2":
                loader = _MSMARCODocumentRankingV2Loader(
                    split=config.split, data_path=config.data_path
                )
            case _:
                raise ValueError(f"Unsupported data_name: {config.data_name}")

        # load corpus, queries, and qrels
        self._context_data = {}
        if config.load_corpus:
            p_logger = SimpleProgressLogger(logger=logger, interval=100_000)
            for context in loader.load_corpus():
                p_logger.update(1, desc="Loading corpus")
                self._context_data[context.context_id] = context
        self._qrels_data = loader.load_qrels()
        self._queries_data = loader.load_queries()
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


RESOURCES = {
    "msmarco_passage_ranking_v1": {
        "corpus": "https://msmarco.z22.web.core.windows.net/msmarcoranking/collection.tar.gz",
        "train": {
            "queries": "https://msmarco.z22.web.core.windows.net/msmarcoranking/queries.tar.gz",
            "qrels": "https://msmarco.z22.web.core.windows.net/msmarcoranking/qrels.train.tsv",
        },
        "dev": {
            "queries": "https://msmarco.z22.web.core.windows.net/msmarcoranking/queries.tar.gz",
            "qrels": "https://msmarco.z22.web.core.windows.net/msmarcoranking/qrels.dev.tsv",
        },
    },
    "msmarco_passage_ranking_v2": {
        "corpus": "https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco_v2_passage.tar",
        "train": {
            "queries": "https://msmarco.z22.web.core.windows.net/msmarcoranking/passv2_train_queries.tsv",
            "qrels": "https://msmarco.z22.web.core.windows.net/msmarcoranking/passv2_train_qrels.tsv",
        },
        "dev1": {
            "queries": "https://msmarco.z22.web.core.windows.net/msmarcoranking/passv2_dev_queries.tsv",
            "qrels": "https://msmarco.z22.web.core.windows.net/msmarcoranking/passv2_dev_qrels.tsv",
        },
        "dev2": {
            "queries": "https://msmarco.z22.web.core.windows.net/msmarcoranking/passv2_dev2_queries.tsv",
            "qrels": "https://msmarco.z22.web.core.windows.net/msmarcoranking/passv2_dev2_qrels.tsv",
        },
    },
    "msmarco_document_ranking_v1": {
        "corpus": "https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-docs.tsv.gz",
        "train": {
            "queries": "https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-doctrain-queries.tsv.gz",
            "qrels": "https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-doctrain-qrels.tsv.gz",
        },
        "dev": {
            "queries": "https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-docdev-queries.tsv.gz",
            "qrels": "https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-docdev-qrels.tsv.gz",
        },
    },
    "msmarco_document_ranking_v2": {
        "corpus": "https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco_v2_doc.tar",
        "train": {
            "queries": "https://msmarco.z22.web.core.windows.net/msmarcoranking/docv2_train_queries.tsv",
            "qrels": "https://msmarco.z22.web.core.windows.net/msmarcoranking/docv2_train_qrels.tsv",
        },
        "dev1": {
            "queries": "https://msmarco.z22.web.core.windows.net/msmarcoranking/docv2_dev_queries.tsv",
            "qrels": "https://msmarco.z22.web.core.windows.net/msmarcoranking/docv2_dev_qrels.tsv",
        },
        "dev2": {
            "queries": "https://msmarco.z22.web.core.windows.net/msmarcoranking/docv2_dev2_queries.tsv",
            "qrels": "https://msmarco.z22.web.core.windows.net/msmarcoranking/docv2_dev2_qrels.tsv",
        },
    },
}


class _MSMARCOPassageRankingV1Loader:
    """Dataset for loading MSMARCO Passage Ranking V1 Dataset."""

    def __init__(self, split: Literal["train", "dev"], data_path: str = None) -> None:
        self.data_path = data_path
        self.split = split
        return

    def load_corpus(
        self, return_type: Literal["Context", "dict"] = "Context"
    ) -> Iterator[Context]:
        """Load the corpus from the given path."""
        if self.data_path is not None:
            corpus_path = Path(self.data_path, "collection.tsv")
        else:
            corpus_path = Path(
                FLEXRAG_CACHE_DIR,
                "datasets",
                "msmarco_passage_ranking_v1",
                "collection.tsv",
            )
        # download the corpus if not exists
        if not corpus_path.exists():
            url = RESOURCES["msmarco_passage_ranking_v1"]["corpus"]
            logger.info(f"Downloading corpus from {url} to {corpus_path.parent}.")
            download_and_extract(url, str(corpus_path.parent), show_progress=True)
        # load the corpus
        reader = LineDelimitedReader(
            corpus_path,
            titles=["_id", "text"],
            encoding="utf-8",
            file_format="tsv",
        )
        for data in reader:
            if return_type == "dict":
                yield data
            elif return_type == "Context":
                ctx_id = data.pop("_id")
                yield Context(
                    context_id=ctx_id,
                    data={"text": data["text"]},
                    source="msmarco_passage_ranking_v1",
                )
        return

    def load_queries(self) -> dict[str, str]:
        """Load the queries from the given path."""
        if self.data_path is not None:
            queries_path = Path(self.data_path, f"queries.{self.split}.tsv")
        else:
            queries_path = Path(
                FLEXRAG_CACHE_DIR,
                "datasets",
                "msmarco_passage_ranking_v1",
                f"queries.{self.split}.tsv",
            )
        # download the queries if not exists
        if not queries_path.exists():
            url = RESOURCES["msmarco_passage_ranking_v1"][self.split]["queries"]
            logger.info(f"Downloading queries from {url} to {queries_path.parent}.")
            download_and_extract(url, queries_path.parent)
        # load the queries
        queries = {}
        reader = LineDelimitedReader(
            queries_path,
            titles=["_id", "query"],
            encoding="utf-8",
            file_format="tsv",
        )
        for data in reader:
            queries[data["_id"]] = data["query"]
        return queries

    def load_qrels(self) -> dict[str, dict[str, float]]:
        """Load the qrels from the given path.

        :return: A dictionary mapping from query id to a set of relevant context ids.
        :rtype: dict[str, dict[str, float]]
        """
        if self.data_path is not None:
            qrels_path = Path(self.data_path, f"qrels.{self.split}.tsv")
        else:
            qrels_path = Path(
                FLEXRAG_CACHE_DIR,
                "datasets",
                "msmarco_passage_ranking_v1",
                f"qrels.{self.split}.tsv",
            )
        # download the qrels if not exists
        if not qrels_path.exists():
            url = RESOURCES["msmarco_passage_ranking_v1"][self.split]["qrels"]
            logger.info(f"Downloading qrels from {url} to {qrels_path}.")
            download(url, qrels_path)
        # load the qrels
        qrels = defaultdict(dict)
        reader = LineDelimitedReader(
            qrels_path,
            titles=["qid", "_", "ctx_id", "rel"],
            encoding="utf-8",
            file_format="tsv",
            delimiter=r"\s+",
        )
        for data in reader:
            qrels[data["qid"]][data["ctx_id"]] = float(data["rel"])
        return qrels


class _MSMARCODocumentRankingV1Loader:
    """Dataset for loading MSMARCO Document Ranking V1 Dataset."""

    def __init__(self, split: Literal["train", "dev"], data_path: str = None) -> None:
        self.data_path = data_path
        self.split = split
        return

    def load_corpus(
        self, return_type: Literal["Context", "dict"] = "Context"
    ) -> Iterator[Context]:
        """Load the corpus from the given path."""
        if self.data_path is not None:
            corpus_path = Path(self.data_path, "msmarco-docs.tsv.gz")
        else:
            corpus_path = Path(
                FLEXRAG_CACHE_DIR,
                "datasets",
                "msmarco_document_ranking_v1",
                "msmarco-docs.tsv.gz",
            )
        # download the corpus if not exists
        if not corpus_path.exists():
            url = RESOURCES["msmarco_document_ranking_v1"]["corpus"]
            logger.info(f"Downloading corpus from {url} to {corpus_path}.")
            download(url, corpus_path, show_progress=True)
        # load the corpus
        reader = LineDelimitedReader(
            corpus_path,
            titles=["_id", "url", "title", "text"],
            encoding="utf-8",
            file_format="tsv",
        )
        for data in reader:
            if return_type == "dict":
                yield data
            elif return_type == "Context":
                ctx_id = data.pop("_id")
                yield Context(
                    context_id=ctx_id,
                    data=data,
                    source="msmarco_document_ranking_v1",
                )
        return

    def load_queries(self) -> dict[str, str]:
        """Load the queries from the given path."""
        # msmarco-doctrain-queries.tsv
        if self.data_path is not None:
            queries_path = Path(
                self.data_path, f"msmarco-doc{self.split}-queries.tsv.gz"
            )
        else:
            queries_path = Path(
                FLEXRAG_CACHE_DIR,
                "datasets",
                "msmarco_document_ranking_v1",
                f"msmarco-doc{self.split}-queries.tsv.gz",
            )
        # download the queries if not exists
        if not queries_path.exists():
            url = RESOURCES["msmarco_document_ranking_v1"][self.split]["queries"]
            logger.info(f"Downloading queries from {url} to {queries_path}.")
            download(url, queries_path)
        # load the queries
        queries = {}
        reader = LineDelimitedReader(
            queries_path,
            titles=["_id", "query"],
            encoding="utf-8",
            file_format="tsv",
        )
        for data in reader:
            queries[data["_id"]] = data["query"]
        return queries

    def load_qrels(self) -> dict[str, dict[str, float]]:
        """Load the qrels from the given path.

        :return: A dictionary mapping from query id to a set of relevant context ids.
        :rtype: dict[str, dict[str, float]]
        """
        if self.data_path is not None:
            qrels_path = Path(self.data_path, f"msmarco-doc{self.split}-qrels.tsv.gz")
        else:
            qrels_path = Path(
                FLEXRAG_CACHE_DIR,
                "datasets",
                "msmarco_document_ranking_v1",
                f"msmarco-doc{self.split}-qrels.tsv.gz",
            )
        # download the qrels if not exists
        if not qrels_path.exists():
            url = RESOURCES["msmarco_document_ranking_v1"][self.split]["qrels"]
            logger.info(f"Downloading qrels from {url} to {qrels_path}.")
            download(url, qrels_path)
        # load the qrels
        qrels = defaultdict(dict)
        reader = LineDelimitedReader(
            qrels_path,
            titles=["qid", "_", "ctx_id", "rel"],
            encoding="utf-8",
            file_format="tsv",
            delimiter=r"\s+",
        )
        for data in reader:
            qrels[data["qid"]][data["ctx_id"]] = float(data["rel"])
        return qrels


class _MSMARCOPassageRankingV2Loader:
    """Dataset for loading MSMARCO Passage Ranking V2 Dataset."""

    def __init__(
        self, split: Literal["train", "dev1", "dev2"], data_path: str = None
    ) -> None:
        self.data_path = data_path
        self.split = split
        return

    def load_corpus(
        self, return_type: Literal["Context", "dict"] = "Context"
    ) -> Iterator[Context]:
        """Load the corpus from the given path."""
        if self.data_path is not None:
            corpus_dir = Path(self.data_path, "corpus")
        else:
            corpus_dir = Path(
                FLEXRAG_CACHE_DIR,
                "datasets",
                "msmarco_passage_ranking_v2",
                "corpus",
            )
        # download the corpus if not exists
        if not corpus_dir.exists():
            url = RESOURCES["msmarco_passage_ranking_v2"]["corpus"]
            logger.info(f"Downloading corpus from {url} to {corpus_dir}.")
            download_and_extract(url, corpus_dir, show_progress=True)
        # load the corpus
        for corpus_path in corpus_dir.glob("*"):
            reader = LineDelimitedReader(
                corpus_path, encoding="utf-8", file_format="jsonl"
            )
            for data in reader:
                if return_type == "dict":
                    yield data
                elif return_type == "Context":
                    _id = data.pop("pid")
                    yield Context(
                        context_id=_id,
                        data=data,
                        source="msmarco_passage_ranking_v2",
                    )
        return

    def load_queries(self) -> dict[str, str]:
        """Load the queries from the given path."""
        if self.data_path is not None:
            queries_path = Path(self.data_path, f"passv2_{self.split}_queries.tsv")
        else:
            queries_path = Path(
                FLEXRAG_CACHE_DIR,
                "datasets",
                "msmarco_passage_ranking_v2",
                f"passv2_{self.split}_queries.tsv",
            )
        # download the queries if not exists
        if not queries_path.exists():
            url = RESOURCES["msmarco_passage_ranking_v2"][self.split]["queries"]
            logger.info(f"Downloading queries from {url} to {queries_path}.")
            download(url, queries_path)
        # load the queries
        queries = {}
        reader = LineDelimitedReader(
            queries_path,
            titles=["_id", "query"],
            encoding="utf-8",
            file_format="tsv",
        )
        for data in reader:
            queries[data["_id"]] = data["query"]
        return queries

    def load_qrels(self) -> dict[str, dict[str, float]]:
        """Load the qrels from the given path.

        :return: A dictionary mapping from query id to a set of relevant context ids.
        :rtype: dict[str, dict[str, float]]
        """
        if self.data_path is not None:
            qrels_path = Path(self.data_path, f"passv2_{self.split}_qrels.tsv")
        else:
            qrels_path = Path(
                FLEXRAG_CACHE_DIR,
                "datasets",
                "msmarco_passage_ranking_v2",
                f"passv2_{self.split}_qrels.tsv",
            )
        # download the qrels if not exists
        if not qrels_path.exists():
            url = RESOURCES["msmarco_passage_ranking_v2"][self.split]["qrels"]
            logger.info(f"Downloading qrels from {url} to {qrels_path}.")
            download(url, qrels_path)
        # load the qrels
        qrels = defaultdict(dict)
        reader = LineDelimitedReader(
            qrels_path,
            titles=["qid", "_", "ctx_id", "rel"],
            encoding="utf-8",
            file_format="tsv",
            delimiter=r"\s+",
        )
        for data in reader:
            qrels[data["qid"]][data["ctx_id"]] = float(data["rel"])
        return qrels


class _MSMARCODocumentRankingV2Loader:
    """Dataset for loading MSMARCO Document Ranking V2 Dataset."""

    def __init__(
        self, split: Literal["train", "dev1", "dev2"], data_path: str = None
    ) -> None:
        self.data_path = data_path
        self.split = split
        return

    def load_corpus(
        self, return_type: Literal["Context", "dict"] = "Context"
    ) -> Iterator[Context]:
        """Load the corpus from the given path."""
        if self.data_path is not None:
            corpus_dir = Path(self.data_path, "corpus")
        else:
            corpus_dir = Path(
                FLEXRAG_CACHE_DIR,
                "datasets",
                "msmarco_document_ranking_v2",
                "corpus",
            )
        # download the corpus if not exists
        if not corpus_dir.exists():
            url = RESOURCES["msmarco_document_ranking_v2"]["corpus"]
            logger.info(f"Downloading corpus from {url} to {corpus_dir}.")
            download_and_extract(url, corpus_dir, show_progress=True)
        # load the corpus
        for corpus_path in corpus_dir.glob("*"):
            reader = LineDelimitedReader(
                corpus_path, encoding="utf-8", file_format="jsonl"
            )
            for data in reader:
                if return_type == "dict":
                    yield data
                elif return_type == "Context":
                    docid = data.pop("docid")
                    yield Context(
                        context_id=docid,
                        data=data,
                        source="msmarco_document_ranking_v2",
                    )
        return

    def load_queries(self) -> dict[str, str]:
        """Load the queries from the given path."""
        if self.data_path is not None:
            queries_path = Path(self.data_path, f"docv2_{self.split}_queries.tsv")
        else:
            queries_path = Path(
                FLEXRAG_CACHE_DIR,
                "datasets",
                "msmarco_document_ranking_v2",
                f"docv2_{self.split}_queries.tsv",
            )
        # download the queries if not exists
        if not queries_path.exists():
            url = RESOURCES["msmarco_document_ranking_v2"][self.split]["queries"]
            logger.info(f"Downloading queries from {url} to {queries_path}.")
            download(url, queries_path)
        # load the queries
        queries = {}
        reader = LineDelimitedReader(
            queries_path,
            titles=["_id", "query"],
            encoding="utf-8",
            file_format="tsv",
        )
        for data in reader:
            queries[data["_id"]] = data["query"]
        return queries

    def load_qrels(self) -> dict[str, dict[str, float]]:
        """Load the qrels from the given path.

        :return: A dictionary mapping from query id to a set of relevant context ids.
        :rtype: dict[str, dict[str, float]]
        """
        if self.data_path is not None:
            qrels_path = Path(self.data_path, f"docv2_{self.split}_qrels.tsv")
        else:
            qrels_path = Path(
                FLEXRAG_CACHE_DIR,
                "datasets",
                "msmarco_document_ranking_v2",
                f"docv2_{self.split}_qrels.tsv",
            )
        # download the qrels if not exists
        if not qrels_path.exists():
            url = RESOURCES["msmarco_document_ranking_v2"][self.split]["qrels"]
            logger.info(f"Downloading qrels from {url} to {qrels_path}.")
            download(url, qrels_path)
        # load the qrels
        qrels = defaultdict(dict)
        reader = LineDelimitedReader(
            qrels_path,
            titles=["qid", "_", "ctx_id", "rel"],
            encoding="utf-8",
            file_format="tsv",
            delimiter=r"\s+",
        )
        for data in reader:
            qrels[data["qid"]][data["ctx_id"]] = float(data["rel"])
        return qrels
