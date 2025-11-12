import glob
from collections.abc import Iterable
from pathlib import Path
from typing import Generator, Literal

from flexrag.utils import (
    FLEXRAG_CACHE_DIR,
    LOGGER_MANAGER,
    SimpleProgressLogger,
    configure,
    download,
    download_and_extract,
)
from flexrag.utils.dataclasses import Context

from ..reader import LineDelimitedReader
from .retrieval_dataset import RETRIEVAL_DATASETS, IREvalData, RetrievalDataset

logger = LOGGER_MANAGER.get_logger("flexrag.datasets.msmarco_dataset")


@configure
class MSMARCOConfig:
    """Configuration for loading `MS MARCO <https://microsoft.github.io/msmarco>`_ Retrieval Dataset.
    The __getitem__ method will return `IREvalData` objects.

    Specificly, this class is designed to load the following datasets:

        - msmarco_passage_ranking_v1
        - msmarco_passage_ranking_v2
        - msmarco_document_ranking_v1
        - msmarco_document_ranking_v2

    :param corpus_path: The repository path of the MS MARCO dataset. Required.
        This could be a glob pattern to match multiple files.
    :type corpus_path: str
    :param qrels_path: The repository path of the MS MARCO qrels file. Required.
    :type qrels_path: str
    :param queries_path: The repository path of the MS MARCO queries file. Required.
    :type queries_path: str
    :param load_corpus: Whether to load the corpus of the dataset. Default: False.
        If set to False, the contexts in the `IREvalData` will not contain the actual data.
        If set to True, it will take more time to load the dataset.
    :type load_corpus: bool

    For example, you can use the following code to load the train subset of the msmarco_passage_ranking_v1:

        >>> config = MSMARCOConfig(
        ...     data_name="msmarco_passage_ranking_v1",
        ...     subset="train",
        ...     load_corpus=True,
        ... )
        >>> dataset = MSMARCODataset(config)

    For more information about the of the MS MARCO dataset,
    please refer to the `MS MARCO repository <https://github.com/microsoft/MSMARCO>`_.
    """

    data_name: str
    subset: str
    data_path: str | None = None
    load_corpus: bool = False


@RETRIEVAL_DATASETS("msmarco", MSMARCOConfig)
class MSMARCODataset(RetrievalDataset):
    """Dataset for loading MSMARCO Retrieval Dataset."""

    def __init__(self, config: MSMARCOConfig) -> None:
        match config.data_name:
            case "msmarco_passage_ranking_v1":
                loader = _MSMARCOPassageRankingV1Loader(
                    subset=config.subset, data_path=config.data_path
                )
            case "msmarco_document_ranking_v1":
                loader = _MSMARCODocumentRankingV1Loader(
                    subset=config.subset, data_path=config.data_path
                )
            case "msmarco_passage_ranking_v2":
                loader = _MSMARCOPassageRankingV2Loader(
                    subset=config.subset, data_path=config.data_path
                )
            case "msmarco_document_ranking_v2":
                loader = _MSMARCODocumentRankingV2Loader(
                    subset=config.subset, data_path=config.data_path
                )
            case _:
                raise ValueError(f"Unsupported data_name: {config.data_name}")

        # load corpus, queries, and qrels
        if config.load_corpus:
            p_logger = SimpleProgressLogger(logger=logger, interval=10000)
            corpus = {}
            for context in loader.load_corpus():
                p_logger.update(1, desc="Loading corpus")
                corpus[context.context_id] = context
            self._corpus = corpus
        else:
            self._corpus = None
        self._qrels = loader.load_qrels()
        self._queries = loader.load_queries()
        self._query_map: dict[str, int] = {
            query["_id"]: index for index, query in enumerate(self._queries)
        }

        # merge qrels, queries, and corpus into RetrievalData
        dataset_map: dict[str, IREvalData] = {}

        for qrel in self.qrels:
            # construct the context
            context = Context(context_id=qrel["corpus-id"])
            if self._corpus is not None:
                context.data = self.corpus[self._corpus_map[qrel["corpus-id"]]]
            if "score" in qrel:  # relevance level of the context
                context.meta_data["score"] = int(qrel["score"])
            # construct the query
            query = self.queries[self._query_map[qrel["query-id"]]]["text"]

            if qrel["query-id"] not in dataset_map:
                dataset_map[qrel["query-id"]] = IREvalData(
                    question=query,
                    contexts=[context],
                    meta_data={"query-id": qrel["query-id"]},
                )
            else:
                dataset_map[qrel["query-id"]].contexts.append(context)
        self.dataset: list[IREvalData] = list(dataset_map.values())
        return

    @property
    def corpus(self) -> Generator[Context, None, None]:
        """The corpus of the dataset."""
        if self._corpus is None:
            raise ValueError(
                "Corpus is not loaded. Please set `load_corpus=True` in the configuration."
            )
        for data in self._corpus:
            yield self._corpus[data]
        return

    @property
    def queries(self) -> list[dict]:
        """The queries of the dataset."""
        return self._queries

    @property
    def qrels(self) -> list[dict]:
        """The qrels of the dataset."""
        return self._qrels

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> IREvalData:
        return self.dataset[index]


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

    def __init__(self, subset: Literal["train", "dev"], data_path: str = None) -> None:
        self.data_path = data_path
        self.subset = subset
        return

    def load_corpus(self) -> Iterable[Context]:
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
            queries_path = Path(self.data_path, f"queries.{self.subset}.tsv")
        else:
            queries_path = Path(
                FLEXRAG_CACHE_DIR,
                "datasets",
                "msmarco_passage_ranking_v1",
                f"queries.{self.subset}.tsv",
            )
        # download the queries if not exists
        if not queries_path.exists():
            url = RESOURCES["msmarco_passage_ranking_v1"][self.subset]["queries"]
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

    def load_qrels(self) -> dict[str, str]:
        """Load the qrels from the given path."""
        if self.data_path is not None:
            qrels_path = Path(self.data_path, f"qrels.{self.subset}.tsv")
        else:
            qrels_path = Path(
                FLEXRAG_CACHE_DIR,
                "datasets",
                "msmarco_passage_ranking_v1",
                f"qrels.{self.subset}.tsv",
            )
        # download the qrels if not exists
        if not qrels_path.exists():
            url = RESOURCES["msmarco_passage_ranking_v1"][self.subset]["qrels"]
            logger.info(f"Downloading qrels from {url} to {qrels_path}.")
            download(url, qrels_path)
        # load the qrels
        qrels = {}
        reader = LineDelimitedReader(
            qrels_path,
            titles=["qid", "_", "ctx_id", "rel"],
            encoding="utf-8",
            file_format="tsv",
            delimiter=r"\s+",
        )
        for data in reader:
            if data["rel"] == "0":
                continue
            qrels[data["qid"]] = data["ctx_id"]
        return qrels


class _MSMARCODocumentRankingV1Loader:
    """Dataset for loading MSMARCO Document Ranking V1 Dataset."""

    def __init__(self, subset: Literal["train", "dev"], data_path: str = None) -> None:
        self.data_path = data_path
        self.subset = subset
        return

    def load_corpus(self) -> Iterable[Context]:
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
            titles=["_id", "text"],
            encoding="utf-8",
            file_format="tsv",
        )
        for data in reader:
            ctx_id = data.pop("_id")
            yield Context(
                context_id=ctx_id,
                data={"text": data["text"]},
                source="msmarco_document_ranking_v1",
            )
        return

    def load_queries(self) -> dict[str, str]:
        """Load the queries from the given path."""
        # msmarco-doctrain-queries.tsv
        if self.data_path is not None:
            queries_path = Path(
                self.data_path, f"msmarco-doc{self.subset}-queries.tsv.gz"
            )
        else:
            queries_path = Path(
                FLEXRAG_CACHE_DIR,
                "datasets",
                "msmarco_document_ranking_v1",
                f"msmarco-doc{self.subset}-queries.tsv.gz",
            )
        # download the queries if not exists
        if not queries_path.exists():
            url = RESOURCES["msmarco_document_ranking_v1"][self.subset]["queries"]
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

    def load_qrels(self) -> dict[str, str]:
        """Load the qrels from the given path."""
        if self.data_path is not None:
            qrels_path = Path(self.data_path, f"msmarco-doc{self.subset}-qrels.tsv.gz")
        else:
            qrels_path = Path(
                FLEXRAG_CACHE_DIR,
                "datasets",
                "msmarco_document_ranking_v1",
                f"msmarco-doc{self.subset}-qrels.tsv.gz",
            )
        # download the qrels if not exists
        if not qrels_path.exists():
            url = RESOURCES["msmarco_document_ranking_v1"][self.subset]["qrels"]
            logger.info(f"Downloading qrels from {url} to {qrels_path}.")
            download(url, qrels_path)
        # load the qrels
        qrels = {}
        reader = LineDelimitedReader(
            qrels_path,
            titles=["qid", "_", "ctx_id", "rel"],
            encoding="utf-8",
            file_format="tsv",
            delimiter=r"\s+",
        )
        for data in reader:
            if data["rel"] == "0":
                continue
            qrels[data["qid"]] = data["ctx_id"]
        return qrels


class _MSMARCOPassageRankingV2Loader:
    """Dataset for loading MSMARCO Passage Ranking V2 Dataset."""

    def __init__(
        self, subset: Literal["train", "dev1", "dev2"], data_path: str = None
    ) -> None:
        self.data_path = data_path
        self.subset = subset
        return

    def load_corpus(self) -> Iterable[Context]:
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
            queries_path = Path(self.data_path, f"passv2_{self.subset}_queries.tsv")
        else:
            queries_path = Path(
                FLEXRAG_CACHE_DIR,
                "datasets",
                "msmarco_passage_ranking_v2",
                f"passv2_{self.subset}_queries.tsv",
            )
        # download the queries if not exists
        if not queries_path.exists():
            url = RESOURCES["msmarco_passage_ranking_v2"][self.subset]["queries"]
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

    def load_qrels(self) -> dict[str, str]:
        """Load the qrels from the given path."""
        if self.data_path is not None:
            qrels_path = Path(self.data_path, f"passv2_{self.subset}_qrels.tsv")
        else:
            qrels_path = Path(
                FLEXRAG_CACHE_DIR,
                "datasets",
                "msmarco_passage_ranking_v2",
                f"passv2_{self.subset}_qrels.tsv",
            )
        # download the qrels if not exists
        if not qrels_path.exists():
            url = RESOURCES["msmarco_passage_ranking_v2"][self.subset]["qrels"]
            logger.info(f"Downloading qrels from {url} to {qrels_path}.")
            download(url, qrels_path)
        # load the qrels
        qrels = {}
        reader = LineDelimitedReader(
            qrels_path,
            titles=["qid", "_", "ctx_id", "rel"],
            encoding="utf-8",
            file_format="tsv",
            delimiter=r"\s+",
        )
        for data in reader:
            if data["rel"] == "0":
                continue
            qrels[data["qid"]] = data["ctx_id"]
        return qrels


class _MSMARCODocumentRankingV2Loader:
    """Dataset for loading MSMARCO Document Ranking V2 Dataset."""

    def __init__(
        self, subset: Literal["train", "dev1", "dev2"], data_path: str = None
    ) -> None:
        self.data_path = data_path
        self.subset = subset
        return

    def load_corpus(self) -> Iterable[Context]:
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
            queries_path = Path(self.data_path, f"docv2_{self.subset}_queries.tsv")
        else:
            queries_path = Path(
                FLEXRAG_CACHE_DIR,
                "datasets",
                "msmarco_document_ranking_v2",
                f"docv2_{self.subset}_queries.tsv",
            )
        # download the queries if not exists
        if not queries_path.exists():
            url = RESOURCES["msmarco_document_ranking_v2"][self.subset]["queries"]
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

    def load_qrels(self) -> dict[str, str]:
        """Load the qrels from the given path."""
        if self.data_path is not None:
            qrels_path = Path(self.data_path, f"docv2_{self.subset}_qrels.tsv")
        else:
            qrels_path = Path(
                FLEXRAG_CACHE_DIR,
                "datasets",
                "msmarco_document_ranking_v2",
                f"docv2_{self.subset}_qrels.tsv",
            )
        # download the qrels if not exists
        if not qrels_path.exists():
            url = RESOURCES["msmarco_document_ranking_v2"][self.subset]["qrels"]
            logger.info(f"Downloading qrels from {url} to {qrels_path}.")
            download(url, qrels_path)
        # load the qrels
        qrels = {}
        reader = LineDelimitedReader(
            qrels_path,
            titles=["qid", "_", "ctx_id", "rel"],
            encoding="utf-8",
            file_format="tsv",
            delimiter=r"\s+",
        )
        for data in reader:
            if data["rel"] == "0":
                continue
            qrels[data["qid"]] = data["ctx_id"]
        return qrels
