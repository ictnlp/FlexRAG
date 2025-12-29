from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path
from types import MappingProxyType
from typing import Annotated, Any, Literal, Mapping

from flexrag.common import (
    FLEXRAG_CACHE_DIR,
    LOGGER_MANAGER,
    Choices,
    Context,
    SimpleProgressLogger,
    configure,
    download,
    download_and_extract,
)

from ...core import DATASETS
from ...reader import LineDelimitedReader
from .retrieval_dataset_base import RetrievalDatasetBase

logger = LOGGER_MANAGER.get_logger("flexrag.datasets.msmarco_dataset")


@configure
class MSMARCODatasetConfig:
    """Configuration for loading `MS MARCO <https://microsoft.github.io/msmarco>`_ Retrieval Dataset.
    The __getitem__ method will return `IREvalData` objects.

    :param subset: The name of the dataset to load. Required.
        Supported values are:

            - msmarco_passage_ranking_v1
            - msmarco_passage_ranking_v2
            - msmarco_document_ranking_v1
            - msmarco_document_ranking_v2

    :type subset: str
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

        >>> config = MSMARCODatasetConfig(
        ...     subset="msmarco_passage_ranking_v1",
        ...     split="train",
        ...     load_corpus=True,
        ... )
        >>> dataset = MSMARCODataset(config)

    For more information about the of the MS MARCO dataset,
    please refer to the `MS MARCO repository <https://github.com/microsoft/MSMARCO>`_.
    """

    subset: Annotated[
        str,
        Choices(
            "msmarco_passage_ranking_v1",
            "msmarco_passage_ranking_v2",
            "msmarco_document_ranking_v1",
            "msmarco_document_ranking_v2",
        ),
    ]
    split: str
    data_path: str | None = None
    load_corpus: bool = False


@DATASETS("msmarco", MSMARCODatasetConfig)
class MSMARCODataset(RetrievalDatasetBase):
    """Dataset for loading MSMARCO Retrieval Dataset."""

    def __init__(self, config: MSMARCODatasetConfig) -> None:
        match config.subset:
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
                raise ValueError(f"Unsupported subset: {config.subset}")

        # load corpus, queries, and qrels
        self._context_data = {}
        if config.load_corpus:
            p_logger = SimpleProgressLogger(logger=logger, interval=100_000)
            for context in loader.load_corpus():
                p_logger.update(1, desc="Loading corpus")
                self._context_data[context.context_id] = context
        self._qrels_data = loader.load_qrels()
        self._queries_data = loader.load_queries()
        self._candidates = loader.load_scoreddocs()
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

    @property
    def candidates(self) -> Mapping[str, list[Mapping[str, Any]]]:
        """Return a mapping from query_id to a set of candidate context_ids and their scores."""
        return MappingProxyType(self._candidates)


RESOURCES = {
    "msmarco_passage_ranking_v1": {
        "corpus": "https://msmarco.z22.web.core.windows.net/msmarcoranking/collection.tar.gz",
        "train": {
            "queries": "https://msmarco.z22.web.core.windows.net/msmarcoranking/queries.tar.gz",
            "qrels": "https://msmarco.z22.web.core.windows.net/msmarcoranking/qrels.train.tsv",
            "scoreddocs": "https://msmarco.z22.web.core.windows.net/msmarcoranking/top1000.train.tar.gz",
        },
        "dev": {
            "queries": "https://msmarco.z22.web.core.windows.net/msmarcoranking/queries.tar.gz",
            "qrels": "https://msmarco.z22.web.core.windows.net/msmarcoranking/qrels.dev.tsv",
            "scoreddocs": "https://msmarco.z22.web.core.windows.net/msmarcoranking/top1000.dev.tar.gz",
        },
        "trec-dl-2019": {
            "queries": "https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-test2019-queries.tsv.gz",
            "qrels": "https://trec.nist.gov/data/deep/2019qrels-pass.txt",
            "scoreddocs": "https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-passagetest2019-top1000.tsv.gz",
        },
        "trec-dl-2020": {
            "queries": "https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-test2020-queries.tsv.gz",
            "qrels": "https://trec.nist.gov/data/deep/2020qrels-pass.txt",
            "scoreddocs": "https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-passagetest2020-top1000.tsv.gz",
        },
        # queries and scoreddocs can be loaded from trec-dl-2019 and trec-dl-2020
        "trec-dl-hard": {
            "qrels": "https://raw.githubusercontent.com/grill-lab/DL-Hard/main/dataset/dl_hard-passage.qrels"
        },
    },
    "msmarco_passage_ranking_v2": {
        "corpus": "https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco_v2_passage.tar",
        "train": {
            "queries": "https://msmarco.z22.web.core.windows.net/msmarcoranking/passv2_train_queries.tsv",
            "qrels": "https://msmarco.z22.web.core.windows.net/msmarcoranking/passv2_train_qrels.tsv",
            "scoreddocs": "https://msmarco.z22.web.core.windows.net/msmarcoranking/passv2_train_top100.txt.gz",
        },
        "dev1": {
            "queries": "https://msmarco.z22.web.core.windows.net/msmarcoranking/passv2_dev_queries.tsv",
            "qrels": "https://msmarco.z22.web.core.windows.net/msmarcoranking/passv2_dev_qrels.tsv",
            "scoreddocs": "https://msmarco.z22.web.core.windows.net/msmarcoranking/passv2_dev_top100.txt.gz",
        },
        "dev2": {
            "queries": "https://msmarco.z22.web.core.windows.net/msmarcoranking/passv2_dev2_queries.tsv",
            "qrels": "https://msmarco.z22.web.core.windows.net/msmarcoranking/passv2_dev2_qrels.tsv",
            "scoreddocs": "https://msmarco.z22.web.core.windows.net/msmarcoranking/passv2_dev2_top100.txt.gz",
        },
        "trec-dl-2021": {
            "queries": "https://msmarco.z22.web.core.windows.net/msmarcoranking/2021_queries.tsv",
            "qrels": "https://trec.nist.gov/data/deep/2021.qrels.pass.final.txt",
            "scoreddocs": "https://msmarco.z22.web.core.windows.net/msmarcoranking/2021_passage_top100.txt.gz",
        },
        "trec-dl-2022": {
            "queries": "https://msmarco.z22.web.core.windows.net/msmarcoranking/2022_queries.tsv",
            "qrels": "https://trec.nist.gov/data/deep/2022.qrels.pass.withDupes.txt",
            "scoreddocs": "https://msmarco.z22.web.core.windows.net/msmarcoranking/2022_passage_top100.txt.gz",
        },
    },
    "msmarco_document_ranking_v1": {
        "corpus": "https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-docs.tsv.gz",
        "train": {
            "queries": "https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-doctrain-queries.tsv.gz",
            "qrels": "https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-doctrain-qrels.tsv.gz",
            "scoreddocs": "https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-doctrain-top100.gz",
        },
        "dev": {
            "queries": "https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-docdev-queries.tsv.gz",
            "qrels": "https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-docdev-qrels.tsv.gz",
            "scoreddocs": "https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-docdev-top100.gz",
        },
        "trec-dl-2019": {
            "queries": "https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-test2019-queries.tsv.gz",
            "qrels": "https://trec.nist.gov/data/deep/2019qrels-docs.txt",
            "scoreddocs": "https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-doctest2019-top100.gz",
        },
        "trec-dl-2020": {
            "queries": "https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-test2020-queries.tsv.gz",
            "qrels": "https://trec.nist.gov/data/deep/2020qrels-docs.txt",
            "scoreddocs": "https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-doctest2020-top100.gz",
        },
        "orcas": {
            "queries": "https://msmarco.z22.web.core.windows.net/msmarcoranking/orcas-doctrain-queries.tsv.gz",
            "qrels": "https://msmarco.z22.web.core.windows.net/msmarcoranking/orcas-doctrain-qrels.tsv.gz",
            "scoreddocs": "https://msmarco.z22.web.core.windows.net/msmarcoranking/orcas-doctrain-top100.gz",
        },
        # queries and scoreddocs can be loaded from trec-dl-2019 and trec-dl-2020
        "trec-dl-hard": {
            "qrels": "https://raw.githubusercontent.com/grill-lab/DL-Hard/main/dataset/dl_hard-doc.qrels",
        },
    },
    "msmarco_document_ranking_v2": {
        "corpus": "https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco_v2_doc.tar",
        "train": {
            "queries": "https://msmarco.z22.web.core.windows.net/msmarcoranking/docv2_train_queries.tsv",
            "qrels": "https://msmarco.z22.web.core.windows.net/msmarcoranking/docv2_train_qrels.tsv",
            "scoreddocs": "https://msmarco.z22.web.core.windows.net/msmarcoranking/docv2_train_top100.txt.gz",
        },
        "dev1": {
            "queries": "https://msmarco.z22.web.core.windows.net/msmarcoranking/docv2_dev_queries.tsv",
            "qrels": "https://msmarco.z22.web.core.windows.net/msmarcoranking/docv2_dev_qrels.tsv",
            "scoreddocs": "https://msmarco.z22.web.core.windows.net/msmarcoranking/docv2_dev_top100.txt.gz",
        },
        "dev2": {
            "queries": "https://msmarco.z22.web.core.windows.net/msmarcoranking/docv2_dev2_queries.tsv",
            "qrels": "https://msmarco.z22.web.core.windows.net/msmarcoranking/docv2_dev2_qrels.tsv",
            "scoreddocs": "https://msmarco.z22.web.core.windows.net/msmarcoranking/docv2_dev2_top100.txt.gz",
        },
        "trec-dl-2019": {
            "queries": "https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-test2019-queries.tsv.gz",
            "qrels": "https://msmarco.z22.web.core.windows.net/msmarcoranking/docv2_trec2019_qrels.txt.gz",
        },
        "trec-dl-2020": {
            "queries": "https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-test2020-queries.tsv.gz",
            "qrels": "https://msmarco.z22.web.core.windows.net/msmarcoranking/docv2_trec2020_qrels.txt.gz",
        },
        "trec-dl-2021": {
            "queries": "https://msmarco.z22.web.core.windows.net/msmarcoranking/2021_queries.tsv",
            "qrels": "https://trec.nist.gov/data/deep/2021.qrels.docs.final.txt",
            "scoreddocs": "https://msmarco.z22.web.core.windows.net/msmarcoranking/2021_document_top100.txt.gz",
        },
        "trec-dl-2022": {
            "queries": "https://msmarco.z22.web.core.windows.net/msmarcoranking/2022_queries.tsv",
            "qrels": "https://trec.nist.gov/data/deep/2022.qrels.docs.inferred.txt",
            "scoreddocs": "https://msmarco.z22.web.core.windows.net/msmarcoranking/2022_document_top100.txt.gz",
        },
        "trec-dl-2023": {
            "queries": "https://msmarco.z22.web.core.windows.net/msmarcoranking/2023_queries.tsv",
            "scoreddocs": "https://msmarco.z22.web.core.windows.net/msmarcoranking/2023_document_top100.txt.gz",
        },
    },
}


class _MSMARCOPassageRankingV1Loader:
    """Dataset for loading MSMARCO Passage Ranking V1 Dataset."""

    def __init__(
        self,
        split: Literal[
            "train",
            "dev",
            "trec-dl-2019",
            "trec-dl-2020",
            "trec-dl-hard",
        ],
        data_path: str = None,
    ) -> None:
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

    def load_queries(self, split: str = None) -> dict[str, str]:
        """Load the queries from the given path."""
        split = split if split else self.split
        if split == "trec-dl-hard":
            return self.load_queries(split="trec-dl-2019") | self.load_queries(
                split="trec-dl-2020"
            )
        query_url = RESOURCES["msmarco_passage_ranking_v1"][split]["queries"]
        query_name = Path(query_url).name
        if self.data_path is not None:
            queries_path = Path(self.data_path, query_name)
        else:
            queries_path = Path(
                FLEXRAG_CACHE_DIR,
                "datasets",
                "msmarco_passage_ranking_v1",
                query_name,
            )
        # download the queries if not exists
        if split in {"train", "dev"}:
            queries_path = queries_path.parent / f"queries.{split}.tsv"
            if not queries_path.exists():
                logger.info(f"Downloading queries from {query_url} to {queries_path}.")
                download_and_extract(query_url, queries_path.parent)
        else:
            if not queries_path.exists():
                logger.info(f"Downloading queries from {query_url} to {queries_path}.")
                download(query_url, queries_path)
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

    def load_qrels(self, split: str = None) -> dict[str, dict[str, float]]:
        """Load the qrels from the given path.

        :return: A dictionary mapping from query id to a set of relevant context ids.
        :rtype: dict[str, dict[str, float]]
        """
        split = split if split else self.split
        qrels_url = RESOURCES["msmarco_passage_ranking_v1"][split]["qrels"]
        qrels_name = Path(qrels_url).name
        if self.data_path is not None:
            qrels_path = Path(self.data_path, qrels_name)
        else:
            qrels_path = Path(
                FLEXRAG_CACHE_DIR,
                "datasets",
                "msmarco_passage_ranking_v1",
                qrels_name,
            )
        # download the qrels if not exists
        if not qrels_path.exists():
            logger.info(f"Downloading qrels from {qrels_url} to {qrels_path}.")
            download(qrels_url, qrels_path)
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

    def load_scoreddocs(self, split: str = None) -> dict[str, list[dict[str, Any]]]:
        """Load the scoredocs from the given path.

        :return: A dictionary mapping from query id to a dictionary of context ids and their scores.
        :rtype: dict[str, list[dict[str, Any]]]
        """
        # parse split
        split = split if split else self.split
        if split == "trec-dl-hard":
            return self.load_scoreddocs(split="trec-dl-2019") | self.load_scoreddocs(
                split="trec-dl-2020"
            )
        if "scoreddocs" not in RESOURCES["msmarco_passage_ranking_v1"][split]:
            return {}
        cands_url = RESOURCES["msmarco_passage_ranking_v1"][split]["scoreddocs"]
        cands_name = Path(cands_url).name
        if self.data_path is not None:
            cands_path = Path(self.data_path, cands_name)
        else:
            cands_path = Path(
                FLEXRAG_CACHE_DIR,
                "datasets",
                "msmarco_passage_ranking_v1",
                cands_name,
            )
        # download the scoredocs if not exists
        if split in {"train", "dev"}:
            cands_path = cands_path.parent / f"top1000.{split}"
            if not cands_path.exists():
                logger.info(f"Downloading scoredocs from {cands_url} to {cands_path}.")
                download_and_extract(cands_url, cands_path.parent)
            reader = LineDelimitedReader(
                cands_path,
                titles=["qid", "ctx_id", "query", "ctx"],
                encoding="utf-8",
                file_format="tsv",
            )
        else:
            if not cands_path.exists():
                logger.info(f"Downloading scoredocs from {cands_url} to {cands_path}.")
                download(cands_url, cands_path)
            reader = LineDelimitedReader(
                cands_path,
                titles=["qid", "ctx_id", "query", "ctx"],
                encoding="utf-8",
                file_format="tsv",
            )
        # load the scoredocs
        scoredocs = defaultdict(list)
        for data in reader:
            scoredocs[data["qid"]].append({"ctx_id": data["ctx_id"]})
        return scoredocs


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

    def load_queries(self, split: str = None) -> dict[str, str]:
        """Load the queries from the given path."""
        split = split if split else self.split
        if split == "trec-dl-hard":
            return self.load_queries(split="trec-dl-2019") | self.load_queries(
                split="trec-dl-2020"
            )
        query_url = RESOURCES["msmarco_document_ranking_v1"][split]["queries"]
        query_name = Path(query_url).name
        if self.data_path is not None:
            queries_path = Path(self.data_path, query_name)
        else:
            queries_path = Path(
                FLEXRAG_CACHE_DIR,
                "datasets",
                "msmarco_document_ranking_v1",
                query_name,
            )
        # download the queries if not exists
        if not queries_path.exists():
            logger.info(f"Downloading queries from {query_url} to {queries_path}.")
            download(query_url, queries_path)
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
        qrels_url = RESOURCES["msmarco_document_ranking_v1"][self.split]["qrels"]
        qrels_name = Path(qrels_url).name
        if self.data_path is not None:
            qrels_path = Path(self.data_path, qrels_name)
        else:
            qrels_path = Path(
                FLEXRAG_CACHE_DIR,
                "datasets",
                "msmarco_document_ranking_v1",
                qrels_name,
            )
        # download the qrels if not exists
        if not qrels_path.exists():
            logger.info(f"Downloading qrels from {qrels_url} to {qrels_path}.")
            download(qrels_url, qrels_path)
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

    def load_scoreddocs(self, split: str = None) -> dict[str, list[dict[str, Any]]]:
        """Load the scoredocs from the given path.

        :return: A dictionary mapping from query id to a dictionary of context ids and their scores.
        :rtype: dict[str, list[dict[str, Any]]]
        """
        split = split if split else self.split
        if split == "trec-dl-hard":
            return self.load_scoreddocs(split="trec-dl-2019") | self.load_scoreddocs(
                split="trec-dl-2020"
            )
        if "scoreddocs" not in RESOURCES["msmarco_document_ranking_v1"][split]:
            return {}
        cands_url = RESOURCES["msmarco_document_ranking_v1"][split]["scoreddocs"]
        cands_name = Path(cands_url).name
        if self.data_path is not None:
            cands_path = Path(self.data_path, cands_name)
        else:
            cands_path = Path(
                FLEXRAG_CACHE_DIR,
                "datasets",
                "msmarco_document_ranking_v1",
                cands_name,
            )
        # download the scoredocs if not exists
        if not cands_path.exists():
            logger.info(f"Downloading scoredocs from {cands_url} to {cands_path}.")
            download(cands_url, cands_path)
        # load the scoredocs
        scoredocs = defaultdict(list)
        reader = LineDelimitedReader(
            cands_path,
            titles=["qid", "q0", "ctx_id", "rank", "score", "retriever"],
            encoding="utf-8",
            file_format="tsv",
            delimiter=r"\s+",
        )
        for data in reader:
            scoredocs[data["qid"]].append(
                {
                    "ctx_id": data["ctx_id"],
                    "score": float(data["score"]),
                    "rank": int(data["rank"]),
                    "retriever": data["retriever"],
                }
            )
        return scoredocs


class _MSMARCOPassageRankingV2Loader:
    """Dataset for loading MSMARCO Passage Ranking V2 Dataset."""

    def __init__(
        self,
        split: Literal[
            "train",
            "dev1",
            "dev2",
            "trec-dl-2021",
            "trec-dl-2022",
        ],
        data_path: str = None,
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
        query_url = RESOURCES["msmarco_passage_ranking_v2"][self.split]["queries"]
        query_name = Path(query_url).name
        if self.data_path is not None:
            queries_path = Path(self.data_path, query_name)
        else:
            queries_path = Path(
                FLEXRAG_CACHE_DIR,
                "datasets",
                "msmarco_passage_ranking_v2",
                query_name,
            )
        # download the queries if not exists
        if not queries_path.exists():
            logger.info(f"Downloading queries from {query_url} to {queries_path}.")
            download(query_url, queries_path)
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
        qrels_url = RESOURCES["msmarco_passage_ranking_v2"][self.split]["qrels"]
        qrels_name = Path(qrels_url).name
        if self.data_path is not None:
            qrels_path = Path(self.data_path, qrels_name)
        else:
            qrels_path = Path(
                FLEXRAG_CACHE_DIR,
                "datasets",
                "msmarco_passage_ranking_v2",
                qrels_name,
            )
        # download the qrels if not exists
        if not qrels_path.exists():
            logger.info(f"Downloading qrels from {qrels_url} to {qrels_path}.")
            download(qrels_url, qrels_path)
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

    def load_scoreddocs(self) -> dict[str, list[dict[str, Any]]]:
        """Load the scoredocs from the given path.

        :return: A dictionary mapping from query id to a dictionary of context ids and their scores.
        :rtype: dict[str, list[dict[str, Any]]]
        """
        if "scoreddocs" not in RESOURCES["msmarco_passage_ranking_v2"][self.split]:
            return {}
        cands_url = RESOURCES["msmarco_passage_ranking_v2"][self.split]["scoreddocs"]
        cands_name = Path(cands_url).name
        if self.data_path is not None:
            cands_path = Path(self.data_path, cands_name)
        else:
            cands_path = Path(
                FLEXRAG_CACHE_DIR,
                "datasets",
                "msmarco_passage_ranking_v2",
                cands_name,
            )
        # download the scoredocs if not exists
        if not cands_path.exists():
            logger.info(f"Downloading scoredocs from {cands_url} to {cands_path}.")
            download(cands_url, cands_path)
        # load the scoredocs
        scoredocs = defaultdict(list)
        reader = LineDelimitedReader(
            cands_path,
            titles=["qid", "q0", "ctx_id", "rank", "score", "retriever"],
            encoding="utf-8",
            file_format="tsv",
            delimiter=r"\s+",
        )
        for data in reader:
            scoredocs[data["qid"]].append(
                {
                    "ctx_id": data["ctx_id"],
                    "score": float(data["score"]),
                    "rank": int(data["rank"]),
                    "retriever": data["retriever"],
                }
            )
        return scoredocs


class _MSMARCODocumentRankingV2Loader:
    """Dataset for loading MSMARCO Document Ranking V2 Dataset."""

    def __init__(
        self,
        split: Literal[
            "train",
            "dev1",
            "dev2",
            "trec-dl-2019",
            "trec-dl-2020",
            "trec-dl-2021",
            "trec-dl-2022",
        ],
        data_path: str = None,
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
        query_url = RESOURCES["msmarco_document_ranking_v2"][self.split]["queries"]
        query_name = Path(query_url).name
        if self.data_path is not None:
            queries_path = Path(self.data_path, query_name)
        else:
            queries_path = Path(
                FLEXRAG_CACHE_DIR,
                "datasets",
                "msmarco_document_ranking_v2",
                query_name,
            )
        # download the queries if not exists
        if not queries_path.exists():
            logger.info(f"Downloading queries from {query_url} to {queries_path}.")
            download(query_url, queries_path)
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
        qrels_url = RESOURCES["msmarco_document_ranking_v2"][self.split]["qrels"]
        qrels_name = Path(qrels_url).name
        if self.data_path is not None:
            qrels_path = Path(self.data_path, qrels_name)
        else:
            qrels_path = Path(
                FLEXRAG_CACHE_DIR,
                "datasets",
                "msmarco_document_ranking_v2",
                qrels_name,
            )
        # download the qrels if not exists
        if not qrels_path.exists():
            logger.info(f"Downloading qrels from {qrels_url} to {qrels_path}.")
            download(qrels_url, qrels_path)
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

    def load_scoreddocs(self) -> dict[str, list[dict[str, Any]]]:
        """Load the scoredocs from the given path.

        :return: A dictionary mapping from query id to a dictionary of context ids and their scores.
        :rtype: dict[str, list[dict[str, Any]]]
        """
        if "scoreddocs" not in RESOURCES["msmarco_document_ranking_v2"][self.split]:
            return {}
        cands_url = RESOURCES["msmarco_document_ranking_v2"][self.split]["scoreddocs"]
        cands_name = Path(cands_url).name
        if self.data_path is not None:
            cands_path = Path(self.data_path, cands_name)
        else:
            cands_path = Path(
                FLEXRAG_CACHE_DIR,
                "datasets",
                "msmarco_document_ranking_v2",
                cands_name,
            )
        # download the scoredocs if not exists
        if not cands_path.exists():
            logger.info(f"Downloading scoredocs from {cands_url} to {cands_path}.")
            download(cands_url, cands_path)
        # load the scoredocs
        scoredocs = defaultdict(list)
        reader = LineDelimitedReader(
            cands_path,
            titles=["qid", "q0", "ctx_id", "rank", "score", "retriever"],
            encoding="utf-8",
            file_format="tsv",
            delimiter=r"\s+",
        )
        for data in reader:
            scoredocs[data["qid"]].append(
                {
                    "ctx_id": data["ctx_id"],
                    "score": float(data["score"]),
                    "rank": int(data["rank"]),
                    "retriever": data["retriever"],
                }
            )
        return scoredocs
