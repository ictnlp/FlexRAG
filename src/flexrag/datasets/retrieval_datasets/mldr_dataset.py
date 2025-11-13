from collections.abc import Iterator
from functools import cached_property
from pathlib import Path

from huggingface_hub import snapshot_download

from flexrag.utils import FLEXRAG_CACHE_DIR, LOGGER_MANAGER, configure
from flexrag.utils.dataclasses import Context

from ..reader import LineDelimitedReader
from .retrieval_dataset import RETRIEVAL_DATASETS, IREvalData, RetrievalDataset

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
        self._corpus = {}
        for item in corpus_reader:
            docid = str(item["docid"])
            self._corpus[docid] = Context(context_id=docid, data=item, source="mldr")

        # load the subset
        subset_path = (
            data_path / f"mldr-v1.0-{config.lang}" / f"{config.subset}.jsonl.gz"
        )
        self._subset = list(LineDelimitedReader(subset_path))

        # for training subset, we need to add documents from self._subset to the corpus
        for item in self._subset:
            for p in item["positive_passages"]:
                docid = str(p["docid"])
                if docid not in self._corpus:
                    self._corpus[docid] = Context(
                        context_id=docid,
                        data=p,
                        source="mldr_subset",
                    )
            for n in item["negative_passages"]:
                docid = str(n["docid"])
                if docid not in self._corpus:
                    self._corpus[docid] = Context(
                        context_id=docid,
                        data=n,
                        source="mldr_subset",
                    )
        return

    def __getitem__(self, index: int) -> IREvalData:
        item = self._subset[index]
        question = item["query"]
        ctx_ids = [p["docid"] for p in item["positive_passages"]]
        neg_ids = [n["docid"] for n in item["negative_passages"]]
        return IREvalData(
            question=question,
            contexts=[self._corpus[ctx_id] for ctx_id in ctx_ids],
            hard_negatives=[self._corpus[neg_id] for neg_id in neg_ids],
            meta_data={"id": item["query_id"]},
        )

    @property
    def corpus(self) -> Iterator[Context]:
        """The corpus of the dataset."""
        for context in self._corpus.values():
            yield context

    @cached_property
    def queries(self) -> list[dict]:
        """The queries of the dataset."""
        queries = []
        for item in self._subset:
            queries.append(
                {
                    "query_id": item["query_id"],
                    "query": item["query"],
                }
            )
        return queries

    @cached_property
    def qrels(self) -> list[dict]:
        """The qrels of the dataset."""
        qrels = []
        for item in self._subset:
            query_id = item["query_id"]
            for p in item["positive_passages"]:
                qrels.append(
                    {
                        "query_id": query_id,
                        "docid": p["docid"],
                        "relevance": 1,
                    }
                )
            for n in item["negative_passages"]:
                qrels.append(
                    {
                        "query_id": query_id,
                        "docid": n["docid"],
                        "relevance": 0,
                    }
                )
        return qrels

    def __len__(self) -> int:
        return len(self._subset)
