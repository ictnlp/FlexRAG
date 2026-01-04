from collections import defaultdict
from pathlib import Path
from types import MappingProxyType
from typing import Annotated, Mapping, Optional

from huggingface_hub import snapshot_download

from flexrag.common import FLEXRAG_CACHE_DIR, LOGGER_MANAGER, Choices, configure
from flexrag.common.dataclasses import Context

from ...core import DATASETS
from ...reader import LineDelimitedReader
from .retrieval_dataset_base import RetrievalDatasetBase

logger = LOGGER_MANAGER.get_logger("flexrag.datasets.mldr_dataset")


@configure
class MultiLongDocRetrievalDatasetConfig:
    """Configuration for loading `MLDR <https://huggingface.co/datasets/Shitao/MLDR>`_ Retrieval Dataset.
    The __getitem__ method will return `IREvalData` objects.

    :param split: The split of the dataset to load. Required.
        Available choices are: `train`, `dev`, `test`.
    :type split: str
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

    split: str
    lang: Annotated[
        str,
        Choices(
            "ar",
            "de",
            "en",
            "es",
            "fr",
            "hi",
            "it",
            "ja",
            "ko",
            "pt",
            "ru",
            "th",
            "zh",
        ),
    ] = "en"
    data_path: Optional[str] = None


@DATASETS("mldr", config_class=MultiLongDocRetrievalDatasetConfig)
class MultiLongDocRetrievalDataset(RetrievalDatasetBase):
    def __init__(self, config: MultiLongDocRetrievalDatasetConfig) -> None:
        # prepare dataset path
        if config.data_path is None:
            data_path = FLEXRAG_CACHE_DIR / "datasets" / "MLDR"
        else:
            data_path = Path(config.data_path)

        # download the dataset if not exists
        if not data_path.exists():
            logger.info(f"Downloading MLDR dataset to {data_path.as_posix()}...")
            snapshot_download(
                repo_id="Shitao/MLDR",
                local_dir=data_path.as_posix(),
                repo_type="dataset",
            )

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

        # load the split
        split_path = data_path / f"mldr-v1.0-{config.lang}" / f"{config.split}.jsonl.gz"
        self._split = list(LineDelimitedReader(split_path))

        # for training split, we need to add documents from self._split to the corpus
        for item in self._split:
            for p in item["positive_passages"]:
                docid = str(p["docid"])
                if docid not in self._context_data:
                    self._context_data[docid] = Context(
                        context_id=docid,
                        data=p,
                        source="mldr",
                    )
            for n in item["negative_passages"]:
                docid = str(n["docid"])
                if docid not in self._context_data:
                    self._context_data[docid] = Context(
                        context_id=docid,
                        data=n,
                        source="mldr",
                    )

        # set up queries and qrels
        self._queries_data = {}
        for item in self._split:
            self._queries_data[item["query_id"]] = item["query"]
        self._qrels_data = defaultdict(dict)
        for item in self._split:
            query_id = item["query_id"]
            for p in item["positive_passages"]:
                self._qrels_data[query_id][p["docid"]] = 1.0
            for n in item["negative_passages"]:
                self._qrels_data[query_id][n["docid"]] = 0.0
        return

    @property
    def qrels(self) -> Mapping[str, Mapping[str, float]]:
        return MappingProxyType(self._qrels_data)

    @property
    def queries(self) -> Mapping[str, str]:
        return MappingProxyType(self._queries_data)

    @property
    def contexts(self) -> Mapping[str, Context]:
        return MappingProxyType(self._context_data)
