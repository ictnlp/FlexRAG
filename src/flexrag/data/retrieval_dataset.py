import json
import os
from dataclasses import dataclass, field
from typing import Optional

from omegaconf import MISSING

from .dataset import MappingDataset


@dataclass
class RetrievalData:
    question: str
    contexts: Optional[list[str]] = None
    context_ids: Optional[list[str]] = None
    meta_data: dict = field(default_factory=dict)


@dataclass
class MTEBDatasetConfig:
    """Configuration for loading MTEB Retrieval Dataset.

    For example, to load the NQ dataset, you can use the following configuration:
    ```python
    config = MTEBDatasetConfig(
        data_path="mteb/nq",
        subset="test",
        load_corpus=False,
    )
    dataset = MTEBDataset(config)
    ```

    :param data_path: Path to the data directory. Required.
    :type data_path: str
    :param subset: Subset of the dataset to load. Required.
    :type subset: str
    :param encoding: Encoding of the data files. Default is 'utf-8'.
    :type encoding: str
    :param load_corpus: Whether to load the corpus data. Default is False.
    :type load_corpus: bool
    """

    data_path: str = MISSING
    subset: str = MISSING
    encoding: str = "utf-8"
    load_corpus: bool = False


class MTEBDataset(MappingDataset[RetrievalData]):
    def __init__(self, config: MTEBDatasetConfig) -> None:
        self.dataset = [
            json.loads(line)
            for line in open(
                os.path.join(config.data_path, "qrels", f"{config.subset}.jsonl"),
                "r",
                encoding=config.encoding,
            )
        ]
        self.queries = [
            json.loads(line)
            for line in open(
                os.path.join(config.data_path, "queries.jsonl"),
                "r",
                encoding=config.encoding,
            )
        ]
        self.queries = {query["_id"]: query for query in self.queries}

        if config.load_corpus:
            self.corpus = [
                json.loads(line)
                for line in open(
                    os.path.join(config.data_path, "corpus.jsonl"),
                    "r",
                    encoding=config.encoding,
                )
            ]
            self.corpus = {doc["_id"]: doc for doc in self.corpus}
        else:
            self.corpus = None
        return

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> RetrievalData:
        data = self.dataset[index]
        query = self.queries[data["query-id"]]
        if self.corpus is not None:
            context = [self.corpus[data["corpus-id"]]]
        else:
            context = None
        return RetrievalData(
            question=query["text"],
            context_ids=data["corpus-id"],
            contexts=context,
            meta_data={"query-id": data["query-id"]},
        )
