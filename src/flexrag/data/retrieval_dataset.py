import json
import os
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Optional

from omegaconf import MISSING

from flexrag.common_dataclass import Context

from .dataset import MappingDataset


@dataclass
class RetrievalData:
    question: str
    contexts: Optional[list[Context]] = None
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
        qrels: list[dict] = [
            json.loads(line)
            for line in open(
                os.path.join(config.data_path, "qrels", f"{config.subset}.jsonl"),
                "r",
                encoding=config.encoding,
            )
        ]
        queries = [
            json.loads(line)
            for line in open(
                os.path.join(config.data_path, "queries.jsonl"),
                "r",
                encoding=config.encoding,
            )
        ]
        queries = {query["_id"]: query for query in queries}

        if config.load_corpus:
            corpus = [
                json.loads(line)
                for line in open(
                    os.path.join(config.data_path, "corpus.jsonl"),
                    "r",
                    encoding=config.encoding,
                )
            ]
            corpus = {doc["_id"]: doc for doc in corpus}
        else:
            corpus = None

        # merge qrels, queries, and corpus into RetrievalData
        dataset_map: dict[str, int] = {}
        self.dataset: list[RetrievalData] = []
        for qrel in qrels:
            # construct the context
            context = Context(
                context_id=qrel["corpus-id"],
                score=float(qrel.get("score", 0.0)),
            )
            if corpus is not None:
                context.data = corpus[qrel["corpus-id"]]
            query = queries[qrel["query-id"]]["text"]

            if qrel["query-id"] not in dataset_map:
                dataset_map[qrel["query-id"]] = len(self.dataset)
                self.dataset.append(
                    RetrievalData(
                        question=query,
                        contexts=[context],
                        meta_data={"query-id": qrel["query-id"]},
                    )
                )
            else:
                index = dataset_map[qrel["query-id"]]
                self.dataset[index].contexts.append(context)
        return

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> RetrievalData:
        return self.dataset[index]
