from dataclasses import field
from typing import Generator, Optional

from datasets import load_dataset

from flexrag.utils import configure, data
from flexrag.utils.dataclasses import Context

from .dataset import MappingDataset


@configure
class MTEBDatasetConfig:
    """Configuration for loading `MTEB <https://huggingface.co/mteb>`_ Retrieval Dataset.
    The __getitem__ method will return `IREvalData` objects.

    :param path: The repository path of the MTEB dataset. Required.
        This could be a local path or a HuggingFace repository path.
    :type path: str
    :param split: The split of the dataset to load. Required.
    :type split: str
    :param load_corpus: Whether to load the corpus of the dataset. Default: False.
        If set to False, the contexts in the `IREvalData` will not contain the actual data.
        If set to True, it will take more time to load the dataset.
    :type load_corpus: bool

    You can use the following code to load the dataset directly from the MTEB repository:

        >>> config = MTEBDatasetConfig(
        ...     path="mteb/nq",
        ...     split="test",
        ... )
        >>> dataset = MTEBDataset(config)

    For more information about the MTEB datasets,
    please refer to the `MTEB repository <https://huggingface.co/mteb>`_.
    """

    path: str
    split: str
    load_corpus: bool = False


@data
class IREvalData:
    """The dataclass for Information Retrieval evaluation data.

    :param question: The question for evaluation. Required.
    :type question: str
    :param contexts: The contexts related to the question. Default: None.
    :type contexts: Optional[list[Context]]
    :param meta_data: The metadata of the evaluation data. Default: {}.
    :type meta_data: dict
    """

    question: str
    contexts: Optional[list[Context]] = None
    meta_data: dict = field(default_factory=dict)


class MTEBDataset(MappingDataset[IREvalData]):
    """Dataset for loading MTEB Retrieval Dataset."""

    def __init__(self, config: MTEBDatasetConfig) -> None:
        # load necessary datasets
        self.data_name = f"{config.path} ({config.split})"
        if config.load_corpus:
            self._corpus = load_dataset(
                path=config.path,
                name="corpus",
                split="corpus",
            )
            self._corpus_map: dict[str, int] = {
                doc["_id"]: index for index, doc in enumerate(self._corpus)
            }
        else:
            self._corpus = None
        self._queries = load_dataset(
            path=config.path,
            name="queries",
            split="queries",
        )
        self._qrels = load_dataset(
            path=config.path,
            name="default",
            split=config.split,
        )
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
            yield Context(
                context_id=data["_id"],
                data={"text": data["text"]},
                source=self.data_name,
            )
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
