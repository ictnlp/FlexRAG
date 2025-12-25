from collections import defaultdict
from collections.abc import Mapping

from datasets import load_dataset

from flexrag.common import configure
from flexrag.common.dataclasses import Context

from ...core import DATASETS
from .retrieval_dataset_base import RetrievalDatasetBase


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


@DATASETS("mteb", MTEBDatasetConfig)
class MTEBDataset(RetrievalDatasetBase):
    """Dataset for loading MTEB Retrieval Dataset."""

    def __init__(self, config: MTEBDatasetConfig) -> None:
        # load corpus if needed
        self.data_name = f"{config.path} ({config.split})"
        self._context_data = {}
        if config.load_corpus:
            corpus = load_dataset(
                path=config.path,
                name="corpus",
                split="corpus",
            )

            for item in corpus:
                self._context_data[item["_id"]] = Context(
                    context_id=item["_id"],
                    data=item,
                    source=config.path,
                )

        # load queries
        queries = load_dataset(
            path=config.path,
            name="queries",
            split="queries",
        )
        self._queries_data = {query["_id"]: query["text"] for query in queries}

        # load qrels
        qrels = load_dataset(
            path=config.path,
            name="default",
            split=config.split,
        )
        self._qrels_data = defaultdict(dict)
        for qrel in qrels:
            if qrel["query-id"] in self._queries:
                self._qrels_data[qrel["query-id"]][qrel["corpus-id"]] = float(
                    qrel["score"]
                )
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
