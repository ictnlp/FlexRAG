from abc import abstractmethod
from dataclasses import field
from typing import Generator, Optional

from flexrag.utils import Register, data
from flexrag.utils.dataclasses import Context

from ..dataset import MappingDataset


@data
class IREvalData:
    """The dataclass for Information Retrieval evaluation data.

    :param question: The question for evaluation. Required.
    :type question: str
    :param contexts: The contexts related to the question. Default: None.
    :type contexts: Optional[list[Context]]
    :param hard_negatives: The hard negatives related to the question. Default: None.
    :type hard_negatives: Optional[list[Context]]
    :param meta_data: The metadata of the evaluation data. Default: {}.
    :type meta_data: dict
    """

    question: str
    contexts: Optional[list[Context]] = None
    hard_negatives: Optional[list[Context]] = None
    meta_data: dict = field(default_factory=dict)


class RetrievalDataset(MappingDataset[IREvalData]):
    """Interface for Information Retrieval dataset."""

    @property
    @abstractmethod
    def corpus(self) -> Generator[Context, None, None]:
        """The corpus of the dataset."""
        return


RETRIEVAL_DATASETS = Register[RetrievalDataset]("retrieval_dataset")
