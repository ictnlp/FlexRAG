from abc import abstractmethod
from dataclasses import field
from functools import cached_property
from typing import Mapping, Optional

from flexrag.common import Register, data

from ..dataset import MappingDataset
from ..retrieval_datasets import IREvalData, RetrievalDatasetBase


@data
class MultipleChoiceEvalData:
    """The dataclass for Multiple Choice evaluation data.

    :param question: The question for evaluation. Required.
    :type question: str
    :param choices: The list of answer choices. Required.
    :type choices: list[str]
    :param meta_data: The metadata of the evaluation data. Default: {}.
    :type meta_data: dict
    :param answers: The golden answers for the question. Default: None.
    :type answers: Optional[list[int]]
    """

    question: str
    choices: list[str]
    meta_data: dict = field(default_factory=dict)
    answers: Optional[list[int]] = None


class MultipleChoiceDatasetBase(MappingDataset[MultipleChoiceEvalData]):
    """Base class for Multiple Choice datasets.

    This class provides a unified interface for accessing Multiple Choice datasets, which typically consist of:
    1. A set of questions.
    2. A set of answer choices for each question.
    3. Golden answers for each question.

    Subclasses must implement the following abstract properties to define the data source:

        >>> @property
        >>> def _queries(self) -> Mapping[str, str]:
        >>>     # Return a mapping from question_id to question string
        >>>     ...

        >>> @property
        >>> def _choices(self) -> Mapping[str, list[str]]:
        >>>     # Return a mapping from question_id to list of answer choices
        >>>     ...

        >>> @property
        >>> def _answers(self) -> Mapping[str, list[int]] | None:
        >>>     # Return a mapping from question_id to list of golden answer indices
        >>>     ...

    Subclasses can also optionally implement the `_meta_data` property to provide additional
    information about the dataset. This information will be retrieved along with the question
    and answers via the `get_item` method.

        >>> @property
        >>> def _meta_data(self) -> Mapping[str, dict]:
        >>>     # Return a mapping from question_id to metadata dictionary
        >>>     ...
    """

    @property
    @abstractmethod
    def _queries(self) -> Mapping[str, str]:
        """Mapping from question_id to question string."""
        return

    @property
    @abstractmethod
    def _choices(self) -> Mapping[str, list[str]]:
        """Mapping from question_id to list of answer choices."""
        return

    @property
    @abstractmethod
    def _answers(self) -> Mapping[str, list[int]] | None:
        """Mapping from question_id to list of golden answer indices."""
        return

    @cached_property
    def _qids(self) -> list[str]:
        return sorted(self._queries.keys())

    def __len__(self) -> int:
        return len(self._queries)

    def get_item(self, idx: int) -> MultipleChoiceEvalData:
        qid = self._qids[idx]
        question = self._queries[qid]
        choices = self._choices[qid]
        if self._answers is not None:
            answers = self._answers.get(qid)
        else:
            answers = None
        if hasattr(self, "_meta_data"):
            meta_data = self._meta_data.get(qid, {})
        else:
            meta_data = {}
        return MultipleChoiceEvalData(
            question=question,
            choices=choices,
            answers=answers,
            meta_data=meta_data,
        )


@data
class KnowledgeMultipleChoiceData(IREvalData, MultipleChoiceEvalData):
    """The dataclass for Knowledge-Intensive Multiple Choice evaluation data."""


class KnowledgeMultipleChoiceDatasetBase(
    RetrievalDatasetBase, MultipleChoiceDatasetBase
):
    """Base class for Knowledge-Intensive Multiple Choice (KIMC) datasets.

    This class combines the features of `RetrievalDatasetBase` and `MultipleChoiceDatasetBase` to support
    tasks that require both contextual information and multiple choice answering. It is suitable for
    datasets where questions are associated with both relevant contexts (from a corpus) and
    multiple choice options with golden answers.

    It inherits from both `RetrievalDatasetBase` and `MultipleChoiceDatasetBase`, providing a unified
    interface to access `KnowledgeMultipleChoiceData` objects. Each `KnowledgeMultipleChoiceData` contains:
    1. The question.
    2. Relevant contexts and hard negatives (from `RetrievalDatasetBase`).
    3. Answer choices and golden answers (from `MultipleChoiceDatasetBase`).

    Subclasses must implement the abstract properties required by both parent classes:
    - `_contexts` (from `RetrievalDatasetBase`)
    - `_queries` (from both, usually shared)
    - `_qrels` (from `RetrievalDatasetBase`)
    - `_choices` (from `MultipleChoiceDatasetBase`)
    - `_answers` (from `MultipleChoiceDatasetBase`)
    """

    def get_item(self, index: int) -> KnowledgeMultipleChoiceData:
        ir_data = RetrievalDatasetBase.get_item(self, index)
        mc_data = MultipleChoiceDatasetBase.get_item(self, index)
        return KnowledgeMultipleChoiceData(
            question=mc_data.question,
            choices=mc_data.choices,
            answers=mc_data.answers,
            contexts=ir_data.contexts,
            hard_negatives=ir_data.hard_negatives,
            meta_data=ir_data.meta_data | mc_data.meta_data,
        )


MULTIPLE_CHOICE_DATASETS = Register[MultipleChoiceDatasetBase](
    "multiple_choice_dataset"
)
KNOWLEDGE_MULTIPLE_CHOICE_DATASETS = Register[KnowledgeMultipleChoiceDatasetBase](
    "knowledge_multiple_choice_dataset"
)
