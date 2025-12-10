from abc import abstractmethod
from dataclasses import field
from functools import cached_property
from typing import Mapping, Optional

from flexrag.common import Register, data

from ..dataset import MappingDataset
from ..retrieval_datasets import IREvalData, RetrievalDatasetBase


@data
class QAEvalData:
    """The dataclass for QA evaluation data.

    :param question: The question for evaluation. Required.
    :type question: str
    :param meta_data: The metadata of the evaluation data. Default: {}.
    :type meta_data: dict
    :param answers: The golden answers for the question. Default: None.
    :type answers: Optional[list[str]]
    """

    question: str
    meta_data: dict = field(default_factory=dict)
    answers: Optional[list[str]] = None


class QADatasetBase(MappingDataset[QAEvalData]):
    """Base class for Question Answering (QA) datasets.

    This class provides a unified interface for accessing QA datasets, which typically consist of:
    1. A set of questions.
    2. A set of golden answers for each question.

    It inherits from `MappingDataset[QAEvalData]`, allowing iteration and random access
    to `QAEvalData` objects. Each `QAEvalData` contains a question and its associated
    golden answers.

    Subclasses must implement the following abstract properties to define the data source:

        >>> @property
        >>> def _queries(self) -> Mapping[str, str]:
        >>>     # Return a mapping from question_id to question string
        >>>     ...

        >>> @property
        >>> def _answers(self) -> Mapping[str, list[str]] | None:
        >>>     # Return a mapping from question_id to golden answers list
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
    def _answers(self) -> Mapping[str, list[str]] | None:
        """Mapping from question_id to golden answers list."""
        return

    @cached_property
    def _qids(self) -> list[str]:
        """List of question IDs in the dataset."""
        return sorted(self._answers.keys())

    def get_item(self, index: int) -> QAEvalData:
        qid = self._qids[index]
        question = self._queries[qid]
        if hasattr(self, "_meta_data"):
            meta_data = self._meta_data.get(qid, {})
        else:
            meta_data = {}
        if self._answers is not None:
            answers = self._answers.get(qid)
        else:
            answers = None
        return QAEvalData(question=question, answers=answers, meta_data=meta_data)

    def __len__(self) -> int:
        return len(self._qids)


@data
class KnowledgeQAEvalData(IREvalData, QAEvalData):
    """The dataclass for Knowledge-Intensive QA task."""


class KnowledgeQADatasetBase(RetrievalDatasetBase, QADatasetBase):
    """Base class for Knowledge-Intensive Question Answering (KIQA) datasets.

    This class combines the features of `RetrievalDatasetBase` and `QADatasetBase` to support
    tasks that require both information retrieval and question answering. It is suitable for
    datasets where questions are associated with both relevant contexts (from a corpus) and
    golden answers.

    It inherits from both `RetrievalDatasetBase` and `QADatasetBase`, providing a unified
    interface to access `KnowledgeQAEvalData` objects. Each `KnowledgeQAEvalData` contains:
    1. The question.
    2. Relevant contexts and hard negatives (from `RetrievalDatasetBase`).
    3. Golden answers (from `QADatasetBase`).

    Subclasses must implement the abstract properties required by both parent classes:
    - `_contexts` (from `RetrievalDatasetBase`)
    - `_queries` (from both, usually shared)
    - `_qrels` (from `RetrievalDatasetBase`)
    - `_answers` (from `QADatasetBase`)
    """

    def get_item(self, index: int) -> QAEvalData:
        ir_data = RetrievalDatasetBase.get_item(self, index)
        qa_data = QADatasetBase.get_item(self, index)
        return KnowledgeQAEvalData(
            question=ir_data.question,
            contexts=ir_data.contexts,
            hard_negatives=ir_data.hard_negatives,
            meta_data=ir_data.meta_data | qa_data.meta_data,
            answers=qa_data.answers,
        )


QA_DATASETS = Register[QADatasetBase]("qa_dataset")
KNOWLEDGE_QA_DATASETS = Register[KnowledgeQADatasetBase]("knowledge_qa_dataset")
