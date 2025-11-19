from dataclasses import field
from typing import Optional

from flexrag.common import data
from flexrag.common.dataclasses import Context

from .dataset import MappingDataset


@data
class MultipleChoiceEvalData:
    """The dataclass for multiple choice task.

    :param question: The question for evaluation. Required.
    :type question: str
    :param options: The options for the question. Required.
    :type options: list[str]
    :param golden_option: The golden option for the question. Default: None.
    :type golden_option: Optional[list[int]]
    :param golden_contexts: The contexts related to the question. Default: None.
    :type golden_contexts: Optional[list[Context]]
    :param meta_data: The metadata of the evaluation data. Default: {}.
    :type meta_data: dict
    """

    question: str
    options: list[str]
    golden_options: Optional[list[int]] = None
    golden_contexts: Optional[list[Context]] = None
    meta_data: dict = field(default_factory=dict)


@data
class TrueFalseEvalData:
    """The dataclass for true/false task.

    :param question: The question for evaluation. Required.
    :type question: str
    :param golden_contexts: The contexts related to the question. Default: None.
    :type golden_contexts: Optional[list[Context]]
    :param golden_answer: The golden answer for the question. Default: None.
    :type golden_answer: Optional[bool]
    :param meta_data: The metadata of the evaluation data. Default: {}.
    :type meta_data: dict
    """

    question: str
    golden_contexts: Optional[list[Context]] = None
    golden_answer: Optional[bool] = None
    meta_data: dict = field(default_factory=dict)


class MultipleChoiceDataset(MappingDataset[MultipleChoiceEvalData]):
    """Interface for knowledge intensive multiple choice dataset."""


class TrueFalseDataset(MappingDataset[TrueFalseEvalData]):
    """Interface for knowledge intensive true/false dataset."""


class MIRAGEDatset(MultipleChoiceDataset): ...
