from dataclasses import field
from typing import Optional

from flexrag.common import ChatMessages, ChatTurn, Context, data


@data(kw_only=True)
class IRSample:
    """The dataclass for Information Retrieval evaluation. The IRSample does
    not have the candidates to be ranked, the contexts should be retrieved from
    the whole corpus.

    :param question: The question for evaluation. Required.
    :type question: str
    :param question_id: The unique identifier for the question. Default: None.
    :type question_id: Optional[str]
    :param contexts: The contexts related to the question. Default: None.
    :type contexts: Optional[list[Context]]
    :param meta_data: The metadata of the evaluation data. Default: None.
    :type meta_data: Optional[dict]
    """

    question: str
    question_id: Optional[str] = None
    contexts: Optional[list[Context]] = None
    meta_data: Optional[dict] = None


@data(kw_only=True)
class RankingSample(IRSample):
    """The dataclass for Passage / Document Ranking evaluation. Different from
    IRSample, RankingSample has a list of candidate contexts to be ranked, and
    the candidates are also included in the contexts field.

    :param question: The question for evaluation. Required.
    :type question: str
    :param candidates: The candidate contexts to be ranked. Required.
    :type candidates: list[Context]
    :param question_id: The unique identifier for the question. Default: None.
    :type question_id: Optional[str]
    :param contexts: The contexts related to the question. Default: None.
    :type contexts: Optional[list[Context]]
    :param meta_data: The metadata of the evaluation data. Default: None.
    :type meta_data: Optional[dict]
    """

    candidates: list[Context]


@data(kw_only=True)
class MultipleChoiceSample:
    """The dataclass for Multiple Choice evaluation.

    :param question: The question for evaluation. Required.
    :type question: str
    :param question_id: The unique identifier for the question. Default: None.
    :type question_id: Optional[str]
    :param choices: The list of answer choices. Required.
    :type choices: list[str]
    :param meta_data: The metadata of the evaluation data. Default: None.
    :type meta_data: Optional[dict]
    :param answers: The golden answers for the question. Default: None.
    :type answers: Optional[list[int]]
    """

    question: str
    choices: list[str]
    question_id: Optional[str] = None
    meta_data: Optional[dict] = None
    answers: Optional[list[int]] = None


@data(kw_only=True)
class ContextualMCSample(MultipleChoiceSample):
    """The dataclass for Contextualized Multiple Choice evaluation data.

    :param question: The question for evaluation. Required.
    :type question: str
    :param question_id: The unique identifier for the question. Default: None.
    :type question_id: Optional[str]
    :param choices: The list of answer choices. Required.
    :type choices: list[str]
    :param meta_data: The metadata of the evaluation data. Default: None.
    :type meta_data: Optional[dict]
    :param answers: The golden answers for the question. Default: None.
    :type answers: Optional[list[int]]
    :param contexts: The contexts related to the question. Default: [].
    :type contexts: list[Context]
    """

    contexts: list[Context] = field(default_factory=list)


@data(kw_only=True)
class QASample:
    """The dataclass for QA evaluation.

    :param question: The question for evaluation. Required.
    :type question: str
    :param question_id: The unique identifier for the question. Default: None.
    :type question_id: Optional[str]
    :param meta_data: The metadata of the evaluation data. Default: None.
    :type meta_data: Optional[dict]
    :param answers: The golden answers for the question. Default: None.
    :type answers: Optional[list[str]]
    """

    question: str
    question_id: Optional[str] = None
    meta_data: Optional[dict] = None
    answers: Optional[list[str]] = None


@data(kw_only=True)
class ContextualQASample(QASample):
    """The dataclass for Contextualized QA evaluation data.

    :param question: The question for evaluation. Required.
    :type question: str
    :param question_id: The unique identifier for the question. Default: None.
    :type question_id: Optional[str]
    :param meta_data: The metadata of the evaluation data. Default: None.
    :type meta_data: Optional[dict]
    :param answers: The golden answers for the question. Default: None.
    :type answers: Optional[list[str]]
    :param contexts: The contexts related to the question. Default: [].
    :type contexts: list[Context]
    """

    contexts: list[Context] = field(default_factory=list)


@data(kw_only=True)
class MultiSessionQASample(QASample):
    """The dataclass for Multi-Session QA evaluation data.

    :param question: The question for evaluation. Required.
    :type question: str
    :param question_id: The unique identifier for the question. Default: None.
    :type question_id: Optional[str]
    :param answers: The golden answers for the question. Default: None.
    :type answers: Optional[list[str]]
    :param sessions: A list of completed conversation sessions. Default: [].
    :type sessions: list[ChatMessages]
    :param meta_data: The metadata of the evaluation data. Default: None.
    :type meta_data: Optional[dict]
    """

    sessions: list[ChatMessages] = field(default_factory=list)


@data(kw_only=True)
class DialogueSample:
    """The dataclass for Dialogue evaluation.

    :param dialogue_id: The unique identifier for the dialogue. Default: None.
    :type dialogue_id: Optional[str]
    :param messages: The history messages of the dialogue. Required.
    :type messages: ChatMessages
    :param golden_responses: The golden responses for the dialogue. Default: None.
    :type golden_responses: Optional[list[ChatTurn]]
    :param meta_data: The metadata of the evaluation data. Default: None.
    :type meta_data: Optional[dict]
    """

    messages: ChatMessages
    dialogue_id: Optional[str] = None
    golden_responses: Optional[list[ChatTurn]] = None
    meta_data: Optional[dict] = None


@data(kw_only=True)
class ContextualDialogueSample(DialogueSample):
    """The dataclass for Contextualized Dialogue evaluation data.

    :param dialogue_id: The unique identifier for the dialogue. Default: None.
    :type dialogue_id: Optional[str]
    :param messages: The history messages of the dialogue. Required.
    :type messages: ChatMessages
    :param golden_responses: The golden responses for the dialogue. Default: None.
    :type golden_responses: Optional[list[ChatTurn]]
    :param contexts: The contexts related to the dialogue. Default: [].
    :type contexts: list[Context]
    :param meta_data: The metadata of the evaluation data. Default: None.
    :type meta_data: Optional[dict]
    """

    contexts: list[Context] = field(default_factory=list)
