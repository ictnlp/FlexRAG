from dataclasses import field
from typing import Optional

from flexrag.common import ChatMessages, ChatTurn, Context, RetrievedContext, data


@data(kw_only=True)
class IRSample:
    """The dataclass for Information Retrieval evaluation. The IRSample does
    not have candidates to be ranked. Its ``contexts`` are optional materialized
    golden contexts and are not retrieval inputs; ``qrels`` is the authoritative
    relevance mapping for evaluation.

    :param question: The question for evaluation. Required.
    :param question_id: The unique identifier for the question. Default: None.
    :param contexts: Materialized golden contexts related to the question.
        Default: None.
    :param qrels: Mapping from context IDs to their relevance grades for this
        question. Defaults to an empty mapping.
    :param metadata: The metadata of the evaluation data. Default: None.
    """

    question: str
    question_id: Optional[str] = None
    contexts: Optional[list[Context]] = None
    qrels: dict[str, float] = field(default_factory=dict)
    metadata: Optional[dict] = None


@data(kw_only=True)
class RankingSample(IRSample):
    """The dataclass for Passage / Document Ranking evaluation. Different from
    IRSample, RankingSample has a list of retrieved candidate contexts to be
    ranked. The ``contexts`` field still contains only golden contexts.

    :param question: The question for evaluation. Required.
    :param candidates: Retrieved candidate contexts to be ranked. Required.
    :param question_id: The unique identifier for the question. Default: None.
    :param contexts: The contexts related to the question. Default: None.
    :param metadata: The metadata of the evaluation data. Default: None.
    """

    candidates: list[RetrievedContext]


@data(kw_only=True)
class MultipleChoiceSample:
    """The dataclass for Multiple Choice evaluation.

    :param question: The question for evaluation. Required.
    :param question_id: The unique identifier for the question. Default: None.
    :param choices: The list of answer choices. Required.
    :param metadata: The metadata of the evaluation data. Default: None.
    :param answers: The golden answers for the question. Default: None.
    """

    question: str
    choices: list[str]
    question_id: Optional[str] = None
    metadata: Optional[dict] = None
    answers: Optional[list[int]] = None


@data(kw_only=True)
class ContextualMCSample(MultipleChoiceSample):
    """The dataclass for Contextualized Multiple Choice evaluation data.

    :param question: The question for evaluation. Required.
    :param question_id: The unique identifier for the question. Default: None.
    :param choices: The list of answer choices. Required.
    :param metadata: The metadata of the evaluation data. Default: None.
    :param answers: The golden answers for the question. Default: None.
    :param contexts: The contexts related to the question. Default: [].
    """

    contexts: list[Context] = field(default_factory=list)


@data(kw_only=True)
class IRMCSample(MultipleChoiceSample, IRSample):
    """The dataclass for Information Retrieval Multiple Choice evaluation data.
    Different from ContextualMCSample, the `contexts` field in IRMCSample is
    golden contexts that are relevant to the question, which should not be
    provided as input. The ContextualMCSample can be used for both IR task
    and RAG tasks.

    :param question: The question for evaluation. Required.
    :param question_id: The unique identifier for the question. Default: None.
    :param choices: The list of answer choices. Required.
    :param metadata: The metadata of the evaluation data. Default: None.
    :param answers: The golden answers for the question. Default: None.
    :param contexts: The golden contexts related to the question. Default: [].
    """


@data(kw_only=True)
class QASample:
    """The dataclass for QA evaluation.

    :param question: The question for evaluation. Required.
    :param question_id: The unique identifier for the question. Default: None.
    :param metadata: The metadata of the evaluation data. Default: None.
    :param answers: The golden answers for the question. Default: None.
    """

    question: str
    question_id: Optional[str] = None
    metadata: Optional[dict] = None
    answers: Optional[list[str]] = None


@data(kw_only=True)
class ContextualQASample(QASample):
    """The dataclass for Contextualized QA evaluation data.

    :param question: The question for evaluation. Required.
    :param question_id: The unique identifier for the question. Default: None.
    :param metadata: The metadata of the evaluation data. Default: None.
    :param answers: The golden answers for the question. Default: None.
    :param contexts: The contexts related to the question. Default: [].
    """

    contexts: list[Context] = field(default_factory=list)


@data(kw_only=True)
class IRQASample(QASample, IRSample):
    """The dataclass for Information Retrieval QA evaluation data.
    Different from ContextualQASample, the `contexts` field in IRQASample is
    golden contexts that are relevant to the question, which should not be
    provided as input. The ContextualQASample can be used for both IR task
    and RAG tasks.

    :param question: The question for evaluation. Required.
    :param question_id: The unique identifier for the question. Default: None.
    :param metadata: The metadata of the evaluation data. Default: None.
    :param answers: The golden answers for the question. Default: None.
    :param contexts: The golden contexts related to the question. Default: [].
    """


@data(kw_only=True)
class MultiSessionQASample(QASample):
    """The dataclass for Multi-Session QA evaluation data.

    :param question: The question for evaluation. Required.
    :param question_id: The unique identifier for the question. Default: None.
    :param answers: The golden answers for the question. Default: None.
    :param sessions: A list of completed conversation sessions. Default: [].
    :param sessions_id: The unique identifier for the conversation sessions.
        Default: None.
    :param metadata: The metadata of the evaluation data. Default: None.
    """

    sessions: list[ChatMessages] = field(default_factory=list)
    sessions_id: Optional[str] = None


@data(kw_only=True)
class DialogueSample:
    """The dataclass for Dialogue evaluation.

    :param dialogue_id: The unique identifier for the dialogue. Default: None.
    :param messages: The history messages of the dialogue. Required.
    :param golden_responses: The golden responses for the dialogue. Default: None.
    :param metadata: The metadata of the evaluation data. Default: None.
    """

    messages: ChatMessages
    dialogue_id: Optional[str] = None
    golden_responses: Optional[list[ChatTurn]] = None
    metadata: Optional[dict] = None


@data(kw_only=True)
class ContextualDialogueSample(DialogueSample):
    """The dataclass for Contextualized Dialogue evaluation data.

    :param dialogue_id: The unique identifier for the dialogue. Default: None.
    :param messages: The history messages of the dialogue. Required.
    :param golden_responses: The golden responses for the dialogue. Default: None.
    :param contexts: The contexts related to the dialogue. Default: [].
    :param metadata: The metadata of the evaluation data. Default: None.
    """

    contexts: list[Context] = field(default_factory=list)


@data(kw_only=True)
class IRDialogueSample(DialogueSample):
    """The dataclass for Information Retrieval Dialogue evaluation data.
    Different from ContextualDialogueSample, the `contexts` field in IRDialogueSample
    is golden contexts that are relevant to the dialogue, which should not be provided
    as input. The ContextualDialogueSample can be used for RAG tasks.

    :param dialogue_id: The unique identifier for the dialogue. Default: None.
    :param messages: The history messages of the dialogue. Required.
    :param golden_responses: The golden responses for the dialogue. Default: None.
    :param contexts: The golden contexts related to the dialogue. Default: [].
    :param qrels: Mapping from context IDs to their relevance grades for this
        dialogue. Defaults to an empty mapping.
    :param metadata: The metadata of the evaluation data. Default: None.
    """

    contexts: list[Context] = field(default_factory=list)
    qrels: dict[str, float] = field(default_factory=dict)
