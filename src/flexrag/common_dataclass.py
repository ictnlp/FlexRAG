from dataclasses import dataclass, field
from typing import Optional

from omegaconf import MISSING


@dataclass
class Context:
    """The dataclass for retrieved context.

    :param context_id: The unique identifier of the context. Default: None.
    :type context_id: Optional[str]
    :param data: The context data. Default: {}.
    :type data: dict
    :param source: The source of the retrieved data. Default: None.
    :type source: Optional[str]
    :param meta_data: The metadata of the context. Default: {}.
    :type meta_data: dict
    """

    context_id: Optional[str] = None
    data: dict = field(default_factory=dict)
    source: Optional[str] = None
    meta_data: dict = field(default_factory=dict)

    def to_dict(self):
        return {
            "context_id": self.context_id,
            "source": self.source,
            "data": self.data,
            "meta_data": self.meta_data,
        }


@dataclass
class RetrievedContext(Context):
    """The dataclass for retrieved context.

    :param retriever: The name of the retriever. Required.
    :type retriever: str
    :param query: The query for retrieval. Required.
    :type query: str
    :param score: The relevance score of the retrieved data. Default: 0.0.
    :type score: float
    """

    retriever: str = MISSING
    query: str = MISSING
    score: float = 0.0

    def to_dict(self):
        return {
            **super().to_dict(),
            "retriever": self.retriever,
            "query": self.query,
            "score": self.score,
        }


@dataclass
class RAGEvalData:
    """The dataclass for RAG evaluation data.

    :param question: The question for evaluation. Required.
    :type question: str
    :param golden_contexts: The contexts related to the question. Default: None.
    :type golden_contexts: Optional[list[Context]]
    :param golden_answers: The golden answers for the question. Default: None.
    :type golden_answers: Optional[list[str]]
    :param meta_data: The metadata of the evaluation data. Default: {}.
    :type meta_data: dict
    """

    question: str = MISSING
    golden_contexts: Optional[list[Context]] = None
    golden_answers: Optional[list[str]] = None
    meta_data: dict = field(default_factory=dict)


@dataclass
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
