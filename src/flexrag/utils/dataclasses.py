from dataclasses import field
from typing import Optional

from .configure import data


@data
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


@data
class RetrievedContext(Context):
    """The dataclass for retrieved context.

    :param retriever: The name of the retriever. Required.
    :type retriever: str
    :param query: The query for retrieval. Required.
    :type query: str
    :param score: The relevance score of the retrieved data. Default: 0.0.
    :type score: float
    """

    retriever: str = ""
    query: str = ""
    score: float = 0.0
