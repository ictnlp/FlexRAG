from dataclasses import field
from typing import Any, Optional

from flexrag.utils import data


@data
class WebResource:
    """The web resource dataclass.
    ``WebResource`` is the fundamental component for information transmission in the ``web_retrievers`` module of FlexRAG.
    The ``WebSeeker`` retrieves the corresponding ``WebResource`` based on the user's query,
    while the ``WebDownloader`` downloads the resource based on the URL in the ``WebResource`` and stores it in the ``data`` field of the ``WebResource``.
    The ``WebReader`` then converts the ``data`` field of the ``WebResource`` into a LLM friendly format and returns the ``RetrievedContext``.

    :param url: The URL of the resource.
    :type url: str
    :param query: The query for the resource. Default is None.
    :type query: Optional[str]
    :param metadata: The metadata of the resource, offen provided by the WebSeeker. Default is {}.
    :type metadata: dict
    :param data: The content of the resource, offen filled by the WebDownloader. Default is None.
    :type data: Any
    """

    url: str
    query: Optional[str] = None
    metadata: dict = field(default_factory=dict)
    data: Any = None
