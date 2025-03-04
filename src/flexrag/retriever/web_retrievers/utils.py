from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class WebResource:
    """The web resource dataclass.

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
