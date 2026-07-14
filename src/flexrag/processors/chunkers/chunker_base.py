from typing import Optional, Protocol

from flexrag.common import Register, data


@data
class Chunk:
    """The dataclass for a chunk of text.

    :param text: The text of the chunk.
    :type text: str
    :param start: The start index of the chunk in the original text.
    :type start: Optional[int]
    :param end: The end index of the chunk in the original text.
    :type end: Optional[int]
    :param metadata: Optional metadata associated with the chunk.
    :type metadata: Optional[dict]
    """

    text: str
    start: Optional[int] = None
    end: Optional[int] = None
    metadata: Optional[dict] = None


class ChunkerProtocol(Protocol):
    """Protocol for directly usable text chunkers.

    Raw chunkers and managed chunker handles both implement this structural
    interface.
    """

    def chunk(self, text: str) -> list[Chunk]:
        """Split text into chunks.

        :param text: Text to split.
        :return: Chunks produced from the text.
        """
        ...


CHUNKERS = Register[ChunkerProtocol]("chunker")
