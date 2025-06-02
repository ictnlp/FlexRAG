from abc import ABC, abstractmethod
from typing import Optional

from flexrag.utils import Register, data


@data
class Chunk:
    """The dataclass for a chunk of text.

    :param text: The text of the chunk.
    :type text: str
    :param start: The start index of the chunk in the original text.
    :type start: Optional[int]
    :param end: The end index of the chunk in the original text.
    :type end: Optional[int]
    """

    text: str
    start: Optional[int] = None
    end: Optional[int] = None


class ChunkerBase(ABC):
    """Chunker that splits text into chunks of fixed size.
    This is an abstract class that defines the interface for all chunkers.
    The subclasses should implement the `chunk` method to split the text.
    """

    @abstractmethod
    def chunk(self, text: str, return_str: bool = False) -> list[Chunk]:
        """Chunk the given text into smaller chunks.

        :param text: The text to chunk.
        :type text: str
        :param return_str: If True, return the chunks as strings instead of Chunk objects.
            Default is False.
        :type return_str: bool
        :return: The chunks of the text.
        :rtype: list[Chunk]
        """
        return


CHUNKERS = Register[ChunkerBase]("chunker")
