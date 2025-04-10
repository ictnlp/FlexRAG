from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from flexrag.utils import Register


@dataclass
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
    def chunk(self, text: str) -> list[Chunk]:
        """Chunk the given text into smaller chunks.

        :param text: The text to chunk.
        :type text: str
        :return: The chunks of the text.
        :rtype: list[Chunk]
        """
        return


CHUNKERS = Register[ChunkerBase]("chunker")
