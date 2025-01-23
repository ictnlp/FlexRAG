from abc import ABC, abstractmethod

from flexrag.utils import Register


class ChunkerBase(ABC):
    @abstractmethod
    def chunk(self, text: str) -> list[str]:
        """Chunk the given text into smaller chunks.

        :param text: The text to chunk.
        :type text: str
        :return: The chunks of the text.
        :rtype: list[str]
        """
        return


CHUNKERS = Register[ChunkerBase]("chunker")
