from abc import ABC, abstractmethod

from librarian.utils import Register


class ChunkerBase(ABC):
    @abstractmethod
    def chunk(self, text: str) -> list[str]:
        """Chunk the given text.

        Args:
            text (str): The text to chunk.

        Returns:
            list[str]: The chunks of the text.
        """
        return


CHUNKERS = Register[ChunkerBase]("chunker")
