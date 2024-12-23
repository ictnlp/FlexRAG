from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

from PIL.Image import Image

from librarian.utils import Register


@dataclass
class Document:
    source_file_path: str
    text: Optional[str] = None
    screenshots: list[Image] = field(default_factory=list)
    images: list[Image] = field(default_factory=list)


class DocumentParserBase(ABC):
    @abstractmethod
    def parse(self, document_path: str) -> Document:
        """Parse the document at the given path.

        Args:
            document_path (str): The path to the document to parse.

        Returns:
            Document: The parsed document.
        """
        return


DOCUMENTPARSERS = Register[DocumentParserBase]("document_parser")
