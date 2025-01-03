from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

from PIL.Image import Image

from flexrag.utils import Register


@dataclass
class Document:
    """A document parsed by a DocumentParser."""

    source_file_path: str
    text: Optional[str] = None
    screenshots: list[Image] = field(default_factory=list)
    images: list[Image] = field(default_factory=list)


class DocumentParserBase(ABC):
    @abstractmethod
    def parse(self, document_path: str) -> Document:
        """Parse the document at the given path.

        :param document_path: The path to the document to parse.
        :type document_path: str
        :return: The parsed document.
        :rtype: Document
        """
        return


DOCUMENTPARSERS = Register[DocumentParserBase]("document_parser")
