import re
from collections.abc import Iterable

from .file_mixin import FileReaderMixin


class TRECReader(Iterable[dict[str, str]], FileReaderMixin):
    """The iterative reader for loading TREC formatted file."""

    def __init__(self, file_path: str, encoding: str = "utf-8") -> None:
        """Initialize the TRECReader.

        :param file_path: The path to the TREC formatted file.
        :type file_path: str
        :param encoding: The encoding of the file.
        :type encoding: str

        Example:

            >>> reader = TRECReader(
            ...     file_path="data/data.trec",
            ...     encoding="utf-8",
            ... )
            >>> items = [i for i in reader]
        """
        self.file_path = file_path
        self.encoding = encoding
        self._doc_pattern = re.compile(r"<DOC>(.*?)</DOC>", re.DOTALL)
        self._docno_pattern = re.compile(r"<DOCNO>\s*(.*?)\s*</DOCNO>")
        self._title_pattern = re.compile(r"<TITLE>\s*(.*?)\s*</TITLE>", re.DOTALL)
        self._text_pattern = re.compile(r"<TEXT>\s*(.*?)\s*</TEXT>", re.DOTALL)
        self._url_pattern = re.compile(r"<URL>\s*(.*?)\s*</URL>")
        return

    def __iter__(self):
        buffer = []
        with open(self.file_path, "r", encoding=self.encoding, errors="ignore") as f:
            for line in f:
                if "<DOC>" in line:
                    inside_doc = True
                    buffer = [line]
                elif "</DOC>" in line:
                    buffer.append(line)
                    doc_block = "".join(buffer)
                    yield self._parse_doc(doc_block)
                    inside_doc = False
                    buffer = []
                elif inside_doc:
                    buffer.append(line)

    def _parse_doc(self, doc_block: str) -> dict:
        """Parse a single TREC document block."""
        docid = self._get_match(self._docno_pattern, doc_block)
        title = self._get_match(self._title_pattern, doc_block)
        text = self._get_match(self._text_pattern, doc_block)
        return {"docid": docid, "title": title, "text": text}

    @staticmethod
    def _get_match(pattern: re.Pattern, text: str) -> str:
        match = pattern.search(text)
        return match.group(1).strip() if match else ""
