from dataclasses import dataclass, field
from glob import glob
from typing import Iterator

from flexrag.chunking import CHUNKERS, ChunkerConfig
from flexrag.document_parser import DOCUMENTPARSERS, Document, DocumentParserConfig

from .dataset import IterableDataset


@dataclass
class DocumentDatasetConfig(DocumentParserConfig, ChunkerConfig):
    """The configuration for DocumentDataset.
    The documents will be parsed by the DocumentParser specified in the config and then chunked by the Chunker.
    If the Chunker is not specified, the dataset will yield the whole document.

    :param document_paths: The paths to the documents.
    :type document_paths: list[str] | str
    """

    document_paths: list[str] | str = field(default_factory=list)


class DocumentDataset(IterableDataset):
    """DocumentDataset is a dataset that iterates over documents (PDFs, Words, HTMLs, etc.)."""

    def __init__(self, cfg: DocumentDatasetConfig) -> None:
        # parse paths
        if isinstance(cfg.document_paths, str):
            document_paths = [cfg.document_paths]
        else:
            document_paths = cfg.document_paths
        document_paths = [glob(p) for p in document_paths]
        self.document_paths = [p for doc_path in document_paths for p in doc_path]
        # prepare document parser
        self.parser = DOCUMENTPARSERS.load(cfg)
        self.chunker = CHUNKERS.load(cfg)
        return

    def __iter__(self) -> Iterator[Document | str]:
        for path in self.document_paths:
            document = self.parser.parse(path)
            if self.chunker is not None:
                chunks = self.chunker.chunk(document.text)
                yield from chunks
            else:
                yield document
