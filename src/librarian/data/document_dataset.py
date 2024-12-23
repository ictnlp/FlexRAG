from dataclasses import dataclass, field
from glob import glob
from typing import Iterator

from .chunking import CHUNKERS
from .dataset import Dataset
from .document_parser import DOCUMENTPARSERS, Document

ParserConfig = DOCUMENTPARSERS.make_config(default="markitdown")
ChunkerConfig = CHUNKERS.make_config(default=None)


@dataclass
class DocumentDatasetConfig(ParserConfig, ChunkerConfig):
    document_paths: list[str] | str = field(default_factory=list)


class DocumentDataset(Dataset):
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
