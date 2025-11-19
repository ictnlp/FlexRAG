from .docling_parser import DoclingConfig, DoclingParser
from .document_parser_base import DOCUMENTPARSERS, Document, DocumentParserBase
from .markitdown_parser import MarkItDownParser

DocumentParserConfig = DOCUMENTPARSERS.make_config(default="markitdown")


__all__ = [
    "DocumentParserBase",
    "Document",
    "DOCUMENTPARSERS",
    "DocumentParserConfig",
    "DoclingParser",
    "DoclingConfig",
    "MarkItDownParser",
]
