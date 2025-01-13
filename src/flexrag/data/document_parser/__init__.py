from .document_parser_base import DocumentParserBase, Document, DOCUMENTPARSERS
from .docling_parser import DoclingParser, DoclingConfig
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
