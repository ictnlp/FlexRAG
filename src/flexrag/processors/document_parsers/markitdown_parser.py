from .document_parser_base import DOCUMENTPARSERS, Document, DocumentParserBase


@DOCUMENTPARSERS("markitdown")
class MarkItDownParser(DocumentParserBase):
    def __init__(self):
        try:
            from markitdown import MarkItDown
        except ImportError:
            raise ImportError(
                "MarkItDown is not installed. Please install it via `pip install markitdown`."
            )
        finally:
            self.parser = MarkItDown()
        return

    def parse(self, path: str) -> Document:
        doc = self.parser.convert(path)
        return Document(source_file_path=path, text=doc.text_content, title=doc.title)
