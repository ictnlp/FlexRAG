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
        content = self.parser.convert(path).text_content
        return Document(source_file_path=path, text=content)
