import pytest

from flexrag.document_parser import (
    Document,
    TwelveLabsVideoParser,
    TwelveLabsVideoParserConfig,
)


class TestTwelveLabsVideoParser:
    def test_parse_url(self, mock_twelvelabs_client):
        parser = TwelveLabsVideoParser(
            TwelveLabsVideoParserConfig(model="pegasus1.5", api_key="test")
        )
        url = "https://example.com/sample.mp4"
        document = parser.parse(url)
        assert isinstance(document, Document)
        assert document.source_file_path == url
        assert isinstance(document.text, str)
        assert "Pegasus" in document.text
        return
