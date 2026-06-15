import pytest

from parsers.base import ParseResult
from parsers.markdown import MarkdownParser
from parsers.text import TextParser
from parsers.registry import ParserRegistry


class TestTextParser:
    @pytest.fixture
    def parser(self):
        return TextParser()

    def test_supported_content_types(self, parser):
        assert "text/plain" in parser.supported_content_types()

    @pytest.mark.asyncio
    async def test_parse_utf8(self, parser):
        content = "Hello, world! This is a test document.".encode("utf-8")
        result = await parser.parse(content, "test.txt")
        assert isinstance(result, ParseResult)
        assert result.text == "Hello, world! This is a test document."
        assert result.metadata["parser"] == "text_passthrough"

    @pytest.mark.asyncio
    async def test_parse_empty(self, parser):
        result = await parser.parse(b"", "empty.txt")
        assert result.text == ""


class TestMarkdownParser:
    @pytest.fixture
    def parser(self):
        return MarkdownParser()

    def test_supported_content_types(self, parser):
        types = parser.supported_content_types()
        assert "text/markdown" in types
        assert "text/x-markdown" in types

    @pytest.mark.asyncio
    async def test_parse_markdown(self, parser):
        content = "# Title\n\nParagraph content.".encode("utf-8")
        result = await parser.parse(content, "test.md")
        assert "# Title" in result.text
        assert "Paragraph content." in result.text


class TestParserRegistry:
    @pytest.fixture
    def registry(self):
        return ParserRegistry.default()

    def test_get_pdf_parser(self, registry):
        parser = registry.get_parser("application/pdf", "test.pdf")
        assert parser is not None

    def test_get_text_parser(self, registry):
        parser = registry.get_parser("text/plain", "test.txt")
        assert parser is not None

    def test_get_markdown_parser(self, registry):
        parser = registry.get_parser("text/markdown", "test.md")
        assert parser is not None

    def test_extension_fallback(self, registry):
        # Fallback to text for unknown content type but .txt extension
        parser = registry.get_parser("application/octet-stream", "readme.txt")
        assert parser is not None

    def test_unknown_format_raises(self, registry):
        from core.exceptions import ParserNotFoundError
        with pytest.raises(ParserNotFoundError):
            registry.get_parser("application/x-custom-binary", "data.bin")
