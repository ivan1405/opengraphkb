from parsers.base import DocumentParser, ParseResult


class MarkdownParser(DocumentParser):
    """Passthrough parser for markdown files — no transformation needed."""

    def supported_content_types(self) -> set[str]:
        return {"text/markdown", "text/x-markdown"}

    async def parse(self, content: bytes, filename: str) -> ParseResult:
        text = content.decode("utf-8", errors="replace")
        return ParseResult(text=text, metadata={"parser": "markdown_passthrough"})
