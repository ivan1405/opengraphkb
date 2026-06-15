from parsers.base import DocumentParser, ParseResult


class TextParser(DocumentParser):
    """Passthrough parser for plain text files."""

    def supported_content_types(self) -> set[str]:
        return {"text/plain"}

    async def parse(self, content: bytes, filename: str) -> ParseResult:
        text = content.decode("utf-8", errors="replace")
        return ParseResult(text=text, metadata={"parser": "text_passthrough"})
