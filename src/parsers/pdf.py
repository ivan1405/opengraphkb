import asyncio

from core.logging import get_logger
from parsers.base import DocumentParser, ParseResult

log = get_logger("parser.pdf")


class PDFParser(DocumentParser):
    """Extract text from PDF files using pymupdf4llm for high-quality markdown output."""

    def supported_content_types(self) -> set[str]:
        return {"application/pdf"}

    async def parse(self, content: bytes, filename: str) -> ParseResult:
        return await asyncio.to_thread(self._parse_sync, content, filename)

    def _parse_sync(self, content: bytes, filename: str) -> ParseResult:
        import pymupdf  # type: ignore[import-untyped]
        import pymupdf4llm  # type: ignore[import-untyped]

        doc = pymupdf.open(stream=content, filetype="pdf")
        text = pymupdf4llm.to_markdown(doc)
        page_count = len(doc)
        doc.close()

        log.info("pdf_parsed", filename=filename, pages=page_count, chars=len(text))
        return ParseResult(
            text=text,
            metadata={"page_count": page_count, "parser": "pymupdf4llm"},
        )
