import asyncio
import io

from core.logging import get_logger
from parsers.base import DocumentParser, ParseResult

log = get_logger("parser.docx")


class DOCXParser(DocumentParser):
    """Extract text from DOCX files using python-docx."""

    def supported_content_types(self) -> set[str]:
        return {
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "application/msword",
        }

    async def parse(self, content: bytes, filename: str) -> ParseResult:
        return await asyncio.to_thread(self._parse_sync, content, filename)

    def _parse_sync(self, content: bytes, filename: str) -> ParseResult:
        from docx import Document  # type: ignore[import-untyped]

        doc = Document(io.BytesIO(content))
        paragraphs: list[str] = []
        for paragraph in doc.paragraphs:
            text = paragraph.text.strip()
            if text:
                # Preserve heading structure as markdown
                if paragraph.style and paragraph.style.name.startswith("Heading"):
                    level = paragraph.style.name.replace("Heading", "").strip()
                    try:
                        heading_level = int(level)
                    except ValueError:
                        heading_level = 1
                    paragraphs.append(f"{'#' * heading_level} {text}")
                else:
                    paragraphs.append(text)

        full_text = "\n\n".join(paragraphs)
        log.info(
            "docx_parsed",
            filename=filename,
            paragraphs=len(paragraphs),
            chars=len(full_text),
        )
        return ParseResult(
            text=full_text,
            metadata={"paragraph_count": len(paragraphs), "parser": "python-docx"},
        )
