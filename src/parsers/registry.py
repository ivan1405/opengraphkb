from core.exceptions import ParserNotFoundError
from core.logging import get_logger
from parsers.base import DocumentParser
from parsers.docx import DOCXParser
from parsers.markdown import MarkdownParser
from parsers.pdf import PDFParser
from parsers.text import TextParser

log = get_logger("parser.registry")

# File-extension to content-type mapping for fallback detection
_EXTENSION_MAP: dict[str, str] = {
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".doc": "application/msword",
    ".md": "text/markdown",
    ".markdown": "text/markdown",
    ".txt": "text/plain",
    ".text": "text/plain",
    ".csv": "text/plain",
    ".json": "text/plain",
    ".xml": "text/plain",
    ".html": "text/plain",
    ".htm": "text/plain",
}


class ParserRegistry:
    """Maps content types to parser implementations.

    Uses a two-pass strategy:
    1. Match by declared content type.
    2. Fall back to file extension if content type is generic or missing.
    """

    def __init__(self) -> None:
        self._parsers: dict[str, DocumentParser] = {}

    def register(self, parser: DocumentParser) -> None:
        """Register a parser for all its supported content types."""
        for content_type in parser.supported_content_types():
            self._parsers[content_type] = parser
            log.debug("parser_registered", content_type=content_type, parser=type(parser).__name__)

    def get_parser(self, content_type: str, filename: str) -> DocumentParser:
        """Return the appropriate parser for the given content type and filename.

        Raises:
            ParserNotFoundError: If no parser is registered for the content type
                or inferred extension.
        """
        # Direct content-type match
        parser = self._parsers.get(content_type)
        if parser is not None:
            return parser

        # Extension-based fallback
        extension = _get_extension(filename)
        if extension:
            inferred_type = _EXTENSION_MAP.get(extension)
            if inferred_type:
                parser = self._parsers.get(inferred_type)
                if parser is not None:
                    log.debug(
                        "parser_fallback_to_extension",
                        filename=filename,
                        extension=extension,
                        inferred_type=inferred_type,
                    )
                    return parser

        # Last resort — try text parser for anything text-like
        if content_type.startswith("text/"):
            text_parser = self._parsers.get("text/plain")
            if text_parser is not None:
                return text_parser

        raise ParserNotFoundError(content_type, filename)

    @classmethod
    def default(cls) -> "ParserRegistry":
        """Create a registry pre-loaded with all built-in parsers."""
        registry = cls()
        registry.register(PDFParser())
        registry.register(DOCXParser())
        registry.register(MarkdownParser())
        registry.register(TextParser())
        return registry


def _get_extension(filename: str) -> str:
    """Extract the lowercase file extension including the dot."""
    dot_index = filename.rfind(".")
    if dot_index == -1:
        return ""
    return filename[dot_index:].lower()
