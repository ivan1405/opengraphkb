from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ParseResult:
    """Output of a document parser.

    Attributes:
        text: Extracted plain text or markdown content.
        metadata: Parser-specific metadata (e.g. page count, detected language).
    """

    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


class DocumentParser(ABC):
    """Abstract base class for document parsers.

    Each implementation handles one or more file formats, identified by MIME type.
    To add support for a new format, create a new class inheriting from
    ``DocumentParser``, implement the two abstract methods, and register
    the instance in :class:`parsers.registry.ParserRegistry`.
    """

    @abstractmethod
    def supported_content_types(self) -> set[str]:
        """Return the MIME types this parser can handle."""
        ...

    @abstractmethod
    async def parse(self, content: bytes, filename: str) -> ParseResult:
        """Extract text from raw file bytes.

        Args:
            content: Raw file content.
            filename: Original filename (used for extension-based fallback).

        Returns:
            A ``ParseResult`` with extracted text and optional metadata.
        """
        ...
