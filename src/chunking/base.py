from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Chunk:
    """A single text chunk produced by a chunking strategy.

    Attributes:
        index: Zero-based chunk position within the document.
        text: Chunk text content.
        metadata: Strategy-specific metadata (e.g. character offsets).
    """

    index: int
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


class ChunkingStrategy(ABC):
    """Abstract base class for text chunking strategies.

    To add a new strategy, inherit from this class and implement ``chunk()``.
    Then register the strategy name in :func:`chunking.factory.create_chunking_strategy`.
    """

    @abstractmethod
    def chunk(self, text: str) -> list[Chunk]:
        """Split text into chunks.

        Args:
            text: Full document text to chunk.

        Returns:
            Ordered list of ``Chunk`` objects.
        """
        ...
