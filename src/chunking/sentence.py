import re

from chunking.base import Chunk, ChunkingStrategy

# Sentence-ending punctuation followed by whitespace or end-of-string.
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


class SentenceChunker(ChunkingStrategy):
    """Sentence-boundary-aware chunker.

    Splits text on sentence boundaries (regex-based) and groups sentences
    into chunks that do not exceed ``max_size`` characters. This produces
    more semantically coherent chunks at the cost of variable chunk size.
    """

    def __init__(self, max_size: int = 1000) -> None:
        if max_size <= 0:
            raise ValueError(f"Max chunk size must be positive, got {max_size}")
        self._max_size = max_size

    def chunk(self, text: str) -> list[Chunk]:
        sentences = _SENTENCE_BOUNDARY.split(text.strip())
        if not sentences:
            return []

        chunks: list[Chunk] = []
        current_parts: list[str] = []
        current_length = 0
        index = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            # If a single sentence exceeds max_size, emit it as its own chunk.
            if len(sentence) > self._max_size:
                if current_parts:
                    chunks.append(Chunk(index=index, text=" ".join(current_parts)))
                    index += 1
                    current_parts = []
                    current_length = 0
                chunks.append(Chunk(index=index, text=sentence))
                index += 1
                continue

            projected = current_length + len(sentence) + (1 if current_parts else 0)
            if projected > self._max_size and current_parts:
                chunks.append(Chunk(index=index, text=" ".join(current_parts)))
                index += 1
                current_parts = []
                current_length = 0

            current_parts.append(sentence)
            current_length += len(sentence) + (1 if len(current_parts) > 1 else 0)

        if current_parts:
            chunks.append(Chunk(index=index, text=" ".join(current_parts)))

        return chunks
