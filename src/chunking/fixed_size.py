from chunking.base import Chunk, ChunkingStrategy


class FixedSizeChunker(ChunkingStrategy):
    """Character-based fixed-size chunking with configurable overlap.

    Splits text into chunks of ``size`` characters, stepping forward by
    ``size - overlap`` characters each iteration. Trailing whitespace is
    stripped from each chunk; empty chunks are discarded.
    """

    def __init__(self, size: int = 1000, overlap: int = 100) -> None:
        if size <= 0:
            raise ValueError(f"Chunk size must be positive, got {size}")
        if overlap < 0:
            raise ValueError(f"Chunk overlap must be non-negative, got {overlap}")
        if overlap >= size:
            raise ValueError(f"Chunk overlap ({overlap}) must be less than size ({size})")
        self._size = size
        self._overlap = overlap

    def chunk(self, text: str) -> list[Chunk]:
        chunks: list[Chunk] = []
        start = 0
        index = 0
        while start < len(text):
            end = start + self._size
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(
                    Chunk(
                        index=index,
                        text=chunk_text,
                        metadata={"start": start, "end": min(end, len(text))},
                    )
                )
                index += 1
            start = end - self._overlap
        return chunks
