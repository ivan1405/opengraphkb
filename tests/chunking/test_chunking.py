import pytest

from chunking.fixed_size import FixedSizeChunker
from chunking.sentence import SentenceChunker


class TestFixedSizeChunker:
    def test_basic_chunking(self):
        chunker = FixedSizeChunker(size=20, overlap=5)
        text = "a" * 50
        chunks = chunker.chunk(text)
        assert len(chunks) >= 3
        assert all(len(c.text) <= 20 for c in chunks)

    def test_overlap(self):
        chunker = FixedSizeChunker(size=10, overlap=3)
        text = "0123456789ABCDEFGHIJ"
        chunks = chunker.chunk(text)
        assert len(chunks) >= 2
        # Second chunk should start at position 7 (10 - 3)
        assert chunks[1].text.startswith("789")

    def test_empty_text(self):
        chunker = FixedSizeChunker(size=100, overlap=10)
        assert chunker.chunk("") == []
        assert chunker.chunk("   ") == []

    def test_text_smaller_than_chunk(self):
        chunker = FixedSizeChunker(size=100, overlap=10)
        chunks = chunker.chunk("Short text")
        assert len(chunks) == 1
        assert chunks[0].text == "Short text"

    def test_chunk_indices(self):
        chunker = FixedSizeChunker(size=10, overlap=0)
        chunks = chunker.chunk("a" * 30)
        assert [c.index for c in chunks] == [0, 1, 2]

    def test_invalid_size_raises(self):
        with pytest.raises(ValueError):
            FixedSizeChunker(size=0, overlap=0)

    def test_overlap_ge_size_raises(self):
        with pytest.raises(ValueError):
            FixedSizeChunker(size=10, overlap=10)


class TestSentenceChunker:
    def test_basic_sentence_chunking(self):
        chunker = SentenceChunker(max_size=50)
        text = "First sentence. Second sentence. Third sentence."
        chunks = chunker.chunk(text)
        assert len(chunks) >= 1
        for chunk in chunks:
            assert len(chunk.text) <= 50 or chunk.text.count(".") <= 1

    def test_long_sentence_alone(self):
        chunker = SentenceChunker(max_size=20)
        text = "A very long sentence that exceeds the maximum size. Short."
        chunks = chunker.chunk(text)
        # The long sentence should be its own chunk
        assert len(chunks) >= 2

    def test_empty_text(self):
        chunker = SentenceChunker(max_size=100)
        assert chunker.chunk("") == []
        assert chunker.chunk("   ") == []

    def test_invalid_max_size_raises(self):
        with pytest.raises(ValueError):
            SentenceChunker(max_size=0)
