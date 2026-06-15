from chunking.base import ChunkingStrategy
from chunking.fixed_size import FixedSizeChunker
from chunking.sentence import SentenceChunker
from core.config import Settings


def create_chunking_strategy(config: Settings) -> ChunkingStrategy:
    """Instantiate the chunking strategy specified in configuration.

    Args:
        config: Application settings.

    Returns:
        A ``ChunkingStrategy`` instance.

    Raises:
        ValueError: If the configured strategy name is not recognized.
    """
    match config.chunking_strategy:
        case "fixed_size":
            return FixedSizeChunker(size=config.chunk_size, overlap=config.chunk_overlap)
        case "sentence":
            return SentenceChunker(max_size=config.chunk_size)
        case _:
            raise ValueError(
                f"Unknown chunking strategy '{config.chunking_strategy}'. "
                f"Supported: 'fixed_size', 'sentence'."
            )
