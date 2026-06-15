from abc import ABC, abstractmethod


class LLMProvider(ABC):
    """Abstract base class for LLM providers.

    Covers both chat completions and embedding generation. Each provider
    implementation wraps a specific SDK (OpenAI, Anthropic, etc.) behind
    this uniform interface so that the rest of the codebase never depends
    on a specific vendor.

    To add a new provider:
    1. Create a new file in ``llm/`` (e.g. ``llm/anthropic.py``).
    2. Implement ``LLMProvider`` with the vendor SDK.
    3. Register the provider name in :func:`llm.factory.create_llm_provider`.
    """

    @abstractmethod
    async def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        model: str | None = None,
    ) -> str:
        """Send a chat completion request.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.
            temperature: Sampling temperature.
            model: Override the default chat model.

        Returns:
            The assistant's response content as a string.
        """
        ...

    @abstractmethod
    async def embed(
        self,
        texts: list[str],
        *,
        model: str | None = None,
        dimensions: int | None = None,
    ) -> list[list[float]]:
        """Generate embeddings for a batch of texts.

        Args:
            texts: Texts to embed.
            model: Override the default embedding model.
            dimensions: Override the default embedding dimensions.

        Returns:
            List of embedding vectors, one per input text.
        """
        ...

    @abstractmethod
    async def close(self) -> None:
        """Release any held resources (HTTP connections, etc.)."""
        ...
