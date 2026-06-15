import asyncio

from openai import AsyncOpenAI

from core.logging import get_logger
from llm.base import LLMProvider

log = get_logger("llm.openai")

_EMBED_CACHE_MAXSIZE = 512
_MAX_RETRIES = 3
_EMBED_BATCH_SIZE = 16


class OpenAILLMProvider(LLMProvider):
    """OpenAI-compatible LLM provider.

    Works with any endpoint that implements the OpenAI API contract:
    OpenAI, Azure OpenAI (via proxy), vLLM, Ollama, LiteLLM, etc.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str,
        chat_model: str,
        embedding_model: str,
        embedding_dimensions: int,
    ) -> None:
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self._chat_model = chat_model
        self._embedding_model = embedding_model
        self._embedding_dimensions = embedding_dimensions
        self._embed_cache: dict[str, list[float]] = {}

    async def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        model: str | None = None,
    ) -> str:
        effective_model = model or self._chat_model
        last_exc: Exception | None = None

        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                response = await self._client.chat.completions.create(
                    model=effective_model,
                    temperature=temperature,
                    messages=messages,  # type: ignore[arg-type]
                )
                content = response.choices[0].message.content or ""
                return content
            except Exception as exc:
                last_exc = exc
                if attempt < _MAX_RETRIES:
                    delay = 2 ** attempt
                    log.warning(
                        "llm_chat_retry",
                        attempt=attempt,
                        delay=delay,
                        error=str(exc),
                    )
                    await asyncio.sleep(delay)

        assert last_exc is not None
        raise last_exc

    async def embed(
        self,
        texts: list[str],
        *,
        model: str | None = None,
        dimensions: int | None = None,
    ) -> list[list[float]]:
        effective_model = model or self._embedding_model
        effective_dimensions = dimensions or self._embedding_dimensions

        # Check cache for all texts
        results: list[list[float] | None] = []
        uncached_indices: list[int] = []
        uncached_texts: list[str] = []

        for i, text in enumerate(texts):
            cached = self._embed_cache.get(text)
            if cached is not None:
                results.append(cached)
            else:
                results.append(None)
                uncached_indices.append(i)
                uncached_texts.append(text)

        # Batch-embed uncached texts
        if uncached_texts:
            all_embeddings: list[list[float]] = []
            for batch_start in range(0, len(uncached_texts), _EMBED_BATCH_SIZE):
                batch = uncached_texts[batch_start : batch_start + _EMBED_BATCH_SIZE]

                kwargs: dict = {}
                if "ada" not in effective_model.lower():
                    kwargs["dimensions"] = effective_dimensions

                response = await self._client.embeddings.create(
                    model=effective_model,
                    input=batch,
                    **kwargs,
                )
                for item in response.data:
                    all_embeddings.append(item.embedding)

            # Populate results and cache
            for idx, embedding in zip(uncached_indices, all_embeddings, strict=True):
                results[idx] = embedding
                text = texts[idx]
                # Evict oldest if cache is full
                if len(self._embed_cache) >= _EMBED_CACHE_MAXSIZE:
                    self._embed_cache.pop(next(iter(self._embed_cache)))
                self._embed_cache[text] = embedding

        return results  # type: ignore[return-value]

    async def close(self) -> None:
        await self._client.close()
