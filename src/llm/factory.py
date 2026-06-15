from core.config import Settings
from llm.base import LLMProvider
from llm.openai import OpenAILLMProvider


def create_llm_provider(config: Settings) -> LLMProvider:
    """Instantiate the LLM provider specified in configuration.

    Args:
        config: Application settings.

    Returns:
        An ``LLMProvider`` instance.

    Raises:
        ValueError: If the configured provider name is not recognized.
    """
    match config.llm_provider:
        case "openai":
            return OpenAILLMProvider(
                api_key=config.llm_api_key,
                base_url=config.llm_base_url,
                chat_model=config.llm_chat_model,
                embedding_model=config.llm_embedding_model,
                embedding_dimensions=config.llm_embedding_dimensions,
            )
        case "anthropic":
            raise NotImplementedError(
                "Anthropic LLM provider is not yet implemented. "
                "Create llm/anthropic.py implementing LLMProvider to add support."
            )
        case _:
            raise ValueError(
                f"Unknown LLM provider '{config.llm_provider}'. "
                f"Supported: 'openai'. Coming soon: 'anthropic'."
            )
