"""Provider adapter construction and model compatibility checks."""

from __future__ import annotations

from agent.config import AgentConfig
from agent.errors import UnsupportedModelError
from agent.providers.base import LLMProvider
from agent.providers.gemini import GeminiProvider
from agent.providers.ollama_cloud import OllamaCloudProvider
from agent.providers.openai_responses import OpenAIProvider


def create_provider(config: AgentConfig) -> LLMProvider:
    """Build only the explicitly selected provider; never change models implicitly."""
    constructors = {
        "ollama": OllamaCloudProvider,
        "gemini": GeminiProvider,
        "openai": OpenAIProvider,
    }
    constructor = constructors.get(config.provider)
    if constructor is None:
        raise UnsupportedModelError(f"Unsupported LLM provider '{config.provider}'.")
    try:
        return constructor(model=config.model, api_key=config.api_key)
    except UnsupportedModelError:
        raise
    except ValueError:
        raise
