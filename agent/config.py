"""Environment-backed agent configuration."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

from dotenv import load_dotenv

from agent.errors import ConfigurationError


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROVIDERS = ("ollama", "gemini", "openai")


@dataclass(frozen=True)
class AgentConfig:
    provider: str
    model: str
    max_iterations: int = 6
    temperature: float = 0.2
    api_key: str = field(default="", repr=False)

    @classmethod
    def from_env(
        cls,
        environ: Mapping[str, str] | None = None,
        dotenv_path: str | Path | None = None,
    ) -> AgentConfig:
        """Load and validate settings; explicit mappings avoid process-env leakage in tests."""
        if environ is None:
            load_dotenv(
                dotenv_path=Path(dotenv_path) if dotenv_path else PROJECT_ROOT / ".env",
                override=False,
            )
            values: Mapping[str, str] = os.environ
        else:
            values = environ

        provider = values.get("LLM_PROVIDER", "").strip().lower()
        if not provider:
            raise ConfigurationError("LLM_PROVIDER is required.")
        if provider not in PROVIDERS:
            raise ConfigurationError(
                f"Unsupported LLM_PROVIDER '{provider}'. Choose: {', '.join(PROVIDERS)}."
            )

        model = values.get("LLM_MODEL", "").strip()
        if not model:
            raise ConfigurationError("LLM_MODEL is required.")

        key_name = {
            "ollama": "OLLAMA_API_KEY",
            "gemini": "GEMINI_API_KEY",
            "openai": "OPENAI_API_KEY",
        }[provider]
        api_key = values.get(key_name, "").strip()
        if not api_key or api_key.startswith("replace_with_"):
            raise ConfigurationError(
                f"{key_name} is required when LLM_PROVIDER is '{provider}'."
            )

        try:
            max_iterations = int(values.get("LLM_MAX_ITERATIONS", "6"))
        except ValueError as error:
            raise ConfigurationError("LLM_MAX_ITERATIONS must be an integer.") from error
        if not 1 <= max_iterations <= 50:
            raise ConfigurationError("LLM_MAX_ITERATIONS must be between 1 and 50.")

        try:
            temperature = float(values.get("LLM_TEMPERATURE", "0.2"))
        except ValueError as error:
            raise ConfigurationError("LLM_TEMPERATURE must be a number.") from error
        if not math.isfinite(temperature) or not 0 <= temperature <= 2:
            raise ConfigurationError("LLM_TEMPERATURE must be between 0 and 2.")

        return cls(
            provider=provider,
            model=model,
            max_iterations=max_iterations,
            temperature=temperature,
            api_key=api_key,
        )
