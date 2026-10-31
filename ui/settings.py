"""Provider settings, validated against the existing provider adapters."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import streamlit as st
from dotenv import load_dotenv

from agent.config import AgentConfig, PROJECT_ROOT, PROVIDERS
from agent.errors import ConfigurationError, UnsupportedModelError
from agent.providers.gemini import SUPPORTED_MODELS as GEMINI_MODELS
from agent.providers.ollama_cloud import SUPPORTED_MODELS as OLLAMA_MODELS
from agent.providers.openai_responses import SUPPORTED_MODELS as OPENAI_MODELS
from ui.styles import render_page_heading


SESSION_CONFIG_KEY = "bizbridge_session_agent_config"
_API_KEY_WIDGET_KEY = "bizbridge_settings_api_key"
_PROVIDER_WIDGET_KEY = "bizbridge_settings_provider"
_ERROR_KEY = "bizbridge_settings_error"
_SESSION_CONFIG_ERROR_KEY = "bizbridge_session_agent_config_error"

PROVIDER_LABELS = {
    "ollama": "Ollama Cloud",
    "gemini": "Gemini",
    "openai": "OpenAI",
}
API_KEY_ENV_VARS = {
    "ollama": "OLLAMA_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "openai": "OPENAI_API_KEY",
}
PROVIDER_MODELS = {
    "ollama": OLLAMA_MODELS,
    "gemini": tuple(sorted(GEMINI_MODELS)),
    "openai": tuple(sorted(OPENAI_MODELS)),
}


@dataclass(frozen=True)
class ProviderSettingsStatus:
    provider: str | None
    model: str | None
    credentials_available: bool
    provider_supported: bool
    model_supported: bool


@dataclass(frozen=True)
class SessionAgentConfig:
    provider: str
    model: str
    max_iterations: int
    temperature: float
    api_key: str = field(repr=False)

    def as_agent_config(self) -> AgentConfig:
        return AgentConfig(
            provider=self.provider,
            model=self.model,
            max_iterations=self.max_iterations,
            temperature=self.temperature,
            api_key=self.api_key,
        )


def _environment(environ: Mapping[str, str] | None = None) -> Mapping[str, str]:
    if environ is not None:
        return environ
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    return os.environ


def get_settings_status(
    environ: Mapping[str, str] | None = None,
) -> ProviderSettingsStatus:
    """Return safe configuration metadata without exposing credential values."""
    values = _environment(environ)
    provider = values.get("LLM_PROVIDER", "").strip().lower() or None
    model = values.get("LLM_MODEL", "").strip() or None
    provider_supported = provider in PROVIDERS
    model_supported = bool(
        provider_supported
        and model in PROVIDER_MODELS.get(provider or "", ())
    )
    key_name = API_KEY_ENV_VARS.get(provider or "")
    credentials_available = bool(
        key_name
        and values.get(key_name, "").strip()
        and not values.get(key_name, "").strip().startswith("replace_with_")
    )
    return ProviderSettingsStatus(
        provider=provider,
        model=model,
        credentials_available=credentials_available,
        provider_supported=provider_supported,
        model_supported=model_supported,
    )


def get_session_agent_config(
    state: Mapping[str, Any] | None = None,
) -> AgentConfig | None:
    """Return a selected in-session config without consulting or changing CLI config."""
    values = st.session_state if state is None else state
    config_error = values.get(_SESSION_CONFIG_ERROR_KEY)
    if isinstance(config_error, str) and config_error:
        raise ConfigurationError(config_error)
    configured = values.get(SESSION_CONFIG_KEY)
    if isinstance(configured, SessionAgentConfig):
        return configured.as_agent_config()
    return None


def build_session_config(
    provider: str,
    model: str,
    api_key: str | None = None,
    *,
    environ: Mapping[str, str] | None = None,
) -> SessionAgentConfig:
    """Validate a session choice using the provider adapters and shared config parser."""
    if provider not in PROVIDERS:
        raise ConfigurationError("Choose one of the implemented providers.")
    if model not in PROVIDER_MODELS[provider]:
        raise UnsupportedModelError(
            "The selected model is not in this provider's supported model set."
        )

    values = dict(_environment(environ))
    values["LLM_PROVIDER"] = provider
    values["LLM_MODEL"] = model
    key_name = API_KEY_ENV_VARS[provider]
    key = (api_key or "").strip() or values.get(key_name, "").strip()
    if not key or key.startswith("replace_with_"):
        raise ConfigurationError(
            f"Credentials for {PROVIDER_LABELS[provider]} are not available. "
            f"Set {key_name} in your environment or enter a temporary key."
        )
    values[key_name] = key
    config = AgentConfig.from_env(environ=values)
    return SessionAgentConfig(
        provider=config.provider,
        model=config.model,
        max_iterations=config.max_iterations,
        temperature=config.temperature,
        api_key=config.api_key,
    )


def _apply_session_configuration() -> None:
    """Apply validated settings from form widgets and discard the key widget value."""
    provider = st.session_state.get(_PROVIDER_WIDGET_KEY)
    model = st.session_state.get(f"bizbridge_settings_model_{provider}")
    entered_key = st.session_state.get(_API_KEY_WIDGET_KEY, "")
    try:
        current = get_session_agent_config()
    except ConfigurationError:
        current = None
    if not entered_key and current is not None and current.provider == provider:
        entered_key = current.api_key
    try:
        if provider is None or model is None:
            raise ConfigurationError("Select both a provider and a supported model.")
        selected = build_session_config(provider, model, entered_key)
    except (ConfigurationError, UnsupportedModelError, ValueError) as error:
        st.session_state[_ERROR_KEY] = str(error)
        st.session_state.pop(SESSION_CONFIG_KEY, None)
        st.session_state[_SESSION_CONFIG_ERROR_KEY] = str(error)
    else:
        st.session_state[SESSION_CONFIG_KEY] = selected
        st.session_state.pop(_ERROR_KEY, None)
        st.session_state.pop(_SESSION_CONFIG_ERROR_KEY, None)
    finally:
        st.session_state.pop(_API_KEY_WIDGET_KEY, None)


def render_settings_page() -> None:
    """Render provider/model settings without persisting secrets to disk or chat."""
    render_page_heading(
        "Settings",
        "Choose the model that powers this Streamlit session. Credentials stay in the current session or your local environment.",
        "WORKSPACE CONFIGURATION",
    )
    try:
        active_config = get_session_agent_config()
        session_config_error = None
    except ConfigurationError as error:
        active_config = None
        session_config_error = str(error)
    current = get_settings_status()

    if session_config_error:
        active_provider = None
        active_model = None
        credentials_available = False
        source = "Session configuration needs attention"
    elif active_config is not None:
        active_provider = active_config.provider
        active_model = active_config.model
        credentials_available = bool(active_config.api_key)
        source = "This Streamlit session"
    else:
        active_provider = current.provider
        active_model = current.model
        credentials_available = current.credentials_available
        source = "Environment / .env"

    st.subheader("Active configuration")
    with st.container(border=True):
        status_columns = st.columns((1, 1, 1.4))
        with status_columns[0]:
            st.caption("PROVIDER")
            st.markdown(
                f"**{PROVIDER_LABELS.get(active_provider or '', 'Not configured')}**"
            )
        with status_columns[1]:
            st.caption("MODEL")
            st.markdown(f"**{active_model or 'Not configured'}**")
        with status_columns[2]:
            st.caption("CREDENTIAL STATUS")
            credential_status = "Available" if credentials_available else "Missing"
            credential_class = (
                "bb-status-success" if credentials_available else "bb-status-warning"
            )
            st.markdown(
                f'<span class="bb-status {credential_class}">{credential_status}</span>',
                unsafe_allow_html=True,
            )
        st.caption(
            f"Configuration source: {source}. API key values are never displayed."
        )
    if session_config_error:
        st.error("The selected session configuration is invalid. Apply a valid provider and model to continue.")
    if (
        active_config is None
        and current.provider_supported
        and current.model
        and not current.model_supported
    ):
        st.warning(
            "The environment-configured model is not in this Settings model list. "
            "It has not been selected for this session; choose a listed model "
            "explicitly to change the application configuration."
        )

    st.subheader("Provider for this session")
    st.caption(
        "This selection applies only to the current Streamlit session. "
        "The command-line agent continues to use environment configuration."
    )
    provider_options = tuple(PROVIDERS)
    initial_provider = active_provider if active_provider in provider_options else None
    provider_index = (
        provider_options.index(initial_provider) if initial_provider is not None else None
    )

    with st.container(border=True):
        selection_columns = st.columns(2)
        with selection_columns[0]:
            provider = st.selectbox(
                "Provider",
                provider_options,
                index=provider_index,
                format_func=lambda value: PROVIDER_LABELS[value],
                placeholder="Select a provider",
                key=_PROVIDER_WIDGET_KEY,
            )
        models = PROVIDER_MODELS.get(provider, ())
        if provider == active_provider and active_model in models:
            model_index = models.index(active_model)
        else:
            model_index = None
        with selection_columns[1]:
            model = st.selectbox(
                "Model",
                models,
                index=model_index,
                placeholder="Select a supported model",
                key=f"bizbridge_settings_model_{provider}",
                help=(
                    "Gemini and OpenAI choices come from their adapter allowlists. "
                    "Ollama Cloud checks the selected model's tools capability on "
                    "the first assistant request."
                ),
            )
    key_name = API_KEY_ENV_VARS.get(provider or "")
    values = _environment()
    env_key = values.get(key_name, "").strip() if key_name else ""
    env_key_available = bool(
        env_key and not env_key.startswith("replace_with_")
    )
    session_key_available = bool(
        active_config
        and active_config.provider == provider
        and active_config.api_key
    )
    if provider and not (env_key_available or session_key_available):
        st.text_input(
            f"{PROVIDER_LABELS[provider]} API key (optional if set in environment)",
            type="password",
            key=_API_KEY_WIDGET_KEY,
            help=(
                "Used only in this Streamlit session. It is not written to "
                ".env, the database, or chat history."
            ),
        )
    elif provider:
        st.caption(
            "Credentials are available for this provider. The value is not shown."
        )
    if provider and not (env_key_available or session_key_available):
        st.caption(
            "The key is used only for this session and is discarded from the input "
            "widget after applying."
        )
    st.button(
        "Apply configuration",
        on_click=_apply_session_configuration,
        type="primary",
    )

    error_message = st.session_state.pop(_ERROR_KEY, None)
    if session_config_error:
        st.error(session_config_error)
    elif error_message:
        st.error(error_message)
    elif active_config is not None and active_config.provider == provider and active_config.model == model:
        st.success(
            f"Using {PROVIDER_LABELS[active_config.provider]} / {active_config.model} "
            "for this Streamlit session."
        )

    with st.expander("Credential safety and connection checks", expanded=False):
        st.info(
            "A uniform, non-generative connection check is not available in the "
            "current provider adapters. The first assistant request performs the "
            "provider's normal validation; no business tool runs solely to test "
            "credentials."
        )
        st.caption(
            "For persistent setup, put the provider, model, and API key in your "
            "local .env file or environment variables. Never commit .env."
        )
