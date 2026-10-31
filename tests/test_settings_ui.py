from __future__ import annotations

import asyncio
import unittest
from pathlib import Path
from unittest.mock import AsyncMock
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from agent.config import AgentConfig, ConfigurationError
from agent.errors import UnsupportedModelError
from ui.settings import (
    API_KEY_ENV_VARS,
    PROVIDER_LABELS,
    PROVIDER_MODELS,
    SESSION_CONFIG_KEY,
    SessionAgentConfig,
    build_session_config,
    get_session_agent_config,
    get_settings_status,
)


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def _is_password_input(widget: object) -> bool:
    proto = getattr(widget, "proto")
    field = proto.DESCRIPTOR.fields_by_name["type"]
    return field.enum_type.values_by_number[proto.type].name == "PASSWORD"


class SettingsConfigurationTests(unittest.TestCase):
    def test_provider_and_model_choices_come_from_implemented_adapters(self) -> None:
        self.assertEqual(
            set(PROVIDER_LABELS),
            {"ollama", "gemini", "openai"},
        )
        self.assertEqual(set(PROVIDER_MODELS), set(PROVIDER_LABELS))
        self.assertEqual(
            set(API_KEY_ENV_VARS),
            set(PROVIDER_LABELS),
        )
        self.assertIn("beam:cloud", PROVIDER_MODELS["ollama"])
        self.assertIn("nemotron-3-nano:30b-cloud", PROVIDER_MODELS["ollama"])
        self.assertIn("nemotron-3-super:cloud", PROVIDER_MODELS["ollama"])

    def test_model_provider_pair_is_validated(self) -> None:
        for provider, model, key_name in (
            ("ollama", "beam:cloud", "OLLAMA_API_KEY"),
            ("gemini", "gemini-2.5-flash", "GEMINI_API_KEY"),
            ("openai", "gpt-6-astra", "OPENAI_API_KEY"),
        ):
            with self.subTest(provider=provider):
                result = build_session_config(
                    provider,
                    model,
                    environ={
                        "LLM_MAX_ITERATIONS": "4",
                        "LLM_TEMPERATURE": "0.5",
                    },
                    api_key=f"{provider}-temporary-key",
                )
                self.assertEqual(result.provider, provider)
                self.assertEqual(result.model, model)
                self.assertEqual(result.max_iterations, 4)
                self.assertEqual(result.temperature, 0.5)
                self.assertEqual(result.api_key, f"{provider}-temporary-key")
                self.assertNotIn(result.api_key, repr(result))
                self.assertEqual(API_KEY_ENV_VARS[provider], key_name)

        with self.assertRaises(UnsupportedModelError):
            build_session_config(
                "gemini",
                "beam:cloud",
                "temporary-key",
                environ={},
            )
        with self.assertRaises(ConfigurationError):
            build_session_config(
                "unknown",
                "model",
                "temporary-key",
                environ={},
            )

    def test_missing_credentials_are_rejected_without_printing_values(self) -> None:
        with self.assertRaisesRegex(ConfigurationError, "OLLAMA_API_KEY"):
            build_session_config(
                "ollama",
                "beam:cloud",
                environ={
                    "LLM_PROVIDER": "openai",
                    "OPENAI_API_KEY": "existing-secret",
                },
            )

    def test_status_reports_metadata_only(self) -> None:
        status = get_settings_status(
            {
                "LLM_PROVIDER": "gemini",
                "LLM_MODEL": "gemini-2.5-flash",
                "GEMINI_API_KEY": "never-display-this",
            }
        )
        self.assertEqual(status.provider, "gemini")
        self.assertEqual(status.model, "gemini-2.5-flash")
        self.assertTrue(status.credentials_available)
        self.assertTrue(status.provider_supported)
        self.assertTrue(status.model_supported)
        self.assertNotIn("never-display-this", repr(status))

        missing = get_settings_status(
            {"LLM_PROVIDER": "openai", "LLM_MODEL": "gpt-unsupported"}
        )
        self.assertFalse(missing.credentials_available)
        self.assertFalse(missing.model_supported)
        ollama_status = get_settings_status(
            {
                "LLM_PROVIDER": "ollama",
                "LLM_MODEL": "nemotron-3-super:cloud",
                "OLLAMA_API_KEY": "configured-key",
            }
        )
        self.assertTrue(ollama_status.model_supported)

    def test_session_config_is_separate_from_conversation_history(self) -> None:
        config = SessionAgentConfig(
            "ollama",
            "beam:cloud",
            6,
            0.2,
            "session-secret",
        )
        state = {
            SESSION_CONFIG_KEY: config,
            "bizbridge_chat_history": [
                {"role": "user", "content": "question"},
            ],
        }
        resolved = get_session_agent_config(state)
        self.assertIsInstance(resolved, AgentConfig)
        self.assertEqual(resolved.api_key, "session-secret")
        self.assertNotIn("session-secret", repr(config))
        self.assertNotIn("session-secret", repr(state["bizbridge_chat_history"]))

    def test_invalid_session_configuration_does_not_fall_back_to_environment(self) -> None:
        with self.assertRaisesRegex(ConfigurationError, "missing credentials"):
            get_session_agent_config(
                {"bizbridge_session_agent_config_error": "missing credentials"}
            )


class SettingsStreamlitTests(unittest.TestCase):
    def _open_settings(self) -> AppTest:
        app = AppTest.from_file(APP_PATH).run()
        app.radio[0].set_value("Settings").run()
        return app

    def test_settings_page_shows_safe_status_and_password_input(self) -> None:
        with patch(
            "ui.settings._environment",
            return_value={
                "LLM_PROVIDER": "ollama",
                "LLM_MODEL": "beam:cloud",
                "OLLAMA_API_KEY": "do-not-display",
            },
        ):
            app = self._open_settings()
        self.assertFalse(app.exception)
        rendered = " ".join(
            [item.value for item in app.markdown]
            + [item.label for item in app.selectbox]
        )
        self.assertNotIn("do-not-display", rendered)
        self.assertTrue(any("Available" in item.value for item in app.markdown))
        self.assertIn(
            "non-generative connection check",
            " ".join(item.value for item in app.info),
        )
        self.assertEqual(
            app.selectbox[0].options,
            ["Ollama Cloud", "Gemini", "OpenAI"],
        )
        self.assertFalse(any(_is_password_input(item) for item in app.text_input))

    def test_session_settings_apply_and_secret_is_not_left_in_chat_or_widget_state(self) -> None:
        with patch("ui.settings._environment", return_value={}):
            app = self._open_settings()
            app.selectbox[0].select("Ollama Cloud").run()
            app.selectbox[1].select("beam:cloud").run()
            password_inputs = [
                item for item in app.text_input if _is_password_input(item)
            ]
            self.assertEqual(len(password_inputs), 1)
            password_inputs[0].set_value("temporary-secret").run()
            app.button[0].click().run()

        selected = app.session_state[SESSION_CONFIG_KEY]
        self.assertEqual(selected.provider, "ollama")
        self.assertEqual(selected.model, "beam:cloud")
        self.assertEqual(selected.api_key, "temporary-secret")
        self.assertNotIn(
            "temporary-secret",
            repr(app.session_state.get("bizbridge_chat_history", [])),
        )
        self.assertNotEqual(
            app.session_state.get("bizbridge_settings_api_key"),
            "temporary-secret",
        )
        self.assertFalse(any("temporary-secret" in item.value for item in app.markdown))
        self.assertTrue(any("Using Ollama Cloud / beam:cloud" in item.value for item in app.success))

    def test_settings_shows_missing_credentials_and_does_not_apply(self) -> None:
        with patch("ui.settings._environment", return_value={}):
            app = self._open_settings()
            app.selectbox[0].select("Ollama Cloud").run()
            app.selectbox[1].select("beam:cloud").run()
            self.assertTrue(any(_is_password_input(item) for item in app.text_input))
            app.button[0].click().run()

        self.assertFalse(app.session_state.get(SESSION_CONFIG_KEY))
        self.assertTrue(
            any("Credentials for Ollama Cloud are not available" in item.value for item in app.error)
        )

    def test_failed_provider_change_clears_active_session_without_fallback(self) -> None:
        from ui.settings import _SESSION_CONFIG_ERROR_KEY

        with patch("ui.settings._environment", return_value={}):
            app = self._open_settings()
            app.session_state[SESSION_CONFIG_KEY] = SessionAgentConfig(
                "ollama", "beam:cloud", 6, 0.2, "previous-key"
            )
            app.selectbox[0].select("Gemini").run()
            app.selectbox[1].select("gemini-2.5-flash").run()
            app.button[0].click().run()

        self.assertFalse(app.exception)
        self.assertNotIn(SESSION_CONFIG_KEY, app.session_state)
        self.assertTrue(app.session_state[_SESSION_CONFIG_ERROR_KEY])
        self.assertTrue(
            any("Credentials for Gemini are not available" in item.value for item in app.error)
        )

    def test_existing_cli_path_still_uses_environment_configuration(self) -> None:
        from agent.cli import _run
        from agent.orchestrator import AgentResult

        config = AgentConfig("ollama", "beam:cloud", api_key="cli-secret")
        with (
            patch("agent.cli.AgentConfig.from_env", return_value=config) as from_env,
            patch("agent.cli.create_provider", return_value="cli-provider"),
            patch("agent.cli.Agent") as agent_class,
        ):
            agent = agent_class.return_value
            agent.run = AsyncMock(return_value=AgentResult("CLI answer.", 1, 0))
            answer = asyncio.run(_run("A CLI question"))

        self.assertEqual(answer, "CLI answer.")
        from_env.assert_called_once_with()
        agent_class.assert_called_once()

    def test_assistant_uses_session_provider_configuration(self) -> None:
        from ui.assistant import _run_agent_question
        from agent.orchestrator import AgentResult

        session_config = AgentConfig(
            "gemini",
            "gemini-2.5-flash",
            api_key="temporary-secret",
        )
        with (
            patch("ui.assistant.get_session_agent_config", return_value=session_config),
            patch("ui.assistant.AgentConfig.from_env") as from_env,
            patch("ui.assistant.create_provider", return_value="selected-provider") as factory,
            patch("ui.assistant.Agent") as agent_class,
        ):
            agent = agent_class.return_value
            agent.run = AsyncMock(return_value=AgentResult("Done.", 1, 0))
            _run_agent_question("Question?")

        from_env.assert_not_called()
        factory.assert_called_once_with(session_config)
        agent_class.assert_called_once()


if __name__ == "__main__":
    unittest.main()
