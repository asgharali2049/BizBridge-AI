from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from streamlit.testing.v1 import AppTest

from agent.config import AgentConfig, ConfigurationError
from agent.errors import (
    AgentIterationLimitError,
    ProviderProtocolError,
    ProviderRequestError,
    UnsupportedModelError,
)
from agent.orchestrator import AgentResult, ToolActivity
from ui.assistant import CHAT_HISTORY_KEY, _friendly_error, _run_agent_question


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


class AssistantAgentIntegrationTests(unittest.TestCase):
    def test_uses_public_agent_entry_point_and_existing_configuration(self) -> None:
        config = AgentConfig(
            provider="ollama",
            model="beam:cloud",
            api_key="test-key",
            max_iterations=4,
            temperature=0.4,
        )
        expected = AgentResult("Two orders.", 2, 1)
        with (
            patch("ui.assistant.AgentConfig.from_env", return_value=config),
            patch("ui.assistant.create_provider", return_value="provider") as factory,
            patch("ui.assistant.Agent") as agent_class,
        ):
            agent = agent_class.return_value
            agent.run = AsyncMock(return_value=expected)

            result = _run_agent_question("How many completed orders?")

        self.assertIs(result, expected)
        factory.assert_called_once_with(config)
        agent_class.assert_called_once()
        args = agent_class.call_args.kwargs
        self.assertEqual(args["provider"], "provider")
        self.assertEqual(args["max_iterations"], 4)
        self.assertEqual(args["temperature"], 0.4)
        agent.run.assert_awaited_once_with(
            "How many completed orders?",
            conversation_history=[],
        )


class AssistantErrorHandlingTests(unittest.TestCase):
    def test_provider_and_agent_errors_are_safe_and_actionable(self) -> None:
        cases = (
            (
                ConfigurationError("OLLAMA_API_KEY is required."),
                "matching provider API key",
            ),
            (
                UnsupportedModelError("model unsupported"),
                "not supported for tool calling",
            ),
            (
                ProviderRequestError("secret details"),
                "provider could not complete",
            ),
            (
                AgentIterationLimitError("iteration limit"),
                "tool-call limit",
            ),
            (
                ProviderProtocolError("empty response"),
                "empty or unusable response",
            ),
            (
                RuntimeError("Unable to connect to the BizBridge MCP server."),
                "analytics service could not be reached",
            ),
        )
        for error, expected in cases:
            with self.subTest(error=type(error).__name__):
                message = _friendly_error(error)
                self.assertIn(expected, message)
                self.assertNotIn("secret details", message)
                self.assertNotIn("OLLAMA_API_KEY", message)

    def test_invalid_session_configuration_is_not_replaced_by_environment_config(self) -> None:
        with (
            patch(
                "ui.assistant.get_session_agent_config",
                side_effect=ConfigurationError("missing credentials"),
            ),
            patch("ui.assistant.AgentConfig.from_env") as from_env,
        ):
            with self.assertRaises(ConfigurationError):
                _run_agent_question("Question?")
        from_env.assert_not_called()


class AssistantStreamlitTests(unittest.TestCase):
    def _open_assistant(self) -> AppTest:
        app = AppTest.from_file(APP_PATH).run()
        app.radio[0].set_value("AI Assistant").run()
        return app

    def test_chat_history_activity_and_rerun_persistence(self) -> None:
        results = [
            AgentResult(
                "Revenue was PKR 500.",
                2,
                1,
                (
                    ToolActivity(
                        "sales_summary",
                        {"start_date": "2025-01-01", "end_date": "2025-01-31"},
                        '{"revenue_pkr":500}',
                        False,
                        True,
                    ),
                ),
            ),
            AgentResult("There were 4 units.", 1, 0),
        ]
        app = self._open_assistant()
        with patch(
            "ui.assistant._run_agent_question", side_effect=results
        ) as run:
            app.chat_input[0].set_value("Summarize January sales.").run()
            self.assertEqual(run.call_count, 1)
            self.assertEqual(len(app.session_state[CHAT_HISTORY_KEY]), 2)
            self.assertEqual(
                app.session_state[CHAT_HISTORY_KEY][1]["tool_activities"][0]["name"],
                "sales_summary",
            )
            self.assertTrue(any("Tool activity" in item.label for item in app.expander))
            self.assertTrue(any("Revenue was PKR 500." in item.value for item in app.markdown))

            app.run()
            self.assertEqual(run.call_count, 1)
            self.assertEqual(len(app.session_state[CHAT_HISTORY_KEY]), 2)

            app.chat_input[0].set_value("How many units?").run()
            self.assertEqual(run.call_count, 2)
            self.assertEqual(len(app.session_state[CHAT_HISTORY_KEY]), 4)
            self.assertEqual(
                run.call_args.args,
                (
                    "How many units?",
                    app.session_state[CHAT_HISTORY_KEY][:2],
                ),
            )
            self.assertEqual(
                app.session_state[CHAT_HISTORY_KEY][-1]["content"],
                "There were 4 units.",
            )

    def test_new_conversation_clears_the_session_history(self) -> None:
        app = self._open_assistant()
        with patch(
            "ui.assistant._run_agent_question",
            return_value=AgentResult("Done.", 1, 0),
        ):
            app.chat_input[0].set_value("Say hello.").run()
            self.assertEqual(len(app.session_state[CHAT_HISTORY_KEY]), 2)
            app.button[0].click().run()
        self.assertEqual(app.session_state[CHAT_HISTORY_KEY], [])

    def test_empty_agent_result_is_shown_as_safe_error(self) -> None:
        app = self._open_assistant()
        with patch(
            "ui.assistant._run_agent_question",
            return_value=AgentResult("  ", 1, 0),
        ):
            app.chat_input[0].set_value("Summarize sales.").run()
        history = app.session_state[CHAT_HISTORY_KEY]
        self.assertTrue(history[-1]["error"])
        self.assertIn("empty or unusable response", history[-1]["content"])

    def test_provider_mcp_and_iteration_errors_are_displayed_without_details(self) -> None:
        cases = (
            (ProviderRequestError("secret provider response"), "AI provider could not"),
            (RuntimeError("MCP internal path"), "analytics service could not"),
            (AgentIterationLimitError("private details"), "tool-call limit"),
        )
        for error, expected_text in cases:
            with self.subTest(error=type(error).__name__):
                app = self._open_assistant()
                with patch("ui.assistant._run_agent_question", side_effect=error):
                    app.chat_input[0].set_value("Summarize sales.").run()
                history = app.session_state[CHAT_HISTORY_KEY]
                self.assertIn(expected_text, history[-1]["content"])
                self.assertNotIn("private details", history[-1]["content"])
                self.assertNotIn("secret provider response", history[-1]["content"])

    def test_busy_guard_prevents_duplicate_agent_submission(self) -> None:
        app = self._open_assistant()
        app.session_state["bizbridge_chat_busy"] = True
        app.run()
        with patch("ui.assistant._run_agent_question") as run:
            self.assertEqual(app.session_state[CHAT_HISTORY_KEY], [])
            run.assert_not_called()

    def test_dashboard_does_not_initialize_agent_or_provider(self) -> None:
        with (
            patch("ui.assistant.Agent") as agent,
            patch("ui.assistant.create_provider") as provider,
        ):
            app = AppTest.from_file(APP_PATH).run()
        self.assertFalse(app.exception)
        agent.assert_not_called()
        provider.assert_not_called()


if __name__ == "__main__":
    unittest.main()
