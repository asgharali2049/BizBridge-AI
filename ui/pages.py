"""Placeholder pages for the initial Streamlit application."""

import streamlit as st

from ui.assistant import render_assistant_page as render_chat_assistant
from ui.dashboard import render_dashboard_page as render_analytics_dashboard
from ui.settings import render_settings_page as render_provider_settings


def render_dashboard_page() -> None:
    """Render the read-only analytics dashboard."""
    render_analytics_dashboard()


def render_assistant_page() -> None:
    """Render the session-based chat interface."""
    render_chat_assistant()


def render_settings_page() -> None:
    """Render session-only provider and model configuration."""
    render_provider_settings()
