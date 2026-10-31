"""Application navigation and shared page layout."""

from collections.abc import Callable

import streamlit as st

from ui.pages import render_assistant_page, render_dashboard_page, render_settings_page
from ui.styles import inject_styles, render_brand


_PAGES: dict[str, Callable[[], None]] = {
    "Dashboard": render_dashboard_page,
    "AI Assistant": render_assistant_page,
    "Settings": render_settings_page,
}


def render_app() -> None:
    """Render the shared design system, navigation, and selected page."""
    inject_styles()

    with st.sidebar:
        render_brand()
        selected_page = st.radio(
            "Navigation",
            options=tuple(_PAGES),
            label_visibility="collapsed",
            key="bizbridge_navigation",
        )

    _PAGES[selected_page]()
