"""Streamlit entry point for the BizBridge AI application."""

import streamlit as st

from ui.app import render_app


st.set_page_config(
    page_title="BizBridge AI",
    page_icon="B",
    layout="wide",
    initial_sidebar_state="expanded",
)

render_app()
