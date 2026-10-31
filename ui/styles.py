"""Shared visual system and small presentation helpers for the Streamlit UI."""

from __future__ import annotations

from html import escape
from typing import Any

import streamlit as st


def inject_styles() -> None:
    """Apply the shared BizBridge dark SaaS theme."""
    st.markdown(
        """
        <style>
        :root {
            color-scheme: dark;
            --primary-color: #45d6c5;
            --bb-bg: #0a101b;
            --bb-sidebar: #0d1522;
            --bb-panel: #111c2c;
            --bb-panel-raised: #142235;
            --bb-border: #24354a;
            --bb-text: #e7eef7;
            --bb-muted: #93a4b8;
            --bb-accent: #45d6c5;
            --bb-accent-soft: rgba(69, 214, 197, .12);
            --bb-success: #55d6a5;
            --bb-warning: #f3bb64;
            --bb-danger: #fa7c83;
        }

        [data-testid="stAppViewContainer"] {
            background:
                radial-gradient(ellipse at 82% 0%, rgba(35, 83, 106, .12), transparent 34rem),
                var(--bb-bg);
            color: var(--bb-text);
        }
        [data-testid="stHeader"] {
            background: rgba(10, 16, 27, .88);
        }
        [data-testid="stSidebar"] {
            background: var(--bb-sidebar);
            border-right: 1px solid var(--bb-border);
        }
        [data-testid="stSidebar"] > div:first-child {
            padding-top: 1.4rem;
        }
        [data-testid="stAppViewContainer"] .main .block-container {
            max-width: 1440px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }
        [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
            gap: 1rem;
        }
        [data-testid="stSidebar"] [role="radiogroup"] {
            gap: .45rem;
        }
        [data-testid="stSidebar"] input[type="radio"] {
            accent-color: var(--bb-accent);
        }
        [data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {
            border-color: rgba(69, 214, 197, .32);
            background: var(--bb-accent-soft);
        }
        [data-testid="stSidebar"] [role="radiogroup"] label {
            border: 1px solid transparent;
            border-radius: .7rem;
            padding: .62rem .8rem;
            transition: background-color .18s ease, border-color .18s ease;
        }
        [data-testid="stSidebar"] [role="radiogroup"] label:hover {
            background: rgba(255, 255, 255, .045);
            border-color: var(--bb-border);
        }
        [data-testid="stSidebar"] [role="radio"][aria-checked="true"] + div {
            color: var(--bb-accent);
        }
        h1, h2, h3, h4 {
            color: var(--bb-text);
            letter-spacing: -.025em;
        }
        [data-testid="stMarkdownContainer"] p {
            color: var(--bb-text);
        }
        [data-testid="stCaptionContainer"] {
            color: var(--bb-muted);
        }
        [data-testid="stDivider"] {
            border-color: var(--bb-border);
        }
        [data-testid="stMetric"],
        [data-testid="stChatMessage"],
        [data-testid="stDataFrame"],
        [data-testid="stPlotlyChart"],
        [data-testid="stExpander"] {
            border: 1px solid var(--bb-border);
            border-radius: .9rem;
            background: rgba(17, 28, 44, .72);
        }
        [data-testid="stChatMessage"] {
            padding: 1rem 1.15rem;
            margin-bottom: .7rem;
        }
        [data-testid="stPlotlyChart"] {
            overflow: hidden;
            padding: .35rem;
        }
        [data-testid="stDataFrame"] {
            overflow: hidden;
        }
        [data-testid="stSelectbox"] [data-baseweb="select"] > div,
        [data-testid="stDateInput"] [data-baseweb="input"] > div,
        [data-testid="stTextInput"] [data-baseweb="input"] > div,
        [data-testid="stTextArea"] [data-baseweb="textarea"] {
            background: var(--bb-panel) !important;
            border-color: var(--bb-border) !important;
            border-radius: .65rem;
        }
        [data-testid="stDateInput"] input,
        [data-testid="stTextInput"] input,
        [data-testid="stSelectbox"] [role="combobox"] {
            background: var(--bb-panel) !important;
            color: var(--bb-text) !important;
        }
        [data-testid="stDateInput"] div[data-baseweb="input"],
        [data-testid="stTextInput"] div[data-baseweb="input"] {
            border-color: var(--bb-border) !important;
            background: var(--bb-panel) !important;
        }
        [data-testid="stButton"] > button {
            border: 1px solid var(--bb-border);
            border-radius: .65rem;
            background: var(--bb-panel-raised);
            color: var(--bb-text);
            transition: transform .16s ease, border-color .16s ease, background-color .16s ease;
        }
        [data-testid="stButton"] > button:hover {
            border-color: var(--bb-accent);
            background: #192b3f;
            color: var(--bb-text);
            transform: translateY(-1px);
        }
        [data-testid="stChatInput"] {
            border-color: var(--bb-border);
            border-radius: .9rem;
            background: var(--bb-panel);
        }
        [data-testid="stAlert"] {
            border-radius: .8rem;
        }
        .bb-brand {
            display: flex;
            align-items: center;
            gap: .72rem;
            padding: .35rem .2rem .9rem;
        }
        .bb-brand-mark {
            display: grid;
            width: 2.45rem;
            height: 2.45rem;
            place-items: center;
            border: 1px solid rgba(69, 214, 197, .35);
            border-radius: .78rem;
            background: var(--bb-accent-soft);
            color: var(--bb-accent);
            font-size: 1rem;
            font-weight: 800;
        }
        .bb-brand-name {
            color: var(--bb-text);
            font-size: 1rem;
            font-weight: 700;
            letter-spacing: -.02em;
        }
        .bb-brand-subtitle {
            margin-top: .13rem;
            color: var(--bb-muted);
            font-size: .73rem;
        }
        .bb-page-heading {
            margin: 0 0 1.35rem;
        }
        .bb-eyebrow {
            margin-bottom: .35rem;
            color: var(--bb-accent);
            font-size: .7rem;
            font-weight: 750;
            letter-spacing: .13em;
            text-transform: uppercase;
        }
        .bb-page-title {
            margin: 0;
            color: var(--bb-text);
            font-size: clamp(1.7rem, 3vw, 2.25rem);
            font-weight: 720;
            letter-spacing: -.045em;
            line-height: 1.15;
        }
        .bb-page-subtitle {
            max-width: 48rem;
            margin: .55rem 0 0;
            color: var(--bb-muted);
            font-size: .96rem;
            line-height: 1.55;
        }
        .bb-kpi-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: .85rem;
            margin: .35rem 0 1.9rem;
        }
        .bb-kpi {
            min-height: 7rem;
            padding: 1rem 1.1rem;
            border: 1px solid var(--bb-border);
            border-radius: .9rem;
            background: linear-gradient(145deg, rgba(20, 34, 53, .95), rgba(15, 25, 40, .9));
            box-shadow: 0 10px 24px rgba(0, 0, 0, .13);
            transition: border-color .18s ease, transform .18s ease;
        }
        .bb-kpi:hover {
            border-color: rgba(69, 214, 197, .48);
            transform: translateY(-2px);
        }
        .bb-kpi-label {
            color: var(--bb-muted);
            font-size: .78rem;
            font-weight: 600;
        }
        .bb-kpi-value {
            margin-top: .65rem;
            color: var(--bb-text);
            font-size: clamp(1.35rem, 2vw, 1.8rem);
            font-weight: 730;
            letter-spacing: -.045em;
            line-height: 1.2;
            overflow-wrap: anywhere;
        }
        .bb-section-label {
            margin: 1.4rem 0 .45rem;
            color: var(--bb-text);
            font-size: 1rem;
            font-weight: 680;
        }
        .bb-status {
            display: inline-flex;
            align-items: center;
            gap: .42rem;
            padding: .25rem .55rem;
            border: 1px solid currentColor;
            border-radius: 999px;
            font-size: .72rem;
            font-weight: 680;
            line-height: 1.2;
        }
        .bb-status-success { color: var(--bb-success); background: rgba(85, 214, 165, .08); }
        .bb-status-warning { color: var(--bb-warning); background: rgba(243, 187, 100, .08); }
        .bb-status-danger { color: var(--bb-danger); background: rgba(250, 124, 131, .08); }
        .bb-empty-state {
            padding: 1rem 1.1rem;
            border: 1px dashed var(--bb-border);
            border-radius: .8rem;
            color: var(--bb-muted);
            background: rgba(17, 28, 44, .4);
        }
        @media (max-width: 1150px) {
            .bb-kpi-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
        }
        @media (max-width: 520px) {
            [data-testid="stAppViewContainer"] .main .block-container {
                padding-left: 1rem;
                padding-right: 1rem;
            }
            .bb-kpi-grid { gap: .55rem; }
            .bb-kpi { min-height: 6rem; padding: .85rem; }
            .bb-kpi-value { font-size: 1.25rem; }
        }
        @media (prefers-reduced-motion: reduce) {
            *, *::before, *::after {
                scroll-behavior: auto !important;
                animation-duration: .01ms !important;
                animation-iteration-count: 1 !important;
                transition-duration: .01ms !important;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_brand() -> None:
    """Render the static sidebar brand lockup."""
    st.markdown(
        """
        <div class="bb-brand">
            <div class="bb-brand-mark" aria-hidden="true">B</div>
            <div>
                <div class="bb-brand-name">BizBridge AI</div>
                <div class="bb-brand-subtitle">Business intelligence</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_page_heading(title: str, subtitle: str, eyebrow: str = "WORKSPACE") -> None:
    """Render a consistent title block for a page."""
    st.markdown(
        (
            '<div class="bb-page-heading">'
            f'<div class="bb-eyebrow">{escape(eyebrow)}</div>'
            f'<h1 class="bb-page-title">{escape(title)}</h1>'
            f'<p class="bb-page-subtitle">{escape(subtitle)}</p>'
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def render_kpi_cards(items: list[tuple[str, Any]]) -> None:
    """Render real, preformatted KPI values in a responsive card grid."""
    cards = []
    for label, value in items:
        cards.append(
            '<div class="bb-kpi">'
            f'<div class="bb-kpi-label">{escape(str(label))}</div>'
            f'<div class="bb-kpi-value">{escape(str(value))}</div>'
            "</div>"
        )
    st.markdown(
        f'<div class="bb-kpi-grid">{"".join(cards)}</div>',
        unsafe_allow_html=True,
    )


def render_empty_state(message: str) -> None:
    """Render a quiet, consistent empty-state panel."""
    st.markdown(
        f'<div class="bb-empty-state">{escape(message)}</div>',
        unsafe_allow_html=True,
    )
