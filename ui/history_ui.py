"""Render saved chat messages."""

import html
from datetime import datetime

import streamlit as st

from ui.action_ui import display_actions
from ui.source_ui import build_inline_source_chips


def safe_html_text(text):
    # Escape plain chat text before HTML render.
    return html.escape(str(text or "")).replace("\n", "<br>")


def format_message_timestamp(value):
    # Convert DB timestamp into compact time.
    if not value:
        return ""

    text = str(value).strip()

    for fmt in [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S.%f",
    ]:
        try:
            return datetime.strptime(text[:26], fmt).strftime("%I:%M %p").lstrip("0")
        except ValueError:
            continue

    return text


def render_timestamp(timestamp, css_class):
    # Timestamp HTML below bubble.
    timestamp = format_message_timestamp(timestamp)

    if not timestamp:
        return ""

    return f'<div class="message-timestamp {css_class}">{html.escape(timestamp)}</div>'


def display_user_bubble(message, timestamp=None):
    # User bubble.
    st.markdown(
        "\n".join([
            '<div class="user-message-wrapper">',
            f'<div class="user-message-bubble">{safe_html_text(message)}</div>',
            render_timestamp(timestamp, "user-message-timestamp"),
            '</div>',
        ]),
        unsafe_allow_html=True,
    )


def display_assistant_bubble(message, timestamp=None, sources=None):
    # Assistant bubble with compact source chips at the end.
    source_chips = build_inline_source_chips(sources or [])

    st.markdown(
        "\n".join([
            '<div class="assistant-message-wrapper">',
            f'<div class="assistant-message-bubble">{safe_html_text(message)} {source_chips}</div>',
            render_timestamp(timestamp, "assistant-message-timestamp"),
            '</div>',
        ]),
        unsafe_allow_html=True,
    )

def display_chat_history(messages):
    # Render all visible messages and assistant action rows.
    hide_actions = st.session_state.get("hide_actions", False)

    for index, message in enumerate(messages or []):
        role = message.get("role")
        content = message.get("content", "")
        timestamp = message.get("created_at")

        if role == "user":
            display_user_bubble(content, timestamp=timestamp)
            continue

        if role != "assistant":
            continue

        display_assistant_bubble(content, timestamp=timestamp, sources=message.get("sources", []))

        if not hide_actions:
            display_actions(
                answer=content,
                sources=message.get("sources", []),
                message_index=index,
            )
