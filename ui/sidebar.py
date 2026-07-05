import base64
import html
from pathlib import Path

import streamlit as st

from retrieval.context_config import read_query_config
from storage.chat_history_store import list_conversations
from ui.source_ui import (
    clean_preview_text,
    deduplicate_sources,
    make_static_file_link,
    normalize_source_item,
)

MAX_HISTORY_ITEMS = 25
MAX_SIDEBAR_SOURCES = 5
SIDEBAR_SOURCE_PREVIEW_LIMIT = 100
PENDING_SIDEBAR_ACTION_KEY = "pending_sidebar_action"

# Native Streamlit scroll containers.
# These are safer than CSS-only scroll hacks because Streamlit controls the height.
# Balanced heights: bigger Sources area while keeping the bottom session block visible.
HISTORY_LIST_HEIGHT = 360
SOURCES_LIST_HEIGHT = 0

LOGO_PATHS = [
    Path("assets/iknowva_icon.png"),
    Path("assets/iknowva_icon(1).png"),
    Path("iknowva_icon.png"),
]


def queue_sidebar_action(action_type, conversation_id=None):
    # Store the sidebar click before the next script body runs.
    action = {"type": action_type}

    if conversation_id is not None:
        action["conversation_id"] = conversation_id

    st.session_state[PENDING_SIDEBAR_ACTION_KEY] = action


def make_scroll_container(key, height):
    # Use Streamlit native height-based scroll containers when available.
    try:
        return st.container(key=key, height=height, border=False)
    except TypeError:
        try:
            return st.container(key=key, height=height)
        except TypeError:
            return st.container(key=key)


def get_logo_data_uri():
    # Optional logo as browser-safe data URI.
    for logo_path in LOGO_PATHS:
        if not logo_path.exists() or not logo_path.is_file():
            continue

        mime_type = "image/png"
        suffix = logo_path.suffix.lower()

        if suffix in [".jpg", ".jpeg"]:
            mime_type = "image/jpeg"
        elif suffix == ".webp":
            mime_type = "image/webp"
        elif suffix == ".svg":
            mime_type = "image/svg+xml"

        encoded = base64.b64encode(logo_path.read_bytes()).decode("utf-8")
        return f"data:{mime_type};base64,{encoded}"

    return ""


def truncate_title(title, limit=42):
    # Keep titles compact.
    title = " ".join(str(title or "New chat").split())

    if len(title) > limit:
        return title[:limit].rstrip() + "..."

    return title


def safe_html(value):
    # Escape user/source text before putting it in custom HTML.
    return html.escape(str(value or ""), quote=True)


def display_sidebar_banner():
    # Top brand block. CSS is in main.css.
    logo_uri = get_logo_data_uri()
    logo_html = ""

    if logo_uri:
        logo_html = f'<img class="sidebar-brand-logo" src="{logo_uri}" alt="InknowVa logo">'

    st.markdown(
        f"""
<div class="sidebar-banner">
    <div class="sidebar-decorative-line"></div>
    <div class="sidebar-brand-row">
        {logo_html}
        <div class="sidebar-brand-text">
            <div class="sidebar-banner-title">InknowVa</div>
            <div class="sidebar-banner-subtitle">Internal Knowledge Virtual Assistant</div>
        </div>
    </div>
</div>
""",
        unsafe_allow_html=True,
    )


def display_sidebar_actions():
    # Keep the buttons in separate rows so CSS can lock their order.
    # on_click makes the action visible at the very start of the next rerun.
    with st.container(key="sidebar_new_chat_row"):
        st.button(
            "＋ New chat",
            key="sidebar_new_chat",
            on_click=queue_sidebar_action,
            args=("new_chat",),
        )

    with st.container(key="sidebar_clear_chat_row"):
        st.button(
            "⌫ Clear chat",
            key="sidebar_clear_chat",
            on_click=queue_sidebar_action,
            args=("clear_chat",),
        )

    return False, False


def display_fixed_title(title):
    # Fixed title above a native scroll container.
    st.markdown(f'<div class="sidebar-fixed-title">{safe_html(title)}</div>', unsafe_allow_html=True)


def display_chat_history_items(browser_id, active_conversation_id=None):
    # Scrollable saved conversations for current browser only.
    conversations = list_conversations(browser_id=browser_id, limit=MAX_HISTORY_ITEMS)

    if not conversations:
        st.markdown('<div class="sidebar-history-empty">No saved chats yet.</div>', unsafe_allow_html=True)
        return None

    selected_id = None

    for conversation in conversations:
        conversation_id = conversation.get("id")
        title = truncate_title(conversation.get("title"))
        is_active = str(conversation_id) == str(active_conversation_id)
        key = f"sidebar_history_chat{'_active' if is_active else ''}_{conversation_id}"

        st.button(
            title,
            key=key,
            use_container_width=True,
            on_click=queue_sidebar_action,
            args=("open_chat", conversation_id),
        )

    return selected_id


def build_sidebar_source_card(index, source):
    # Small source card shown in the left sidebar.
    item = normalize_source_item(source)

    if not item:
        return ""

    link = make_static_file_link(item.get("source_path"), page=item.get("page")) or ""
    name = safe_html(item.get("source") or "Unknown source")
    page = safe_html(item.get("page") or "N/A")
    preview = safe_html(clean_preview_text(item.get("preview"), limit=SIDEBAR_SOURCE_PREVIEW_LIMIT))

    if link:
        safe_link = safe_html(link)
        title_html = (
            f'<a class="sidebar-source-title" href="{safe_link}" '
            f'target="_blank" rel="noopener noreferrer">{name}</a>'
        )
        preview_html = (
            f'<a class="sidebar-source-preview" href="{safe_link}" '
            f'target="_blank" rel="noopener noreferrer">{preview}</a>'
        )
    else:
        title_html = f'<span class="sidebar-source-title">{name}</span>'
        preview_html = f'<div class="sidebar-source-preview">{preview}</div>'

    return "\n".join([
        '<div class="sidebar-source-card">',
        '<div class="sidebar-source-header">',
        f'<span class="sidebar-source-number">{index}</span>',
        title_html,
        '</div>',
        f'<div class="sidebar-source-page">Page: {page}</div>',
        preview_html,
        '</div>',
    ])


def display_source_items(sources):
    # Scrollable answer sources.
    sources = deduplicate_sources(sources or [], max_sources=MAX_SIDEBAR_SOURCES)

    if not sources:
        st.markdown(
            '<div class="sidebar-source-empty">Sources will appear here after an answer.</div>',
            unsafe_allow_html=True,
        )
        return

    cards = [
        build_sidebar_source_card(index, source)
        for index, source in enumerate(sources, start=1)
    ]
    cards = [card for card in cards if card]

    st.markdown(
        '<div class="sidebar-source-list">' + "\n".join(cards) + "</div>",
        unsafe_allow_html=True,
    )


def get_sidebar_source_fallback_keys():
    config = read_query_config()
    ui_config = config.get("ui", {})
    sidebar_config = ui_config.get("sidebar", {})
    keys = sidebar_config.get("source_fallback_keys", ())

    if isinstance(keys, str):
        keys = keys.split("|")

    if not isinstance(keys, (list, tuple, set)):
        return []

    return [str(key).strip() for key in keys if str(key).strip()]


def get_sidebar_sources(sources=None):
    # Use explicit sources first.
    if sources:
        return sources

    for key in get_sidebar_source_fallback_keys():
        value = st.session_state.get(key)

        if value:
            return value

    return []


def display_sidebar_bottom_status(browser_id=None):
    # Fixed bottom browser-session status.
    browser_tail = str(browser_id or "")[-6:] if browser_id else "local"

    st.markdown(
        f"""
<div class="sidebar-bottom-status">
    <div class="sidebar-decorative-line sidebar-decorative-line-bottom"></div>
    <div class="sidebar-session-status">
        <span class="sidebar-status-dot"></span>
        <span>Browser session</span>
    </div>
    <div class="sidebar-bottom-note">Local history · {browser_tail}</div>
</div>
""",
        unsafe_allow_html=True,
    )


def display_sidebar(browser_id, active_conversation_id=None, sources=None):
    # Render full sidebar and return clicked actions. Sources are shown inline per answer.
    result = {
        "new_chat_clicked": False,
        "clear_chat_clicked": False,
        "selected_conversation_id": None,
    }

    with st.sidebar:
        with st.container(key="sidebar_header_area"):
            display_sidebar_banner()

        with st.container(key="sidebar_actions_area"):
            new_clicked, clear_clicked = display_sidebar_actions()

        with st.container(key="sidebar_scroll_area"):
            display_fixed_title("Recent chats")

            with make_scroll_container("sidebar_history_list", HISTORY_LIST_HEIGHT):
                selected_id = display_chat_history_items(
                    browser_id=browser_id,
                    active_conversation_id=active_conversation_id,
                )

        with st.container(key="sidebar_bottom_area"):
            display_sidebar_bottom_status(browser_id=browser_id)

    result["new_chat_clicked"] = new_clicked
    result["clear_chat_clicked"] = clear_clicked
    result["selected_conversation_id"] = selected_id

    return result
