from pathlib import Path
import time

import streamlit as st

from chains.chatbot import ask_rag_stream, load_chatbot_components
from memory.conversation_memory import (
    clear_conversation_memory,
    get_conversation_history,
    init_memory,
    rebuild_conversation_memory,
)
from storage.browser_identity import get_browser_id
from storage.chat_history_store import (
    clean_title,
    get_now_text,
    init_history_db,
    load_messages,
    replace_conversation_messages,
)

try:
    from storage.chat_history_store import delete_conversation
except Exception:
    delete_conversation = None
from ui.history_ui import display_assistant_bubble, display_chat_history, display_user_bubble
from ui.sidebar import display_sidebar
from ui.source_ui import normalize_sources_for_state
from ui.styles import load_css

try:
    from config.settings import NO_ANSWER_TEXT
except Exception:
    NO_ANSWER_TEXT = "I cannot find the answer in the provided documents."

PAGE_TITLE = "InknowVa"
PAGE_ICON_PATH = Path("assets/iknowva_icon.png")
PAGE_ICON_FALLBACK = "🤖"
CHAT_INPUT_PLACEHOLDER = "Message InKnowVa"
MEMORY_LIMIT = None
DEBUG_RAG = True
FRONTEND_FLUSH_SECONDS = 0.12
ACTIVE_CONVERSATION_KEY = "active_conversation_id"
PENDING_SIDEBAR_ACTION_KEY = "pending_sidebar_action"
UI_FLOW_DEBUG_FILE = Path(__file__).resolve().parents[1] / "reports" / "ui_rag_debug.txt"


def append_ui_flow_debug(title, lines=None):
    # Lightweight UI-level debug log.
    try:
        UI_FLOW_DEBUG_FILE.parent.mkdir(parents=True, exist_ok=True)
        log_lines = [
            "",
            "=" * 80,
            str(title or "UI DEBUG"),
            "=" * 80,
            f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        ]

        for line in lines or []:
            log_lines.append(str(line))

        with UI_FLOW_DEBUG_FILE.open("a", encoding="utf-8") as file:
            file.write("\n".join(log_lines))
            file.write("\n")

        print(f"UI DEBUG: {title} -> {UI_FLOW_DEBUG_FILE.resolve()}", flush=True)
    except Exception as error:
        print(f"UI DEBUG WRITE FAILED: {error}", flush=True)


print(f"ACTIVE CHAT_UI FILE: {Path(__file__).resolve()}", flush=True)
print(f"UI DEBUG FILE TARGET: {UI_FLOW_DEBUG_FILE.resolve()}", flush=True)


def is_no_answer(answer):
    # Check if the backend returned the configured fallback answer.
    # Use a contains check to stay synced with chains.chatbot.is_no_answer().
    answer_text = str(answer or "").strip().lower()
    no_answer_text = str(NO_ANSWER_TEXT).strip().lower()

    if not answer_text or not no_answer_text:
        return False

    return no_answer_text in answer_text


def get_page_icon():
    # Optional tab icon.
    if not PAGE_ICON_PATH.exists():
        return PAGE_ICON_FALLBACK

    try:
        from PIL import Image
        return Image.open(PAGE_ICON_PATH)
    except Exception:
        return PAGE_ICON_FALLBACK


@st.cache_resource(show_spinner="Starting InknowVa...")
def load_components():
    # Load heavy RAG components once per Streamlit server process.
    # This is the only function that is allowed to call load_chatbot_components().
    return load_chatbot_components()


def get_loaded_components():
    # Keep already-loaded components in the current browser session too.
    # After the first load, clicks and submits reuse the same objects.
    if "components" not in st.session_state:
        st.session_state.components = load_components()

    return st.session_state.components


def initialize_session_state():
    # Initialize all state keys used by the UI.
    defaults = {
        "messages": [],
        "is_generating": False,
        "latest_sources": [],
        "latest_sources_clean": [],
        "pending_question": None,
        "pending_conversation_history": "",
        PENDING_SIDEBAR_ACTION_KEY: None,
        ACTIVE_CONVERSATION_KEY: None,
    }

    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def clear_answer_ui_state():
    # Clear UI state that belongs only to the previous assistant answer.
    # This prevents stale action buttons during streaming.
    for key in [
        "last_answer",
        "last_question",
        "last_sources",
        "open_export_panel_key",
    ]:
        st.session_state.pop(key, None)



def begin_answer_generation():
    # Hide answer-only controls while a new answer is being generated.
    st.session_state.is_generating = True
    st.session_state.hide_actions = True
    clear_answer_ui_state()


def finish_answer_generation():
    # Restore answer controls after generation completes.
    st.session_state.is_generating = False
    st.session_state.pop("hide_actions", None)


def should_hide_answer_controls():
    # True while an answer is being generated.
    return (
        st.session_state.get("hide_actions", False)
        or st.session_state.get("is_generating", False)
    )


def inject_answer_control_hider():
    # Add a small marker only.
    # The actual styling lives in main.css.
    if not should_hide_answer_controls():
        return

    st.markdown(
        '<div class="hide-answer-controls-active"></div>',
        unsafe_allow_html=True,
    )


def is_empty_chat_state():
    # True only on the landing page before the user sends the first message.
    return (
        not st.session_state.get("messages")
        and not st.session_state.get("is_generating", False)
    )


def inject_empty_state_chat_input_position():
    # Add a small marker only.
    # The actual landing input position is handled in main.css.
    if not is_empty_chat_state():
        return

    st.markdown(
        '<div class="landing-chat-input-active"></div>',
        unsafe_allow_html=True,
    )




def set_latest_sources(sources):
    # Save latest sources for optional sidebar/future use.
    sources = normalize_sources_for_state(sources or [])
    st.session_state.latest_sources = sources
    st.session_state.latest_sources_clean = sources
    st.session_state.sidebar_sources = sources


def get_last_assistant_sources(messages):
    # Get sources from latest assistant answer.
    for message in reversed(messages or []):
        if isinstance(message, dict) and message.get("role") == "assistant":
            return normalize_sources_for_state(message.get("sources", []))

    return []


def sanitize_existing_source_state():
    # Clean source state loaded from older saved sessions.
    for key in ["latest_sources", "latest_sources_clean", "last_sources"]:
        if key in st.session_state:
            st.session_state[key] = normalize_sources_for_state(st.session_state.get(key, []))

    for message in st.session_state.get("messages", []):
        if isinstance(message, dict) and message.get("role") == "assistant":
            message["sources"] = normalize_sources_for_state(message.get("sources", []))



def reset_chat_state(clear_active_conversation=True):
    # Clear visible chat and temporary UI state.
    clear_conversation_memory(st)

    for key in [
        "hide_actions",
        "open_export_panel_key",
        "last_sources",
        "last_answer",
        "last_question",
        "sidebar_sources",
        "pending_question",
        "pending_conversation_history",
        PENDING_SIDEBAR_ACTION_KEY,
        "selected_conversation_id",
    ]:
        st.session_state.pop(key, None)

    st.session_state.messages = []
    st.session_state.latest_sources = []
    st.session_state.latest_sources_clean = []
    st.session_state.is_generating = False

    if clear_active_conversation:
        st.session_state[ACTIVE_CONVERSATION_KEY] = None


# Public names exported by this compatibility/refactor module.
__all__ = [
    'Path',
    'time',
    'st',
    'ask_rag_stream',
    'load_chatbot_components',
    'clear_conversation_memory',
    'get_conversation_history',
    'init_memory',
    'rebuild_conversation_memory',
    'get_browser_id',
    'clean_title',
    'get_now_text',
    'init_history_db',
    'load_messages',
    'replace_conversation_messages',
    'delete_conversation',
    'display_assistant_bubble',
    'display_chat_history',
    'display_user_bubble',
    'display_sidebar',
    'normalize_sources_for_state',
    'load_css',
    'NO_ANSWER_TEXT',
    'PAGE_TITLE',
    'PAGE_ICON_PATH',
    'PAGE_ICON_FALLBACK',
    'CHAT_INPUT_PLACEHOLDER',
    'MEMORY_LIMIT',
    'DEBUG_RAG',
    'FRONTEND_FLUSH_SECONDS',
    'ACTIVE_CONVERSATION_KEY',
    'PENDING_SIDEBAR_ACTION_KEY',
    'UI_FLOW_DEBUG_FILE',
    'append_ui_flow_debug',
    'is_no_answer',
    'get_page_icon',
    'load_components',
    'get_loaded_components',
    'initialize_session_state',
    'clear_answer_ui_state',
    'begin_answer_generation',
    'finish_answer_generation',
    'should_hide_answer_controls',
    'inject_answer_control_hider',
    'is_empty_chat_state',
    'inject_empty_state_chat_input_position',
    'set_latest_sources',
    'get_last_assistant_sources',
    'sanitize_existing_source_state',
    'reset_chat_state',
]
