#Main Streamlit chat UI for InknowVa.

from pathlib import Path

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
    delete_all_conversations,
    get_now_text,
    init_history_db,
    load_messages,
    replace_conversation_messages,
)
from suggestions.suggestion_generator import generate_suggestions
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
MEMORY_LIMIT = 6
DEBUG_RAG = False
SUGGESTION_LIMIT = 3
ACTIVE_CONVERSATION_KEY = "active_conversation_id"


def is_no_answer(answer):
    # Check if the model returned the configured fallback answer.
    return str(answer or "").strip().lower() == str(NO_ANSWER_TEXT).strip().lower()


def get_page_icon():
    # Optional tab icon.
    if not PAGE_ICON_PATH.exists():
        return PAGE_ICON_FALLBACK

    try:
        from PIL import Image
        return Image.open(PAGE_ICON_PATH)
    except Exception:
        return PAGE_ICON_FALLBACK


@st.cache_resource
def load_components():
    # Load heavy RAG components once per server process.
    return load_chatbot_components()


def initialize_session_state():
    # Initialize all state keys used by the UI.
    defaults = {
        "messages": [],
        "is_generating": False,
        "latest_sources": [],
        "latest_sources_clean": [],
        ACTIVE_CONVERSATION_KEY: None,
    }

    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def clear_answer_ui_state():
    # Clear UI state that belongs only to the previous assistant answer.
    # This prevents stale action buttons and suggestion chips during regenerate/streaming.
    for key in [
        "suggestions",
        "last_answer",
        "last_question",
        "last_sources",
        "open_export_panel_key",
    ]:
        st.session_state.pop(key, None)


def clear_message_suggestions(messages=None, keep_latest_assistant=False):
    # Remove old suggestion chips saved inside assistant messages.
    # Only the latest assistant answer should show suggestions.
    if messages is None:
        messages = st.session_state.get("messages", [])

    keep_index = None

    if keep_latest_assistant:
        for index in range(len(messages) - 1, -1, -1):
            message = messages[index]

            if isinstance(message, dict) and message.get("role") == "assistant":
                keep_index = index
                break

    for index, message in enumerate(messages or []):
        if not isinstance(message, dict):
            continue

        if message.get("role") != "assistant":
            continue

        if keep_index is not None and index == keep_index:
            continue

        message["suggestions"] = []


def begin_answer_generation():
    # Hide answer-only controls while a new answer is being generated.
    st.session_state.is_generating = True
    st.session_state.hide_actions = True
    st.session_state.hide_suggestions = True
    clear_answer_ui_state()
    clear_message_suggestions(keep_latest_assistant=False)


def finish_answer_generation():
    # Restore answer controls after generation completes.
    st.session_state.is_generating = False
    st.session_state.pop("hide_actions", None)
    st.session_state.pop("hide_suggestions", None)


def should_hide_answer_controls():
    # True as soon as regenerate or a suggested question is clicked.
    return (
        st.session_state.get("hide_actions", False)
        or st.session_state.get("hide_suggestions", False)
        or st.session_state.get("is_generating", False)
        or st.session_state.get("regenerate_index") is not None
        or st.session_state.get("regenerate_data") is not None
        or st.session_state.get("suggested_query") is not None
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
        and st.session_state.get("regenerate_data") is None
        and st.session_state.get("suggested_query") is None
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


def normalize_suggestion_list(suggestions, limit=SUGGESTION_LIMIT):
    # Clean suggestions returned by backend.
    if not suggestions:
        return []

    if isinstance(suggestions, str):
        suggestions = [suggestions]

    result = []

    for item in suggestions:
        text = str(item or "").strip().lstrip("-•123456789. )").strip()

        if text:
            result.append(text)

        if len(result) >= limit:
            break

    return result


def safe_generate_ui_suggestions(question, answer, llm):
    # Fallback suggestion generation when backend returns none.
    if not question or not answer:
        return []

    if NO_ANSWER_TEXT.lower() in str(answer).lower():
        return []

    try:
        return normalize_suggestion_list(
            generate_suggestions(question=question, answer=answer, llm=llm)
        )
    except Exception:
        return []


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

    clear_message_suggestions(keep_latest_assistant=True)


def reset_chat_state(clear_active_conversation=True):
    # Clear visible chat and temporary UI state.
    clear_conversation_memory(st)

    for key in [
        "regenerate_index",
        "regenerate_data",
        "suggested_query",
        "hide_actions",
        "hide_suggestions",
        "open_export_panel_key",
        "last_sources",
        "last_answer",
        "last_question",
        "suggestions",
        "sidebar_sources",
        "pending_question",
        "selected_conversation_id",
    ]:
        st.session_state.pop(key, None)

    st.session_state.messages = []
    st.session_state.latest_sources = []
    st.session_state.latest_sources_clean = []
    st.session_state.is_generating = False

    if clear_active_conversation:
        st.session_state[ACTIVE_CONVERSATION_KEY] = None


def load_saved_conversation(browser_id, conversation_id):
    # Load conversation into current session.
    messages = load_messages(browser_id=browser_id, conversation_id=conversation_id)

    if not messages:
        return False

    clear_message_suggestions(messages=messages, keep_latest_assistant=True)

    st.session_state.messages = messages
    st.session_state[ACTIVE_CONVERSATION_KEY] = conversation_id
    set_latest_sources(get_last_assistant_sources(messages))
    rebuild_conversation_memory(st, messages)

    return True


def persist_current_conversation(browser_id, title=None):
    # Persist current visible chat to SQLite.
    messages = st.session_state.get("messages", [])

    if not messages:
        return None

    conversation_id = replace_conversation_messages(
        browser_id=browser_id,
        conversation_id=st.session_state.get(ACTIVE_CONVERSATION_KEY),
        messages=messages,
        title=clean_title(title),
    )
    st.session_state[ACTIVE_CONVERSATION_KEY] = conversation_id

    return conversation_id


def handle_sidebar_action(sidebar_action, browser_id):
    # Apply sidebar button/history clicks.
    if not sidebar_action:
        return

    selected_id = sidebar_action.get("selected_conversation_id")

    if selected_id and load_saved_conversation(browser_id, selected_id):
        st.rerun()

    if sidebar_action.get("clear_chat_clicked"):
        delete_all_conversations(browser_id=browser_id)
        reset_chat_state(clear_active_conversation=True)
        st.rerun()

    if sidebar_action.get("new_chat_clicked"):
        reset_chat_state(clear_active_conversation=True)
        st.rerun()


def display_streaming_assistant_bubble(message):
    # Streaming assistant bubble with cursor.
    from ui.history_ui import safe_html_text

    st.markdown(
        "\n".join([
            '<div class="assistant-message-wrapper">',
            f'<div class="assistant-message-bubble">{safe_html_text(message)}<span class="streaming-cursor">▌</span></div>',
            '</div>',
        ]),
        unsafe_allow_html=True,
    )


def display_thinking():
    # Retrieval/rerank loading indicator.
    st.markdown(
        """
<div class="assistant-message-wrapper">
    <div class="thinking-box">
        <div class="thinking-dots"><span></span><span></span><span></span></div>
        <span>Thinking...</span>
    </div>
</div>
""",
        unsafe_allow_html=True,
    )


def stream_assistant_answer(question, components, conversation_history):
    # Stream one RAG answer and return final metadata.
    answer_slot = st.empty()
    thinking_slot = st.empty()
    answer = ""
    sources = []
    suggestions = []
    first_chunk = True

    with thinking_slot:
        display_thinking()

    for event in ask_rag_stream(
        question=question,
        vectorstore=components["vectorstore"],
        bm25_retriever=components["bm25_retriever"],
        reranker=components["reranker"],
        llm=components["llm"],
        chat_history=conversation_history,
        debug=DEBUG_RAG,
        all_chunks=components.get("chunks"),
    ):
        event_type = event.get("type")

        if event_type == "chunk":
            if first_chunk:
                thinking_slot.empty()
                first_chunk = False

            answer += event.get("content", "")

            with answer_slot:
                display_streaming_assistant_bubble(answer)

            continue

        if event_type == "done":
            answer = event.get("answer", answer)
            sources = normalize_sources_for_state(event.get("sources", []))
            suggestions = normalize_suggestion_list(event.get("suggestions", []))

            if is_no_answer(answer):
                sources = []
                suggestions = []

            set_latest_sources(sources)

    thinking_slot.empty()

    if is_no_answer(answer):
        sources = []
        suggestions = []
        set_latest_sources([])

    if not suggestions:
        suggestions = safe_generate_ui_suggestions(
            question=question,
            answer=answer,
            llm=components["llm"],
        )

    created_at = get_now_text()

    with answer_slot:
        display_assistant_bubble(answer, timestamp=created_at)

    return answer, sources, suggestions, created_at


def build_visible_conversation_history(limit=MEMORY_LIMIT):
    # Build chat history from visible messages.
    lines = []

    for message in st.session_state.get("messages", []):
        if not isinstance(message, dict):
            continue

        role = message.get("role")
        content = str(message.get("content", "")).strip()

        if not content:
            continue

        if role == "user":
            lines.append(f"User: {content}")
        elif role == "assistant":
            lines.append(f"Assistant: {content}")

    if limit is not None:
        lines = lines[-limit:]

    return "\n".join(lines)


def get_current_history():
    # Prefer visible history, fallback to session memory.
    return build_visible_conversation_history(MEMORY_LIMIT) or get_conversation_history(st, MEMORY_LIMIT)


def display_chat_history_without_controls(messages):
    # Render messages without answer-only controls.
    # Used during regeneration so old action buttons and suggestions cannot remain visible.
    for message in messages or []:
        if not isinstance(message, dict):
            continue

        role = message.get("role")
        content = message.get("content", "")
        timestamp = message.get("created_at")

        if role == "user":
            display_user_bubble(content, timestamp=timestamp)
        elif role == "assistant":
            display_assistant_bubble(content, timestamp=timestamp)


def handle_regenerate_request():
    # Prepare assistant answer regeneration.
    if "regenerate_index" not in st.session_state:
        return

    index = st.session_state.regenerate_index
    messages = st.session_state.get("messages", [])

    if index is None or index < 0 or index >= len(messages):
        st.error("Cannot regenerate: message index is invalid.")
        st.session_state.pop("regenerate_index", None)
        st.session_state.pop("hide_actions", None)
        st.stop()

    old_message = messages[index]
    question = old_message.get("question")

    if not question:
        st.error("Cannot regenerate: original question not found.")
        st.session_state.pop("regenerate_index", None)
        st.session_state.pop("hide_actions", None)
        st.stop()

    st.session_state.regenerate_data = {"index": index, "question": question}
    begin_answer_generation()

    # Remove only the old assistant response.
    # Keep the user question and previous conversation visible during regeneration.
    messages.pop(index)

    # Clear old answer-only UI state so old actions/suggestions/sources do not remain visible.
    st.session_state.pop("regenerate_index", None)
    set_latest_sources([])

    rebuild_conversation_memory(st, messages)
    st.rerun()


def handle_regenerate_answer(components, history_slot, browser_id):
    # Generate replacement answer after regenerate request.
    if "regenerate_data" not in st.session_state:
        return

    regen_data = st.session_state.regenerate_data
    index = regen_data["index"]
    question = regen_data["question"]

    history_slot.empty()

    with history_slot.container():
        # Re-render current visible messages without old controls/suggestions.
        # The old assistant response was already removed, so only the answer area regenerates.
        display_chat_history_without_controls(st.session_state.messages)

        answer, sources, suggestions, created_at = stream_assistant_answer(
            question=question,
            components=components,
            conversation_history=get_current_history(),
        )

        if is_no_answer(answer):
            sources = []
            suggestions = []
            set_latest_sources([])

    st.session_state.messages.insert(index, {
        "role": "assistant",
        "content": answer,
        "sources": sources,
        "question": question,
        "suggestions": suggestions,
        "created_at": created_at,
    })

    set_latest_sources(sources)
    rebuild_conversation_memory(st, st.session_state.messages)
    persist_current_conversation(browser_id, title=question)
    st.session_state.pop("regenerate_data", None)
    finish_answer_generation()
    st.rerun()


def display_empty_state():
    # Welcome title when there is no active chat.
    if st.session_state.messages or st.session_state.get("is_generating", False):
        return

    st.markdown(
        '<div class="main-title">How can I help you today?</div>',
        unsafe_allow_html=True,
    )


def display_chat_footer_note():
    # Small fixed disclaimer below input.
    st.markdown(
        '<div class="chat-footer-note">InknowVa may make mistakes. Verify important information with the official source document.</div>',
        unsafe_allow_html=True,
    )


def get_user_query():
    # Typed query or clicked suggestion.
    if "suggested_query" in st.session_state:
        return st.session_state.pop("suggested_query")

    return st.chat_input(CHAT_INPUT_PLACEHOLDER)


def append_user_message(query):
    # Append user query and return timestamp.
    created_at = get_now_text()
    st.session_state.messages.append({
        "role": "user",
        "content": query,
        "created_at": created_at,
    })
    return created_at


def append_assistant_message(answer, sources, question, suggestions, created_at=None):
    # Append assistant answer to visible history.
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
        "question": question,
        "suggestions": suggestions,
        "created_at": created_at or get_now_text(),
    })


def handle_new_query(query, components, browser_id):
    # Render, save, and persist a new Q&A turn.
    begin_answer_generation()
    set_latest_sources([])
    conversation_history = get_current_history()

    user_created_at = append_user_message(query)
    display_user_bubble(query, timestamp=user_created_at)

    answer, sources, suggestions, assistant_created_at = stream_assistant_answer(
        question=query,
        components=components,
        conversation_history=conversation_history,
    )

    if is_no_answer(answer):
        sources = []
        suggestions = []
        set_latest_sources([])

    append_assistant_message(
        answer=answer,
        sources=sources,
        question=query,
        suggestions=suggestions,
        created_at=assistant_created_at,
    )

    set_latest_sources(sources)
    rebuild_conversation_memory(st, st.session_state.messages)
    persist_current_conversation(browser_id, title=query)
    finish_answer_generation()
    st.rerun()


def run_chat_ui():
    # Main UI entrypoint called by app.py/main.py.
    st.set_page_config(
        page_title=PAGE_TITLE,
        page_icon=get_page_icon(),
        layout="wide",
        initial_sidebar_state="expanded",
    )

    load_css()
    initialize_session_state()
    inject_answer_control_hider()
    init_memory(st)
    sanitize_existing_source_state()
    init_history_db()

    browser_id = get_browser_id(st)
    components = load_components()

    sidebar_action = display_sidebar(
        browser_id=browser_id,
        active_conversation_id=st.session_state.get(ACTIVE_CONVERSATION_KEY),
        sources=st.session_state.get("sidebar_sources")
        or st.session_state.get("latest_sources_clean")
        or [],
    )
    handle_sidebar_action(sidebar_action, browser_id)

    # Read the chat input before rendering the empty landing title.
    # This hides the landing title as soon as the user presses Enter,
    # instead of waiting until the assistant answer finishes.
    query = get_user_query()

    if query:
        begin_answer_generation()
        inject_answer_control_hider()
    else:
        clear_message_suggestions(keep_latest_assistant=True)
        inject_empty_state_chat_input_position()

    display_empty_state()
    handle_regenerate_request()

    history_slot = st.empty()

    # During regeneration, do not render the full chat history here.
    # handle_regenerate_answer() renders a no-controls version so stale suggestions disappear.
    if "regenerate_data" not in st.session_state:
        with history_slot.container():
            display_chat_history(st.session_state.messages)

    handle_regenerate_answer(
        components=components,
        history_slot=history_slot,
        browser_id=browser_id,
    )

    display_chat_footer_note()

    if query:
        handle_new_query(
            query=query,
            components=components,
            browser_id=browser_id,
        )
