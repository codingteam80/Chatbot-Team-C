#Main Streamlit chat UI for InknowVa.

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
    delete_conversation,
    get_now_text,
    init_history_db,
    load_messages,
    replace_conversation_messages,
)
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
    # This confirms whether the Regenerate button reaches chat_ui.py.
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
    # Use contains check para sync sa chains.chatbot.is_no_answer().
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
    # True as soon as regenerate is clicked or an answer is streaming.
    return (
        st.session_state.get("hide_actions", False)
        or st.session_state.get("is_generating", False)
        or st.session_state.get("regenerate_index") is not None
        or st.session_state.get("regenerate_data") is not None
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
        "regenerate_index",
        "regenerate_data",
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


def load_saved_conversation(browser_id, conversation_id):
    # Load conversation into current session.
    messages = load_messages(browser_id=browser_id, conversation_id=conversation_id)

    if not messages:
        return False

    st.session_state.messages = messages
    st.session_state[ACTIVE_CONVERSATION_KEY] = conversation_id
    set_latest_sources(get_last_assistant_sources(messages))
    rebuild_conversation_memory(st, messages)

    return True


def get_first_user_question(messages):
    # Use the first user question as the stable conversation title.
    for message in messages or []:
        if not isinstance(message, dict):
            continue

        if message.get("role") != "user":
            continue

        content = str(message.get("content", "")).strip()

        if content:
            return content

    return None


def persist_current_conversation(browser_id, title=None):
    # Persist current visible chat to SQLite.
    # Keep the history title stable by using the first user question in the session.
    messages = st.session_state.get("messages", [])

    if not messages:
        return None

    conversation_title = get_first_user_question(messages) or title

    conversation_id = replace_conversation_messages(
        browser_id=browser_id,
        conversation_id=st.session_state.get(ACTIVE_CONVERSATION_KEY),
        messages=messages,
        title=clean_title(conversation_title),
    )
    st.session_state[ACTIVE_CONVERSATION_KEY] = conversation_id

    return conversation_id


def get_pending_sidebar_action():
    # Sidebar button callbacks set this before the next script run starts.
    action = st.session_state.get(PENDING_SIDEBAR_ACTION_KEY)

    if not isinstance(action, dict):
        return None

    return action


def clear_pending_sidebar_action():
    # Clear queued sidebar navigation/action.
    st.session_state.pop(PENDING_SIDEBAR_ACTION_KEY, None)


def handle_pending_sidebar_action(browser_id):
    # Handle New chat, Clear chat, and History clicks before rendering heavy UI.
    action = get_pending_sidebar_action()

    if not action:
        return False

    action_type = action.get("type")

    if action_type == "new_chat":
        display_loading_state("Starting new chat...")
        flush_frontend_render()
        reset_chat_state(clear_active_conversation=True)
        clear_pending_sidebar_action()
        st.rerun()
        return True

    if action_type == "clear_chat":
        display_loading_state("Clearing active chat...")
        flush_frontend_render()
        delete_conversation(
            browser_id=browser_id,
            conversation_id=st.session_state.get(ACTIVE_CONVERSATION_KEY),
        )
        reset_chat_state(clear_active_conversation=True)
        clear_pending_sidebar_action()
        st.rerun()
        return True

    if action_type == "open_chat":
        display_loading_state("Opening chat...")
        flush_frontend_render()
        conversation_id = action.get("conversation_id")

        if conversation_id:
            load_saved_conversation(browser_id, conversation_id)

        clear_pending_sidebar_action()
        st.rerun()
        return True

    clear_pending_sidebar_action()
    return False


def handle_sidebar_action(sidebar_action, browser_id):
    # Legacy fallback for sidebar versions that still return clicked actions.
    if not sidebar_action:
        return

    selected_id = sidebar_action.get("selected_conversation_id")

    if selected_id:
        st.session_state[PENDING_SIDEBAR_ACTION_KEY] = {
            "type": "open_chat",
            "conversation_id": selected_id,
        }
        st.rerun()

    if sidebar_action.get("clear_chat_clicked"):
        st.session_state[PENDING_SIDEBAR_ACTION_KEY] = {"type": "clear_chat"}
        st.rerun()

    if sidebar_action.get("new_chat_clicked"):
        st.session_state[PENDING_SIDEBAR_ACTION_KEY] = {"type": "new_chat"}
        st.rerun()


def flush_frontend_render():
    # Give the browser a tiny window to paint the user bubble and Thinking state
    # before CPU-heavy retrieval/reranking/LLM work starts.
    if FRONTEND_FLUSH_SECONDS > 0:
        time.sleep(FRONTEND_FLUSH_SECONDS)


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


def display_loading_state(message="Loading..."):
    # Small generic loading state for sidebar/export/history actions.
    safe_message = str(message or "Loading...")
    st.markdown(
        f"""
<div class="assistant-message-wrapper">
    <div class="thinking-box">
        <div class="thinking-dots"><span></span><span></span><span></span></div>
        <span>{safe_message}</span>
    </div>
</div>
""",
        unsafe_allow_html=True,
    )




def stream_assistant_answer(
    question,
    components,
    conversation_history,
    thinking_slot=None,
    regenerate=False,
    regenerate_attempt=0,
    previous_answer="",
):
    # Stream one RAG answer and return final metadata.
    # When regenerate=True, retrieval/rerank still runs, but the final prompt asks for a different wording.
    answer_slot = st.empty()

    if thinking_slot is None:
        thinking_slot = st.empty()
    answer = ""
    sources = []
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
        regenerate=regenerate,
        regenerate_attempt=regenerate_attempt,
        previous_answer=previous_answer,
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

            if is_no_answer(answer):
                sources = []

            set_latest_sources(sources)

    thinking_slot.empty()

    if is_no_answer(answer):
        sources = []
        set_latest_sources([])

    created_at = get_now_text()

    with answer_slot:
        display_assistant_bubble(answer, timestamp=created_at, sources=sources)

    return answer, sources, created_at




def get_message_source_titles(message):
    # Keep source titles in memory so follow-up references can resolve to the cited topic.
    titles = []
    seen = set()

    for source in message.get("sources", []) or []:
        if not isinstance(source, dict):
            continue

        title = str(
            source.get("title")
            or source.get("source")
            or source.get("file_name")
            or ""
        ).strip()

        if not title:
            continue

        key = title.lower()
        if key in seen:
            continue

        seen.add(key)
        titles.append(title)

    return titles


def build_conversation_history_from_messages(messages, limit=MEMORY_LIMIT, exclude_indexes=None):
    # Build history for follow-up rewriting only.
    # Do not include raw assistant answers because short dates/numbers can become false pronoun targets.
    exclude_indexes = set(exclude_indexes or [])
    lines = []

    for index, message in enumerate(messages or []):
        if index in exclude_indexes:
            continue

        if not isinstance(message, dict):
            continue

        role = message.get("role")
        content = str(message.get("content", "")).strip()

        if role == "user" and content:
            lines.append(f"User: {content}")
            continue

        if role == "assistant":
            resolved_question = str(message.get("question") or "").strip()

            if resolved_question:
                lines.append(f"Resolved question: {resolved_question}")

            for title in get_message_source_titles(message):
                lines.append(f"Source: {title}")

    if limit is not None:
        lines = lines[-limit:]

    return "\n".join(lines)

def find_current_user_message_index(messages, assistant_insert_index, question):
    # During regenerate, the old assistant answer is removed but the original user message remains.
    # Exclude that current user message from chat_history so RAG sees it as the active question, not history.
    question_key = str(question or "").strip().lower()

    if not question_key:
        return None

    start_index = min(int(assistant_insert_index or 0) - 1, len(messages or []) - 1)

    for index in range(start_index, -1, -1):
        message = messages[index]

        if not isinstance(message, dict):
            continue

        role = message.get("role")
        content_key = str(message.get("content") or "").strip().lower()

        if role == "user" and content_key == question_key:
            return index

        # Stop after crossing into an older assistant turn.
        if role == "assistant":
            break

    return None


def build_regenerate_conversation_history(assistant_insert_index, question, limit=MEMORY_LIMIT):
    # Full RAG regenerate should use the same original question, but history must exclude that same user turn.
    messages = st.session_state.get("messages", [])
    current_user_index = find_current_user_message_index(
        messages=messages,
        assistant_insert_index=assistant_insert_index,
        question=question,
    )
    exclude_indexes = {current_user_index} if current_user_index is not None else set()

    return build_conversation_history_from_messages(
        messages=messages,
        limit=limit,
        exclude_indexes=exclude_indexes,
    )


def build_visible_conversation_history(limit=MEMORY_LIMIT):
    # Build full chat history from visible messages by default.
    return build_conversation_history_from_messages(
        messages=st.session_state.get("messages", []),
        limit=limit,
    )


def get_current_history():
    # Prefer full visible history, fallback to full session memory.
    return build_visible_conversation_history(MEMORY_LIMIT) or get_conversation_history(st, MEMORY_LIMIT)


def display_chat_history_without_controls(messages):
    # Render messages without answer-only controls.
    # Used during regeneration so old action buttons cannot remain visible.
    for message in messages or []:
        if not isinstance(message, dict):
            continue

        role = message.get("role")
        content = message.get("content", "")
        timestamp = message.get("created_at")

        if role == "user":
            display_user_bubble(content, timestamp=timestamp)
        elif role == "assistant":
            display_assistant_bubble(content, timestamp=timestamp, sources=message.get("sources", []))


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
    previous_answer = str(old_message.get("original_answer") or old_message.get("content") or "").strip()
    regenerate_attempt = int(old_message.get("regen_count") or 0) + 1

    append_ui_flow_debug(
        "UI REGENERATE REQUEST RECEIVED",
        [
            f"index: {index}",
            f"question: {question}",
            f"previous_answer_chars: {len(previous_answer)}",
            f"regenerate_attempt: {regenerate_attempt}",
            "regenerate_mode: full_rag_different_wording",
        ],
    )

    if not question:
        st.error("Cannot regenerate: original question not found.")
        st.session_state.pop("regenerate_index", None)
        st.session_state.pop("hide_actions", None)
        st.stop()

    st.session_state.regenerate_data = {
        "index": index,
        "question": question,
        "previous_answer": previous_answer,
        "original_answer": str(old_message.get("original_answer") or previous_answer),
        "previous_sources": normalize_sources_for_state(old_message.get("sources", [])),
        "regenerate_attempt": regenerate_attempt,
    }
    begin_answer_generation()

    # Remove only the old assistant response.
    # Keep the user question and previous conversation visible during regeneration.
    messages.pop(index)

    # Clear old answer-only UI state so old actions/sources do not remain visible.
    st.session_state.pop("regenerate_index", None)
    set_latest_sources([])

    rebuild_conversation_memory(st, messages)
    return


def handle_regenerate_answer(components, history_slot, browser_id, thinking_slot=None, pre_rendered=False):
    # Generate replacement answer after regenerate request.
    # Components are passed from run_chat_ui so loading is centralized.
    if "regenerate_data" not in st.session_state:
        return

    regen_data = st.session_state.regenerate_data
    index = regen_data["index"]
    question = regen_data["question"]
    previous_answer = regen_data.get("previous_answer", "")
    regenerate_attempt = int(regen_data.get("regenerate_attempt") or 1)
    conversation_history = build_regenerate_conversation_history(
        assistant_insert_index=index,
        question=question,
        limit=MEMORY_LIMIT,
    )

    if not pre_rendered:
        with history_slot.container():
            display_chat_history_without_controls(st.session_state.messages)
            thinking_slot = st.empty()
            with thinking_slot:
                display_thinking()
            flush_frontend_render()

    append_ui_flow_debug(
        "UI REGENERATE ANSWER START",
        [
            f"index: {index}",
            f"question: {question}",
            f"previous_answer_chars: {len(str(previous_answer or ''))}",
            f"conversation_history_chars: {len(str(conversation_history or ''))}",
            f"regenerate_attempt: {regenerate_attempt}",
            "regenerate_route: full_rag",
        ],
    )

    answer, sources, created_at = stream_assistant_answer(
        question=question,
        components=components,
        conversation_history=conversation_history,
        thinking_slot=thinking_slot,
        regenerate=True,
        regenerate_attempt=regenerate_attempt,
        previous_answer=previous_answer,
    )

    append_ui_flow_debug(
        "UI REGENERATE ANSWER DONE",
        [
            f"new_answer_chars: {len(str(answer or ''))}",
            f"sources_count: {len(sources or [])}",
        ],
    )

    if is_no_answer(answer):
        sources = []
        set_latest_sources([])

    st.session_state.messages.insert(index, {
        "role": "assistant",
        "content": answer,
        "sources": sources,
        "question": question,
        "created_at": created_at,
        "regen_count": regenerate_attempt,
        "original_answer": answer,
    })

    set_latest_sources(sources)
    rebuild_conversation_memory(st, st.session_state.messages)
    persist_current_conversation(browser_id)
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
    # Return typed query from the fixed chat input.
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


def append_assistant_message(answer, sources, question, created_at=None):
    # Append assistant answer to visible history.
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
        "question": question,
        "created_at": created_at or get_now_text(),
        "regen_count": 0,
        "original_answer": answer,
    })


def queue_new_query(query):
    # Save the user message immediately, then generate after the UI has rendered.
    # This prevents blank/dark repaint before the Thinking indicator appears.
    query = str(query or "").strip()

    if not query:
        return False

    if st.session_state.get("pending_question") or st.session_state.get("regenerate_data"):
        return False

    begin_answer_generation()
    set_latest_sources([])
    st.session_state.pending_conversation_history = get_current_history()
    st.session_state.pending_question = query
    append_user_message(query)
    rebuild_conversation_memory(st, st.session_state.messages)

    return True


def render_pending_generation_view(history_slot):
    # Render the visible user bubble/history and Thinking before component loading.
    with history_slot.container():
        display_chat_history_without_controls(st.session_state.messages)
        thinking_slot = st.empty()
        with thinking_slot:
            display_thinking()

    flush_frontend_render()
    return thinking_slot


def handle_pending_new_query(components, history_slot, browser_id, thinking_slot=None, pre_rendered=False):
    # Generate a queued user question.
    # Components are passed from run_chat_ui so loading is centralized.
    question = str(st.session_state.get("pending_question") or "").strip()

    if not question:
        return

    conversation_history = st.session_state.get("pending_conversation_history", "")

    if not pre_rendered:
        thinking_slot = render_pending_generation_view(history_slot)

    answer, sources, assistant_created_at = stream_assistant_answer(
        question=question,
        components=components,
        conversation_history=conversation_history,
        thinking_slot=thinking_slot,
    )

    if is_no_answer(answer):
        sources = []
        set_latest_sources([])

    append_assistant_message(
        answer=answer,
        sources=sources,
        question=question,
        created_at=assistant_created_at,
    )

    set_latest_sources(sources)
    rebuild_conversation_memory(st, st.session_state.messages)
    persist_current_conversation(browser_id, title=question)
    st.session_state.pop("pending_question", None)
    st.session_state.pop("pending_conversation_history", None)
    finish_answer_generation()
    st.rerun()


def handle_new_query(query, browser_id, components):
    # Direct new-query path kept for compatibility.
    # Components are passed from run_chat_ui so loading is centralized.
    begin_answer_generation()
    set_latest_sources([])
    conversation_history = get_current_history()

    user_created_at = append_user_message(query)
    display_user_bubble(query, timestamp=user_created_at)

    thinking_slot = st.empty()

    with thinking_slot:
        display_thinking()

    flush_frontend_render()

    answer, sources, assistant_created_at = stream_assistant_answer(
        question=query,
        components=components,
        conversation_history=conversation_history,
        thinking_slot=thinking_slot,
    )

    if is_no_answer(answer):
        sources = []
        set_latest_sources([])

    append_assistant_message(
        answer=answer,
        sources=sources,
        question=query,
        created_at=assistant_created_at,
    )

    set_latest_sources(sources)
    rebuild_conversation_memory(st, st.session_state.messages)
    persist_current_conversation(browser_id, title=query)
    finish_answer_generation()
    st.rerun()

def run_chat_ui():
    # Main UI entrypoint called by app.py/main.py.
    # This function is the central controller for component loading.
    st.set_page_config(
        page_title=PAGE_TITLE,
        page_icon=get_page_icon(),
        layout="wide",
        initial_sidebar_state="expanded",
    )

    load_css()
    initialize_session_state()
    init_memory(st)
    sanitize_existing_source_state()
    init_history_db()

    browser_id = get_browser_id(st)

    # Sidebar callbacks run before the script body.
    # Handle them before rendering the expensive sidebar/history list.
    if handle_pending_sidebar_action(browser_id):
        return

    # Read chat input early so submit can show user bubble + Thinking before sidebar work.
    query = get_user_query()

    if query:
        queue_new_query(query)

    # Regenerate callbacks also run before the script body.
    # Convert regenerate_index into regenerate_data before rendering sidebar/history.
    handle_regenerate_request()

    has_pending_query = bool(st.session_state.get("pending_question"))
    has_regenerate_data = st.session_state.get("regenerate_data") is not None
    has_pending_work = has_pending_query or has_regenerate_data

    if has_pending_work:
        inject_answer_control_hider()
    else:
        inject_empty_state_chat_input_position()

    display_empty_state()

    history_slot = st.empty()
    display_chat_footer_note()

    # Pending work path:
    # 1. render history/user bubble + Thinking
    # 2. then load/reuse components centrally here
    # 3. then pass components downward
    if has_pending_work:
        thinking_slot = render_pending_generation_view(history_slot)
        components = get_loaded_components()

        if has_regenerate_data:
            handle_regenerate_answer(
                components=components,
                history_slot=history_slot,
                browser_id=browser_id,
                thinking_slot=thinking_slot,
                pre_rendered=True,
            )
            return

        handle_pending_new_query(
            components=components,
            history_slot=history_slot,
            browser_id=browser_id,
            thinking_slot=thinking_slot,
            pre_rendered=True,
        )
        return

    # Idle path only: render sidebar/history and preload/reuse heavy components once.
    sidebar_action = display_sidebar(
        browser_id=browser_id,
        active_conversation_id=st.session_state.get(ACTIVE_CONVERSATION_KEY),
    )
    handle_sidebar_action(sidebar_action, browser_id)

    with history_slot.container():
        display_chat_history(st.session_state.messages)

    get_loaded_components()
