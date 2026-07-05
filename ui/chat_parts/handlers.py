from ui.chat_parts.render import (
    Path,
    time,
    st,
    ask_rag_stream,
    load_chatbot_components,
    clear_conversation_memory,
    get_conversation_history,
    init_memory,
    rebuild_conversation_memory,
    get_browser_id,
    clean_title,
    get_now_text,
    init_history_db,
    load_messages,
    replace_conversation_messages,
    delete_conversation,
    display_assistant_bubble,
    display_chat_history,
    display_user_bubble,
    display_sidebar,
    normalize_sources_for_state,
    load_css,
    NO_ANSWER_TEXT,
    PAGE_TITLE,
    PAGE_ICON_PATH,
    PAGE_ICON_FALLBACK,
    CHAT_INPUT_PLACEHOLDER,
    MEMORY_LIMIT,
    DEBUG_RAG,
    FRONTEND_FLUSH_SECONDS,
    ACTIVE_CONVERSATION_KEY,
    PENDING_SIDEBAR_ACTION_KEY,
    UI_FLOW_DEBUG_FILE,
    append_ui_flow_debug,
    is_no_answer,
    get_page_icon,
    load_components,
    get_loaded_components,
    initialize_session_state,
    clear_answer_ui_state,
    begin_answer_generation,
    finish_answer_generation,
    should_hide_answer_controls,
    inject_answer_control_hider,
    is_empty_chat_state,
    inject_empty_state_chat_input_position,
    set_latest_sources,
    get_last_assistant_sources,
    sanitize_existing_source_state,
    reset_chat_state,
    delete_active_conversation_from_store,
    clear_active_chat_session,
    load_saved_conversation,
    get_first_user_question,
    persist_current_conversation,
    get_pending_sidebar_action,
    clear_pending_sidebar_action,
    handle_pending_sidebar_action,
    handle_sidebar_action,
    flush_frontend_render,
    display_streaming_assistant_bubble,
    display_thinking,
    display_loading_state,
    stream_assistant_answer,
    get_current_history,
    display_chat_history_without_controls,
    display_empty_state,
    display_chat_footer_note,
)



def get_user_query():
    # Read typed query only.
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
    })


def queue_new_query(query):
    # Save the user message immediately, then generate after the UI has rendered.
    # This prevents blank/dark repaint before the Thinking indicator appears.
    query = str(query or "").strip()

    if not query:
        return False

    if st.session_state.get("pending_question"):
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
    'delete_active_conversation_from_store',
    'clear_active_chat_session',
    'load_saved_conversation',
    'get_first_user_question',
    'persist_current_conversation',
    'get_pending_sidebar_action',
    'clear_pending_sidebar_action',
    'handle_pending_sidebar_action',
    'handle_sidebar_action',
    'flush_frontend_render',
    'display_streaming_assistant_bubble',
    'display_thinking',
    'display_loading_state',
    'stream_assistant_answer',
    'get_current_history',
    'display_chat_history_without_controls',
    'display_empty_state',
    'display_chat_footer_note',
    'get_user_query',
    'append_user_message',
    'append_assistant_message',
    'queue_new_query',
    'render_pending_generation_view',
    'handle_pending_new_query',
    'handle_new_query',
]
