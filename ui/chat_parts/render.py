from ui.chat_parts.storage_actions import (
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
)



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
):
    # Stream one RAG answer and return final metadata.
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




def get_current_history():
    # One source of truth for memory: visible session messages.
    # The memory helper converts them into prompt-ready User/Assistant/Source lines.
    return get_conversation_history(st, MEMORY_LIMIT)


def display_chat_history_without_controls(messages):
    # Render messages without answer-only controls.
    # Used while pending generation so old action buttons cannot remain visible.
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
]
