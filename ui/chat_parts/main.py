from ui.chat_parts.handlers import (
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
    get_user_query,
    append_user_message,
    append_assistant_message,
    queue_new_query,
    render_pending_generation_view,
    handle_pending_new_query,
    handle_new_query,
)


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

    has_pending_work = bool(st.session_state.get("pending_question"))

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
    'run_chat_ui',
]
