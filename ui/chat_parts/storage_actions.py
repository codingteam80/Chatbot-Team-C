from ui.chat_parts.state import (
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
)




def delete_active_conversation_from_store(browser_id):
    # Delete or clear only the active saved conversation.
    conversation_id = st.session_state.get(ACTIVE_CONVERSATION_KEY)

    if not conversation_id:
        return False

    if delete_conversation is not None:
        try:
            delete_conversation(browser_id=browser_id, conversation_id=conversation_id)
            return True
        except TypeError:
            try:
                delete_conversation(conversation_id)
                return True
            except Exception as error:
                append_ui_flow_debug(
                    "Delete active conversation fallback failed",
                    [f"conversation_id={conversation_id}", f"error={error}"],
                )
                return False
        except Exception as error:
            append_ui_flow_debug(
                "Delete active conversation failed",
                [f"conversation_id={conversation_id}", f"error={error}"],
            )
            return False

    # Fallback for older storage modules that do not have delete_conversation().
    # This clears the active conversation's saved messages instead of deleting all chats.
    try:
        replace_conversation_messages(
            browser_id=browser_id,
            conversation_id=conversation_id,
            messages=[],
            title="Cleared chat",
        )
        return True
    except Exception as error:
        append_ui_flow_debug(
            "Clear active conversation fallback failed",
            [f"conversation_id={conversation_id}", f"error={error}"],
        )
        return False


def clear_active_chat_session(browser_id):
    # Clear only the current visible chat and its active saved conversation.
    delete_active_conversation_from_store(browser_id)
    reset_chat_state(clear_active_conversation=True)


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
    from ui.chat_parts.render import display_loading_state

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
        display_loading_state("Clearing current chat...")
        flush_frontend_render()
        clear_active_chat_session(browser_id)
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
]
