"""Small session-memory helper for follow-up questions."""

MEMORY_KEY = "chat_memory"


def init_memory(st):
    # Gumawa ng memory list once per Streamlit session.
    st.session_state.setdefault(MEMORY_KEY, [])


def get_conversation_history(st, limit=None):
    # Ibalik ang current chat memory bilang prompt-ready text.
    init_memory(st)
    memory = st.session_state.get(MEMORY_KEY, [])

    if limit is not None:
        memory = memory[-limit:]

    return "\n".join(memory)


def get_message_source_titles(message):
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


def save_to_memory(st, question, answer):
    # Keep only the user question in fallback memory.
    # Raw assistant answers may be short dates/numbers and can poison follow-up rewriting.
    init_memory(st)
    question = str(question or "").strip()

    if not question:
        return

    st.session_state[MEMORY_KEY].append(f"User: {question}")


def rebuild_conversation_memory(st, messages):
    # Rebuild prompt memory from visible messages using topics/sources, not raw answers.
    rebuilt = []
    last_question = None

    for message in messages or []:
        if not isinstance(message, dict):
            continue

        role = message.get("role")
        content = str(message.get("content", "")).strip()

        if role == "user" and content:
            last_question = content
            rebuilt.append(f"User: {content}")
            continue

        if role == "assistant":
            resolved_question = str(message.get("question") or last_question or "").strip()

            if resolved_question:
                rebuilt.append(f"Resolved question: {resolved_question}")

            for title in get_message_source_titles(message):
                rebuilt.append(f"Source: {title}")

            last_question = None

    st.session_state[MEMORY_KEY] = rebuilt


def clear_conversation_memory(st):
    # Burahin ang memory ng current Streamlit session.
    st.session_state[MEMORY_KEY] = []
