from config.settings import MAX_HISTORY_CHARS, MEMORY_KEY


ROLE_LABELS = {
    "user": "User",
    "assistant": "Assistant",
}


def normalize_space(text):
    # Collapse whitespace so memory stays compact and prompt-friendly.
    return " ".join(str(text or "").split()).strip()


def trim_history_text(text, max_chars=MAX_HISTORY_CHARS):
    # Keep the most recent turns if the session becomes long.
    text = str(text or "").strip()

    if not max_chars or len(text) <= max_chars:
        return text

    return text[-max_chars:].lstrip()


def get_source_title(source):
    # Keep this helper for older imports, but sources are not used as rewrite memory.
    if not isinstance(source, dict):
        return ""

    metadata = source.get("metadata") if isinstance(source.get("metadata"), dict) else {}

    title = (
        source.get("title")
        or source.get("source")
        or source.get("file_name")
        or source.get("filename")
        or metadata.get("title")
        or metadata.get("source")
        or metadata.get("file_name")
        or metadata.get("filename")
        or ""
    )

    return normalize_space(title)


def build_memory_lines_from_messages(messages):
    # Convert saved visible messages into User/Assistant lines only.
    lines = []

    for message in messages or []:
        if not isinstance(message, dict):
            continue

        role = str(message.get("role") or "").strip().lower()
        label = ROLE_LABELS.get(role)
        content = normalize_space(message.get("content", ""))

        if label and content:
            lines.append(f"{label}: {content}")

    return lines


def history_lines_to_text(lines, limit=None, max_chars=MAX_HISTORY_CHARS):
    # Turn memory lines into prompt-ready text.
    lines = [normalize_space(line) for line in lines or [] if normalize_space(line)]

    if limit is not None:
        lines = lines[-int(limit):]

    return trim_history_text("\n".join(lines), max_chars=max_chars)


def init_memory(st):
    # Create the compatibility memory list once per Streamlit session.
    st.session_state.setdefault(MEMORY_KEY, [])


def get_conversation_history(st, limit=None):
    # Prefer visible chat messages, then fallback to compatibility memory.
    init_memory(st)

    messages = st.session_state.get("messages", [])
    lines = build_memory_lines_from_messages(messages)

    if not lines:
        lines = st.session_state.get(MEMORY_KEY, [])

    return history_lines_to_text(lines, limit=limit)


def save_to_memory(st, question, answer, sources=None):
    # Keep sources parameter for older call sites, but do not store source titles.
    init_memory(st)

    question = normalize_space(question)
    answer = normalize_space(answer)

    if question:
        st.session_state[MEMORY_KEY].append(f"User: {question}")

    if answer:
        st.session_state[MEMORY_KEY].append(f"Assistant: {answer}")


def rebuild_conversation_memory(st, messages):
    # Rebuild compatibility memory from the visible chat.
    st.session_state[MEMORY_KEY] = build_memory_lines_from_messages(messages)


def clear_conversation_memory(st):
    # Clear memory for the current Streamlit browser session.
    st.session_state[MEMORY_KEY] = []
