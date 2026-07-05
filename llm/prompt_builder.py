import json
import re
from pathlib import Path

from config.settings import (
    MAX_DOC_CHARS,
    MAX_PROMPT_CONTEXT_CHARS,
    NO_ANSWER_TEXT,
)


EMPTY_VALUES = {"", "none", "nan", "null"}
DEFAULT_PROMPT_CONFIG_PATH = Path("config") / "query_expansion_config.json"


def read_prompt_json(config_path=DEFAULT_PROMPT_CONFIG_PATH):
    # Read shared query/prompt config. If missing/invalid, use an empty config.
    try:
        config_path = Path(config_path)

        if not config_path.exists():
            return {}

        return json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return {}


def normalize_config_list(values):
    if isinstance(values, str):
        values = [values]

    if not isinstance(values, list):
        return []

    cleaned_values = []

    for value in values:
        value = str(value or "").strip().lower()
        value = " ".join(value.split())

        if value and value not in cleaned_values:
            cleaned_values.append(value)

    return cleaned_values


def normalize_config_patterns(values):
    if isinstance(values, str):
        values = [values]

    if not isinstance(values, list):
        return []

    cleaned_values = []

    for value in values:
        value = str(value or "").strip()

        if value and value not in cleaned_values:
            cleaned_values.append(value)

    return cleaned_values


def load_prompt_mode_detection_config():
    # List/enumeration cues come from JSON so Python stays generic.
    raw_config = read_prompt_json()
    mode_config = raw_config.get("mode_detection", {}) if isinstance(raw_config, dict) else {}
    prompt_config = raw_config.get("prompt_builder", {}) if isinstance(raw_config, dict) else {}

    if not isinstance(mode_config, dict):
        mode_config = {}

    if not isinstance(prompt_config, dict):
        prompt_config = {}

    return {
        "list_phrases": normalize_config_list(mode_config.get("list_phrases", [])),
        "list_patterns": normalize_config_patterns(mode_config.get("list_patterns", [])),
        "list_question_fallback_cues": normalize_config_list(prompt_config.get("list_question_fallback_cues", [])),
    }


def load_prompt_output_config():
    # Optional lightweight output-format rules from JSON.
    raw_config = read_prompt_json()

    if not isinstance(raw_config, dict):
        return {"list_answer_format_rules": []}

    answer_output = raw_config.get("answer_output", {})
    list_output = answer_output.get("list_answer", {}) if isinstance(answer_output, dict) else {}

    if not isinstance(list_output, dict):
        list_output = {}

    return {
        "list_answer_format_rules": normalize_config_list(list_output.get("format_rules", [])),
    }


PROMPT_OUTPUT_CONFIG = load_prompt_output_config()
PROMPT_MODE_CONFIG = load_prompt_mode_detection_config()


def clean_value(value, default_value="N/A"):
    # Return a safe display value for prompt text.
    if value is None:
        return default_value

    text = str(value).strip()

    if not text or text.lower() in EMPTY_VALUES:
        return default_value

    return text


def get_file_name(source):
    # Keep only the file name so the prompt stays readable.
    source = clean_value(source, "Unknown source")

    if source == "Unknown source":
        return source

    return Path(source).name


def trim_text(text, max_chars):
    # Trim long chunks so the prompt does not become too large.
    text = " ".join(str(text or "").split())

    if len(text) <= max_chars:
        return text

    return text[:max_chars].rstrip() + "..."


def build_context_block(doc, index, max_doc_chars=MAX_DOC_CHARS):
    # Excerpt content is evidence; metadata is only for source/location questions.
    metadata = dict(getattr(doc, "metadata", {}) or {})

    source = get_file_name(metadata.get("source") or metadata.get("file_name"))
    page = clean_value(metadata.get("page"), "N/A")
    title = clean_value(metadata.get("title"), "N/A")
    content = trim_text(getattr(doc, "page_content", ""), max_doc_chars)

    reference = " | ".join([
        f"file={source}",
        f"page={page}",
        f"title={title}",
    ])

    return "\n".join([
        f"[Document Excerpt {index}]",
        f"Reference metadata for source/location questions only, not answer facts: {reference}",
        "Excerpt:",
        content,
    ])


def build_context(docs, max_doc_chars=MAX_DOC_CHARS, max_context_chars=MAX_PROMPT_CONTEXT_CHARS):
    # Combine final retrieved documents into one context block.
    if not docs:
        return "No document excerpts provided."

    blocks = []
    total_chars = 0

    for index, doc in enumerate(docs, start=1):
        block = build_context_block(
            doc=doc,
            index=index,
            max_doc_chars=max_doc_chars,
        )

        if blocks and total_chars + len(block) > max_context_chars:
            break

        blocks.append(block)
        total_chars += len(block)

    if not blocks:
        return "No document excerpts provided."

    return "\n\n".join(blocks)


def build_chat_history(chat_history):
    # Chat history is only for follow-up understanding, not evidence.
    text = str(chat_history or "").strip()

    if not text:
        return "No previous conversation."

    return text


def normalize_question_for_prompt(question):
    # Normalize question text for lightweight prompt routing.
    return " ".join(str(question or "").lower().split())


def is_list_question(question):
    # Detect broad list/enumeration questions using JSON-configured cues.
    question = normalize_question_for_prompt(question)

    if not question:
        return False

    for cue in PROMPT_MODE_CONFIG.get("list_phrases", []):
        if cue and cue in question:
            return True

    for pattern in PROMPT_MODE_CONFIG.get("list_patterns", []):
        try:
            if re.search(pattern, question, flags=re.IGNORECASE):
                return True
        except re.error:
            continue

    for cue in PROMPT_MODE_CONFIG.get("list_question_fallback_cues", []):
        if cue and cue in question:
            return True

    return False


def build_question_specific_rules(question):
    # Keep the main prompt simple. Generic list/false-premise/short-query rules are already in build_prompt_rules().
    return []


def add_rule_section(lines, title, rules):
    # Add one readable prompt rule section.
    lines.append(f"{title}:")

    for rule in rules:
        lines.append(f"- {rule}")

    lines.append("")


def build_prompt_rules():
    # Simple generic prompt rules. Keep the assistant grounded without over-filtering valid answers.
    fallback_text = str(NO_ANSWER_TEXT or "I cannot find the answer in the provided documents.").strip()

    return [
        "You are InknowVa, a document-grounded assistant.",
        "",
        "Use only the provided excerpts as evidence.",
        "Answer in the same language as the question.",
        "Answer directly and concisely.",
        "Do not use outside knowledge.",
        "",
        "If the excerpts contain relevant information, answer using that information.",
        f'Use "{fallback_text}" only when all excerpts have no relevant information for answering, partially answering, or correcting the question.',
        "",
        "For list questions:",
        "- Read all excerpts.",
        "- Include all distinct items that are clearly listed or clearly connected to the requested subject.",
        "- Use bullets.",
        "- Do not drop listed items just because the exact relationship wording is not repeated beside every item.",
        "- Do not add items that are only background, references, source titles, or unrelated mentions.",
        "",
        "For false-premise questions:",
        "- First check whether the exact role, title, action, status, date, or relationship in the question is supported.",
        '- If the exact premise is not supported, start with: "The premise is not supported."',
        '- Do not answer "why" or "how" as if the unsupported premise is true.',
        "- Do not replace the asked role/title/action with a different related role/title/action.",
        "- After rejecting the premise, give the corrected supported fact if available.",
        "",
        "For yes/no or actor questions:",
        "- If the exact personal actor is not stated, do not claim a person personally did the action.",
        "- If the excerpt only supports a group, army, team, organization, process, or event, answer using that careful wording.",
        "- If the excerpt supports only part of the claim, state the supported part and clearly say what is not stated.",
        "",
        "For short keyword, name, acronym, code, title, or date queries:",
        "- Identify what the query refers to in the excerpts.",
        "- Do not return an empty answer.",
        "- If a date appears with a person, event, policy, deadline, effective date, or record, say what the date is associated with.",
        "",
        "For Tagalog/Filipino questions:",
        "- Answer in simple Filipino.",
        "- Keep names, dates, titles, and facts exact.",
        "- Do not mistranslate events, dates, or relationships.",
        "",
        "Formatting:",
        "- One complete sentence for single-fact answers.",
        "- Bullets for list answers.",
        '- No labels like "Answer:", "Evidence:", or "Final Answer:".',
        "- Do not mention excerpts, context, retrieval, chunks, or metadata unless the user asks for sources.",
    ]


def build_rag_prompt(
    question,
    docs,
    chat_history="",
    max_doc_chars=MAX_DOC_CHARS,
    max_context_chars=MAX_PROMPT_CONTEXT_CHARS,
):
    # Final prompt passed to the LLM.
    context = build_context(
        docs=docs,
        max_doc_chars=max_doc_chars,
        max_context_chars=max_context_chars,
    )

    prompt_lines = [
        *build_prompt_rules(),
        "",
        "CHAT HISTORY:",
        build_chat_history(chat_history),
        "",
        "DOCUMENT EXCERPTS:",
        context,
        "",
        "QUESTION:",
        str(question or "").strip(),
        "",
        *build_question_specific_rules(question),
        "FINAL ANSWER:",
    ]

    return "\n".join(prompt_lines)
