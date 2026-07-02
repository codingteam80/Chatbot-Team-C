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
    # Read shared query/prompt config. If missing/invalid, use safe fallback cues.
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

    if not isinstance(mode_config, dict):
        mode_config = {}

    return {
        "list_phrases": normalize_config_list(mode_config.get("list_phrases", [])),
        "list_patterns": normalize_config_patterns(mode_config.get("list_patterns", [])),
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

    fallback_cues = [
        "who are",
        "what are",
        "which are",
        "list",
        "enumerate",
        "name all",
        "sino ang mga",
        "ano ang mga",
        "alin ang mga",
    ]

    return any(cue in question for cue in fallback_cues)


def build_question_specific_rules(question):
    # Lightweight list formatting only. No candidate checklist and no strict coverage audit.
    if not is_list_question(question):
        return []

    rules = [
        "LIST QUESTION CHECK:",
        "- Scan all Document Excerpts before answering; do not stop after the first matching excerpt.",
        "- Start with one short direct sentence that names the requested category, actor, relationship, or topic from the QUESTION.",
        "- After the short lead-in sentence, use bullets for multiple distinct supported items.",
        "- Include distinct supported items that directly answer the question.",
        "- If support is partial or unclear, state the limitation briefly instead of overstating it.",
    ]

    for rule in PROMPT_OUTPUT_CONFIG.get("list_answer_format_rules", []):
        rules.append("- " + rule)

    rules.append("")
    return rules


def add_rule_section(lines, title, rules):
    # Add one readable prompt rule section.
    lines.append(f"{title}:")

    for rule in rules:
        lines.append(f"- {rule}")

    lines.append("")


def build_prompt_rules():
    # Balanced document-grounded rules for factual, policy, SOP, and manual answers.
    sections = [
        (
            "CORE RULES",
            [
                "You are InknowVa, a professional document-grounded RAG assistant.",
                "Use only the text under Excerpt as evidence; do not use outside knowledge.",
                "Use CHAT HISTORY only to understand follow-up questions, not as evidence.",
                "Use the same language as the QUESTION.",
                "Treat file names, pages, titles, chunk IDs, source labels, and section labels as metadata, not answer facts unless the QUESTION asks for source or location.",
            ],
        ),
        (
            "ANSWER RULES",
            [
                "Answer the actual QUESTION directly and concisely, but do not make the answer incomplete just to be short.",
                "Match the answer type to the question: who/what needs the direct entity, yes/no needs a clear judgment, and why/how needs a supported reason or process.",
                "For list questions, start with one short lead-in sentence that names the requested category, actor, relationship, or topic from the QUESTION, then use bullets for the distinct supported items.",
                "If multiple Excerpts provide complementary supported details for the same direct question, combine those details concisely instead of using only the first Excerpt.",
                "Use one complete sentence for a single supported answer, bullets after a short lead-in sentence for multiple distinct items, and numbered steps only for ordered procedures or sequences.",
                "Do not add a second version of the answer, a summary paragraph, or a support/evidence section unless the QUESTION asks for it.",
            ],
        ),
        (
            "CLAIM AND EVIDENCE RULES",
            [
                "A claim is supported only when the Excerpt directly supports the same actor, action, object, scope, condition, date/time, status, and relationship asked in the QUESTION.",
                "Do not upgrade partial, related, nearby, passive, circumstantial, or background evidence into an exact claim.",
                "Do not infer motives, causes, responsibilities, permissions, approvals, ownership, membership, dates, or statuses unless they are directly supported.",
                "If the exact claim is not directly supported, state only the supported part and clearly say that the exact claim is not directly supported.",
            ],
        ),
        (
            "FALSE PREMISE AND QUESTION CHECK RULES",
            [
                "Before answering, check whether the QUESTION assumes an unsupported actor, action, role, title, status, requirement, permission, approval, ownership, date, cause, or relationship.",
                "For yes/no questions, answer yes only when the exact claim is directly supported, and no only when the exact claim is directly contradicted.",
                "For why/how questions, verify the premise first. If the premise is not directly supported, correct the premise briefly and stop.",
                "For date/time questions, use the date, schedule, deadline, effective date, expiry date, review date, celebration date, or observed date that directly matches what the QUESTION asks.",
            ],
        ),
        (
            "FALLBACK AND FORMAT RULES",
            [
                "Do not guess, invent, speculate, or fill missing details.",
                "Use the fallback only when the Excerpt has no relevant information to answer, partially answer, or correct the QUESTION.",
                "If there is no relevant information, reply exactly: " + NO_ANSWER_TEXT,
                "Do not mention sources, file names, pages, titles, metadata, excerpts, documents, context, or retrieval details unless the QUESTION asks for source or location.",
                "Do not write phrases like 'the context says', 'the documents indicate', 'the excerpt states', 'according to the excerpt', 'based on the provided context', or 'provided documents'.",
                "Do not add labels such as Support, Evidence, Claim, Final Answer, Answer, Note, or Explanation unless the QUESTION asks for that format.",
                "Return only one final answer body and finish the final sentence completely.",
            ],
        ),
    ]

    lines = []

    for title, rules in sections:
        add_rule_section(lines, title, rules)

    if lines and lines[-1] == "":
        lines.pop()

    return lines



def normalize_prompt_metadata_value(value):
    # Normalize metadata values used only for prompt-control decisions.
    return " ".join(str(value or "").strip().lower().split())


def get_prompt_doc_metadata_values(docs, keys):
    # Collect metadata values from final context docs without treating them as answer facts.
    values = []

    for doc in docs or []:
        metadata = dict(getattr(doc, "metadata", {}) or {})

        for key in keys:
            value = normalize_prompt_metadata_value(metadata.get(key))

            if value and value not in values:
                values.append(value)

    return values


def is_filipino_or_tagalog_question(question):
    # Lightweight language cue for Filipino/Tagalog questions.
    question_key = normalize_question_for_prompt(question)
    filipino_cues = [
        "kailan",
        "sino",
        "ano",
        "alin",
        "bakit",
        "paano",
        "ipinagdiriwang",
        "ginugunita",
        "ng pilipinas",
        "ang ",
        "mga ",
    ]

    return any(cue in question_key for cue in filipino_cues)


def looks_like_date_or_time_question(question):
    question_key = normalize_question_for_prompt(question)
    date_cues = [
        "when",
        "date",
        "birthday",
        "kailan",
        "ipinagdiriwang",
        "ginugunita",
        "schedule",
        "deadline",
        "effective date",
        "expiry",
        "expiration",
    ]

    return any(cue in question_key for cue in date_cues)


def build_context_control_rules(question, docs):
    # Extra instructions derived only from system-detected final-context metadata.
    # These are not test hints; they are produced by the retrieval/context-filter layer.
    modes = set(get_prompt_doc_metadata_values(docs, ["context_mode"]))
    intents = set(get_prompt_doc_metadata_values(docs, ["dynamic_answer_intent", "answer_intent"]))
    rules = []

    if "false_premise" in modes:
        rules.extend([
            "SYSTEM-DETECTED PREMISE RISK:",
            "- The selected evidence was marked as false_premise by the context filter.",
            "- Do not answer the why/how question as if its premise is true.",
            "- Start by saying the premise is not directly supported, then give only the supported correction.",
            "- Do not invent a motive, reason, honorary title, role, membership, or status.",
            "",
        ])

    if "date_fact" in intents or looks_like_date_or_time_question(question):
        rules.extend([
            "DATE/TIME ANSWER CONTROL:",
            "- Put the exact date/time from the Excerpt first.",
            "- Do not replace a full date with only a year unless the QUESTION asks only for a year.",
            "- If the Excerpt contains both a date and the event/significance, answer both parts directly.",
            "",
        ])

    if is_filipino_or_tagalog_question(question):
        rules.extend([
            "FILIPINO/TAGALOG ANSWER CONTROL:",
            "- Answer in simple, natural Filipino/Tagalog.",
            "- Preserve exact dates, names, and proper nouns from the Excerpt.",
            "- Do not invent or use awkward translated words; use simple phrasing instead.",
            "",
        ])

    if not rules:
        return []

    return rules


def build_rag_prompt(
    question,
    docs,
    chat_history="",
    max_doc_chars=MAX_DOC_CHARS,
    max_context_chars=MAX_PROMPT_CONTEXT_CHARS,
):
    # Final prompt passed to the LLM. One prompt only; no candidate checklist/validator.
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
        *build_context_control_rules(question, docs),
        "QUESTION:",
        str(question or "").strip(),
        "",
        *build_question_specific_rules(question),
        "FINAL ANSWER CHECK:",
        "Give one final answer only. Verify actor, action, object, scope, condition, date/time, status, and relationship against the QUESTION. If the premise or exact claim is not directly supported, correct it briefly. For list questions, start with one short direct sentence using the QUESTION's key topic/category, then use bullets for distinct supported items. For single direct questions, answer in one complete sentence, not just a label. Do not speculate, do not add labels, and do not repeat the answer.",
        "",
        "FINAL ANSWER:",
    ]

    return "\n".join(prompt_lines)
