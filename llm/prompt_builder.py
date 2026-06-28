from pathlib import Path

from config.settings import (
    MAX_DOC_CHARS,
    MAX_PROMPT_CONTEXT_CHARS,
    NO_ANSWER_TEXT,
)


EMPTY_VALUES = {"", "none", "nan", "null"}


def safe_metadata_value(value, default_value):
    # Safe display value para sa metadata.
    if value is None:
        return default_value

    value = str(value).strip()

    if not value or value.lower() in EMPTY_VALUES:
        return default_value

    return value


def get_source_name(source):
    # File name lang ang ilagay sa context para readable.
    source = safe_metadata_value(source, "Unknown source")

    if source == "Unknown source":
        return source

    return Path(source).name


def trim_text(text, max_chars):
    # Putulin ang text kapag lampas sa limit.
    text = str(text or "").strip()

    if len(text) <= max_chars:
        return text

    return text[:max_chars].rstrip() + "..."


def build_context_block(doc, index, max_doc_chars=MAX_DOC_CHARS):
    # Gawing structured context block ang isang document chunk.
    metadata = doc.metadata or {}

    source = get_source_name(metadata.get("source"))
    page = safe_metadata_value(metadata.get("page"), "N/A")
    chunk = safe_metadata_value(metadata.get("chunk_index"), "N/A")
    content = trim_text(doc.page_content, max_doc_chars)

    return "\n".join([
        f"[Source {index}]",
        f"File: {source}",
        f"Page: {page}",
        f"Chunk: {chunk}",
        "Content:",
        content,
    ])


def build_context(docs, max_doc_chars=MAX_DOC_CHARS, max_context_chars=MAX_PROMPT_CONTEXT_CHARS):
    # Pagsamahin ang retrieved docs bilang final context.
    if not docs:
        return "No context provided."

    context_blocks = []
    total_chars = 0

    for index, doc in enumerate(docs, start=1):
        block = build_context_block(
            doc=doc,
            index=index,
            max_doc_chars=max_doc_chars,
        )

        if total_chars + len(block) > max_context_chars:
            if not context_blocks:
                context_blocks.append(block)
            break

        context_blocks.append(block)
        total_chars += len(block)

    if not context_blocks:
        return "No context provided."

    return "\n\n".join(context_blocks)


def build_chat_history(chat_history):
    # Chat history ay pang-follow-up lang, hindi evidence.
    if not chat_history:
        return "No previous conversation."

    return str(chat_history).strip()


def build_prompt_rules(strict_assumption_check=False, correction_retry=False):
    # Main rules: generic para gumana sa manuals, SOPs, policies, reports, guides, at KB articles.
    rules = [
        "You are a document-grounded RAG assistant.",
        "Use only facts directly supported by the CONTEXT.",
        "Equivalent wording, translation, or paraphrase is allowed if the meaning is clearly supported by the CONTEXT.",
        "Do not require the CONTEXT to use the exact same wording as the QUESTION.",
        "Do not use outside knowledge, memory, assumptions, or common knowledge.",
        "Use CHAT HISTORY only to understand follow-up references, not as evidence.",
        "Answer the exact QUESTION only and ignore unrelated CONTEXT.",
        "Before answering, verify the main claim assumed by the QUESTION against the CONTEXT.",
        "Do not explain an assumed claim as true unless the CONTEXT clearly supports the same meaning, even if the wording is different.",
        "A correction is supported when the CONTEXT states a different correct person, role, owner, date, step, requirement, or responsible party.",
        "Do not require the CONTEXT to explicitly say 'not' or 'hindi' before correcting a false premise.",
        "If the QUESTION assigns a role/action to X but the CONTEXT assigns that same role/action to Y, start with 'No.' or 'Hindi.' and say Y is correct, not X.",
        "When correcting a false premise, the first sentence must contain a clear correction word such as 'No.', 'Hindi.', 'not correct', 'did not', or 'was not'.",
        "Do not only state the corrected fact; explicitly say that the assumed role, action, date, owner, or responsibility in the question is not supported.",
        "False-premise correction has priority over the fallback answer.",
        "Use the fallback answer only when the CONTEXT has no answer and no supported correction.",
        "If the CONTEXT gives partial evidence, answer the supported part first, then clearly say which requested part is not stated or not provided.",
        "For multi-part questions joined by words like and/at/also, evaluate each requested part separately.",
        "A missing requested detail must not erase a supported answer to another part of the same question.",
        "If the CONTEXT contains a related attribute but not the exact requested compound attribute, do not substitute it as the answer.",
        "Example behavior: if the question asks for a specific compound attribute but the CONTEXT only mentions a different related value, say the requested exact attribute is not stated.",
        "Do not use the fallback answer for the whole question when at least one requested part is supported by the CONTEXT.",
        "Give a complete answer, not only a single word or title. Use 1 to 3 concise sentences unless the question asks for a list.",
        "For identification questions, name the answer and include one supported detail explaining why it is the answer.",
        "When the question asks to distinguish between similar topics, explicitly state which option is correct and which option is not supported.",
        "Preserve exact names, dates, numbers, roles, requirements, steps, approvals, thresholds, and technical terms from the CONTEXT.",
        "For procedures, rules, criteria, or listed items, use bullets or numbered steps and keep the same order as the CONTEXT.",
        f"Use the same language as the QUESTION; if no answer or correction is supported, reply exactly: {NO_ANSWER_TEXT}",
        "Do not repeat the QUESTION, add citations, or use labels like QUESTION, ANSWER, or FINAL ANSWER.",
    ]

    if strict_assumption_check:
        rules.extend([
            "For why/how/bakit/paano questions, first verify that the assumed event, role, owner, date, or responsibility is true.",
            "If the CONTEXT supports a different role/owner/responsible party, correct the premise instead of using the fallback answer.",
            "If the question asks why/how something happened but the CONTEXT shows it did not happen, start with 'No.' or 'Hindi.' and explain the supported correction.",
            "If a why/how question assumes that an entity performed an action but the CONTEXT lists other actors for that action, say that the assumption is not supported.",
        ])

    if correction_retry:
        rules.extend([
            "Correction retry: re-check the CONTEXT for a supported contrast before using the fallback answer.",
            "If the wrong name is not explicitly negated but another correct name owns the same role/action, give the correction.",
            "Do not output the fallback answer when the CONTEXT contains the correct founder, leader, date, event, role, or responsible party that contradicts the premise.",
            "A valid correction can be based on a supported positive fact, such as a different owner, founder, approver, date, role, step, or responsible party, even when the CONTEXT does not explicitly mention the wrong premise.",
            "The retry answer must not be only a factual replacement; it must include an explicit correction phrase in the first sentence.",
        ])

    return rules


def build_rag_prompt(
    question,
    docs,
    chat_history="",
    max_doc_chars=MAX_DOC_CHARS,
    max_context_chars=MAX_PROMPT_CONTEXT_CHARS,
    strict_assumption_check=False,
    correction_retry=False,
):
    # Gumawa ng final prompt para sa LLM.
    context = build_context(
        docs=docs,
        max_doc_chars=max_doc_chars,
        max_context_chars=max_context_chars,
    )

    prompt_lines = [
        *build_prompt_rules(
            strict_assumption_check=strict_assumption_check,
            correction_retry=correction_retry,
        ),
        "",
        "CHAT HISTORY:",
        build_chat_history(chat_history),
        "",
        "CONTEXT:",
        context,
        "",
        "QUESTION:",
        str(question or "").strip(),
        "",
        "FINAL ANSWER:",
    ]

    return "\n".join(prompt_lines)
