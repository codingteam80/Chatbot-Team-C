import html
import re
from pathlib import Path

from config.settings import MAX_CONTEXT_CHARS, MAX_PER_SOURCE, PREVIEW_CHARS, RERANK_TOP_N
from llm.prompt_builder import build_rag_prompt
from retrieval.context_filter import select_final_context_docs


NO_SOURCE = "Unknown source"
EMPTY_VALUES = {"", "none", "nan", "null"}
SCORE_KEYS = [
    "semantic_score",
    "hybrid_score",
    "rerank_score",
    "quality_score",
]


def is_empty_value(value):
    # Check kung empty o invalid ang value.
    if value is None:
        return True

    return str(value).strip().lower() in EMPTY_VALUES


def extract_text(response):
    # Kunin text mula sa LangChain response/chunk.
    if hasattr(response, "content"):
        return response.content

    return str(response)


def clean_generated_answer(answer, question=""):
    # Tanggalin common local LLM artifacts.
    answer = str(answer or "").strip()
    question = str(question or "").strip()

    if not answer:
        return ""

    if question and answer.lower().startswith(question.lower()):
        answer = answer[len(question):].strip()

    prefixes = [
        "ANSWER:",
        "Answer:",
        "FINAL ANSWER:",
        "Final answer:",
        "SAGOT:",
        "Sagot:",
        "Response:",
    ]

    changed = True
    while changed:
        changed = False

        for prefix in prefixes:
            if answer.startswith(prefix):
                answer = answer[len(prefix):].strip()
                changed = True

    return answer.strip()


def print_prompt_debug(prompt):
    # Print prompt preview kapag debug mode.
    print("\n" + "=" * 60)
    print("PROMPT SENT TO LLM")
    print("=" * 60)
    print(prompt[:5000])



def prepare_context_docs(question, docs):
    # Piliin ang final context bago gumawa ng prompt.
    # Para sa cross-doc questions, may source diversity bago pumasok sa LLM.
    return select_final_context_docs(
        reranked_docs=docs or [],
        question=question,
        top_n=RERANK_TOP_N,
        max_chars=MAX_CONTEXT_CHARS,
        max_per_source=MAX_PER_SOURCE,
    )


def build_prompt(
    question,
    docs,
    chat_history="",
    strict_assumption_check=False,
    correction_retry=False,
):
    # Gumawa ng final RAG prompt.
    context_docs = prepare_context_docs(
        question=question,
        docs=docs,
    )

    return build_rag_prompt(
        question=question,
        docs=context_docs,
        chat_history=chat_history,
        strict_assumption_check=strict_assumption_check,
        correction_retry=correction_retry,
    )


def generate_answer(
    question,
    docs,
    llm,
    chat_history="",
    debug=False,
    strict_assumption_check=False,
    correction_retry=False,
):
    # Non-streaming answer generation.
    prompt = build_prompt(
        question=question,
        docs=docs,
        chat_history=chat_history,
        strict_assumption_check=strict_assumption_check,
        correction_retry=correction_retry,
    )

    if debug:
        print_prompt_debug(prompt)

    response = llm.invoke(prompt)

    return clean_generated_answer(
        answer=extract_text(response),
        question=question,
    )


def stream_answer(
    question,
    docs,
    llm,
    chat_history="",
    debug=False,
    strict_assumption_check=False,
    correction_retry=False,
):
    # Streaming answer generation.
    prompt = build_prompt(
        question=question,
        docs=docs,
        chat_history=chat_history,
        strict_assumption_check=strict_assumption_check,
        correction_retry=correction_retry,
    )

    if debug:
        print_prompt_debug(prompt)

    for chunk in llm.stream(prompt):
        yield extract_text(chunk)


def clean_preview_text(text, limit=PREVIEW_CHARS):
    # Linisin at paikliin ang source preview.
    if not text:
        return ""

    text = html.unescape(str(text))
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)

    for symbol in ["#", "**", "__", "`"]:
        text = text.replace(symbol, "")

    text = " ".join(text.split())

    if len(text) > limit:
        return text[:limit].rstrip() + "..."

    return text


def normalize_page(page):
    # Gawing safe ang page value.
    if is_empty_value(page):
        return "N/A"

    return str(page).strip()


def normalize_source(raw_source):
    # Gawing safe ang source path/name.
    if is_empty_value(raw_source):
        return NO_SOURCE

    return str(raw_source).strip()


def get_source_display_name(raw_source):
    # File stem lang para malinis sa UI.
    raw_source = normalize_source(raw_source)

    if raw_source == NO_SOURCE:
        return NO_SOURCE

    return Path(raw_source).stem


def get_source_file_name(raw_source):
    # File name with extension para sa debug/evaluation.
    raw_source = normalize_source(raw_source)

    if raw_source == NO_SOURCE:
        return NO_SOURCE

    return Path(raw_source).name


def get_doc_key(raw_source, page, preview, metadata):
    # Stable key para maiwasan duplicate source cards.
    chunk_id = metadata.get("chunk_id") or metadata.get("chunk_index")

    if chunk_id is not None:
        return (raw_source, page, chunk_id)

    return (raw_source, page, preview[:120])


def add_scores(item, metadata):
    # Idagdag optional scores kapag meron.
    for key in SCORE_KEYS:
        if key in metadata:
            item[key] = metadata[key]


def get_sources(docs, preview_limit=PREVIEW_CHARS):
    # Convert retrieved docs into UI source cards.
    sources = []
    seen = set()

    for doc in docs or []:
        metadata = dict(doc.metadata or {})

        raw_source = normalize_source(metadata.get("source"))
        page = normalize_page(metadata.get("page"))
        preview = clean_preview_text(doc.page_content, limit=preview_limit)

        item_key = get_doc_key(
            raw_source=raw_source,
            page=page,
            preview=preview,
            metadata=metadata,
        )

        if item_key in seen:
            continue

        seen.add(item_key)

        source_name = get_source_display_name(raw_source)
        file_name = get_source_file_name(raw_source)

        item = {
            "source": source_name,
            "title": source_name,
            "file_name": file_name,
            "source_path": raw_source,
            "file_path": raw_source,
            "path": raw_source,
            "page": page,
            "preview": preview,
            "metadata": metadata,
        }

        add_scores(item, metadata)
        sources.append(item)

    return sources
