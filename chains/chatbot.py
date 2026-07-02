import inspect
import json
import re
from datetime import datetime
from pathlib import Path

from config.settings import (
    BM25_K,
    DATA_PATH,
    ENABLE_FALLBACK_RETRY,
    ENABLE_MULTI_QUERY_RETRIEVAL,
    ENABLE_QUESTION_REWRITE,
    ENABLE_TRUNCATION_RETRY,
    HYBRID_FINAL_K,
    MAX_CANDIDATES_BEFORE_RERANK,
    MAX_RETRIEVAL_QUERIES,
    NO_ANSWER_TEXT,
    RERANK_POOL_TOP_N,
    RERANK_TOP_N,
    SEMANTIC_K,
)

try:
    import streamlit as st
except ImportError:
    st = None

from embeddings.embedding_model import get_embedding_model
from vectorstore.chroma_store import load_chroma_vectorstore

from utils.bm25_cache import load_or_create_bm25
from utils.chunk_cache import load_or_create_chunks

from retrieval.context_filter import normalize_text
from retrieval.hybrid_retriever import hybrid_search
from retrieval.reranker import load_reranker, rerank_documents

try:
    from retrieval.query_analyzer import analyze_query
except ImportError:
    analyze_query = None

from llm.ollama_llm import load_llm
from chains.rag_chain import (
    build_prompt_from_context,
    clean_generated_answer,
    extract_text as extract_llm_text,
    generate_answer_with_context,
    get_sources,
    prepare_context_docs,
)

from memory.question_rewriter import is_follow_up_question, rewrite_question


MIN_SEARCHABLE_TOKEN_LENGTH = 2
FALLBACK_RETRY_SKIP_CATEGORIES = {"NEGATIVE_UNANSWERABLE", "FOLLOW_UP_AMBIGUOUS"}
TERMINAL_PUNCTUATION = (".", "?", "!", ")", "]", "}", '"', "'")

DEFAULT_QUERY_CONFIG_PATH = Path("config") / "query_expansion_config.json"


def read_query_config(config_path=DEFAULT_QUERY_CONFIG_PATH):
    # Read shared JSON config for generic query guards.
    try:
        config_path = Path(config_path)

        if not config_path.exists():
            return {}

        return json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return {}


def get_nested_config(data, *keys):
    current = data

    for key in keys:
        if not isinstance(current, dict):
            return {}

        current = current.get(key, {})

    return current if isinstance(current, dict) else {}


def normalize_config_list(values):
    if isinstance(values, str):
        values = [values]

    if not isinstance(values, list):
        return []

    cleaned = []

    for value in values:
        value = str(value or "").strip()

        if value and value not in cleaned:
            cleaned.append(value)

    return cleaned


def config_bool(config, key, default_value=False):
    value = config.get(key, default_value)

    if isinstance(value, bool):
        return value

    if value is None:
        return default_value

    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def config_int(config, key, default_value):
    try:
        return int(config.get(key, default_value))
    except (TypeError, ValueError):
        return default_value


def load_value_only_question_config():
    raw_config = read_query_config()
    guard_config = raw_config.get("value_only_question", {})

    if not isinstance(guard_config, dict):
        guard_config = {}

    short_value_config = get_nested_config(raw_config, "value_only_question", "short_value_answer")
    language_markers = get_nested_config(raw_config, "language_markers")
    messages = guard_config.get("clarification_messages", {})

    if not isinstance(messages, dict):
        messages = {}

    return {
        "enabled": config_bool(guard_config, "enabled", True),
        "max_value_words": config_int(
            guard_config,
            "max_value_words",
            config_int(short_value_config, "max_words", 6),
        ),
        "question_prefix_patterns": normalize_config_list(guard_config.get("question_prefix_patterns")),
        "value_patterns": normalize_config_list(
            guard_config.get("value_patterns") or short_value_config.get("patterns")
        ),
        "tagalog_markers": set(normalize_config_list(language_markers.get("tagalog_markers"))),
        "clarification_messages": messages,
    }


VALUE_ONLY_QUESTION_CONFIG = load_value_only_question_config()


def normalize_space(text):
    return " ".join(str(text or "").split()).strip()


def word_count(text):
    return len(re.findall(r"\S+", normalize_space(text)))


def extract_value_only_question_value(question):
    # Extract the value from generic questions like "What is <date/value>?".
    if not VALUE_ONLY_QUESTION_CONFIG.get("enabled", True):
        return ""

    text = normalize_space(question)

    if not text:
        return ""

    for pattern in VALUE_ONLY_QUESTION_CONFIG.get("question_prefix_patterns", []):
        try:
            match = re.search(pattern, text, flags=re.IGNORECASE)
        except re.error:
            continue

        if not match:
            continue

        if match.groups():
            return normalize_space(match.group(1)).strip(" ?.!,:;\"'")

    return ""


def looks_like_configured_short_value(value):
    # The actual value patterns live in config/query_expansion_config.json.
    value = normalize_space(value).strip(" ?.!,:;\"'")

    if not value:
        return False

    max_words = VALUE_ONLY_QUESTION_CONFIG.get("max_value_words", 6)

    if word_count(value) > max_words:
        return False

    for pattern in VALUE_ONLY_QUESTION_CONFIG.get("value_patterns", []):
        try:
            if re.search(pattern, value, flags=re.IGNORECASE):
                return True
        except re.error:
            continue

    return False


def looks_like_value_only_question(question):
    value = extract_value_only_question_value(question)
    return bool(value and looks_like_configured_short_value(value))


def is_tagalog_text(text):
    tokens = set(re.findall(r"[a-zA-ZÀ-ÿ']+", str(text or "").lower()))
    return bool(tokens.intersection(VALUE_ONLY_QUESTION_CONFIG.get("tagalog_markers", set())))


def get_value_only_question_clarification(question):
    messages = VALUE_ONLY_QUESTION_CONFIG.get("clarification_messages", {}) or {}

    if is_tagalog_text(question):
        return messages.get("Tagalog") or "Anong tao, event, policy, o topic ang tinutukoy mo?"

    return messages.get("English") or "Which person, event, policy, or topic do you mean?"



def cache_chatbot_resource(show_spinner=None):
    # Kapag nasa Streamlit app, gamitin ang st.cache_resource.
    if st is not None:
        return st.cache_resource(show_spinner=show_spinner)

    # Kapag test script lang at walang Streamlit, hayaan lang tumakbo normally.
    def decorator(function):
        return function

    return decorator


def build_response(answer, sources=None, documents=None):
    # Standard response format para sa UI.
    return {
        "answer": answer,
        "sources": sources or [],
        "documents": documents or [],
    }


def print_ui_rag_debug(message):
    # Debug print para makita sa terminal kung saan nagfa-fallback ang UI RAG path.
    print(f"UI RAG DEBUG: {message}", flush=True)


def print_debug_docs(docs, label="docs", preview_chars=500):
    # Print docs/chunks para makita kung ano ang ginagamit.
    docs = docs or []
    print_ui_rag_debug(f"{label} count = {len(docs)}")

    for index, doc in enumerate(docs, start=1):
        metadata = getattr(doc, "metadata", {}) or {}
        preview = " ".join(str(getattr(doc, "page_content", "") or "").split())[:preview_chars]

        print_ui_rag_debug(f"{label} #{index}")
        print_ui_rag_debug(f"  source       = {metadata.get('source')}")
        print_ui_rag_debug(f"  file_name    = {metadata.get('file_name')}")
        print_ui_rag_debug(f"  page         = {metadata.get('page')}")
        print_ui_rag_debug(f"  chunk_index  = {metadata.get('chunk_index', metadata.get('chunk_id'))}")
        print_ui_rag_debug(f"  hybrid       = {metadata.get('hybrid_score')}")
        print_ui_rag_debug(f"  rerank       = {metadata.get('rerank_score')}")
        print_ui_rag_debug(f"  context_mode = {metadata.get('context_mode')}")
        print_ui_rag_debug(f"  preview      = {preview}")


UI_RAG_FILE_DEBUG = True
# Use an absolute path based on the project root, not the terminal working directory.
# chatbot.py is inside chains/, so parents[1] is the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
UI_RAG_DEBUG_FILE = PROJECT_ROOT / "reports" / "ui_rag_debug.txt"
UI_RAG_DEBUG_PREVIEW_CHARS = 900
UI_RAG_DEBUG_MAX_DOCS = 30

print(f"ACTIVE CHATBOT FILE: {Path(__file__).resolve()}", flush=True)
print(f"DEBUG FILE TARGET: {UI_RAG_DEBUG_FILE.resolve()}", flush=True)


def append_ui_rag_debug(lines):
    # Append UI RAG debug logs to a file without breaking the chatbot flow.
    if not UI_RAG_FILE_DEBUG:
        return

    try:
        UI_RAG_DEBUG_FILE.parent.mkdir(parents=True, exist_ok=True)

        with UI_RAG_DEBUG_FILE.open("a", encoding="utf-8") as file:
            file.write("\n".join(str(line) for line in lines))
            file.write("\n")

        title = ""
        for line in lines or []:
            line = str(line or "").strip()
            if line and set(line) != {"="}:
                title = line[:90]
                break

        print_ui_rag_debug(f"Wrote debug file: {UI_RAG_DEBUG_FILE.resolve()} | {title}")
    except Exception as error:
        print_ui_rag_debug(f"Failed to write UI debug file: {error}")


def add_ui_debug_section(lines, title):
    # Add a clear section header to the UI debug file.
    lines.append("")
    lines.append("=" * 80)
    lines.append(title)
    lines.append("=" * 80)


def get_ui_debug_source_label(doc):
    # Human-readable stable source label for debug comparison.
    metadata = getattr(doc, "metadata", {}) or {}
    source = metadata.get("file_name") or metadata.get("source") or "Unknown source"
    page = metadata.get("page", "N/A")
    chunk_id = metadata.get("chunk_id") or metadata.get("chunk_index") or "Unknown chunk"
    return f"{source} | page={page} | chunk_id={chunk_id}"


def get_ui_debug_preview(doc, max_chars=UI_RAG_DEBUG_PREVIEW_CHARS):
    # Compact one-line preview of a chunk.
    text = " ".join(str(getattr(doc, "page_content", "") or "").split())

    if len(text) <= max_chars:
        return text

    return text[:max_chars].rstrip() + "..."


def add_ui_debug_docs(lines, title, docs, max_docs=UI_RAG_DEBUG_MAX_DOCS):
    # Add retrieved/ranked/final docs to the UI debug file.
    add_ui_debug_section(lines, title)

    docs = list(docs or [])
    lines.append(f"Count: {len(docs)}")

    if not docs:
        lines.append("No documents.")
        return

    score_keys = [
        "semantic_score",
        "semantic_distance",
        "semantic_rank",
        "bm25_score",
        "bm25_rank",
        "hybrid_score",
        "metadata_boosted_score",
        "rerank_score",
        "quality_score",
        "context_mode",
    ]

    for index, doc in enumerate(docs[:max_docs], start=1):
        metadata = getattr(doc, "metadata", {}) or {}
        lines.append("")
        lines.append(f"Rank    : {index}")
        lines.append(f"Source  : {get_ui_debug_source_label(doc)}")
        lines.append(f"Title   : {metadata.get('title', '')}")
        lines.append(f"Section : {metadata.get('section', '')}")
        lines.append(f"Category: {metadata.get('category', '')}")
        lines.append(f"Doc type: {metadata.get('doc_type', '')}")

        for key in score_keys:
            if key in metadata:
                lines.append(f"{key}: {metadata.get(key)}")

        lines.append(f"Preview : {get_ui_debug_preview(doc)}")
        lines.append("-" * 80)

    if len(docs) > max_docs:
        lines.append(f"... {len(docs) - max_docs} more documents not shown.")


def log_ui_rag_input(
    mode,
    question,
    raw_chat_history,
    effective_chat_history,
    rewritten_question,
    answer_question,
    retrieval_queries,
):
    # Log the exact question transformation used by the UI path.
    lines = []
    add_ui_debug_section(lines, f"UI RAG INPUT - {mode}")
    lines.append(f"Time                  : {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"Original question     : {question}")
    lines.append(f"Raw chat history chars: {len(str(raw_chat_history or ''))}")
    lines.append(f"Effective history chars: {len(str(effective_chat_history or ''))}")
    lines.append(f"Rewritten question    : {rewritten_question}")
    lines.append(f"Answer/Retrieval query: {answer_question}")
    lines.append("")
    lines.append("Retrieval queries:")

    for index, query in enumerate(retrieval_queries or [], start=1):
        lines.append(f"[{index}] {query}")

    append_ui_rag_debug(lines)


def log_ui_retrieval_result(mode, retrieval_result, include_final=False):
    # Log retrieval documents from the UI path.
    lines = []
    add_ui_debug_section(lines, f"UI RETRIEVAL RESULT - {mode}")
    lines.append("Retrieval result keys:")
    lines.append(", ".join(retrieval_result.keys()))

    for key in ["semantic_docs", "bm25_docs", "hybrid_docs", "ranked_docs"]:
        add_ui_debug_docs(lines, f"UI RETRIEVED CHUNKS LOG - {key}", retrieval_result.get(key, []))

    if include_final:
        add_ui_debug_docs(lines, "UI FINAL DOCS SENT TO LLM", retrieval_result.get("final_docs", []))

    append_ui_rag_debug(lines)


def log_ui_final_context(mode, final_docs):
    # Log only the exact final context docs sent to the LLM.
    lines = []
    add_ui_debug_docs(lines, f"UI FINAL CONTEXT SENT TO LLM - {mode}", final_docs)
    append_ui_rag_debug(lines)


def log_ui_prompt(mode, prompt):
    # Log the exact prompt string sent to the LLM in the streaming path.
    lines = []
    add_ui_debug_section(lines, f"UI PROMPT SENT TO LLM - {mode}")
    lines.append(str(prompt or ""))
    append_ui_rag_debug(lines)


def get_regenerate_style(regenerate_attempt=0):
    # Cycle regenerate answer styles so deterministic LLM settings still produce a different answer.
    attempt = int(regenerate_attempt or 1)
    styles = [
        {
            "name": "careful_two_part",
            "instruction": (
                "Use exactly this format:\n"
                "Answer: <direct answer in one sentence>\n"
                "Careful note: <explain what the retrieved context says and what it does not prove>"
            ),
        },
        {
            "name": "evidence_first",
            "instruction": (
                "Use exactly this format:\n"
                "According to the retrieved context: <state the evidence first>\n"
                "Therefore: <answer the user's question carefully>"
            ),
        },
        {
            "name": "short_bullets",
            "instruction": (
                "Use exactly two bullet points:\n"
                "- Direct answer: <answer the question>\n"
                "- Context basis: <briefly cite what the context says>"
            ),
        },
        {
            "name": "plain_careful",
            "instruction": (
                "Write one concise paragraph. Start with 'Based on the provided context,'. "
                "Do not start with the same first words as the previous answer."
            ),
        },
    ]

    return styles[(attempt - 1) % len(styles)]


def apply_regenerate_instruction(prompt, regenerate=False, regenerate_attempt=0, previous_answer=""):
    # Keep retrieval deterministic, but force a different answer structure during regeneration.
    if not regenerate:
        return prompt

    attempt = int(regenerate_attempt or 1)
    previous_answer = str(previous_answer or "").strip()
    style = get_regenerate_style(attempt)

    lines = [
        "",
        "REGENERATION MODE:",
        f"This is regenerate attempt #{attempt}.",
        f"Required regenerate style: {style['name']}.",
        "",
        "Hard rules:",
        "1. Use the same retrieved context only.",
        "2. Keep the same facts, sources, and conclusion.",
        "3. Do not invent new information.",
        "4. Do not copy the previous answer word-for-word.",
        "5. The new answer must use the required regenerate style below.",
        "",
        "Required regenerate format/style:",
        style["instruction"],
    ]

    if previous_answer:
        first_sentence = previous_answer.split(".")[0].strip()
        lines.extend([
            "",
            "Previous answer to avoid copying:",
            previous_answer,
        ])

        if first_sentence:
            lines.extend([
                "",
                "Do not reuse this previous opening sentence:",
                first_sentence,
            ])

    return f"{prompt}\n" + "\n".join(lines)


def log_ui_answer(mode, answer, sources=None):
    # Log the final cleaned answer and UI sources.
    lines = []
    add_ui_debug_section(lines, f"UI FINAL ANSWER - {mode}")
    lines.append(str(answer or ""))
    lines.append("")
    lines.append("Sources:")

    for index, source in enumerate(sources or [], start=1):
        lines.append(f"[{index}] {source}")

    append_ui_rag_debug(lines)


def call_supported(function, *args, **kwargs):
    # Call a function while ignoring keyword arguments that the current version does not support.
    parameters = inspect.signature(function).parameters
    accepts_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )

    if accepts_kwargs:
        return function(*args, **kwargs)

    supported_kwargs = {}

    for key, value in kwargs.items():
        if key in parameters:
            supported_kwargs[key] = value

    return function(*args, **supported_kwargs)


def get_metadata(doc):
    # Safe metadata getter.
    return dict(getattr(doc, "metadata", {}) or {})


def get_doc_key(doc):
    # Stable key para hindi maulit ang parehong chunk mula sa semantic, BM25, or multiple queries.
    metadata = get_metadata(doc)
    return (
        metadata.get("source") or metadata.get("file_name") or "",
        metadata.get("page", ""),
        metadata.get("chunk_id") or metadata.get("chunk_index") or id(doc),
    )


def merge_unique_docs(*doc_lists, limit=0):
    # Pagsamahin ang docs habang tinatanggal ang duplicate chunks.
    merged = []
    seen = set()

    for docs in doc_lists:
        for doc in docs or []:
            key = get_doc_key(doc)

            if key in seen:
                continue

            seen.add(key)
            merged.append(doc)

            if limit and len(merged) >= limit:
                return merged

    return merged


def normalize_rerank_result(result):
    # Support list[Document] or list[(Document, score)].
    docs = []

    for item in result or []:
        if isinstance(item, tuple) and item:
            docs.append(item[0])
        else:
            docs.append(item)

    return docs


def is_no_answer(answer):
    # Check kung fallback/no-answer ang sagot.
    if not answer:
        return False

    return str(NO_ANSWER_TEXT).strip().lower() in str(answer).strip().lower()


def is_fallback_answer(answer):
    # Exact fallback checker for retry decisions.
    return str(answer or "").strip() == str(NO_ANSWER_TEXT).strip()


def looks_truncated_answer(answer):
    # Detect answers that likely stopped before the final sentence was complete.
    text = " ".join(str(answer or "").split()).strip()

    if not text or is_fallback_answer(text):
        return False

    if len(text.split()) < 6:
        return False

    return not text.endswith(TERMINAL_PUNCTUATION)


def should_retry_answer(answer, retrieval_result, category="", no_llm=False):
    # Shared retry decision used by UI and tests.
    if no_llm:
        return False

    if not retrieval_result.get("final_docs"):
        return False

    category = str(category or "").strip().upper()

    # Always allow completion retry when the answer looks cut off.
    if ENABLE_TRUNCATION_RETRY and looks_truncated_answer(answer):
        return True

    # Skip fallback retry only for cases where fallback may be intentional.
    if category in FALLBACK_RETRY_SKIP_CATEGORIES:
        return False

    if ENABLE_FALLBACK_RETRY and is_fallback_answer(answer):
        return True

    return False


def has_searchable_question_terms(question):
    # Local lightweight guard para hindi mag-retrieve sa blank/symbol-only input.
    tokens = normalize_text(question).split()

    for token in tokens:
        if len(token) >= MIN_SEARCHABLE_TOKEN_LENGTH:
            return True

    return False


def normalize_query_list(queries, max_queries=MAX_RETRIEVAL_QUERIES):
    # Clean and deduplicate retrieval queries.
    clean_queries = []
    seen = set()

    for query in queries or []:
        query = " ".join(str(query or "").split())
        key = query.lower()

        if not query or key in seen:
            continue

        clean_queries.append(query)
        seen.add(key)

        if max_queries and len(clean_queries) >= max_queries:
            break

    return clean_queries


def get_query_analyzer_terms(question):
    # Use query_analyzer only as optional generic query expansion.
    if analyze_query is None:
        return ""

    try:
        query_info = analyze_query(question)
    except Exception:
        return ""

    terms = []

    for key in ["source_hint", "category", "doc_type", "important_terms", "source_keywords"]:
        value = query_info.get(key)

        if isinstance(value, str):
            values = [value]
        elif isinstance(value, list):
            values = value
        else:
            values = []

        for item in values:
            item = " ".join(str(item or "").split())

            if item and item.lower() not in {term.lower() for term in terms}:
                terms.append(item)

    return " ".join(terms)


def get_keyword_query(question):
    # Fallback keyword query kapag walang query_analyzer terms.
    tokens = normalize_text(question).split()
    useful_tokens = []

    for token in tokens:
        if len(token) < 3:
            continue

        if token in {
            "the",
            "and",
            "or",
            "who",
            "what",
            "when",
            "where",
            "why",
            "how",
            "did",
            "does",
            "do",
            "are",
            "was",
            "were",
            "is",
            "ng",
            "sa",
            "ang",
            "mga",
            "ano",
            "sino",
            "paano",
            "bakit",
        }:
            continue

        if token not in useful_tokens:
            useful_tokens.append(token)

    return " ".join(useful_tokens)


def has_reference_pronoun(question):
    # True when the original user text depends on previous chat context.
    text = normalize_text(question)
    return bool(re.search(r"\b(it|this|that|these|those|there|he|she|they|them|him|her|its|his|their)\b", text))


def get_intent_expanded_query(question):
    # Add generic intent words for single-fact date questions where one verb matters.
    # Example: "When did Jose Rizal die?" must prefer death/execution evidence,
    # not any nearby date about travel or relationships.
    text = normalize_text(question)
    keyword_query = get_keyword_query(question)

    if not keyword_query:
        return ""

    if re.search(r"\b(die|died|death|dead|deaths|execution|executed|execute)\b", text):
        return f"{keyword_query} death died date execution executed"

    if re.search(r"\b(born|birth|birthday|date of birth)\b", text):
        return f"{keyword_query} born birth birthday date"

    return ""


def build_retrieval_queries(
    question,
    rewritten_question=None,
    enabled=ENABLE_MULTI_QUERY_RETRIEVAL,
    max_queries=MAX_RETRIEVAL_QUERIES,
):
    # Shared generic retrieval query builder for UI and tests.
    # If a pronoun follow-up was rewritten, retrieve using the standalone rewrite only.
    # Keeping the raw pronoun query can pull unrelated date chunks and poison final context.
    question = " ".join(str(question or "").split())
    rewritten_question = " ".join(str(rewritten_question or question).split())
    rewritten_changed = bool(rewritten_question and rewritten_question.lower() != question.lower())
    use_rewritten_only = rewritten_changed and has_reference_pronoun(question)
    base_query = rewritten_question if use_rewritten_only else question

    queries = [base_query]

    if rewritten_changed and not use_rewritten_only:
        queries.append(rewritten_question)

    if enabled:
        expansion_base = rewritten_question or base_query
        intent_query = get_intent_expanded_query(expansion_base)
        analyzer_query = get_query_analyzer_terms(expansion_base)
        keyword_query = get_keyword_query(expansion_base)

        if intent_query:
            queries.append(intent_query)

        if analyzer_query:
            queries.append(analyzer_query)

        if keyword_query:
            queries.append(keyword_query)

    effective_max_queries = max_queries

    if use_rewritten_only and max_queries:
        effective_max_queries = max(max_queries, 3)
    elif rewritten_changed and max_queries:
        effective_max_queries = max(max_queries, 2)

    return normalize_query_list(queries, max_queries=effective_max_queries)


def combine_retrieval_queries(retrieval_queries):
    # Debug/display helper.
    return "\n".join(normalize_query_list(retrieval_queries, max_queries=0))


CLARIFY_PREFIX = "CLARIFY:"


def is_clarification_response(text):
    # True kapag sinabi ng rewriter na ambiguous ang follow-up question.
    return str(text or "").strip().upper().startswith(CLARIFY_PREFIX)


def get_clarification_answer(text):
    # User-facing clarification answer, without the internal CLARIFY prefix.
    message = str(text or "").strip()

    if is_clarification_response(message):
        message = message[len(CLARIFY_PREFIX):].strip()

    return message or "Which person, item, or topic do you mean?"


def get_answer_question(question, rewritten_question):
    # Gamitin ang standalone rewritten question para hindi ambiguous ang retrieval/rerank/prompt.
    rewritten_question = str(rewritten_question or "").strip()

    if not rewritten_question or is_clarification_response(rewritten_question):
        return question

    return rewritten_question


def safe_rewrite_question(question, chat_history, llm):
    # Rewrite only when the user question is confirmed as a contextual follow-up.
    # Standalone questions and ambiguous short keyword queries must keep the exact user input.
    question = str(question or "").strip()
    chat_history = str(chat_history or "").strip()

    if not question:
        return question

    if not ENABLE_QUESTION_REWRITE or not chat_history:
        return question

    try:
        if not is_follow_up_question(question):
            return question
    except Exception:
        # If follow-up detection fails, keep the original question instead of risking a bad rewrite.
        return question

    try:
        rewritten_question = rewrite_question(
            question=question,
            chat_history=chat_history,
            llm=llm,
        )

        rewritten_question = str(rewritten_question or "").strip()

        if rewritten_question:
            return rewritten_question

    except Exception:
        pass

    return question


def get_effective_chat_history(question, chat_history):
    # Use memory only for confirmed contextual follow-up questions.
    # Standalone and ambiguous short keyword queries should not inherit old topics.
    chat_history = str(chat_history or "").strip()

    if not chat_history:
        return ""

    try:
        if not is_follow_up_question(question):
            return ""
    except Exception:
        return ""

    return chat_history


@cache_chatbot_resource(show_spinner="Loading chatbot components...")
def load_chatbot_components():
    # Cached ito para first load lang mabigat, succeeding Streamlit reruns mabilis na.
    embedding_model = get_embedding_model()

    vectorstore = load_chroma_vectorstore(
        embedding_model=embedding_model,
    )

    chunks = load_or_create_chunks(DATA_PATH)

    bm25_retriever = load_or_create_bm25(
        chunks=chunks,
        k=BM25_K,
    )

    return {
        "vectorstore": vectorstore,
        "bm25_retriever": bm25_retriever,
        "reranker": load_reranker(),
        "llm": load_llm(),
        "chunks": chunks,
    }




def resolve_all_chunks(all_chunks=None):
    # Ensure context_filter can use neighbor chunk expansion even if the UI caller forgot to pass chunks.
    if all_chunks is not None:
        return all_chunks

    try:
        return load_or_create_chunks(DATA_PATH)
    except Exception as error:
        print_ui_rag_debug(f"Failed to load chunks for neighbor expansion: {error}")
        return []

def clear_chatbot_components_cache():
    # Tawagin ito pagkatapos mag-ingest ng bagong data kung gusto mong i-refresh ang loaded components.
    if hasattr(load_chatbot_components, "clear"):
        load_chatbot_components.clear()


def run_retrieval_core(
    question,
    retrieval_queries,
    vectorstore,
    bm25_retriever,
    reranker,
    use_metadata_boost=False,
    use_reranker=True,
    debug=False,
):
    # Shared retrieval path used by both UI and tests.
    # Flow dito: hybrid -> merge unique -> rerank only.
    # Final context filtering is done only by chains.rag_chain.
    question = str(question or "").strip()

    if isinstance(retrieval_queries, str):
        retrieval_queries = [retrieval_queries]
    else:
        retrieval_queries = list(retrieval_queries or [])

    retrieval_queries = normalize_query_list(retrieval_queries, max_queries=MAX_RETRIEVAL_QUERIES)

    if not retrieval_queries:
        retrieval_queries = [question] if question else []

    semantic_doc_lists = []
    bm25_doc_lists = []
    hybrid_doc_lists = []

    for retrieval_query in retrieval_queries:
        hybrid_result = call_supported(
            hybrid_search,
            query=retrieval_query,
            vectorstore=vectorstore,
            bm25_retriever=bm25_retriever,
            semantic_k=SEMANTIC_K,
            bm25_k=BM25_K,
            final_k=HYBRID_FINAL_K,
            use_metadata_boost=use_metadata_boost,
            return_details=True,
            debug=debug,
        )

        if isinstance(hybrid_result, dict):
            semantic_doc_lists.append(hybrid_result.get("semantic_docs", []))
            bm25_doc_lists.append(hybrid_result.get("bm25_docs", []))
            hybrid_doc_lists.append(hybrid_result.get("hybrid_docs", []))
        else:
            hybrid_doc_lists.append(hybrid_result or [])

        if debug:
            if isinstance(hybrid_result, dict):
                current_docs = hybrid_result.get("hybrid_docs", [])
            else:
                current_docs = hybrid_result or []

            print_ui_rag_debug(
                f"Hybrid candidates for query [{retrieval_query}]: {len(current_docs or [])}"
            )

    semantic_docs = merge_unique_docs(*semantic_doc_lists)
    bm25_docs = merge_unique_docs(*bm25_doc_lists)
    hybrid_docs = merge_unique_docs(
        *hybrid_doc_lists,
        limit=MAX_CANDIDATES_BEFORE_RERANK,
    )

    if use_reranker and hybrid_docs:
        rerank_result = call_supported(
            rerank_documents,
            query=question,
            documents=hybrid_docs,
            reranker=reranker,
            top_n=min(len(hybrid_docs), max(RERANK_POOL_TOP_N, RERANK_TOP_N)),
            return_scores=True,
            show_scores=debug,
            debug=debug,
        )
        ranked_docs = normalize_rerank_result(rerank_result)
    else:
        ranked_docs = hybrid_docs

    if debug:
        print_debug_docs(ranked_docs, label="ranked_docs")

    return {
        "retrieval_queries": retrieval_queries,
        "semantic_docs": semantic_docs,
        "bm25_docs": bm25_docs,
        "hybrid_docs": hybrid_docs,
        "ranked_docs": ranked_docs,
        "final_docs": [],
    }


def ensure_final_context_docs(question, retrieval_result, all_chunks=None, debug=False):
    # Use rag_chain's context filter to get the exact final docs.
    # This is used for UI sources and no-LLM tests.
    final_docs = retrieval_result.get("final_docs")

    if final_docs:
        return final_docs

    final_docs = prepare_context_docs(
        question=question,
        docs=retrieval_result.get("ranked_docs", []),
        semantic_docs=retrieval_result.get("semantic_docs", []),
        bm25_docs=retrieval_result.get("bm25_docs", []),
        debug=debug,
        all_chunks=resolve_all_chunks(all_chunks),
    )

    retrieval_result["final_docs"] = final_docs
    return final_docs


def retrieve_documents(
    retrieval_query,
    vectorstore,
    bm25_retriever,
    reranker,
    all_chunks=None,
    answer_query=None,
    debug=False,
):
    # Backward-compatible wrapper. Returns the same final context docs used by rag_chain.
    question = answer_query or retrieval_query
    retrieval_result = run_retrieval_core(
        question=question,
        retrieval_queries=retrieval_query,
        vectorstore=vectorstore,
        bm25_retriever=bm25_retriever,
        reranker=reranker,
        debug=debug,
    )

    return ensure_final_context_docs(
        question=question,
        retrieval_result=retrieval_result,
        all_chunks=all_chunks,
        debug=debug,
    )


def generate_chatbot_answer(
    question,
    retrieval_result,
    llm,
    chat_history="",
    debug=False,
    correction_retry=False,
    completion_retry=False,
    all_chunks=None,
):
    # Shared answer generation path used by UI and tests.
    # It mutates retrieval_result["final_docs"] with the exact docs used in the prompt.
    ranked_docs = retrieval_result.get("ranked_docs", [])

    if not ranked_docs:
        retrieval_result["final_docs"] = []
        return NO_ANSWER_TEXT

    result = call_supported(
        generate_answer_with_context,
        question=question,
        docs=ranked_docs,
        semantic_docs=retrieval_result.get("semantic_docs", []),
        bm25_docs=retrieval_result.get("bm25_docs", []),
        llm=llm,
        chat_history=chat_history,
        all_chunks=resolve_all_chunks(all_chunks),
        strict_assumption_check=True,
        correction_retry=correction_retry,
        completion_retry=completion_retry,
        debug=debug,
    )

    answer = result.get("answer", NO_ANSWER_TEXT)
    context_docs = result.get("context_docs", [])

    retrieval_result["final_docs"] = context_docs

    return answer


def generate_chatbot_answer_with_retry(
    question,
    retrieval_result,
    llm,
    chat_history="",
    debug=False,
    category="",
    no_llm=False,
    all_chunks=None,
):
    # Shared answer generation plus retry logic used by UI and tests.
    if no_llm:
        ensure_final_context_docs(
            question=question,
            retrieval_result=retrieval_result,
            all_chunks=all_chunks,
            debug=debug,
        )
        return "[NO LLM MODE] Answer generation skipped."

    answer = generate_chatbot_answer(
        question=question,
        retrieval_result=retrieval_result,
        llm=llm,
        chat_history=chat_history,
        debug=debug,
        all_chunks=all_chunks,
    )

    if should_retry_answer(
        answer=answer,
        retrieval_result=retrieval_result,
        category=category,
        no_llm=no_llm,
    ):
        answer = generate_chatbot_answer(
            question=question,
            retrieval_result=retrieval_result,
            llm=llm,
            chat_history=chat_history,
            debug=debug,
            correction_retry=is_fallback_answer(answer),
            completion_retry=looks_truncated_answer(answer),
            all_chunks=all_chunks,
        )

    return answer


def build_retrieval_query_list(question, rewritten_question):
    # Shared helper for UI.
    return build_retrieval_queries(
        question=question,
        rewritten_question=rewritten_question,
        enabled=ENABLE_MULTI_QUERY_RETRIEVAL,
        max_queries=MAX_RETRIEVAL_QUERIES,
    )


def ask_rag(
    question,
    vectorstore,
    bm25_retriever,
    reranker,
    llm,
    chat_history="",
    debug=False,
    all_chunks=None,
):
    # Non-streaming RAG answer.
    question = str(question or "").strip()

    if not question:
        return build_response(answer="No question entered.")

    if looks_like_value_only_question(question):
        answer = get_value_only_question_clarification(question)
        log_ui_answer("ask_rag", answer, sources=[])
        return build_response(answer=answer)

    if not chat_history and not has_searchable_question_terms(question):
        return build_response(answer=NO_ANSWER_TEXT)

    effective_chat_history = get_effective_chat_history(question, chat_history)

    rewritten_question = safe_rewrite_question(
        question=question,
        chat_history=effective_chat_history,
        llm=llm,
    )

    if is_clarification_response(rewritten_question):
        return build_response(answer=get_clarification_answer(rewritten_question))

    answer_question = get_answer_question(question, rewritten_question)

    retrieval_queries = build_retrieval_query_list(
        question=question,
        rewritten_question=rewritten_question,
    )

    log_ui_rag_input(
        mode="ask_rag",
        question=question,
        raw_chat_history=chat_history,
        effective_chat_history=effective_chat_history,
        rewritten_question=rewritten_question,
        answer_question=answer_question,
        retrieval_queries=retrieval_queries,
    )

    retrieval_result = run_retrieval_core(
        question=answer_question,
        retrieval_queries=retrieval_queries,
        vectorstore=vectorstore,
        bm25_retriever=bm25_retriever,
        reranker=reranker,
        debug=debug,
    )

    log_ui_retrieval_result(
        mode="ask_rag",
        retrieval_result=retrieval_result,
    )

    if not retrieval_result.get("ranked_docs"):
        log_ui_answer("ask_rag", NO_ANSWER_TEXT, sources=[])
        return build_response(answer=NO_ANSWER_TEXT)

    answer = generate_chatbot_answer_with_retry(
        question=answer_question,
        retrieval_result=retrieval_result,
        llm=llm,
        chat_history="",
        debug=debug,
        all_chunks=all_chunks,
    )

    answer = clean_generated_answer(
        answer=answer,
        question=answer_question,
    )

    final_docs = retrieval_result.get("final_docs", [])
    sources = get_sources(final_docs)

    log_ui_final_context("ask_rag", final_docs)

    if final_docs:
        debug_prompt = build_prompt_from_context(
            question=answer_question,
            context_docs=final_docs,
            chat_history="",
        )
        log_ui_prompt("ask_rag", debug_prompt)

    if debug:
        print_debug_docs(final_docs, label="final_docs")

    if is_no_answer(answer):
        sources = []

    log_ui_answer("ask_rag", answer, sources=sources)

    return build_response(
        answer=answer,
        sources=sources,
        documents=final_docs,
    )



def get_rephrase_style(regenerate_attempt=0):
    # Cycle wording instructions only for the Regenerate button.
    # These are generic and do not contain sample-specific entities or verbs.
    attempt = int(regenerate_attempt or 1)
    styles = [
        {
            "name": "natural_paraphrase",
            "instruction": "Rewrite naturally with different wording while keeping roughly the same length.",
        },
        {
            "name": "clearer_sentence_order",
            "instruction": "Rewrite using a clearer sentence order. Keep the same facts and caution level.",
        },
        {
            "name": "concise_paraphrase",
            "instruction": "Rewrite more concisely without removing important caveats.",
        },
        {
            "name": "plain_language",
            "instruction": "Rewrite in plain language while preserving the exact meaning.",
        },
    ]

    return styles[(attempt - 1) % len(styles)]


def build_rephrase_prompt(question, previous_answer, regenerate_attempt=0):
    # Prompt used only by the UI Regenerate button.
    # It rephrases the previous answer only; it must not answer from scratch.
    question = str(question or "").strip()
    previous_answer = str(previous_answer or "").strip()
    style = get_rephrase_style(regenerate_attempt)

    return "\n".join([
        "You are rewriting an existing answer for a RAG chatbot.",
        "Your task is paraphrase only, not new answering.",
        "",
        "Rules:",
        "1. Use only the previous answer as the source of truth.",
        "2. Preserve the exact meaning, facts, uncertainty, and caution level.",
        "3. Do not add new claims, names, dates, causes, or conclusions.",
        "4. Do not remove caveats such as 'the documents do not say' or 'not specified'.",
        "5. Do not turn a group attribution into a specific individual attribution unless the previous answer already says that.",
        "6. Do not answer the original question from memory or outside knowledge.",
        "7. Use correct grammar and natural wording.",
        "8. Return only the rewritten answer. No labels, no bullets unless the previous answer is already a list.",
        "",
        f"Rewrite style: {style['name']}",
        style["instruction"],
        "",
        "Original user question for context only:",
        question,
        "",
        "Previous answer to rewrite:",
        previous_answer,
        "",
        "Rewritten answer:",
    ])


def clean_rephrase_output(answer):
    # Remove common LLM wrapper labels while keeping the actual answer text.
    text = str(answer or "").strip()
    text = re.sub(r"^```(?:\w+)?\s*", "", text).strip()
    text = re.sub(r"\s*```$", "", text).strip()

    prefixes = [
        "Rewritten answer:",
        "Rephrased answer:",
        "Paraphrased answer:",
        "Answer:",
        "Final answer:",
        "Response:",
        "Sagót:",
        "Sagot:",
    ]

    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if text.lower().startswith(prefix.lower()):
                text = text[len(prefix):].strip()
                changed = True

    # Remove one accidental surrounding quote pair.
    if len(text) >= 2 and text[0] in {'"', "'"} and text[-1] == text[0]:
        text = text[1:-1].strip()

    return text


def looks_like_bad_rephrase(answer, previous_answer):
    # Safety guard: do not show obviously broken grammar from the rephrase model.
    answer = " ".join(str(answer or "").split()).strip()
    previous_answer = " ".join(str(previous_answer or "").split()).strip()

    if not answer:
        return True

    if len(answer) < 8:
        return True

    # Avoid malformed endings such as "... was." or "... by X was."
    if re.search(r"\b(am|is|are|was|were|be|been|being|by)\s*[.!?]$", answer, re.IGNORECASE):
        return True

    if re.search(r"\bby\s+[^.!?]{1,80}\s+was\s*[.!?]$", answer, re.IGNORECASE):
        return True

    # If previous answer is substantial, do not accept a suspiciously tiny rewrite.
    if len(previous_answer) >= 60 and len(answer) < max(25, int(len(previous_answer) * 0.35)):
        return True

    return False


def stream_rephrased_answer(
    question,
    previous_answer,
    llm,
    regenerate_attempt=0,
    debug=False,
):
    # Regenerate button path only.
    # This does not retrieve/rerank and does not affect normal user submits.
    print_ui_rag_debug("REPHRASE REGENERATE CALLED")
    previous_answer = str(previous_answer or "").strip()

    if not previous_answer:
        response = build_response(answer=NO_ANSWER_TEXT)
        yield {"type": "chunk", "content": response["answer"]}
        yield {"type": "done", **response}
        return

    prompt = build_rephrase_prompt(
        question=question,
        previous_answer=previous_answer,
        regenerate_attempt=regenerate_attempt,
    )

    if debug or UI_RAG_FILE_DEBUG:
        log_ui_prompt("rephrase_only_regenerate", prompt)

    answer_parts = []
    for chunk in llm.stream(prompt):
        text = extract_llm_text(chunk)
        if text:
            answer_parts.append(text)

    answer = clean_rephrase_output("".join(answer_parts))

    # If the rephrase is broken, keep the old answer instead of showing a wrong/awkward claim.
    if looks_like_bad_rephrase(answer, previous_answer):
        answer = previous_answer

    if debug or UI_RAG_FILE_DEBUG:
        lines = []
        add_ui_debug_section(lines, "UI REPHRASE ONLY REGENERATE - LLM")
        lines.append(f"Regenerate attempt: {int(regenerate_attempt or 1)}")
        lines.append("")
        lines.append("Previous answer:")
        lines.append(previous_answer)
        lines.append("")
        lines.append("New answer:")
        lines.append(answer)
        append_ui_rag_debug(lines)

    yield {"type": "chunk", "content": answer}

    response = build_response(
        answer=answer,
        sources=[],
        documents=[],
    )
    log_ui_answer("rephrase_only_regenerate", answer, sources=[])
    yield {"type": "done", **response}



def is_list_like_question(question):
    # Skip second-pass rewording for list/enumeration questions to avoid dropping supported items.
    question_key = normalize_text(question)

    if not question_key:
        return False

    list_patterns = [
        r"^who\s+(are|were)\b",
        r"^what\s+(are|were)\b",
        r"^which\s+(are|were)\b",
        r"^(list|enumerate)\b",
        r"^(name|identify|provide|give|show)\s+(all|the|each|every|a\s+list\s+of)\b",
        r"\b(all|each|every|multiple|several)\b",
        r"\b(steps|requirements|rules|items|examples|responsibilities|actions|conditions|approvers|reviewers|owners|documents|attachments|records|forms)\b",
        r"^sino[\s-]?sino\b",
        r"^(anu|ano)[\s-]?(ano|anu)\b",
        r"\bmga\b",
    ]

    return any(re.search(pattern, question_key, flags=re.IGNORECASE) for pattern in list_patterns)


def is_list_like_answer(answer):
    # Conservative list detector for generated answers.
    text = str(answer or "").strip()

    if not text:
        return False

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    bullet_lines = [line for line in lines if re.match(r"^(?:[-*•]|\d+[.)])\s+", line)]

    if len(bullet_lines) >= 2:
        return True

    # Semicolon/comma-heavy answers often contain multiple people/items in one sentence.
    if text.count(";") >= 2:
        return True

    if text.count(",") >= 3 and len(re.findall(r"\b[A-ZÀ-ÖØ-Þ][A-Za-zÀ-ÖØ-öø-ÿ.'-]+", text)) >= 3:
        return True

    return False


def should_reword_regenerated_answer(question, answer):
    # Reword only simple real answers. Keep list answers unchanged to preserve coverage.
    text = " ".join(str(answer or "").split()).strip()

    if not text:
        return False

    if is_fallback_answer(text) or is_no_answer(text):
        return False

    if is_list_like_question(question) or is_list_like_answer(answer):
        return False

    # Clarification answers are intentionally questions; keep them exact.
    if text.endswith("?"):
        return False

    clarification_markers = [
        "which person",
        "which event",
        "which policy",
        "which topic",
        "do you mean",
        "anong tao",
        "anong event",
        "anong policy",
        "anong topic",
        "tinutukoy mo",
    ]
    text_key = text.lower()

    return not any(marker in text_key for marker in clarification_markers)



def normalize_reword_key(text):
    text = normalize_space(text).lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def extract_date_like_value(text):
    text = normalize_space(text)
    patterns = [
        r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s*\d{4}\b",
        r"\b\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b",
        r"\b\d{4}-\d{2}-\d{2}\b",
        r"\b\d{4}\b",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(0)

    return ""


def clean_question_subject(subject):
    subject = normalize_space(subject).strip(" ?.!,:;\"'")
    subject = re.sub(r"^(?:the|a|an)\s+", "", subject, flags=re.IGNORECASE).strip()
    return subject


def extract_single_fact_subject(question):
    question = normalize_space(question).rstrip("?")
    patterns = [
        r"^when\s+(?:is|was)\s+(.+?)\s*(?:'s|’s)\s+birthday$",
        r"^when\s+(?:is|was)\s+(.+?)\s+(?:birthday|birth\s+date|date\s+of\s+birth)$",
        r"^when\s+(?:was|is)\s+(.+?)\s+born$",
        r"^when\s+did\s+(.+?)\s+(?:die|died)$",
        r"^when\s+(?:was|is)\s+(.+?)\s+(?:death|execution)$",
    ]

    for pattern in patterns:
        match = re.match(pattern, question, flags=re.IGNORECASE)
        if match:
            return clean_question_subject(match.group(1))

    return ""


def build_short_fact_reword_options(question, answer):
    date_value = extract_date_like_value(answer)

    if not date_value:
        return []

    question_key = normalize_text(question)
    subject = extract_single_fact_subject(question)

    if not subject:
        return []

    options = []

    if re.search(r"\b(birthday|birth date|date of birth|born)\b", question_key):
        options.extend([
            f"{subject}'s birthday is {date_value}.",
            f"{subject} was born on {date_value}.",
            f"The birth date of {subject} is {date_value}.",
        ])

    if re.search(r"\b(die|died|death|execution|executed)\b", question_key):
        options.extend([
            f"{subject} died on {date_value}.",
            f"The death date of {subject} is {date_value}.",
            f"{date_value} is when {subject} died.",
        ])

    return options


def deterministic_regenerate_reword(question, answer, previous_answer, regenerate_attempt=0):
    # For short date/value answers, use a deterministic wording change.
    # This avoids returning the exact same visible answer when the LLM is deterministic.
    options = build_short_fact_reword_options(question=question, answer=answer)

    if not options:
        return ""

    previous_key = normalize_reword_key(previous_answer)
    answer_key = normalize_reword_key(answer)
    start = max(0, int(regenerate_attempt or 1) - 1)

    for offset in range(len(options)):
        option = options[(start + offset) % len(options)]
        option_key = normalize_reword_key(option)

        if option_key and option_key not in {previous_key, answer_key}:
            return option

    for option in options:
        option_key = normalize_reword_key(option)
        if option_key and option_key != previous_key:
            return option

    return ""

def build_regenerate_reword_prompt(question, answer, previous_answer, regenerate_attempt=0):
    # Second-pass wording prompt after full RAG already produced the grounded answer.
    # This changes wording only; it does not retrieve or add new evidence.
    question = str(question or "").strip()
    answer = str(answer or "").strip()
    previous_answer = str(previous_answer or "").strip()
    style = get_rephrase_style(regenerate_attempt)

    return "\n".join([
        "You are rewriting a grounded RAG answer after a full retrieval run.",
        "Your task is wording only, not new answering.",
        "",
        "Rules:",
        "1. Use the grounded answer as the source of truth.",
        "2. Keep exactly the same factual meaning, date, number, names, uncertainty, and caution level.",
        "3. Do not add new facts, causes, examples, names, or conclusions.",
        "4. You may use wording from the user question to turn a short value answer into a complete sentence.",
        "5. If the grounded answer is a no-answer or clarification question, return it unchanged.",
        "6. Make the wording different from the previous visible answer when possible.",
        "7. Return only the rewritten answer. No labels.",
        "",
        f"Rewrite style: {style['name']}",
        style["instruction"],
        "",
        "User question:",
        question,
        "",
        "Previous visible answer to avoid copying:",
        previous_answer,
        "",
        "Grounded answer from the current full RAG run:",
        answer,
        "",
        "Rewritten grounded answer:",
    ])


def reword_regenerated_answer(question, answer, previous_answer, llm, regenerate_attempt=0, debug=False):
    # Keep full RAG as the source of truth, then make regenerate visibly different.
    answer = str(answer or "").strip()
    previous_answer = str(previous_answer or "").strip()

    if not should_reword_regenerated_answer(question=question, answer=answer):
        return answer

    deterministic = deterministic_regenerate_reword(
        question=question,
        answer=answer,
        previous_answer=previous_answer,
        regenerate_attempt=regenerate_attempt,
    )

    if deterministic:
        return deterministic

    prompt = build_regenerate_reword_prompt(
        question=question,
        answer=answer,
        previous_answer=previous_answer,
        regenerate_attempt=regenerate_attempt,
    )

    if debug or UI_RAG_FILE_DEBUG:
        log_ui_prompt("full_rag_regenerate_reword", prompt)

    try:
        response = llm.invoke(prompt)
        rewritten = clean_rephrase_output(extract_llm_text(response))
    except Exception as error:
        print_ui_rag_debug(f"Regenerate reword failed: {error}")
        return answer

    if looks_like_bad_rephrase(rewritten, answer):
        return answer

    if previous_answer and normalize_reword_key(rewritten) == normalize_reword_key(previous_answer):
        return answer

    return rewritten or answer

def ask_rag_stream(
    question,
    vectorstore,
    bm25_retriever,
    reranker,
    llm,
    chat_history="",
    debug=False,
    all_chunks=None,
    regenerate=False,
    regenerate_attempt=0,
    previous_answer="",
):
    # UI path synchronized with the non-streaming test RAG answer path.
    # Retrieval/rerank still runs first, then the final shared answer is emitted as one chunk.
    question = str(question or "").strip()

    if not question:
        response = build_response(answer="No question entered.")
        yield {"type": "chunk", "content": response["answer"]}
        yield {"type": "done", **response}
        return

    if looks_like_value_only_question(question):
        response = build_response(answer=get_value_only_question_clarification(question))
        log_ui_answer("ask_rag_stream", response["answer"], sources=[])
        yield {"type": "chunk", "content": response["answer"]}
        yield {"type": "done", **response}
        return

    if not chat_history and not has_searchable_question_terms(question):
        response = build_response(answer=NO_ANSWER_TEXT)
        yield {"type": "chunk", "content": response["answer"]}
        yield {"type": "done", **response}
        return

    effective_chat_history = get_effective_chat_history(question, chat_history)

    rewritten_question = safe_rewrite_question(
        question=question,
        chat_history=effective_chat_history,
        llm=llm,
    )

    if is_clarification_response(rewritten_question):
        response = build_response(answer=get_clarification_answer(rewritten_question))
        yield {"type": "chunk", "content": response["answer"]}
        yield {"type": "done", **response}
        return

    answer_question = get_answer_question(question, rewritten_question)

    retrieval_queries = build_retrieval_query_list(
        question=question,
        rewritten_question=rewritten_question,
    )

    stream_mode = "ask_rag_stream"

    log_ui_rag_input(
        mode=stream_mode,
        question=question,
        raw_chat_history=chat_history,
        effective_chat_history=effective_chat_history,
        rewritten_question=rewritten_question,
        answer_question=answer_question,
        retrieval_queries=retrieval_queries,
    )

    retrieval_result = run_retrieval_core(
        question=answer_question,
        retrieval_queries=retrieval_queries,
        vectorstore=vectorstore,
        bm25_retriever=bm25_retriever,
        reranker=reranker,
        debug=debug,
    )

    log_ui_retrieval_result(
        mode=stream_mode,
        retrieval_result=retrieval_result,
    )

    ranked_docs = retrieval_result.get("ranked_docs", [])

    if not ranked_docs:
        response = build_response(answer=NO_ANSWER_TEXT)
        log_ui_answer(stream_mode, response["answer"], sources=[])
        yield {"type": "chunk", "content": response["answer"]}
        yield {"type": "done", **response}
        return

    answer = generate_chatbot_answer_with_retry(
        question=answer_question,
        retrieval_result=retrieval_result,
        llm=llm,
        chat_history="",
        debug=debug,
        all_chunks=all_chunks,
    )

    answer = clean_generated_answer(
        answer=answer,
        question=answer_question,
    )

    if regenerate:
        answer = reword_regenerated_answer(
            question=answer_question,
            answer=answer,
            previous_answer=previous_answer,
            llm=llm,
            regenerate_attempt=regenerate_attempt,
            debug=debug,
        )
        answer = clean_generated_answer(
            answer=answer,
            question=answer_question,
        )

    final_docs = retrieval_result.get("final_docs", [])
    log_ui_final_context(stream_mode, final_docs)

    if final_docs:
        debug_prompt = build_prompt_from_context(
            question=answer_question,
            context_docs=final_docs,
            chat_history="",
        )
        log_ui_prompt(stream_mode, debug_prompt)

    sources = get_sources(final_docs)

    yield {"type": "chunk", "content": answer}

    if debug:
        print_debug_docs(final_docs, label="final_docs")

    if is_no_answer(answer):
        sources = []

    log_ui_answer(stream_mode, answer, sources=sources)

    response = build_response(
        answer=answer,
        sources=sources,
        documents=final_docs,
    )

    yield {"type": "done", **response}
