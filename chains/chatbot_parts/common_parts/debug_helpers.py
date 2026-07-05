import inspect
import re
from datetime import datetime
from pathlib import Path

from config.settings import (
    BM25_K,
    CLARIFY_PREFIX,
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

from retrieval.context_config import read_query_config
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


def load_fallback_retry_skip_categories():
    config = read_query_config()
    chatbot_config = config.get("chatbot", {})
    categories = chatbot_config.get("fallback_retry_skip_categories", ())

    if isinstance(categories, str):
        categories = categories.split("|")

    if not isinstance(categories, (list, tuple, set)):
        return set()

    return {str(category).strip() for category in categories if str(category).strip()}


FALLBACK_RETRY_SKIP_CATEGORIES = load_fallback_retry_skip_categories()
TERMINAL_PUNCTUATION = (".", "?", "!", ")", "]", "}", '"', "'")


def cache_chatbot_resource(show_spinner=None):
    # When running inside Streamlit, use st.cache_resource.
    if st is not None:
        return st.cache_resource(show_spinner=show_spinner)

    # When running a test script without Streamlit, let it run normally.
    def decorator(function):
        return function

    return decorator


def build_response(answer, sources=None, documents=None):
    # Standard response format for the UI.
    return {
        "answer": answer,
        "sources": sources or [],
        "documents": documents or [],
    }


def print_ui_rag_debug(message):
    # Debug print for checking where the UI RAG path falls back.
    print(f"UI RAG DEBUG: {message}", flush=True)


def print_debug_docs(docs, label="docs", preview_chars=500):
    # Print docs/chunks to check what is being used.
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


# Public names exported by this compatibility/refactor module.
__all__ = [
    'inspect',
    're',
    'datetime',
    'Path',
    'BM25_K',
    'CLARIFY_PREFIX',
    'DATA_PATH',
    'ENABLE_FALLBACK_RETRY',
    'ENABLE_MULTI_QUERY_RETRIEVAL',
    'ENABLE_QUESTION_REWRITE',
    'ENABLE_TRUNCATION_RETRY',
    'HYBRID_FINAL_K',
    'MAX_CANDIDATES_BEFORE_RERANK',
    'MAX_RETRIEVAL_QUERIES',
    'NO_ANSWER_TEXT',
    'RERANK_POOL_TOP_N',
    'RERANK_TOP_N',
    'SEMANTIC_K',
    'st',
    'get_embedding_model',
    'load_chroma_vectorstore',
    'load_or_create_bm25',
    'load_or_create_chunks',
    'normalize_text',
    'hybrid_search',
    'load_reranker',
    'rerank_documents',
    'analyze_query',
    'load_llm',
    'build_prompt_from_context',
    'clean_generated_answer',
    'extract_llm_text',
    'generate_answer_with_context',
    'get_sources',
    'prepare_context_docs',
    'is_follow_up_question',
    'rewrite_question',
    'MIN_SEARCHABLE_TOKEN_LENGTH',
    'FALLBACK_RETRY_SKIP_CATEGORIES',
    'TERMINAL_PUNCTUATION',
    'cache_chatbot_resource',
    'build_response',
    'print_ui_rag_debug',
    'print_debug_docs',
    'UI_RAG_FILE_DEBUG',
    'PROJECT_ROOT',
    'UI_RAG_DEBUG_FILE',
    'UI_RAG_DEBUG_PREVIEW_CHARS',
    'UI_RAG_DEBUG_MAX_DOCS',
    'append_ui_rag_debug',
    'add_ui_debug_section',
    'get_ui_debug_source_label',
    'get_ui_debug_preview',
    'add_ui_debug_docs',
    'log_ui_rag_input',
    'log_ui_retrieval_result',
    'log_ui_final_context',
    'log_ui_prompt',
    'log_ui_answer',
]
