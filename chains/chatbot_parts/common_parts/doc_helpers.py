from chains.chatbot_parts.common_parts.debug_helpers import (
    inspect,
    re,
    datetime,
    Path,
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
    st,
    get_embedding_model,
    load_chroma_vectorstore,
    load_or_create_bm25,
    load_or_create_chunks,
    normalize_text,
    hybrid_search,
    load_reranker,
    rerank_documents,
    analyze_query,
    load_llm,
    build_prompt_from_context,
    clean_generated_answer,
    extract_llm_text,
    generate_answer_with_context,
    get_sources,
    prepare_context_docs,
    is_follow_up_question,
    rewrite_question,
    MIN_SEARCHABLE_TOKEN_LENGTH,
    FALLBACK_RETRY_SKIP_CATEGORIES,
    TERMINAL_PUNCTUATION,
    cache_chatbot_resource,
    build_response,
    print_ui_rag_debug,
    print_debug_docs,
    UI_RAG_FILE_DEBUG,
    PROJECT_ROOT,
    UI_RAG_DEBUG_FILE,
    UI_RAG_DEBUG_PREVIEW_CHARS,
    UI_RAG_DEBUG_MAX_DOCS,
    append_ui_rag_debug,
    add_ui_debug_section,
    get_ui_debug_source_label,
    get_ui_debug_preview,
    add_ui_debug_docs,
    log_ui_rag_input,
    log_ui_retrieval_result,
    log_ui_final_context,
    log_ui_prompt,
    log_ui_answer,
)



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
    # Stable key to avoid repeating the same chunk from semantic, BM25, or multiple queries.
    metadata = get_metadata(doc)
    return (
        metadata.get("source") or metadata.get("file_name") or "",
        metadata.get("page", ""),
        metadata.get("chunk_id") or metadata.get("chunk_index") or id(doc),
    )


def merge_unique_docs(*doc_lists, limit=0):
    # Merge docs while removing duplicate chunks.
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
    # Check whether the answer is a fallback/no-answer response.
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
    # Local lightweight guard to avoid retrieval for blank or symbol-only input.
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
    # Fallback keyword query when query_analyzer terms are missing.
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


def build_retrieval_queries(
    question,
    rewritten_question=None,
    enabled=ENABLE_MULTI_QUERY_RETRIEVAL,
    max_queries=MAX_RETRIEVAL_QUERIES,
):
    # Shared generic retrieval query builder for UI and tests.
    question = " ".join(str(question or "").split())
    rewritten_question = " ".join(str(rewritten_question or question).split())

    queries = [question]

    if rewritten_question and rewritten_question.lower() != question.lower():
        queries.append(rewritten_question)

    if enabled:
        analyzer_query = get_query_analyzer_terms(rewritten_question or question)
        keyword_query = get_keyword_query(rewritten_question or question)

        if analyzer_query:
            queries.append(analyzer_query)

        if keyword_query:
            queries.append(keyword_query)

    # If a follow-up was rewritten, keep both original and rewritten query even when max_queries is low.
    effective_max_queries = max_queries

    if rewritten_question and rewritten_question.lower() != question.lower() and max_queries:
        effective_max_queries = max(max_queries, 2)

    return normalize_query_list(queries, max_queries=effective_max_queries)


def combine_retrieval_queries(retrieval_queries):
    # Debug/display helper.
    return "\n".join(normalize_query_list(retrieval_queries, max_queries=0))


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
    'call_supported',
    'get_metadata',
    'get_doc_key',
    'merge_unique_docs',
    'normalize_rerank_result',
    'is_no_answer',
    'is_fallback_answer',
    'looks_truncated_answer',
    'should_retry_answer',
    'has_searchable_question_terms',
    'normalize_query_list',
    'get_query_analyzer_terms',
    'get_keyword_query',
    'build_retrieval_queries',
    'combine_retrieval_queries',
]
