from chains.chatbot_parts.common_parts.doc_helpers import (
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
    call_supported,
    get_metadata,
    get_doc_key,
    merge_unique_docs,
    normalize_rerank_result,
    is_no_answer,
    is_fallback_answer,
    looks_truncated_answer,
    should_retry_answer,
    has_searchable_question_terms,
    normalize_query_list,
    get_query_analyzer_terms,
    get_keyword_query,
    build_retrieval_queries,
    combine_retrieval_queries,
)




FOLLOWUP_REFERENCE_WORDS = {
    "it", "its", "this", "that", "these", "those", "he", "his", "him",
    "she", "her", "they", "their", "them", "there", "then",
    "ito", "iyan", "yan", "iyon", "yun", "nito", "niyan", "noon",
    "dito", "doon", "diyan", "siya", "sya", "kanya", "kaniya",
    "nila", "sila", "kanila", "ganito", "ganyan", "ganoon",
}

QUESTION_DECISION_STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "to", "in", "on", "for", "from",
    "with", "by", "about", "as", "at", "is", "are", "was", "were", "be",
    "been", "being", "do", "does", "did", "can", "could", "should", "would",
    "will", "what", "when", "where", "why", "how", "who", "which",
    "ano", "anong", "sino", "kanino", "kailan", "kelan", "saan", "san",
    "bakit", "paano", "paaano", "gaano", "alin", "ang", "ng", "sa", "si",
    "ni", "kay", "kina", "mga", "ba", "bang", "naman", "pa", "po",
    "ito", "iyan", "yan", "iyon", "yun", "nito", "niyan", "noon", "dito",
    "doon", "diyan", "siya", "sya", "kanya", "kaniya", "nila", "sila",
    "kanila",
}

MIN_STANDALONE_TERMS = 2
MIN_STANDALONE_TERMS_WITH_REFERENCE = 3
TOKEN_PATTERN = r"[a-z0-9À-ÿ\u3040-\u30ff\u3400-\u9fff]+"


def get_decision_tokens(question):
    # Extract generic searchable terms for deciding if the current question already has its own topic.
    text = str(question or "").lower()

    try:
        tokens = re.findall(TOKEN_PATTERN, text, flags=re.IGNORECASE)
    except Exception:
        tokens = text.split()

    return [token for token in tokens if token]


def has_followup_reference_word(question):
    # Detect pronouns and demonstratives that can make a question contextual.
    tokens = set(get_decision_tokens(question))
    return bool(tokens.intersection(FOLLOWUP_REFERENCE_WORDS))


def get_current_topic_terms(question):
    # Keep only content-like terms so Tagalog/English standalone questions do not inherit old topics.
    terms = []

    for token in get_decision_tokens(question):
        if token in QUESTION_DECISION_STOPWORDS:
            continue

        if len(token) < MIN_SEARCHABLE_TOKEN_LENGTH:
            continue

        if token not in terms:
            terms.append(token)

    return terms


def has_clear_current_question_topic(question):
    # A question with enough current content should not be rewritten from chat history.
    terms = get_current_topic_terms(question)

    if not terms:
        return False

    if has_followup_reference_word(question):
        return len(terms) >= MIN_STANDALONE_TERMS_WITH_REFERENCE

    return len(terms) >= MIN_STANDALONE_TERMS


def should_use_chat_history_for_question(question, chat_history):
    # Use memory only for real contextual follow-ups, not standalone questions with pronouns.
    if not str(chat_history or "").strip():
        return False

    if has_clear_current_question_topic(question):
        return False

    try:
        return bool(is_follow_up_question(question))
    except Exception:
        return False


def is_clarification_response(text):
    # True when the rewriter says the follow-up question is ambiguous.
    return str(text or "").strip().upper().startswith(str(CLARIFY_PREFIX).upper())


def get_clarification_answer(text):
    # User-facing clarification answer, without the internal CLARIFY prefix.
    message = str(text or "").strip()

    if is_clarification_response(message):
        message = message[len(CLARIFY_PREFIX):].strip()

    return message or "Which person, item, or topic do you mean?"


def get_answer_question(question, rewritten_question):
    # Use the standalone rewritten question so retrieval, reranking, and prompt creation are not ambiguous.
    rewritten_question = str(rewritten_question or "").strip()

    if not rewritten_question or is_clarification_response(rewritten_question):
        return question

    return rewritten_question


def safe_rewrite_question(question, chat_history, llm):
    # Rewrite only when the user question is confirmed as a contextual follow-up.
    # Standalone questions and questions with their own subject keep the exact user input.
    question = str(question or "").strip()
    chat_history = str(chat_history or "").strip()

    if not question:
        return question

    if not ENABLE_QUESTION_REWRITE:
        return question

    if not should_use_chat_history_for_question(question, chat_history):
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
    # Return chat history only for confirmed contextual follow-up questions.
    # Standalone questions must not inherit old topics, dates, or sources.
    if should_use_chat_history_for_question(question, chat_history):
        return str(chat_history or "").strip()

    return ""


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
    'should_use_chat_history_for_question',
    'has_clear_current_question_topic',
    'get_current_topic_terms',
    'has_followup_reference_word',
    'get_decision_tokens',
    'TOKEN_PATTERN',
    'MIN_STANDALONE_TERMS_WITH_REFERENCE',
    'MIN_STANDALONE_TERMS',
    'QUESTION_DECISION_STOPWORDS',
    'FOLLOWUP_REFERENCE_WORDS',
    'is_clarification_response',
    'get_clarification_answer',
    'get_answer_question',
    'safe_rewrite_question',
    'get_effective_chat_history',
]
