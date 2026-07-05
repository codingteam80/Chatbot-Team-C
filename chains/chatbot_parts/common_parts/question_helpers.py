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
    'is_clarification_response',
    'get_clarification_answer',
    'get_answer_question',
    'safe_rewrite_question',
    'get_effective_chat_history',
]
