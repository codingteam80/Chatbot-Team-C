from chains.chatbot_parts.common import (
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
    is_clarification_response,
    get_clarification_answer,
    get_answer_question,
    safe_rewrite_question,
    get_effective_chat_history,
)



@cache_chatbot_resource(show_spinner="Loading chatbot components...")
def load_chatbot_components():
    # Cache this so only the first load is heavy and later Streamlit reruns are faster.
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
    # Call this after ingesting new data when loaded components need to be refreshed.
    if hasattr(load_chatbot_components, "clear"):
        load_chatbot_components.clear()


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
    'load_chatbot_components',
    'resolve_all_chunks',
    'clear_chatbot_components_cache',
]
