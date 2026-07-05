from chains.chatbot_parts.components import (
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
    load_chatbot_components,
    resolve_all_chunks,
    clear_chatbot_components_cache,
)



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
    # Flow here: hybrid -> merge unique -> rerank only.
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
    'run_retrieval_core',
    'ensure_final_context_docs',
    'retrieve_documents',
    'generate_chatbot_answer',
    'generate_chatbot_answer_with_retry',
    'build_retrieval_query_list',
]
