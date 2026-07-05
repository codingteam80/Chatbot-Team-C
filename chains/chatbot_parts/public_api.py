from chains.chatbot_parts.retrieval_flow import (
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
    run_retrieval_core,
    ensure_final_context_docs,
    retrieve_documents,
    generate_chatbot_answer,
    generate_chatbot_answer_with_retry,
    build_retrieval_query_list,
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








def ask_rag_stream(
    question,
    vectorstore,
    bm25_retriever,
    reranker,
    llm,
    chat_history="",
    debug=False,
    all_chunks=None,
):
    # UI path synchronized with the non-streaming test RAG answer path.
    # Retrieval/rerank still runs first, then the final shared answer is emitted as one chunk.
    question = str(question or "").strip()

    if not question:
        response = build_response(answer="No question entered.")
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
    'ask_rag',
    'ask_rag_stream',
]
