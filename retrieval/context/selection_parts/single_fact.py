from retrieval.context.selection_parts.list_and_short_query import (
    json,
    re,
    unicodedata,
    defaultdict,
    copy,
    Path,
    MAX_CONTEXT_CHARS,
    MAX_PER_SOURCE,
    SINGLE_FACT_TOP_N,
    CROSS_DOC_TOP_N,
    COMPARISON_TOP_N,
    NEGATIVE_TOP_N,
    FALSE_PREMISE_TOP_N,
    ENABLE_NEIGHBOR_EXPANSION,
    NEIGHBOR_WINDOW,
    analyze_query,
    DEFAULT_CONTEXT_CONFIG_PATH,
    normalize_text,
    normalize_config_list,
    read_context_json,
    load_context_filter_terms,
    CONTEXT_FILTER_TERMS,
    SOURCE_WEAK_TOKENS,
    QUESTION_WEAK_TOKENS,
    SOURCE_ANCHOR_WEAK_TOKENS,
    CROSS_DOC_TERMS,
    load_low_value_filter_config,
    LOW_VALUE_FILTER_CONFIG,
    LOW_VALUE_REFERENCE_MARKERS,
    LOW_VALUE_SECTION_PHRASES,
    LOW_VALUE_BODY_PHRASES,
    get_metadata,
    get_source_name,
    get_source_key,
    get_document_key,
    remove_duplicate_docs,
    get_chunk_number,
    find_chunk_position,
    clone_neighbor_doc,
    get_page_number,
    clone_neighbor_doc_once,
    expand_neighbor_chunks,
    get_context_score,
    get_original_rank,
    get_question_terms,
    get_question_anchor_terms,
    sort_docs_by_rerank_score,
    detect_question_mode,
    limit_context_docs,
    count_source_question_matches,
    source_key_has_exact_question_phrase,
    is_safe_single_token_source_anchor,
    is_strong_source_anchor,
    choose_single_fact_anchor_source,
    apply_confident_filter_inside_anchor_source,
    build_rank_map,
    get_rank_from_metadata,
    get_retrieval_rank,
    get_retrieval_agreement_score,
    get_intro_chunk_score,
    get_direct_window_score,
    has_retrieval_signal,
    load_answer_evidence_config,
    ANSWER_EVIDENCE_CONFIG,
    get_answer_intent_configs,
    get_config_list,
    config_phrase_matches,
    detect_answer_intent,
    get_answer_intent_config,
    safe_int,
    safe_float,
    safe_bool,
    clamp_int,
    get_dynamic_retrieval_settings,
    get_dynamic_context_policy,
    config_phrase_in_text,
    is_reference_like_context_doc,
    is_low_value_context_doc,
    filter_low_value_context_docs,
    has_list_answer_evidence,
    has_any_regex,
    has_multi_answer_cue,
    get_dynamic_multi_answer_settings,
    get_dynamic_single_fact_settings,
    get_doc_body_text,
    get_doc_full_text,
    count_question_term_coverage,
    config_term_matches,
    has_any_evidence_term,
    has_any_evidence_regex,
    get_answer_pattern_score,
    get_top1_agreement_doc,
    score_primary_evidence_doc,
    sort_primary_evidence_docs,
    select_primary_evidence_docs,
    get_primary_evidence_score,
    get_direct_score,
    get_answer_score,
    get_retrieval_score,
    prepare_primary_candidates,
    trim_weak_single_fact_support,
    get_doc_body_without_retrieval_header,
    split_evidence_sentences,
    get_evidence_terms_for_question,
    score_evidence_sentence,
    get_best_evidence_snippet,
    annotate_evidence_snippet,
    has_configured_final_answer_evidence,
    has_direct_final_answer_evidence,
    clean_final_context_docs,
    get_rerank_rank,
    get_body_question_match_count,
    get_full_question_match_count,
    score_related_candidate,
    sort_related_candidates,
    is_good_related_candidate,
    select_dynamic_related_docs,
    get_cross_doc_anchor_sources,
    is_useful_cross_doc_support,
    load_mode_detection_terms,
    phrase_exists,
    has_any_phrase,
    has_wildcard_phrase_match,
    has_any_wildcard_or_regex,
    has_explicit_cross_doc_cue,
    should_force_single_fact_mode,
    select_best_source_only_context,
    has_positive_rerank_signal,
    is_primary_evidence_protected_intent,
    should_prefer_top_rerank_source,
    sort_docs_by_retrieval_priority,
    select_top_rerank_source_context,
    get_anchor_source_keys,
    add_unique_doc,
    annotate_context_docs,
    select_rerank_first_context,
    select_safety_context,
    get_dynamic_list_context_limit,
    get_list_answer_doc_score,
    sort_list_answer_docs,
    select_list_answer_context,
    get_short_source_query_config,
    get_short_source_query_terms,
    source_key_matches_query_terms,
    get_matching_short_query_sources,
    get_short_query_doc_score,
    sort_short_query_source_docs,
    select_short_source_query_context,
)


def select_single_fact_context(
    reranked_docs,
    question,
    semantic_docs=None,
    bm25_docs=None,
    all_chunks=None,
    enable_neighbor_expansion=ENABLE_NEIGHBOR_EXPANSION,
    neighbor_window=NEIGHBOR_WINDOW,
    top_n=3,
    max_chars=MAX_CONTEXT_CHARS,
):
    # Dynamic direct factual question selection.
    #
    # The final behavior depends on answer shape:
    # - normal single-fact and list-style questions can keep up to the dynamic top_n
    # - list/multi-answer questions use rerank-first and no neighbor expansion
    # - false-premise questions stay strict through the separate false_premise mode
    #   so normal direct facts do not get over-trimmed.
    dynamic_settings = get_dynamic_single_fact_settings(
        question=question,
        requested_top_n=top_n,
    )

    intent = dynamic_settings["intent"]
    top_n = dynamic_settings["top_n"]
    selection_strategy = dynamic_settings["selection_strategy"]
    is_multi_answer = bool(dynamic_settings.get("multi_answer"))
    prefer_top_rerank_source = should_prefer_top_rerank_source(
        dynamic_settings=dynamic_settings,
        docs=reranked_docs,
    )

    use_neighbor_expansion = (
        enable_neighbor_expansion
        and dynamic_settings["enable_neighbor_expansion"]
        and not is_multi_answer
        and not prefer_top_rerank_source
    )

    docs = remove_duplicate_docs(reranked_docs)
    docs = filter_low_value_context_docs(
        question=question,
        docs=docs,
        min_keep=1,
    )
    docs = sort_docs_by_rerank_score(docs)

    semantic_docs = filter_low_value_context_docs(
        question=question,
        docs=semantic_docs,
        min_keep=1,
    ) if semantic_docs else semantic_docs
    bm25_docs = filter_low_value_context_docs(
        question=question,
        docs=bm25_docs,
        min_keep=1,
    ) if bm25_docs else bm25_docs

    if not docs and not semantic_docs and not bm25_docs:
        return []

    short_source_docs = select_short_source_query_context(
        reranked_docs=docs,
        question=question,
        semantic_docs=semantic_docs,
        bm25_docs=bm25_docs,
        all_chunks=all_chunks,
        top_n=top_n,
        max_chars=max_chars,
    )

    if short_source_docs:
        return short_source_docs

    # Dynamic source-first path for actor/judgment/list-style facts.
    # This prevents primary-evidence scoring from replacing correct top reranked chunks
    # with generic chunks that only contain loose evidence terms.
    if prefer_top_rerank_source:
        # Use the dynamic top_n for both normal single-fact and multi-answer cases.
        # Only false_premise remains strict through its separate mode path.
        effective_top_n = top_n
        selected_docs = select_top_rerank_source_context(
            reranked_docs=docs,
            question=question,
            semantic_docs=semantic_docs,
            bm25_docs=bm25_docs,
            top_n=effective_top_n,
            max_chars=max_chars,
            multi_answer=is_multi_answer,
        )

    elif selection_strategy == "primary_evidence":
        scored_docs = prepare_primary_candidates(
            question=question,
            candidates=list(docs or []) + list(semantic_docs or []) + list(bm25_docs or []),
            semantic_docs=semantic_docs,
            bm25_docs=bm25_docs,
        )

        selected_docs = select_best_source_only_context(
            scored_docs=scored_docs,
            top_n=top_n,
            strict=dynamic_settings["strict"],
        )

        if not selected_docs and not dynamic_settings["strict"]:
            selected_docs = scored_docs[:1]

        annotate_context_docs(
            docs=selected_docs,
            mode="single_fact",
            anchor_source_key=get_source_key(selected_docs[0]) if selected_docs else "primary_evidence",
            anchor_reason=f"dynamic_{intent}_primary_evidence",
            filter_scope="dynamic_single_fact_primary_evidence",
            keep_reason="dynamic_primary_evidence",
        )

        selected_docs = limit_context_docs(
            docs=selected_docs,
            max_chars=max_chars,
            max_per_source=None,
        )

    elif has_positive_rerank_signal(docs):
        selected_docs = select_rerank_first_context(
            reranked_docs=docs,
            question=question,
            semantic_docs=semantic_docs,
            bm25_docs=bm25_docs,
            top_n=top_n,
            max_chars=max_chars,
        )

    else:
        scored_docs = prepare_primary_candidates(
            question=question,
            candidates=list(docs or []) + list(semantic_docs or []) + list(bm25_docs or []),
            semantic_docs=semantic_docs,
            bm25_docs=bm25_docs,
        )

        selected_docs = select_best_source_only_context(
            scored_docs=scored_docs,
            top_n=top_n,
            strict=dynamic_settings["strict"],
        )

        if not selected_docs and not dynamic_settings["strict"]:
            selected_docs = scored_docs[:1]

        annotate_context_docs(
            docs=selected_docs,
            mode="single_fact",
            anchor_source_key=get_source_key(selected_docs[0]) if selected_docs else "primary_evidence",
            anchor_reason=f"dynamic_{intent}_low_confidence_primary_evidence",
            filter_scope="dynamic_single_fact_low_confidence_primary_evidence",
            keep_reason="dynamic_low_confidence_primary_evidence",
        )

        selected_docs = limit_context_docs(
            docs=selected_docs,
            max_chars=max_chars,
            max_per_source=None,
        )

    # Final generic cleanup: top_n is a maximum, not a target.
    # Remove weak fill chunks that are not related enough to the question.
    selected_docs = select_dynamic_related_docs(
        question=question,
        candidates=selected_docs,
        semantic_docs=semantic_docs,
        bm25_docs=bm25_docs,
        top_n=top_n,
        max_chars=max_chars,
        mode="list_answer" if is_multi_answer else "single_fact",
        min_keep=1,
        preserve_order_docs=selected_docs,
    )

    if use_neighbor_expansion and all_chunks and neighbor_window > 0:
        selected_docs = expand_neighbor_chunks(
            selected_docs=selected_docs,
            all_chunks=all_chunks,
            window=neighbor_window,
        )

    for doc in selected_docs or []:
        doc.metadata = dict(getattr(doc, "metadata", {}) or {})
        doc.metadata["dynamic_answer_intent"] = intent
        doc.metadata["dynamic_single_fact_top_n"] = top_n
        doc.metadata["dynamic_neighbor_expansion"] = bool(use_neighbor_expansion)
        doc.metadata["dynamic_selection_strategy"] = selection_strategy
        doc.metadata["dynamic_strict_single_fact"] = bool(dynamic_settings["strict"])
        doc.metadata["dynamic_multi_answer"] = bool(is_multi_answer)
        doc.metadata["dynamic_prefer_top_rerank_source"] = bool(prefer_top_rerank_source)

    return limit_context_docs(
        docs=selected_docs,
        max_chars=max_chars,
        max_per_source=None,
    )


# Public names exported by this compatibility/refactor module.
__all__ = [
    'json',
    're',
    'unicodedata',
    'defaultdict',
    'copy',
    'Path',
    'MAX_CONTEXT_CHARS',
    'MAX_PER_SOURCE',
    'SINGLE_FACT_TOP_N',
    'CROSS_DOC_TOP_N',
    'COMPARISON_TOP_N',
    'NEGATIVE_TOP_N',
    'FALSE_PREMISE_TOP_N',
    'ENABLE_NEIGHBOR_EXPANSION',
    'NEIGHBOR_WINDOW',
    'analyze_query',
    'DEFAULT_CONTEXT_CONFIG_PATH',
    'normalize_text',
    'normalize_config_list',
    'read_context_json',
    'load_context_filter_terms',
    'CONTEXT_FILTER_TERMS',
    'SOURCE_WEAK_TOKENS',
    'QUESTION_WEAK_TOKENS',
    'SOURCE_ANCHOR_WEAK_TOKENS',
    'CROSS_DOC_TERMS',
    'load_low_value_filter_config',
    'LOW_VALUE_FILTER_CONFIG',
    'LOW_VALUE_REFERENCE_MARKERS',
    'LOW_VALUE_SECTION_PHRASES',
    'LOW_VALUE_BODY_PHRASES',
    'get_metadata',
    'get_source_name',
    'get_source_key',
    'get_document_key',
    'remove_duplicate_docs',
    'get_chunk_number',
    'find_chunk_position',
    'clone_neighbor_doc',
    'get_page_number',
    'clone_neighbor_doc_once',
    'expand_neighbor_chunks',
    'get_context_score',
    'get_original_rank',
    'get_question_terms',
    'get_question_anchor_terms',
    'sort_docs_by_rerank_score',
    'detect_question_mode',
    'limit_context_docs',
    'count_source_question_matches',
    'source_key_has_exact_question_phrase',
    'is_safe_single_token_source_anchor',
    'is_strong_source_anchor',
    'choose_single_fact_anchor_source',
    'apply_confident_filter_inside_anchor_source',
    'build_rank_map',
    'get_rank_from_metadata',
    'get_retrieval_rank',
    'get_retrieval_agreement_score',
    'get_intro_chunk_score',
    'get_direct_window_score',
    'has_retrieval_signal',
    'load_answer_evidence_config',
    'ANSWER_EVIDENCE_CONFIG',
    'get_answer_intent_configs',
    'get_config_list',
    'config_phrase_matches',
    'detect_answer_intent',
    'get_answer_intent_config',
    'safe_int',
    'safe_float',
    'safe_bool',
    'clamp_int',
    'get_dynamic_retrieval_settings',
    'get_dynamic_context_policy',
    'config_phrase_in_text',
    'is_reference_like_context_doc',
    'is_low_value_context_doc',
    'filter_low_value_context_docs',
    'has_list_answer_evidence',
    'has_any_regex',
    'has_multi_answer_cue',
    'get_dynamic_multi_answer_settings',
    'get_dynamic_single_fact_settings',
    'get_doc_body_text',
    'get_doc_full_text',
    'count_question_term_coverage',
    'config_term_matches',
    'has_any_evidence_term',
    'has_any_evidence_regex',
    'get_answer_pattern_score',
    'get_top1_agreement_doc',
    'score_primary_evidence_doc',
    'sort_primary_evidence_docs',
    'select_primary_evidence_docs',
    'get_primary_evidence_score',
    'get_direct_score',
    'get_answer_score',
    'get_retrieval_score',
    'prepare_primary_candidates',
    'trim_weak_single_fact_support',
    'get_doc_body_without_retrieval_header',
    'split_evidence_sentences',
    'get_evidence_terms_for_question',
    'score_evidence_sentence',
    'get_best_evidence_snippet',
    'annotate_evidence_snippet',
    'has_configured_final_answer_evidence',
    'has_direct_final_answer_evidence',
    'clean_final_context_docs',
    'get_rerank_rank',
    'get_body_question_match_count',
    'get_full_question_match_count',
    'score_related_candidate',
    'sort_related_candidates',
    'is_good_related_candidate',
    'select_dynamic_related_docs',
    'get_cross_doc_anchor_sources',
    'is_useful_cross_doc_support',
    'load_mode_detection_terms',
    'phrase_exists',
    'has_any_phrase',
    'has_wildcard_phrase_match',
    'has_any_wildcard_or_regex',
    'has_explicit_cross_doc_cue',
    'should_force_single_fact_mode',
    'select_best_source_only_context',
    'has_positive_rerank_signal',
    'is_primary_evidence_protected_intent',
    'should_prefer_top_rerank_source',
    'sort_docs_by_retrieval_priority',
    'select_top_rerank_source_context',
    'get_anchor_source_keys',
    'add_unique_doc',
    'annotate_context_docs',
    'select_rerank_first_context',
    'select_safety_context',
    'get_dynamic_list_context_limit',
    'get_list_answer_doc_score',
    'sort_list_answer_docs',
    'select_list_answer_context',
    'get_short_source_query_config',
    'get_short_source_query_terms',
    'source_key_matches_query_terms',
    'get_matching_short_query_sources',
    'get_short_query_doc_score',
    'sort_short_query_source_docs',
    'select_short_source_query_context',
    'select_single_fact_context',
]
