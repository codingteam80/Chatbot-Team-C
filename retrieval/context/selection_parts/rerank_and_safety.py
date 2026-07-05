from retrieval.context.selection_parts.cross_doc_helpers import (
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
)



def get_anchor_source_keys(question, docs):
    # Reuse the same strong source-anchor rules.
    # When the question has more than one source/entity, do not lock to only one source.
    anchors = get_cross_doc_anchor_sources(question=question, docs=docs)
    return [anchor["source_key"] for anchor in anchors]


def add_unique_doc(selected_docs, seen_keys, doc):
    # Add one doc while preserving order and avoiding duplicates.
    doc_key = get_document_key(doc)

    if doc_key in seen_keys:
        return False

    seen_keys.add(doc_key)
    selected_docs.append(doc)
    return True


def annotate_context_docs(
    docs,
    mode,
    anchor_source_key,
    anchor_reason,
    filter_scope,
    keep_reason,
):
    # Save context-selection metadata for debug reports.
    for doc in docs or []:
        doc.metadata = dict(getattr(doc, "metadata", {}) or {})
        doc.metadata["context_mode"] = mode
        doc.metadata["context_anchor_source"] = anchor_source_key or "none"
        doc.metadata["context_anchor_reason"] = anchor_reason
        doc.metadata["context_confident_filter_scope"] = filter_scope

        if keep_reason and not doc.metadata.get("context_keep_reason"):
            doc.metadata["context_keep_reason"] = keep_reason

    return docs


def select_rerank_first_context(
    reranked_docs,
    question,
    semantic_docs=None,
    bm25_docs=None,
    top_n=3,
    max_chars=MAX_CONTEXT_CHARS,
):
    # Normal single_fact path.
    # top_n is a maximum only. Final docs are selected from the candidate pool
    # based on generic related/evidence signals.
    docs = remove_duplicate_docs(reranked_docs)
    docs = sort_docs_by_rerank_score(docs)

    candidate_docs = remove_duplicate_docs(
        list(docs or []) + list(semantic_docs or []) + list(bm25_docs or [])
    )
    candidate_docs = filter_low_value_context_docs(
        question=question,
        docs=candidate_docs,
        min_keep=1,
    )

    if not candidate_docs:
        return []

    anchor_source_key, anchor_reason = choose_single_fact_anchor_source(
        docs=candidate_docs,
        question=question,
    )
    anchor_source_keys = get_anchor_source_keys(question=question, docs=candidate_docs)

    if len(anchor_source_keys) >= 2:
        anchor_source_key = None
        anchor_reason = "multiple_source_anchors_no_single_source_lock"

    if anchor_source_key:
        pool = [doc for doc in candidate_docs if get_source_key(doc) == anchor_source_key]
        preserve_order_docs = [doc for doc in docs if get_source_key(doc) == anchor_source_key]
        filter_scope = "single_fact_dynamic_related_anchor_source"
    else:
        pool = candidate_docs
        preserve_order_docs = docs
        filter_scope = "single_fact_dynamic_related_no_source_lock"

    selected_docs = select_dynamic_related_docs(
        question=question,
        candidates=pool,
        semantic_docs=semantic_docs,
        bm25_docs=bm25_docs,
        top_n=top_n,
        max_chars=max_chars,
        mode="single_fact",
        min_keep=1,
        preserve_order_docs=preserve_order_docs,
    )

    if not selected_docs:
        selected_docs = candidate_docs[:1]
        anchor_reason = "fallback_top_available"
        filter_scope = "fallback_top_available"

    annotate_context_docs(
        docs=selected_docs,
        mode="single_fact",
        anchor_source_key=anchor_source_key or ", ".join(anchor_source_keys),
        anchor_reason=anchor_reason,
        filter_scope=filter_scope,
        keep_reason="dynamic_related_evidence",
    )

    return limit_context_docs(
        docs=selected_docs,
        max_chars=max_chars,
        max_per_source=None,
    )


def select_safety_context(
    reranked_docs,
    question,
    semantic_docs=None,
    bm25_docs=None,
    top_n=2,
    max_chars=MAX_CONTEXT_CHARS,
    mode="safety",
):
    # Strict path for negative / unsupported / false-premise questions.
    #
    # Rule:
    # - Small focused evidence only.
    # - No neighbor expansion.
    # - No rerank-first filling.
    # - Do not lock to a single source when multiple entities/sources are in the premise.
    # Build the candidate pool from reranked/semantic results first.
    # BM25 is still used for scoring agreement, but BM25-only chunks are not allowed
    # to enter the final list unless reranked/semantic retrieval already surfaced them.
    # This prevents keyword-only matches from filling list answers with weak chunks.
    candidates = remove_duplicate_docs(
        list(reranked_docs or []) + list(semantic_docs or [])
    )

    if not candidates:
        # Safe fallback only when rerank/semantic produced nothing.
        candidates = remove_duplicate_docs(list(bm25_docs or []))

    candidates = filter_low_value_context_docs(
        question=question,
        docs=candidates,
        min_keep=1,
    )

    if not candidates:
        return []

    scored_docs = prepare_primary_candidates(
        question=question,
        candidates=candidates,
        semantic_docs=semantic_docs,
        bm25_docs=bm25_docs,
    )

    selected_docs = trim_weak_single_fact_support(
        docs=scored_docs,
        top_n=top_n,
        min_keep=1,
    )

    anchor_source_keys = get_anchor_source_keys(question=question, docs=candidates)
    anchor_source_label = ", ".join(anchor_source_keys) if anchor_source_keys else "none"

    annotate_context_docs(
        docs=selected_docs,
        mode=mode,
        anchor_source_key=anchor_source_label,
        anchor_reason="strict_safety_primary_evidence",
        filter_scope="negative_false_premise_safety",
        keep_reason="strict_safety_evidence",
    )

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
]
