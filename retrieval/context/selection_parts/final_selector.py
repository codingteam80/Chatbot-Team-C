from retrieval.context.selection_parts.cross_doc_selection import (
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
    select_single_fact_context,
    get_rerank_score,
    keep_useful_cross_doc_candidates,
    score_cross_doc_candidate,
    sort_cross_doc_candidates,
    select_cross_doc_context,
)


# ============================================================
# 6. MAIN FUNCTION
# ============================================================

def select_final_context_docs(
    reranked_docs,
    question,
    semantic_docs=None,
    bm25_docs=None,
    all_chunks=None,
    enable_neighbor_expansion=ENABLE_NEIGHBOR_EXPANSION,
    neighbor_window=NEIGHBOR_WINDOW,
    top_n=None,
    max_chars=MAX_CONTEXT_CHARS,
    max_per_source=MAX_PER_SOURCE,
    debug=True,
):
    # Main function after reranker.
    # Treatment depends on mode:
    # - single_fact: source/entity anchor or primary evidence fallback
    # - cross_doc/comparison: diverse supporting sources
    # - negative/false_premise: small evidence set only
    mode = detect_question_mode(question)

    if should_force_single_fact_mode(question, mode):
        mode = "single_fact"

    policy = get_dynamic_context_policy(
        question=question,
        mode=mode,
        top_n=top_n,
        max_chars=max_chars,
        max_per_source=max_per_source,
    )
    final_top_n = policy["top_n"]
    final_max_chars = policy["max_chars"]
    final_max_per_source = policy["max_per_source"]
    final_neighbor_expansion = bool(enable_neighbor_expansion and policy["neighbor_expansion"])

    if mode == "cross_doc":
        final_docs = select_cross_doc_context(
            reranked_docs=reranked_docs,
            question=question,
            semantic_docs=semantic_docs,
            bm25_docs=bm25_docs,
            top_n=final_top_n,
            max_chars=final_max_chars,
            max_per_source=final_max_per_source,
        )
    elif mode == "comparison":
        final_docs = select_cross_doc_context(
            reranked_docs=reranked_docs,
            question=question,
            semantic_docs=semantic_docs,
            bm25_docs=bm25_docs,
            top_n=final_top_n,
            max_chars=final_max_chars,
            max_per_source=final_max_per_source,
        )
    elif mode == "negative":
        final_docs = select_safety_context(
            reranked_docs=reranked_docs,
            question=question,
            semantic_docs=semantic_docs,
            bm25_docs=bm25_docs,
            top_n=final_top_n,
            max_chars=final_max_chars,
            mode="negative",
        )
    elif mode == "false_premise":
        final_docs = select_safety_context(
            reranked_docs=reranked_docs,
            question=question,
            semantic_docs=semantic_docs,
            bm25_docs=bm25_docs,
            top_n=final_top_n,
            max_chars=final_max_chars,
            mode="false_premise",
        )
    elif mode == "list_answer":
        final_docs = select_list_answer_context(
            reranked_docs=reranked_docs,
            question=question,
            semantic_docs=semantic_docs,
            bm25_docs=bm25_docs,
            top_n=final_top_n,
            max_chars=final_max_chars,
        )
    else:
        final_docs = select_single_fact_context(
            reranked_docs=reranked_docs,
            question=question,
            semantic_docs=semantic_docs,
            bm25_docs=bm25_docs,
            all_chunks=all_chunks,
            enable_neighbor_expansion=final_neighbor_expansion,
            neighbor_window=neighbor_window,
            top_n=final_top_n,
            max_chars=final_max_chars,
        )

    final_docs = clean_final_context_docs(
        question=question,
        docs=final_docs,
        mode=mode,
        min_keep=1,
    )

    final_docs = limit_context_docs(
        docs=final_docs,
        max_chars=final_max_chars,
        max_per_source=final_max_per_source,
    )

    for doc in final_docs or []:
        doc.metadata = dict(getattr(doc, "metadata", {}) or {})
        doc.metadata["dynamic_context_policy_mode"] = mode
        doc.metadata["dynamic_context_policy_top_n"] = final_top_n
        doc.metadata["dynamic_context_policy_max_chars"] = final_max_chars
        doc.metadata["dynamic_context_policy_max_per_source"] = final_max_per_source

    if debug:
        print_context_debug(
            question=question,
            mode=mode,
            docs=final_docs,
        )

    return final_docs


# ============================================================
# 7. DEBUG HELPER
# ============================================================

def print_context_debug(question, mode, docs):
    # Optional debug print to show what was selected.
    print(f"[CONTEXT FILTER] question={question}", flush=True)
    print(f"[CONTEXT FILTER] mode={mode}", flush=True)
    print(f"[CONTEXT FILTER] final_docs={len(docs or [])}", flush=True)

    for index, doc in enumerate(docs or [], start=1):
        metadata = get_metadata(doc)
        source_name = get_source_name(doc)
        source_key = get_source_key(doc)
        score = get_context_score(doc)
        original_rank = get_original_rank(doc)
        anchor_reason = metadata.get("context_anchor_reason", "")
        filter_scope = metadata.get("context_confident_filter_scope", "")
        primary_score = metadata.get("primary_evidence_score", "")
        retrieval_score = metadata.get("retrieval_agreement_score", "")
        direct_score = metadata.get("direct_window_score", "")
        answer_intent = metadata.get("answer_intent", metadata.get("dynamic_answer_intent", ""))
        answer_score = metadata.get("answer_pattern_score", "")
        dynamic_top_n = metadata.get("dynamic_single_fact_top_n", metadata.get("dynamic_context_policy_top_n", ""))
        dynamic_selection = metadata.get("dynamic_selection_strategy", "")
        dynamic_policy_max_chars = metadata.get("dynamic_context_policy_max_chars", "")
        semantic_rank = metadata.get("semantic_rank_for_primary", "")
        bm25_rank = metadata.get("bm25_rank_for_primary", "")
        final_cleanup = metadata.get("final_low_value_cleanup", "")
        removed_count = metadata.get("final_low_value_removed_count", "")
        evidence_snippet = metadata.get("evidence_snippet", "")

        page = metadata.get("page", "")
        chunk_id = metadata.get("chunk_id") or metadata.get("chunk_index") or ""
        neighbor_flag = metadata.get("neighbor_expanded", False)
        neighbor_reason = metadata.get("neighbor_expand_reason", "")
        neighbor_offset = metadata.get("neighbor_offset", "")

        print(
            f"[CONTEXT FILTER] {index}. "
            f"score={score:.4f} | "
            f"primary_score={primary_score} | "
            f"retrieval_score={retrieval_score} | "
            f"direct_score={direct_score} | "
            f"answer_intent={answer_intent} | "
            f"answer_score={answer_score} | "
            f"dynamic_top_n={dynamic_top_n} | "
            f"dynamic_selection={dynamic_selection} | "
            f"policy_max_chars={dynamic_policy_max_chars} | "
            f"semantic_rank={semantic_rank} | "
            f"bm25_rank={bm25_rank} | "
            f"final_cleanup={final_cleanup} | "
            f"removed_low_value={removed_count} | "
            f"evidence_snippet={evidence_snippet[:120]} | "
            f"original_rank={original_rank} | "
            f"page={page} | "
            f"chunk={chunk_id} | "
            f"neighbor={neighbor_flag} | "
            f"neighbor_offset={neighbor_offset} | "
            f"neighbor_reason={neighbor_reason} | "
            f"source_key={source_key} | "
            f"anchor_reason={anchor_reason} | "
            f"filter_scope={filter_scope} | "
            f"source={source_name}",
            flush=True,
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
    'get_rerank_score',
    'keep_useful_cross_doc_candidates',
    'score_cross_doc_candidate',
    'sort_cross_doc_candidates',
    'select_cross_doc_context',
    'select_final_context_docs',
    'print_context_debug',
]
