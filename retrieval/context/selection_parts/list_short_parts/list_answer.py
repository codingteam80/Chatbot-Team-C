from retrieval.context.selection_parts.rerank_and_safety import (
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
)




def get_dynamic_list_context_limit(question, docs, max_chars=MAX_CONTEXT_CHARS, top_n=None):
    # Dynamic list limit.
    # top_n is a maximum only; weak chunks are not added just to fill it.
    docs = list(docs or [])

    if not docs:
        return 0

    max_docs = safe_int(top_n, 5)

    if max_docs < 1:
        max_docs = 1

    useful_docs = []

    for doc in docs:
        if is_low_value_context_doc(question, doc):
            continue

        if has_list_answer_evidence(question, doc):
            useful_docs.append(doc)

    if not useful_docs:
        useful_docs = docs[:1]

    total_chars = 0
    dynamic_count = 0

    for doc in useful_docs:
        if dynamic_count >= max_docs:
            break

        text_length = len(str(getattr(doc, "page_content", "") or ""))

        if dynamic_count > 0 and total_chars + text_length > max_chars:
            break

        total_chars += text_length
        dynamic_count += 1

    return max(dynamic_count, 1)


def get_list_answer_doc_score(question, doc, semantic_rank_map, bm25_rank_map):
    # Dynamic score for list/list-style questions.
    # Generic: combines answer-shape, direct terms, retrieval agreement, and rerank.
    question_terms = get_question_terms(question)

    scored_doc = score_primary_evidence_doc(
        question=question,
        question_terms=question_terms,
        doc=doc,
        semantic_rank_map=semantic_rank_map,
        bm25_rank_map=bm25_rank_map,
    )

    metadata = get_metadata(scored_doc)
    rerank_score = get_context_score(scored_doc)
    primary_score = float(metadata.get("primary_evidence_score", 0.0))
    direct_score = float(metadata.get("direct_window_score", 0.0))
    answer_score = float(metadata.get("answer_pattern_score", 0.0))
    retrieval_score = float(metadata.get("retrieval_agreement_score", 0.0))

    list_score = (
        primary_score
        + direct_score
        + answer_score
        + retrieval_score
        + (rerank_score * 0.10)
    )

    scored_doc.metadata["list_answer_score"] = float(list_score)
    return scored_doc


def sort_list_answer_docs(docs):
    return sorted(
        docs or [],
        key=lambda doc: (
            float(get_metadata(doc).get("list_answer_score", 0.0)),
            get_context_score(doc),
            -get_original_rank(doc),
        ),
        reverse=True,
    )


def select_list_answer_context(
    reranked_docs,
    question,
    semantic_docs=None,
    bm25_docs=None,
    top_n=5,
    max_chars=MAX_CONTEXT_CHARS,
):
    # Dynamic selector for list/list-style questions.
    # The number of final chunks is based on how many related/good chunks exist
    # in the pool, capped by top_n. BM25-only chunks may be used only when they
    # pass the same generic evidence/relatedness checks.
    candidates = remove_duplicate_docs(
        list(reranked_docs or []) + list(semantic_docs or []) + list(bm25_docs or [])
    )
    candidates = filter_low_value_context_docs(
        question=question,
        docs=candidates,
        min_keep=1,
    )

    if not candidates:
        return []

    semantic_rank_map = build_rank_map(semantic_docs)
    bm25_rank_map = build_rank_map(bm25_docs)

    scored_docs = []

    for doc in candidates:
        scored_docs.append(
            get_list_answer_doc_score(
                question=question,
                doc=doc,
                semantic_rank_map=semantic_rank_map,
                bm25_rank_map=bm25_rank_map,
            )
        )
        score_related_candidate(
            question=question,
            doc=doc,
            semantic_rank_map=semantic_rank_map,
            bm25_rank_map=bm25_rank_map,
            mode="list_answer",
        )

    scored_docs = sort_list_answer_docs(scored_docs)

    source_scores = defaultdict(float)
    source_counts = defaultdict(int)

    for doc in scored_docs:
        source_key = get_source_key(doc)
        list_score = float(get_metadata(doc).get("list_answer_score", 0.0))
        related_score = float(get_metadata(doc).get("related_candidate_score", 0.0))
        score = list_score + related_score

        if score <= 0:
            continue

        source_scores[source_key] += score
        source_counts[source_key] += 1

    if source_scores:
        anchor_source_key = max(
            source_scores,
            key=lambda key: (source_scores[key], source_counts[key]),
        )
        source_pool = [
            doc for doc in scored_docs
            if get_source_key(doc) == anchor_source_key
        ]
        anchor_reason = "dynamic_list_answer_best_related_source_cluster"
    else:
        anchor_source_key = get_source_key(scored_docs[0])
        source_pool = [
            doc for doc in scored_docs
            if get_source_key(doc) == anchor_source_key
        ]
        anchor_reason = "dynamic_list_answer_fallback_top_source"

    preserve_order_docs = remove_duplicate_docs(list(reranked_docs or []) + list(semantic_docs or []) + list(bm25_docs or []))

    selected_docs = select_dynamic_related_docs(
        question=question,
        candidates=source_pool,
        semantic_docs=semantic_docs,
        bm25_docs=bm25_docs,
        top_n=top_n,
        max_chars=max_chars,
        mode="list_answer",
        min_keep=1,
        preserve_order_docs=preserve_order_docs,
    )

    if not selected_docs:
        selected_docs = scored_docs[:1]

    annotate_context_docs(
        docs=selected_docs,
        mode="list_answer",
        anchor_source_key=anchor_source_key,
        anchor_reason=anchor_reason,
        filter_scope="dynamic_list_answer_related_source_cluster",
        keep_reason="dynamic_list_answer_related_evidence",
    )

    for doc in selected_docs:
        doc.metadata = dict(getattr(doc, "metadata", {}) or {})
        doc.metadata["dynamic_multi_answer"] = True
        doc.metadata["dynamic_selection_strategy"] = "dynamic_related_list_answer"
        doc.metadata["dynamic_related_pool_size"] = len(source_pool)
        doc.metadata["dynamic_related_selected_count"] = len(selected_docs)

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
]
