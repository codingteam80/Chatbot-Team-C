from retrieval.context.selection_parts.single_fact import (
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
)


# ============================================================
# 5. CROSS DOCUMENT CONTEXT
# ============================================================


def get_rerank_score(doc):
    # Get only the real reranker score.
    # Important:
    # - Do not mix the Chroma distance score here.
    # - Chroma distance is lower-is-better, but rerank_score is higher-is-better.
    metadata = get_metadata(doc)
    value = metadata.get("rerank_score")

    if value is None:
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None



def keep_useful_cross_doc_candidates(docs, min_rerank_score=0.0):
    # Remove weak reranked chunks when usable positive chunks exist.
    # This is generic and is not based on source name, topic, or keyword.
    #
    # Why is this needed?
    # - When there is a positive rerank_score, it means there are clearly useful chunks.
    # - In that case, do not include negative rerank_score chunks just to fill top_n.
    # - But when all scores are negative or rerank_score is missing, still fall back to the original docs.
    docs = list(docs or [])

    if not docs:
        return []

    has_positive_rerank = False

    for doc in docs:
        rerank_score = get_rerank_score(doc)

        if rerank_score is not None and rerank_score >= min_rerank_score:
            has_positive_rerank = True
            break

    if not has_positive_rerank:
        return docs

    kept_docs = []

    for doc in docs:
        rerank_score = get_rerank_score(doc)

        # Keep docs that have not been reranked yet because they may come from semantic/BM25 retrieval.
        # at useful as support candidate.
        if rerank_score is None:
            kept_docs.append(doc)
            continue

        if rerank_score >= min_rerank_score:
            kept_docs.append(doc)

    if kept_docs:
        return kept_docs

    return docs



def score_cross_doc_candidate(question_terms, doc, semantic_rank_map, bm25_rank_map):
    # Simple and stable candidate score for cross-doc.
    #
    # Priority:
    # 1. Real rerank_score when available.
    # 2. Retrieval agreement when semantic and BM25 retrieval found the same chunk.
    # 3. Direct window score as small tie-breaker only.
    #
    # This is not hardcoded to a specific topic/source.
    rerank_score = get_rerank_score(doc)

    if rerank_score is None:
        safe_rerank_score = 0.0
    else:
        safe_rerank_score = rerank_score

    retrieval_score, semantic_rank, bm25_rank = get_retrieval_agreement_score(
        doc=doc,
        semantic_rank_map=semantic_rank_map,
        bm25_rank_map=bm25_rank_map,
    )
    direct_score = get_direct_window_score(
        question_terms=question_terms,
        doc=doc,
    )

    doc.metadata = dict(getattr(doc, "metadata", {}) or {})
    doc.metadata["cross_doc_safe_rerank_score"] = float(safe_rerank_score)
    doc.metadata["cross_doc_retrieval_score"] = float(retrieval_score)
    doc.metadata["cross_doc_direct_score"] = float(direct_score)
    doc.metadata["semantic_rank_for_cross_doc"] = semantic_rank
    doc.metadata["bm25_rank_for_cross_doc"] = bm25_rank

    return doc



def sort_cross_doc_candidates(docs):
    # Sort by stable signals.
    # safe_rerank_score is first so negative reranked chunks do not beat useful chunks.
    return sorted(
        docs or [],
        key=lambda doc: (
            float(get_metadata(doc).get("cross_doc_safe_rerank_score", 0.0)),
            float(get_metadata(doc).get("cross_doc_retrieval_score", 0.0)),
            float(get_metadata(doc).get("cross_doc_direct_score", 0.0)),
            -get_original_rank(doc),
        ),
        reverse=True,
    )



def select_cross_doc_context(
    reranked_docs,
    question="",
    semantic_docs=None,
    bm25_docs=None,
    top_n=5,
    max_chars=MAX_CONTEXT_CHARS,
    max_per_source=MAX_PER_SOURCE,
):
    # For cross-document/comparison questions.
    #
    # Final context rule:
    # - Do not source-first lock to top 1.
    # - Use multiple explicitly named sources when present.
    # - If explicit sources exist, do not add unrelated sources just to fill top_n.
    # - Fill only with useful support; top_n is maximum only.
    candidate_docs = remove_duplicate_docs(
        list(reranked_docs or []) + list(semantic_docs or []) + list(bm25_docs or [])
    )
    candidate_docs = filter_low_value_context_docs(
        question=question,
        docs=candidate_docs,
        min_keep=1,
    )

    if not candidate_docs:
        return []

    candidate_docs = keep_useful_cross_doc_candidates(candidate_docs)

    question_terms = get_question_terms(question)
    semantic_rank_map = build_rank_map(semantic_docs)
    bm25_rank_map = build_rank_map(bm25_docs)

    scored_docs = []

    for doc in candidate_docs:
        scored_doc = score_cross_doc_candidate(
            question_terms=question_terms,
            doc=doc,
            semantic_rank_map=semantic_rank_map,
            bm25_rank_map=bm25_rank_map,
        )
        scored_docs.append(scored_doc)

    scored_docs = sort_cross_doc_candidates(scored_docs)
    anchor_sources = get_cross_doc_anchor_sources(question=question, docs=scored_docs)
    anchor_source_keys = [item["source_key"] for item in anchor_sources]
    anchor_source_key_set = set(anchor_source_keys)
    anchor_reason = "multi_source_question_anchor" if anchor_sources else "no_explicit_source_anchor"

    selected_docs = []
    selected_doc_keys = set()
    source_counts = defaultdict(int)

    def add_doc(doc, filter_scope):
        doc_key = get_document_key(doc)

        if doc_key in selected_doc_keys:
            return False

        source_key = get_source_key(doc)

        if max_per_source is not None and source_counts[source_key] >= max_per_source:
            return False

        if not is_useful_cross_doc_support(doc, selected_docs):
            return False

        doc.metadata = dict(getattr(doc, "metadata", {}) or {})
        doc.metadata["context_mode"] = "cross_doc"
        doc.metadata["context_anchor_source"] = ", ".join(anchor_source_keys) if anchor_source_keys else "none"
        doc.metadata["context_anchor_reason"] = anchor_reason
        doc.metadata["context_confident_filter_scope"] = filter_scope

        selected_docs.append(doc)
        selected_doc_keys.add(doc_key)
        source_counts[source_key] += 1
        return True

    # Pass 1: one best chunk from every explicit source mentioned in the question.
    for anchor in anchor_sources:
        if len(selected_docs) >= top_n:
            break

        anchor_source_key = anchor["source_key"]

        for doc in scored_docs:
            if get_source_key(doc) != anchor_source_key:
                continue

            if add_doc(doc, "cross_doc_explicit_anchor_source"):
                break

    if anchor_source_keys:
        # Pass 2A: if explicit anchors exist, only add extra support from those same anchor sources.
        for doc in scored_docs:
            if len(selected_docs) >= top_n:
                break

            source_key = get_source_key(doc)

            if source_key not in anchor_source_key_set:
                continue

            add_doc(doc, "cross_doc_anchor_extra_support")
    else:
        # Pass 2B: no explicit anchors, allow one best chunk from each useful source.
        for doc in scored_docs:
            if len(selected_docs) >= top_n:
                break

            source_key = get_source_key(doc)

            if source_counts[source_key] > 0:
                continue

            add_doc(doc, "cross_doc_diversified_support")

        # Pass 3: optional extra support, still respecting max_per_source.
        for doc in scored_docs:
            if len(selected_docs) >= top_n:
                break

            add_doc(doc, "cross_doc_extra_useful_support")

    if not selected_docs:
        selected_docs = scored_docs[:min(top_n, len(scored_docs))]

        for doc in selected_docs:
            doc.metadata = dict(getattr(doc, "metadata", {}) or {})
            doc.metadata["context_mode"] = "cross_doc"
            doc.metadata["context_anchor_source"] = "fallback"
            doc.metadata["context_anchor_reason"] = "fallback_top_scored_docs"
            doc.metadata["context_confident_filter_scope"] = "fallback_top_scored_docs"

    final_docs = limit_context_docs(
        docs=selected_docs,
        max_chars=max_chars,
        max_per_source=max_per_source,
    )

    return final_docs


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
]
