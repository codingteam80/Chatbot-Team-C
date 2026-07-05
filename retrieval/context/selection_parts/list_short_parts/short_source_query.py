from retrieval.context.selection_parts.list_short_parts.list_answer import (
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
)




def get_short_source_query_config():
    # Generic configuration for short source/title queries.
    # The behavior is controlled by JSON and does not depend on sample questions.
    raw_config = read_context_json()
    context_config = raw_config.get("context_filter", {}) if isinstance(raw_config, dict) else {}

    if not isinstance(context_config, dict):
        context_config = {}

    short_config = context_config.get("short_source_query", {})

    if not isinstance(short_config, dict):
        short_config = {}

    return short_config


def get_short_source_query_terms(question):
    # Return source/title-like tokens only for short queries.
    # Numeric-only inputs are handled by answer-intent evidence rules, not source locking.
    short_config = get_short_source_query_config()

    if not safe_bool(short_config.get("enabled"), True):
        return []

    max_terms = safe_int(short_config.get("max_terms"), 3)

    if max_terms < 1:
        max_terms = 1

    terms = []

    for term in get_question_anchor_terms(question):
        term = normalize_text(term)

        if not term:
            continue

        if term.isdigit():
            continue

        if term in SOURCE_ANCHOR_WEAK_TOKENS:
            continue

        if term not in terms:
            terms.append(term)

    if not terms or len(terms) > max_terms:
        return []

    return terms


def source_key_matches_query_terms(source_key, query_terms):
    source_key = normalize_text(source_key)
    source_tokens = set(source_key.split())
    query_terms = [normalize_text(term) for term in query_terms if normalize_text(term)]

    if not source_key or not query_terms:
        return False

    if all(term in source_tokens for term in query_terms):
        return True

    query_phrase = " ".join(query_terms)

    if query_phrase and re.search(rf"\b{re.escape(query_phrase)}\b", source_key):
        return True

    return False


def get_matching_short_query_sources(question, candidate_docs):
    query_terms = get_short_source_query_terms(question)

    if not query_terms:
        return [], []

    source_scores = {}

    for doc in remove_duplicate_docs(candidate_docs):
        source_key = get_source_key(doc)

        if not source_key_matches_query_terms(source_key, query_terms):
            continue

        score = 0.0
        score += len(query_terms) * 10.0
        score += get_intro_chunk_score(doc)
        score += get_direct_window_score(query_terms, doc)
        score += max(get_context_score(doc), 0.0) * 0.10
        score += max(0, 100000 - get_original_rank(doc)) * 0.000001

        if is_low_value_context_doc(question, doc):
            score -= 3.0

        if source_key not in source_scores or score > source_scores[source_key]:
            source_scores[source_key] = score

    ranked_sources = sorted(
        source_scores,
        key=lambda source_key: source_scores[source_key],
        reverse=True,
    )

    return query_terms, ranked_sources


def get_short_query_doc_score(question, query_terms, doc):
    short_config = get_short_source_query_config()

    intro_weight = float(short_config.get("intro_weight", 8.0) or 8.0)
    direct_weight = float(short_config.get("direct_match_weight", 2.0) or 2.0)
    retrieval_weight = float(short_config.get("retrieval_weight", 1.0) or 1.0)
    low_value_penalty = float(short_config.get("low_value_penalty", 8.0) or 8.0)

    retrieval_score = 0.0

    for key in ("semantic_rank", "bm25_rank", "rerank_original_rank"):
        rank = get_rank_from_metadata(doc, (key,))

        if rank is not None:
            retrieval_score += 1.0 / max(rank, 1)

    score = 0.0
    score += get_intro_chunk_score(doc) * intro_weight
    score += get_direct_window_score(query_terms, doc) * direct_weight
    score += retrieval_score * retrieval_weight
    score += max(get_context_score(doc), 0.0) * 0.10

    if is_low_value_context_doc(question, doc):
        score -= low_value_penalty

    return score


def sort_short_query_source_docs(question, query_terms, docs):
    scored_docs = []

    for doc in remove_duplicate_docs(docs):
        doc.metadata = dict(getattr(doc, "metadata", {}) or {})
        doc.metadata["short_source_query_score"] = get_short_query_doc_score(
            question=question,
            query_terms=query_terms,
            doc=doc,
        )
        scored_docs.append(doc)

    return sorted(
        scored_docs,
        key=lambda doc: (
            float(get_metadata(doc).get("short_source_query_score", 0.0)),
            get_intro_chunk_score(doc),
            -get_original_rank(doc),
        ),
        reverse=True,
    )


def select_short_source_query_context(
    reranked_docs,
    question,
    semantic_docs=None,
    bm25_docs=None,
    all_chunks=None,
    top_n=2,
    max_chars=MAX_CONTEXT_CHARS,
):
    # Generic path for short source/title queries.
    # It uses source-title matching and intro/reference scoring, not sample-specific values.
    candidate_docs = remove_duplicate_docs(
        list(reranked_docs or []) + list(semantic_docs or []) + list(bm25_docs or [])
    )

    query_terms, ranked_sources = get_matching_short_query_sources(
        question=question,
        candidate_docs=candidate_docs,
    )

    if not query_terms or not ranked_sources:
        return []

    anchor_source_key = ranked_sources[0]

    source_docs = []

    if all_chunks:
        source_docs.extend(
            doc for doc in all_chunks
            if get_source_key(doc) == anchor_source_key
        )

    source_docs.extend(
        doc for doc in candidate_docs
        if get_source_key(doc) == anchor_source_key
    )

    source_docs = remove_duplicate_docs(source_docs)

    if not source_docs:
        return []

    ranked_docs = sort_short_query_source_docs(
        question=question,
        query_terms=query_terms,
        docs=source_docs,
    )

    filtered_docs = [
        doc for doc in ranked_docs
        if not is_low_value_context_doc(question, doc)
    ]

    if filtered_docs:
        ranked_docs = filtered_docs

    short_config = get_short_source_query_config()
    configured_top_n = safe_int(short_config.get("top_n"), top_n)

    if configured_top_n < 1:
        configured_top_n = 1

    selected_docs = ranked_docs[:configured_top_n]

    annotate_context_docs(
        docs=selected_docs,
        mode="single_fact",
        anchor_source_key=anchor_source_key,
        anchor_reason="short_source_title_match",
        filter_scope="single_fact_short_source_title_match",
        keep_reason="short_source_title_intro_or_evidence",
    )

    for doc in selected_docs:
        doc.metadata = dict(getattr(doc, "metadata", {}) or {})
        doc.metadata["short_source_query"] = True
        doc.metadata["short_source_query_terms"] = list(query_terms)

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
]
