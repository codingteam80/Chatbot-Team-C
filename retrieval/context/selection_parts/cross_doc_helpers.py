from retrieval.context.scoring import (
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
)


def get_cross_doc_anchor_sources(question, docs):
    # For cross-doc questions, more than one source may be explicitly named.
    # Example: Treaty of Paris + Spanish-American War + Philippine-American War.
    question_terms = get_question_anchor_terms(question)
    normalized_question = normalize_text(question)
    anchors = []
    seen_sources = set()

    for doc in docs or []:
        source_key = get_source_key(doc)

        if source_key in seen_sources:
            continue

        is_strong, match_count, reason = is_strong_source_anchor(
            source_key=source_key,
            question_terms=question_terms,
            normalized_question=normalized_question,
        )

        if not is_strong:
            continue

        anchors.append(
            {
                "source_key": source_key,
                "match_count": match_count,
                "reason": reason,
                "score": get_context_score(doc),
                "original_rank": get_original_rank(doc),
            }
        )
        seen_sources.add(source_key)

    anchors.sort(
        key=lambda item: (
            item["match_count"],
            item["score"],
            -item["original_rank"],
        ),
        reverse=True,
    )

    return anchors


def is_useful_cross_doc_support(doc, selected_docs):
    # Cross-doc should prefer relevant chunks from different sources.
    # Do not fill context with completely weak chunks unless nothing else exists.
    direct_score = float(get_metadata(doc).get("cross_doc_direct_score", 0.0))
    retrieval_score = float(get_metadata(doc).get("cross_doc_retrieval_score", 0.0))
    rerank_score = float(get_metadata(doc).get("cross_doc_safe_rerank_score", 0.0))

    if direct_score > 0:
        return True

    if retrieval_score >= 1.0:
        return True

    if rerank_score > 0:
        return True

    # Allow first one if there is no selected doc yet.
    # Otherwise do not fill weak support.
    return not selected_docs


def load_mode_detection_terms(config_path=DEFAULT_CONTEXT_CONFIG_PATH):
    # Read mode detection phrases, regex patterns, and mode aliases from JSON config.
    # List/enumeration cues intentionally live in JSON, not Python.
    raw_config = read_context_json(config_path=config_path)
    mode_config = raw_config.get("mode_detection", {}) if isinstance(raw_config, dict) else {}

    if not isinstance(mode_config, dict):
        mode_config = {}

    mode_aliases = mode_config.get("mode_aliases", {})

    if not isinstance(mode_aliases, dict):
        mode_aliases = {}

    cross_doc_phrases = list(mode_config.get("cross_doc_phrases", []))
    comparison_phrases = list(mode_config.get("comparison_phrases", []))
    false_premise_phrases = list(mode_config.get("false_premise_phrases", []))
    false_premise_patterns = get_config_list(mode_config, "false_premise_patterns")
    list_phrases = list(mode_config.get("list_phrases", []))
    list_patterns = get_config_list(mode_config, "list_patterns")

    return {
        "cross_doc_phrases": normalize_config_list(cross_doc_phrases),
        "comparison_phrases": normalize_config_list(comparison_phrases),
        "false_premise_phrases": normalize_config_list(false_premise_phrases),
        "false_premise_patterns": false_premise_patterns,
        "list_phrases": normalize_config_list(list_phrases),
        "list_patterns": list_patterns,
        "cross_doc_modes": normalize_config_list(mode_aliases.get("cross_doc", [])),
        "comparison_modes": normalize_config_list(mode_aliases.get("comparison", [])),
        "negative_modes": normalize_config_list(mode_aliases.get("negative", [])),
        "false_premise_modes": normalize_config_list(mode_aliases.get("false_premise", [])),
        "list_modes": normalize_config_list(mode_aliases.get("list_answer", [])),
    }


def phrase_exists(normalized_question, phrase):
    # Exact phrase boundary match.
    normalized_question = f" {normalize_text(normalized_question)} "
    phrase = normalize_text(phrase)

    if not phrase:
        return False

    return f" {phrase} " in normalized_question


def has_any_phrase(normalized_question, phrases):
    for phrase in phrases:
        if phrase_exists(normalized_question, phrase):
            return True

    return False


def has_wildcard_phrase_match(normalized_question, pattern):
    # Supports JSON wildcard patterns like "why did * become".
    # This keeps false-premise detection configurable without hardcoding sample topics.
    pattern = str(pattern or "").strip()

    if "*" not in pattern:
        return False

    parts = [normalize_text(part) for part in pattern.split("*")]
    parts = [part for part in parts if part]

    if not parts:
        return False

    cursor = 0

    for part in parts:
        index = normalized_question.find(part, cursor)

        if index < 0:
            return False

        cursor = index + len(part)

    return True


def has_any_wildcard_or_regex(normalized_question, patterns):
    # JSON false-premise patterns may be wildcard-style or regex-style.
    for pattern in patterns or []:
        pattern_text = str(pattern or "").strip()

        if not pattern_text:
            continue

        if "*" in pattern_text and has_wildcard_phrase_match(normalized_question, pattern_text):
            return True

        try:
            if re.search(pattern_text, normalized_question, flags=re.IGNORECASE):
                return True
        except re.error:
            continue

    return False


def has_explicit_cross_doc_cue(question):
    # Cross-doc must come from explicit compare/connect/between phrases.
    # Broad standalone tokens should not force cross-doc mode.
    normalized_question = normalize_text(question)
    mode_terms = load_mode_detection_terms()

    phrases = []
    phrases.extend(mode_terms["comparison_phrases"])
    phrases.extend(mode_terms["cross_doc_phrases"])

    return has_any_phrase(normalized_question, phrases)


def should_force_single_fact_mode(question, detected_mode):
    # Query analyzer may mark broad words as cross_doc.
    # For final context, cross-doc should require a strong comparison/connection cue.
    if detected_mode not in {"cross_doc", "comparison"}:
        return False

    return not has_explicit_cross_doc_cue(question)


def select_best_source_only_context(scored_docs, top_n=3, strict=False):
    # For single-fact questions, once the best answer-bearing source is found,
    # do not add other sources just to fill the context.
    # In strict mode, if the best source has no answer evidence, scan all candidates
    # for the best answer-bearing chunk instead of forcing the top source.
    scored_docs = list(scored_docs or [])

    if not scored_docs:
        return []

    best_doc = scored_docs[0]
    best_source_key = get_source_key(best_doc)

    same_source_docs = []

    for doc in scored_docs:
        if get_source_key(doc) == best_source_key:
            same_source_docs.append(doc)

    selected_docs = trim_weak_single_fact_support(
        docs=same_source_docs,
        top_n=top_n,
        min_keep=1,
        strict=strict,
    )

    if selected_docs or not strict:
        return selected_docs

    # Strict fallback: keep only chunks with JSON-based answer evidence.
    # This is still generic; it does not check file names, topics, or chunk IDs.
    return trim_weak_single_fact_support(
        docs=scored_docs,
        top_n=top_n,
        min_keep=1,
        strict=True,
    )

def has_positive_rerank_signal(docs, min_score=0.0):
    from retrieval.context.selection_parts.cross_doc_selection import get_rerank_score

    # Positive rerank score means the reranker found at least one usable direct candidate.
    # When all scores are negative, it is safer to return to primary evidence scoring.
    for doc in docs or []:
        rerank_score = get_rerank_score(doc)

        if rerank_score is not None and rerank_score >= min_score:
            return True

    return False


def is_primary_evidence_protected_intent(intent):
    # These answer types need exact answer-shape evidence from JSON.
    # Example: birthdate/date questions must prefer chunks with "born on" style evidence,
    # not merely chunks that have a high rerank score.
    normalized_intent = normalize_text(intent)
    protected_markers = {
        "date",
        "deadline",
        "time",
    }

    for marker in protected_markers:
        if marker in normalized_intent:
            return True

    return False


def should_prefer_top_rerank_source(dynamic_settings, docs):
    # Dynamic rule by question type:
    # - list/multi-answer questions: use reranked evidence, not strict primary-evidence trimming
    # - exact date/definition-like questions: keep strict primary evidence
    # - other actor/fact/judgment questions: trust the top reranked source first,
    #   then keep only focused support from that same source.
    docs = list(docs or [])

    if not docs:
        return False

    intent = dynamic_settings.get("intent", "")

    if dynamic_settings.get("multi_answer"):
        return True

    if is_primary_evidence_protected_intent(intent):
        return False

    return True


def sort_docs_by_retrieval_priority(docs):
    # For focused single-answer support, prefer docs that retrieval/RRF already placed high.
    # This prevents low-rerank but high-evidence-looking chunks from replacing the actual source.
    return sorted(
        docs or [],
        key=lambda doc: (
            -get_original_rank(doc),
            get_context_score(doc),
        ),
        reverse=True,
    )


def select_top_rerank_source_context(
    reranked_docs,
    question,
    semantic_docs=None,
    bm25_docs=None,
    top_n=2,
    max_chars=MAX_CONTEXT_CHARS,
    multi_answer=False,
):
    from retrieval.context.selection_parts.rerank_and_safety import annotate_context_docs

    # Use the top reranked source as the anchor, but do not force-fill to top_n.
    # The final size is based on how many chunks in the pool are actually related/good.
    docs = sort_docs_by_rerank_score(remove_duplicate_docs(reranked_docs))

    if not docs:
        return []

    top_doc = docs[0]
    top_source_key = get_source_key(top_doc)

    candidate_docs = remove_duplicate_docs(
        list(docs or []) + list(semantic_docs or []) + list(bm25_docs or [])
    )

    source_pool = [
        doc for doc in candidate_docs
        if get_source_key(doc) == top_source_key
    ]

    if not source_pool:
        source_pool = [top_doc]

    mode = "list_answer" if multi_answer else "single_fact"
    selected_docs = select_dynamic_related_docs(
        question=question,
        candidates=source_pool,
        semantic_docs=semantic_docs,
        bm25_docs=bm25_docs,
        top_n=top_n,
        max_chars=max_chars,
        mode=mode,
        min_keep=1,
        preserve_order_docs=docs,
    )

    if not selected_docs:
        selected_docs = [top_doc]

    keep_reason = "dynamic_related_top_source"
    filter_scope = (
        "list_answer_dynamic_related_top_source"
        if multi_answer
        else "single_fact_dynamic_related_top_source"
    )

    annotate_context_docs(
        docs=selected_docs,
        mode=mode,
        anchor_source_key=top_source_key,
        anchor_reason="dynamic_top_rerank_source_related_pool",
        filter_scope=filter_scope,
        keep_reason=keep_reason,
    )

    for doc in selected_docs:
        doc.metadata = dict(getattr(doc, "metadata", {}) or {})
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
]
