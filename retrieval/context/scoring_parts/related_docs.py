from retrieval.context.scoring_parts.final_evidence import (
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
)



def get_rerank_rank(doc):
    metadata = get_metadata(doc)

    for key in ("rerank_rank", "rerank_original_rank"):
        value = metadata.get(key)

        if value is None:
            continue

        try:
            return int(value)
        except (TypeError, ValueError):
            continue

    return 999999


def get_body_question_match_count(question_terms, doc):
    return count_question_term_coverage(
        question_terms=question_terms,
        text=get_doc_body_without_retrieval_header(doc),
    )


def get_full_question_match_count(question_terms, doc):
    return count_question_term_coverage(
        question_terms=question_terms,
        text=get_doc_full_text(doc),
    )


def score_related_candidate(question, doc, semantic_rank_map=None, bm25_rank_map=None, mode="single_fact"):
    # Score whether a chunk is actually related/good enough for final context.
    # This is generic: no file names, no sample questions, no domain-specific code.
    normalized_question = normalize_text(question)
    metadata = get_metadata(doc)

    # Cache per exact question + mode during one run. This avoids re-scoring the
    # same chunk several times when the selector does cleanup passes.
    if (
        metadata.get("related_candidate_question") == normalized_question
        and metadata.get("related_candidate_mode") == mode
        and "related_candidate_score" in metadata
    ):
        return doc

    question_terms = get_question_terms(question)

    if semantic_rank_map is None:
        semantic_rank_map = {}

    if bm25_rank_map is None:
        bm25_rank_map = {}

    doc = score_primary_evidence_doc(
        question=question,
        question_terms=question_terms,
        doc=doc,
        semantic_rank_map=semantic_rank_map,
        bm25_rank_map=bm25_rank_map,
    )

    metadata = get_metadata(doc)
    answer_score = float(metadata.get("answer_pattern_score", 0.0))
    direct_score = float(metadata.get("direct_window_score", 0.0))
    retrieval_score = float(metadata.get("retrieval_agreement_score", 0.0))
    primary_score = float(metadata.get("primary_evidence_score", 0.0))
    rerank_score = get_context_score(doc)
    body_match_count = get_body_question_match_count(question_terms, doc)
    full_match_count = get_full_question_match_count(question_terms, doc)
    has_evidence = bool(metadata.get("answer_evidence_has_evidence", False))
    required_ok = bool(metadata.get("answer_evidence_required_ok", True))

    # Rerank can be negative, especially for small local models/rerankers.
    # Do not punish negative rerank too much; use it only as a small bonus when positive.
    positive_rerank_bonus = max(0.0, rerank_score) * 0.25

    # True evidence is stronger than loose term overlap. This prevents reference
    # chunks or generic same-source chunks from filling the final max top_n.
    evidence_bonus = 3.0 if (has_evidence and required_ok) else 0.0
    missing_required_penalty = -4.0 if not required_ok else 0.0

    related_score = (
        (answer_score * 1.5)
        + evidence_bonus
        + missing_required_penalty
        + (direct_score * 1.50)
        + (body_match_count * 1.50)
        + (full_match_count * 0.20)
        + (retrieval_score * 0.50)
        + positive_rerank_bonus
        + (primary_score * 0.10)
    )

    doc.metadata = dict(getattr(doc, "metadata", {}) or {})
    doc.metadata["related_body_match_count"] = int(body_match_count)
    doc.metadata["related_full_match_count"] = int(full_match_count)
    doc.metadata["related_candidate_score"] = float(related_score)
    doc.metadata["related_candidate_mode"] = mode
    doc.metadata["related_candidate_question"] = normalized_question
    doc.metadata["related_has_answer_evidence"] = bool(has_evidence and required_ok)

    # Evidence snippets are added only after final selection. Doing it here for
    # every candidate made final filtering slow.
    return doc


def sort_related_candidates(docs):
    return sorted(
        docs or [],
        key=lambda doc: (
            float(get_metadata(doc).get("related_candidate_score", 0.0)),
            float(get_metadata(doc).get("answer_pattern_score", 0.0)),
            float(get_metadata(doc).get("direct_window_score", 0.0)),
            float(get_metadata(doc).get("retrieval_agreement_score", 0.0)),
            get_context_score(doc),
            -get_original_rank(doc),
        ),
        reverse=True,
    )


def is_good_related_candidate(question, doc, mode="single_fact", rank=1, best_related_score=0.0):
    # Decide if a chunk is good enough for final context.
    # top_n is only a max; this function prevents weak chunks from filling the max.
    metadata = get_metadata(doc)
    question_terms = get_question_terms(question)
    term_count = max(1, len(question_terms))

    answer_score = float(metadata.get("answer_pattern_score", 0.0))
    direct_score = float(metadata.get("direct_window_score", 0.0))
    retrieval_score = float(metadata.get("retrieval_agreement_score", 0.0))
    related_score = float(metadata.get("related_candidate_score", 0.0))
    body_match_count = int(metadata.get("related_body_match_count", 0))
    rerank_rank = get_rerank_rank(doc)
    has_evidence = bool(metadata.get("answer_evidence_has_evidence", False))
    required_ok = bool(metadata.get("answer_evidence_required_ok", True))

    # Strong answer evidence means the chunk has the expected evidence terms/regex
    # for the detected answer intent, not just repeated words from the question.
    strong_answer_evidence = has_evidence and required_ok and answer_score >= 2.0

    # References/metadata chunks can pass only when they truly contain answer evidence.
    if is_reference_like_context_doc(doc) and not strong_answer_evidence:
        return False

    # If a question-specific required evidence rule failed, do not keep the chunk.
    if not required_ok:
        return False

    if mode == "list_answer":
        direct_needed = min(2, term_count)
        body_needed = min(2, term_count)

        if strong_answer_evidence:
            return True

        if direct_score >= direct_needed and body_match_count >= 1:
            return True

        if body_match_count >= body_needed and retrieval_score >= 0.5:
            return True

        if rerank_rank <= 4 and body_match_count >= body_needed:
            return True

        if related_score >= max(3.0, best_related_score * 0.45):
            if body_match_count > 0 or direct_score > 0 or retrieval_score >= 1.0:
                return True

        return False

    # Single-fact questions need cleaner context. Max top_n can be 5, but only
    # chunks with direct answer evidence or strong term coverage should enter.
    body_needed = min(3, term_count)
    direct_needed = min(3, term_count)

    if strong_answer_evidence:
        return True

    if body_match_count >= body_needed and direct_score >= min(2, direct_needed):
        return True

    if body_match_count >= body_needed and retrieval_score >= 1.0:
        return True

    if rerank_rank <= 2 and body_match_count >= body_needed:
        return True

    # Last gentle rescue for weak retrieval cases: keep only the very best chunk,
    # and only if it has some body evidence.
    if rank == 1 and body_match_count >= max(1, min(2, term_count)):
        return True

    return False


def select_dynamic_related_docs(
    question,
    candidates,
    semantic_docs=None,
    bm25_docs=None,
    top_n=5,
    max_chars=MAX_CONTEXT_CHARS,
    mode="single_fact",
    min_keep=1,
    preserve_order_docs=None,
):
    # Main generic final-pool selector.
    # 1. Score the whole candidate pool.
    # 2. Count how many chunks are actually good/related.
    # 3. Use that count as the dynamic final size, capped by top_n/max_chars.
    #
    # This keeps the context clean for the LLM without being too strict.
    candidates = filter_low_value_context_docs(
        question=question,
        docs=remove_duplicate_docs(candidates),
        min_keep=min_keep,
    )

    if not candidates:
        return []

    semantic_rank_map = build_rank_map(semantic_docs)
    bm25_rank_map = build_rank_map(bm25_docs)

    scored_docs = []

    for doc in candidates:
        scored_docs.append(
            score_related_candidate(
                question=question,
                doc=doc,
                semantic_rank_map=semantic_rank_map,
                bm25_rank_map=bm25_rank_map,
                mode=mode,
            )
        )

    score_order = sort_related_candidates(scored_docs)
    best_related_score = float(get_metadata(score_order[0]).get("related_candidate_score", 0.0)) if score_order else 0.0
    good_keys = set()

    for rank, doc in enumerate(score_order, start=1):
        if is_good_related_candidate(
            question=question,
            doc=doc,
            mode=mode,
            rank=rank,
            best_related_score=best_related_score,
        ):
            good_keys.add(get_document_key(doc))

    if preserve_order_docs is None:
        ordered_docs = score_order
    else:
        # Preserve rerank/retrieval order when possible, but only for good docs.
        ordered_docs = []
        seen_keys = set()
        scored_by_key = {get_document_key(doc): doc for doc in scored_docs}

        for doc in preserve_order_docs:
            doc_key = get_document_key(doc)

            if doc_key in seen_keys:
                continue

            if doc_key in scored_by_key:
                ordered_docs.append(scored_by_key[doc_key])
                seen_keys.add(doc_key)

        for doc in score_order:
            doc_key = get_document_key(doc)

            if doc_key in seen_keys:
                continue

            ordered_docs.append(doc)
            seen_keys.add(doc_key)

    from retrieval.context.selection_parts.rerank_and_safety import add_unique_doc

    selected_docs = []
    seen_keys = set()
    total_chars = 0
    max_docs = max(1, safe_int(top_n, 5))

    for doc in ordered_docs:
        if len(selected_docs) >= max_docs:
            break

        doc_key = get_document_key(doc)

        if doc_key not in good_keys:
            continue

        text_length = len(str(getattr(doc, "page_content", "") or ""))

        if selected_docs and total_chars + text_length > max_chars:
            break

        if add_unique_doc(selected_docs, seen_keys, doc):
            total_chars += text_length

    if len(selected_docs) >= min_keep:
        selected_docs = selected_docs[:max_docs]
        for selected_doc in selected_docs:
            annotate_evidence_snippet(question, selected_doc)
        return selected_docs

    # Gentle fallback: if all candidates were weak, keep only the best non-reference candidate.
    # This avoids empty context but does not flood the LLM with noise.
    fallback_docs = []

    for doc in score_order:
        if is_reference_like_context_doc(doc) and get_answer_score(doc) <= 0:
            continue

        fallback_docs.append(doc)
        break

    if fallback_docs:
        for selected_doc in fallback_docs:
            annotate_evidence_snippet(question, selected_doc)
        return fallback_docs

    fallback = score_order[:max(1, min_keep)]
    for selected_doc in fallback:
        annotate_evidence_snippet(question, selected_doc)
    return fallback


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
]
