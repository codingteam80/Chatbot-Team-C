from retrieval.context.scoring_parts.config_rules import (
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
)



def get_doc_body_text(doc):
    return str(getattr(doc, "page_content", "") or "")


def get_doc_full_text(doc):
    metadata = get_metadata(doc)

    return " ".join(
        [
            str(metadata.get("title", "")),
            str(metadata.get("section", "")),
            str(metadata.get("file_name", "")),
            str(metadata.get("source", "")),
            get_doc_body_text(doc),
        ]
    )


def count_question_term_coverage(question_terms, text):
    normalized_text = normalize_text(text)
    match_count = 0

    for term in question_terms or []:
        if term in normalized_text:
            match_count += 1

    return match_count


def config_term_matches(text, term):
    normalized_text = f" {normalize_text(text)} "
    normalized_term = normalize_text(term)

    if not normalized_term:
        return False

    return f" {normalized_term} " in normalized_text


def has_any_evidence_term(text, terms):
    for term in terms:
        if config_term_matches(text, term):
            return True

    return False


def has_any_evidence_regex(text, patterns):
    for pattern in patterns:
        try:
            if re.search(pattern, text, flags=re.IGNORECASE):
                return True
        except re.error:
            continue

    return False


def get_answer_pattern_score(question, question_terms, doc):
    # Score whether a chunk contains the expected answer shape for the question.
    # The terms and regex patterns are loaded from JSON, not hardcoded per file/topic.
    _, intent_config = get_answer_intent_config(question)

    if not intent_config:
        return 0.0

    body_text = get_doc_body_text(doc)
    full_text = get_doc_full_text(doc)

    evidence_terms = get_config_list(intent_config, "evidence_terms")
    evidence_regex = get_config_list(intent_config, "evidence_regex")

    has_evidence = False

    if evidence_terms:
        has_evidence = has_any_evidence_term(body_text, evidence_terms)

    if not has_evidence and evidence_regex:
        has_evidence = has_any_evidence_regex(body_text, evidence_regex)

    question_match_count = count_question_term_coverage(question_terms, full_text)
    min_question_matches = safe_int(
        intent_config.get("min_question_term_matches"),
        1,
    )

    required_evidence_ok = True
    required_rules = intent_config.get("required_evidence_terms_by_question_terms", [])

    if not isinstance(required_rules, list):
        required_rules = []

    normalized_question = normalize_text(question)

    for rule in required_rules:
        if not isinstance(rule, dict):
            continue

        rule_question_terms = get_config_list(rule, "question_terms")
        rule_question_phrases = get_config_list(rule, "question_phrases")
        rule_evidence_terms = get_config_list(rule, "evidence_terms")
        rule_evidence_regex = get_config_list(rule, "evidence_regex")
        rule_exclude_terms = get_config_list(rule, "exclude_terms")
        rule_exclude_regex = get_config_list(rule, "exclude_regex")

        question_rule_matched = False

        for term in rule_question_terms:
            normalized_term = normalize_text(term)

            if normalized_term and normalized_term in set(normalized_question.split()):
                question_rule_matched = True
                break

        if not question_rule_matched:
            for phrase in rule_question_phrases:
                if config_phrase_matches(normalized_question, phrase):
                    question_rule_matched = True
                    break

        if question_rule_matched:
            if rule_exclude_terms and has_any_evidence_term(full_text, rule_exclude_terms):
                required_evidence_ok = False
                doc.metadata = dict(getattr(doc, "metadata", {}) or {})
                doc.metadata["answer_evidence_excluded"] = True
                doc.metadata["answer_evidence_exclusion_source"] = "required_rule_exclude_terms"
                break

            if rule_exclude_regex and has_any_evidence_regex(full_text, rule_exclude_regex):
                required_evidence_ok = False
                doc.metadata = dict(getattr(doc, "metadata", {}) or {})
                doc.metadata["answer_evidence_excluded"] = True
                doc.metadata["answer_evidence_exclusion_source"] = "required_rule_exclude_regex"
                break

            required_term_ok = True
            required_regex_ok = True

            if rule_evidence_terms:
                required_term_ok = has_any_evidence_term(body_text, rule_evidence_terms)

            if rule_evidence_regex:
                required_regex_ok = has_any_evidence_regex(body_text, rule_evidence_regex)

            if (rule_evidence_terms or rule_evidence_regex) and not (required_term_ok or required_regex_ok):
                required_evidence_ok = False
                break

    score = 0.0

    if has_evidence and required_evidence_ok and question_match_count >= min_question_matches:
        score += safe_float(intent_config.get("evidence_match_score"), 5.0)
    elif has_evidence and required_evidence_ok:
        score += safe_float(intent_config.get("weak_evidence_match_score"), 1.0)
    elif evidence_terms or evidence_regex:
        score -= safe_float(intent_config.get("missing_evidence_penalty"), 0.0)

    if not required_evidence_ok:
        score -= safe_float(intent_config.get("required_evidence_missing_penalty"), 6.0)

    term_match_score = safe_float(intent_config.get("question_term_match_score"), 0.5)
    max_term_bonus = safe_int(intent_config.get("max_question_term_bonus"), 3)
    score += min(question_match_count, max_term_bonus) * term_match_score

    # Store generic evidence flags so final selection can distinguish
    # true answer evidence from loose question-term overlap.
    doc.metadata = dict(getattr(doc, "metadata", {}) or {})
    doc.metadata["answer_evidence_has_evidence"] = bool(has_evidence)
    doc.metadata["answer_evidence_required_ok"] = bool(required_evidence_ok)
    doc.metadata["answer_evidence_question_match_count"] = int(question_match_count)

    return score



def get_top1_agreement_doc(semantic_docs=None, bm25_docs=None):
    # Strong generic signal: semantic top 1 and BM25 top 1 are the exact same chunk.
    # When this happens for a single_fact question, it is usually a direct-answer chunk.
    if not semantic_docs or not bm25_docs:
        return None

    semantic_top = semantic_docs[0]
    bm25_top = bm25_docs[0]

    if get_document_key(semantic_top) != get_document_key(bm25_top):
        return None

    return semantic_top


def score_primary_evidence_doc(question, question_terms, doc, semantic_rank_map, bm25_rank_map):
    # Keep the formula in one place for consistency.
    retrieval_score, semantic_rank, bm25_rank = get_retrieval_agreement_score(
        doc=doc,
        semantic_rank_map=semantic_rank_map,
        bm25_rank_map=bm25_rank_map,
    )
    direct_window_score = get_direct_window_score(
        question_terms=question_terms,
        doc=doc,
    )
    answer_pattern_score = get_answer_pattern_score(
        question=question,
        question_terms=question_terms,
        doc=doc,
    )
    intro_chunk_score = get_intro_chunk_score(doc)
    rerank_score = get_context_score(doc)

    # Evidence-first scoring:
    # - answer_pattern_score checks if the chunk has the expected answer shape
    # for the question intent, using JSON-configured terms/regex.
    # - direct_window_score keeps question terms close to the evidence.
    # - retrieval/rerank are useful, but only as secondary signals.
    primary_evidence_score = (
        (answer_pattern_score * 4.0)
        + (direct_window_score * 3.0)
        + intro_chunk_score
        + (retrieval_score * 0.35)
        + (rerank_score * 0.05)
    )

    doc.metadata = dict(getattr(doc, "metadata", {}) or {})
    doc.metadata["semantic_rank_for_primary"] = semantic_rank
    doc.metadata["bm25_rank_for_primary"] = bm25_rank
    doc.metadata["retrieval_agreement_score"] = float(retrieval_score)
    doc.metadata["direct_window_score"] = float(direct_window_score)
    doc.metadata["answer_intent"] = detect_answer_intent(question)
    doc.metadata["answer_pattern_score"] = float(answer_pattern_score)
    doc.metadata["intro_chunk_score"] = float(intro_chunk_score)
    doc.metadata["primary_evidence_score"] = float(primary_evidence_score)
    doc.metadata["primary_evidence_score_mode"] = "answer_pattern_weighted_evidence"

    return doc


def sort_primary_evidence_docs(docs):
    return sorted(
        docs or [],
        key=lambda doc: (
            float(get_metadata(doc).get("primary_evidence_score", 0.0)),
            get_context_score(doc),
            -get_original_rank(doc),
        ),
        reverse=True,
    )


def select_primary_evidence_docs(
    question,
    reranked_docs,
    semantic_docs=None,
    bm25_docs=None,
    top_n=3,
):
    # Generic selector for single_fact questions.
    # This is not hardcoded to a specific answer/source.
    # Goal: when many chunks have the same keyword match,
    # choose the one with the most direct evidence.
    base_docs = list(reranked_docs or [])

    if not has_retrieval_signal(base_docs, semantic_docs=semantic_docs, bm25_docs=bm25_docs):
        return []

    question_terms = get_question_terms(question)
    semantic_rank_map = build_rank_map(semantic_docs)
    bm25_rank_map = build_rank_map(bm25_docs)

    candidates = remove_duplicate_docs(
        list(reranked_docs or []) + list(semantic_docs or []) + list(bm25_docs or [])
    )

    agreed_top_doc = get_top1_agreement_doc(
        semantic_docs=semantic_docs,
        bm25_docs=bm25_docs,
    )

    # Strict primary-source mode:
    # When the semantic top 1 and BM25 top 1 are the exact same chunk for single_fact,
    # do not fill the context with other sources.
    # It is safer to use only 1-2 direct chunks than many related but confusing chunks.
    if agreed_top_doc is not None:
        primary_source_key = get_source_key(agreed_top_doc)
        same_source_candidates = []

        for doc in candidates:
            if get_source_key(doc) == primary_source_key:
                same_source_candidates.append(doc)

        scored_same_source_docs = []

        for doc in same_source_candidates:
            scored_doc = score_primary_evidence_doc(
                question=question,
                question_terms=question_terms,
                doc=doc,
                semantic_rank_map=semantic_rank_map,
                bm25_rank_map=bm25_rank_map,
            )
            scored_doc.metadata["primary_source_lock"] = True
            scored_doc.metadata["primary_source_lock_reason"] = "semantic_bm25_top1_exact_chunk_agreement"
            scored_same_source_docs.append(scored_doc)

        scored_same_source_docs = sort_primary_evidence_docs(scored_same_source_docs)

        # Make sure the exact agreed top chunk is included even when there are ties/noise.
        agreed_key = get_document_key(agreed_top_doc)
        ordered_docs = []
        seen_keys = set()

        for doc in scored_same_source_docs:
            if get_document_key(doc) == agreed_key:
                ordered_docs.append(doc)
                seen_keys.add(agreed_key)
                break

        for doc in scored_same_source_docs:
            doc_key = get_document_key(doc)

            if doc_key in seen_keys:
                continue

            ordered_docs.append(doc)
            seen_keys.add(doc_key)

            if len(ordered_docs) >= top_n:
                break

        return ordered_docs[:top_n]

    scored_docs = []

    for doc in candidates:
        scored_doc = score_primary_evidence_doc(
            question=question,
            question_terms=question_terms,
            doc=doc,
            semantic_rank_map=semantic_rank_map,
            bm25_rank_map=bm25_rank_map,
        )
        scored_docs.append(scored_doc)

    scored_docs = sort_primary_evidence_docs(scored_docs)

    return scored_docs[:top_n]


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
]
