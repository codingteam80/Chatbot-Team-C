from retrieval.context.scoring_parts.answer_patterns import (
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
)




# ============================================================
# 4B. FINAL CONTEXT QUALITY HELPERS
# ============================================================

def get_primary_evidence_score(doc):
    # Score created by score_primary_evidence_doc().
    metadata = get_metadata(doc)

    try:
        return float(metadata.get("primary_evidence_score", 0.0))
    except (TypeError, ValueError):
        return 0.0


def get_direct_score(doc):
    # Direct window score = how many important question terms are close together in the chunk.
    metadata = get_metadata(doc)

    try:
        return float(metadata.get("direct_window_score", 0.0))
    except (TypeError, ValueError):
        return 0.0

def get_answer_score(doc):
    # Answer pattern score = whether the chunk has the expected answer shape.
    metadata = get_metadata(doc)

    try:
        return float(metadata.get("answer_pattern_score", 0.0))
    except (TypeError, ValueError):
        return 0.0



def get_retrieval_score(doc):
    # Retrieval agreement score = semantic/BM25 agreement signal.
    metadata = get_metadata(doc)

    try:
        return float(metadata.get("retrieval_agreement_score", 0.0))
    except (TypeError, ValueError):
        return 0.0


def prepare_primary_candidates(question, candidates, semantic_docs=None, bm25_docs=None):
    # Score all available candidates for final context selection.
    # Important: candidates should include reranked + semantic + BM25 docs.
    question_terms = get_question_terms(question)
    semantic_rank_map = build_rank_map(semantic_docs)
    bm25_rank_map = build_rank_map(bm25_docs)

    scored_docs = []

    for doc in remove_duplicate_docs(candidates):
        scored_doc = score_primary_evidence_doc(
            question=question,
            question_terms=question_terms,
            doc=doc,
            semantic_rank_map=semantic_rank_map,
            bm25_rank_map=bm25_rank_map,
        )
        scored_docs.append(scored_doc)

    return sort_primary_evidence_docs(scored_docs)


def trim_weak_single_fact_support(docs, top_n=3, min_keep=1, strict=False):
    # Do not force-fill top_n when the support chunks are already weak.
    # In strict mode, rank 1 is not automatically trusted.
    # It must contain the answer shape required by the dynamic JSON intent.
    docs = list(docs or [])

    if not docs:
        return []

    best_score = get_primary_evidence_score(docs[0])
    best_direct = get_direct_score(docs[0])
    kept_docs = []

    for index, doc in enumerate(docs, start=1):
        if len(kept_docs) >= top_n:
            break

        primary_score = get_primary_evidence_score(doc)
        direct_score = get_direct_score(doc)
        answer_score = get_answer_score(doc)
        retrieval_score = get_retrieval_score(doc)
        rerank_score = get_context_score(doc)

        keep_reason = ""

        if strict:
            # Strict single-fact rule:
            # never keep a chunk only because it is rank 1.
            # The chunk must have the expected answer evidence from JSON.
            if answer_score <= 0:
                continue

            if direct_score > 0:
                keep_reason = "strict_answer_evidence"
            elif primary_score >= max(1.0, best_score * 0.35):
                keep_reason = "strict_primary_answer_evidence"
        else:
            if index == 1:
                keep_reason = "best_primary_evidence"
            elif answer_score > 0 and primary_score >= max(1.0, best_score * 0.35):
                keep_reason = "useful_answer_pattern_support"
            elif direct_score > 0 and answer_score >= 0 and primary_score >= max(1.0, best_score * 0.45):
                keep_reason = "useful_direct_support"
            elif retrieval_score >= 4.0 and answer_score > 0 and primary_score >= max(1.0, best_score * 0.35):
                keep_reason = "semantic_bm25_agreement_answer_support"
            elif best_direct <= 1 and direct_score > 0 and answer_score >= 0:
                # When the top 1 direct match is weak, rescue lower-ranked docs that have direct terms.
                keep_reason = "rescued_lower_rank_direct_evidence"
            elif rerank_score > 0 and answer_score > 0 and primary_score >= max(1.0, best_score * 0.50):
                keep_reason = "positive_rerank_answer_support"

        if keep_reason:
            doc.metadata = dict(getattr(doc, "metadata", {}) or {})
            doc.metadata["context_keep_reason"] = keep_reason
            doc.metadata["strict_single_fact_filter"] = bool(strict)
            kept_docs.append(doc)

    if len(kept_docs) >= min_keep:
        return kept_docs[:top_n]

    if strict:
        return []

    return docs[:min_keep]



# ============================================================
# 4C. DYNAMIC RELATED-CHUNK SELECTION HELPERS
# ============================================================


def get_doc_body_without_retrieval_header(doc):
    # Some ingested chunks start with "Retrieval context: title: ... language: en".
    # For evidence snippets and body matching, prefer the actual body after that header.
    body = str(getattr(doc, "page_content", "") or "")

    if "Retrieval context:" not in body[:120]:
        return body

    language_match = re.search(r"\blanguage\s*:\s*[a-z]{2}\s+", body[:600], flags=re.IGNORECASE)

    if language_match:
        return body[language_match.end():].strip()

    # Fallback: many chunks place the real text after the last metadata pipe.
    header_end = body[:600].rfind("|")

    if header_end >= 0:
        return body[header_end + 1:].strip()

    return body


def split_evidence_sentences(text):
    # Generic sentence splitter for preview/evidence only.
    # It is intentionally simple and language-agnostic enough for English/Japanese chunks.
    text = str(text or "").replace("\r", "\n")
    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return []

    parts = re.split(r"(?<=[.!?。！？])\s+|\n+", text)
    cleaned_parts = []

    for part in parts:
        part = part.strip()

        if len(part) < 20:
            continue

        cleaned_parts.append(part)

    if cleaned_parts:
        return cleaned_parts

    return [text[:500].strip()]


def get_evidence_terms_for_question(question):
    _, intent_config = get_answer_intent_config(question)

    if not isinstance(intent_config, dict):
        return [], []

    return (
        get_config_list(intent_config, "evidence_terms"),
        get_config_list(intent_config, "evidence_regex"),
    )


def score_evidence_sentence(question, sentence):
    # Score a single sentence as an answer-bearing preview.
    # Uses only question terms and JSON-configured evidence terms/regex.
    question_terms = get_question_terms(question)
    normalized_sentence = normalize_text(sentence)
    score = 0.0

    for term in question_terms:
        if term and term in normalized_sentence:
            score += 1.0

    evidence_terms, evidence_regex = get_evidence_terms_for_question(question)

    for term in evidence_terms:
        if config_term_matches(sentence, term):
            score += 2.0

    for pattern in evidence_regex:
        try:
            if re.search(pattern, sentence, flags=re.IGNORECASE):
                score += 2.0
        except re.error:
            continue

    return score


def get_best_evidence_snippet(question, doc, max_chars=360):
    # Return the best answer-bearing sentence/snippet for report/source UI.
    # This fixes misleading section labels without hardcoding any topic.
    body = get_doc_body_without_retrieval_header(doc)
    sentences = split_evidence_sentences(body)

    if not sentences:
        return ""

    best_sentence = max(
        sentences,
        key=lambda sentence: score_evidence_sentence(question, sentence),
    )

    best_sentence = re.sub(r"\s+", " ", best_sentence).strip()

    if len(best_sentence) > max_chars:
        return best_sentence[:max_chars].rstrip() + "..."

    return best_sentence


def annotate_evidence_snippet(question, doc):
    # Store a better preview for downstream reports/source drawer.
    snippet = get_best_evidence_snippet(question, doc)

    if snippet:
        doc.metadata = dict(getattr(doc, "metadata", {}) or {})
        doc.metadata["evidence_snippet"] = snippet

    return doc


def has_configured_final_answer_evidence(question, doc):
    # Strong protection for reference-like chunks.
    # If a chunk looks like bibliography/URL noise, keep it only when JSON
    # answer-evidence rules found the expected answer shape in the body.
    question_terms = get_question_terms(question)
    answer_score = get_answer_pattern_score(
        question=question,
        question_terms=question_terms,
        doc=doc,
    )
    metadata = get_metadata(doc)

    has_configured_evidence = bool(metadata.get("answer_evidence_has_evidence"))
    required_evidence_ok = metadata.get("answer_evidence_required_ok", True) is not False
    evidence_excluded = bool(metadata.get("answer_evidence_excluded"))

    return bool(
        has_configured_evidence
        and required_evidence_ok
        and not evidence_excluded
        and answer_score > 0
    )


def has_direct_final_answer_evidence(question, doc):
    # Generic final evidence check for non-reference chunks.
    # Reference-like chunks are handled more strictly by
    # has_configured_final_answer_evidence().
    if has_configured_final_answer_evidence(question, doc):
        return True

    question_terms = get_question_terms(question)
    body_text = get_doc_body_without_retrieval_header(doc)
    body_match_count = count_question_term_coverage(question_terms, body_text)
    direct_score = get_direct_window_score(question_terms, doc)

    required_body_matches = max(2, min(4, len(question_terms)))

    if direct_score >= 4 and body_match_count >= required_body_matches:
        return True

    return False


def clean_final_context_docs(question, docs, mode="single_fact", min_keep=1):
    # Last gate before the LLM.
    # Remove reference/URL/bibliography-like chunks unless they contain
    # direct answer evidence. top_n remains a maximum, not a fill target.
    docs = remove_duplicate_docs(docs)

    if not docs:
        return []

    cleaned_docs = []
    removed_count = 0

    for doc in docs:
        reference_like = is_reference_like_context_doc(doc)

        # Reference-like chunks are allowed only when they contain configured
        # direct answer evidence. This prevents bibliography/URL chunks from
        # passing because of loose question-term proximity.
        if reference_like and not has_configured_final_answer_evidence(question, doc):
            removed_count += 1
            continue

        if not reference_like and not has_direct_final_answer_evidence(question, doc):
            # Do not over-filter all modes; this is only a last-pass cleanup.
            # Non-reference chunks may still be useful support, so keep them.
            pass

        doc = annotate_evidence_snippet(question, doc)
        doc.metadata = dict(getattr(doc, "metadata", {}) or {})
        doc.metadata["final_low_value_cleanup"] = "kept"
        doc.metadata["final_low_value_cleanup_mode"] = mode
        cleaned_docs.append(doc)

    if cleaned_docs:
        for doc in cleaned_docs:
            doc.metadata = dict(getattr(doc, "metadata", {}) or {})
            doc.metadata["final_low_value_removed_count"] = removed_count

        return cleaned_docs

    non_reference_fallback_docs = [doc for doc in docs if not is_reference_like_context_doc(doc)]
    fallback_source_docs = non_reference_fallback_docs or docs
    fallback_docs = fallback_source_docs[:max(1, min_keep)]

    for doc in fallback_docs:
        doc = annotate_evidence_snippet(question, doc)
        doc.metadata = dict(getattr(doc, "metadata", {}) or {})
        doc.metadata["final_low_value_cleanup"] = "fallback_kept_to_avoid_empty_context"
        doc.metadata["final_low_value_removed_count"] = removed_count

    return fallback_docs


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
]
