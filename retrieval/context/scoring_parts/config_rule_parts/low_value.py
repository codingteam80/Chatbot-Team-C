from retrieval.context.scoring_parts.config_rule_parts.base_settings import (
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
)



def config_phrase_in_text(normalized_text, phrases):
    normalized_text = f" {normalize_text(normalized_text)} "

    for phrase in phrases or []:
        phrase = normalize_text(phrase)

        if phrase and f" {phrase} " in normalized_text:
            return True

    return False


def is_reference_like_context_doc(doc):
    # Generic low-value chunk detector.
    # Reference/infobox-like markers come from query_expansion_config.json.
    metadata = get_metadata(doc)
    section = normalize_text(metadata.get("section", ""))
    title = normalize_text(metadata.get("title", ""))
    body = str(getattr(doc, "page_content", "") or "")
    normalized_body = normalize_text(body)
    combined_header = f" {title} {section} "

    if config_phrase_in_text(combined_header, LOW_VALUE_REFERENCE_MARKERS):
        return True

    if config_phrase_in_text(combined_header, LOW_VALUE_SECTION_PHRASES):
        return True

    if config_phrase_in_text(normalized_body, LOW_VALUE_BODY_PHRASES):
        return True

    marker_hits = 0

    for marker in LOW_VALUE_REFERENCE_MARKERS:
        if marker and marker in normalized_body:
            marker_hits += 1

    url_like_hits = len(re.findall(r"https?://|www\.|\bdoi\b|\bisbn\b", body, flags=re.IGNORECASE))

    if marker_hits >= 2 or url_like_hits >= 2:
        return True

    # Generic URL/query/citation noise detector.
    # This is not source-specific; it catches chunks that are mostly references,
    # copied URL query strings, archive links, or bibliography fragments.
    raw_combined_text = " ".join([
        str(metadata.get("section", "")),
        str(metadata.get("title", "")),
        body,
    ])

    has_url_query_noise = re.search(
        r"(&(?:dq|pg|id|q|source)=|books\.google|archive\.org|wayback|https?://|www\.)",
        raw_combined_text,
        flags=re.IGNORECASE,
    ) is not None

    # Match citation/list-reference fragments even when the date is written as
    # "(23 July 2020)" or when the section starts with a numbered reference.
    # One strong citation-looking section is enough because final cleanup still
    # protects chunks with configured direct answer evidence.
    citation_line_hits = len(re.findall(
        r"(?:^|\n|\s)\d{1,3}\.\s+[A-Z][^\n]{10,220}(?:\([^)]*\d{4}[^)]*\)|Retrieved|Archived|ISBN|doi|http|www\.)",
        raw_combined_text,
        flags=re.IGNORECASE,
    ))

    section_citation_like = re.search(
        r"^\s*\d{1,3}\.\s+.+(?:\([^)]*\d{4}[^)]*\)|Retrieved|Archived|ISBN|doi|http|www\.)",
        str(metadata.get("section", "")),
        flags=re.IGNORECASE,
    ) is not None

    if has_url_query_noise or section_citation_like or citation_line_hits >= 1:
        return True

    words = normalized_body.split()

    if len(words) < 25 and marker_hits > 0:
        return True

    return False


def is_low_value_context_doc(question, doc):
    from retrieval.context.scoring_parts.answer_patterns import (
        count_question_term_coverage,
        get_answer_pattern_score,
        get_doc_body_text,
    )

    # Keep answer-bearing chunks, drop weak/reference-like chunks.
    question_terms = get_question_terms(question)
    body_text = get_doc_body_text(doc)
    body_match_count = count_question_term_coverage(question_terms, body_text)
    direct_score = get_direct_window_score(question_terms, doc)
    answer_score = get_answer_pattern_score(question, question_terms, doc)

    if answer_score > 0:
        return False

    # Drop reference/infobox-like chunks before generic term-count rescue.
    # This prevents metadata/status tables from passing only because they repeat the topic words.
    if is_reference_like_context_doc(doc):
        return True

    if direct_score >= 2:
        return False

    if body_match_count >= max(2, min(3, len(question_terms))):
        return False

    body_words = normalize_text(body_text).split()

    if len(body_words) < 20 and body_match_count == 0:
        return True

    return False


def filter_low_value_context_docs(question, docs, min_keep=1):
    # Filter weak chunks but never return empty when candidates exist.
    docs = remove_duplicate_docs(docs)

    if not docs:
        return []

    kept_docs = []

    for doc in docs:
        if not is_low_value_context_doc(question, doc):
            kept_docs.append(doc)

    if kept_docs:
        return kept_docs

    return docs[:max(1, min_keep)]


def has_list_answer_evidence(question, doc):
    from retrieval.context.scoring_parts.answer_patterns import (
        count_question_term_coverage,
        get_answer_pattern_score,
        get_doc_body_text,
    )

    # Body-focused evidence check for list questions.
    # Metadata/title matches alone should not make a chunk useful.
    question_terms = get_question_terms(question)
    body_text = get_doc_body_text(doc)
    body_match_count = count_question_term_coverage(question_terms, body_text)
    direct_score = get_direct_window_score(question_terms, doc)
    answer_score = get_answer_pattern_score(question, question_terms, doc)

    return answer_score > 0 or direct_score >= 2 or body_match_count >= 2


def has_any_regex(normalized_question, patterns):
    # Regex patterns are read from JSON config.
    for pattern in patterns or []:
        try:
            if re.search(str(pattern), normalized_question, flags=re.IGNORECASE):
                return True
        except re.error:
            continue

    return False


def has_multi_answer_cue(question):
    from retrieval.context.selection_parts.cross_doc_helpers import (
        has_any_phrase,
        load_mode_detection_terms,
    )

    # Detect questions that expect a list, not one exact actor/date/value.
    # Editable list cues stay in query_expansion_config.json.
    normalized_question = normalize_text(question)

    if not normalized_question:
        return False

    mode_terms = load_mode_detection_terms()
    dynamic_config = ANSWER_EVIDENCE_CONFIG.get("dynamic_single_fact", {})

    if not isinstance(dynamic_config, dict):
        dynamic_config = {}

    list_phrases = []
    list_phrases.extend(mode_terms.get("list_phrases", []))
    list_phrases.extend(get_config_list(dynamic_config, "multi_answer_phrases"))

    if has_any_phrase(normalized_question, list_phrases):
        return True

    list_patterns = []
    list_patterns.extend(mode_terms.get("list_patterns", []))
    list_patterns.extend(get_config_list(dynamic_config, "multi_answer_patterns"))

    if has_any_regex(normalized_question, list_patterns):
        return True

    # Optional generic plural rule.
    # The blocked helper words are configurable in JSON using non_list_after_what.
    enable_plural_what_rule = safe_bool(
        dynamic_config.get("enable_plural_what_rule"),
        True,
    )

    if not enable_plural_what_rule:
        return False

    tokens = normalized_question.split()

    if len(tokens) >= 2 and tokens[0] == "what":
        second_token = tokens[1]
        non_list_after_what = set(
            normalize_text(value)
            for value in get_config_list(dynamic_config, "non_list_after_what")
        )

        if second_token not in non_list_after_what and second_token.endswith("s"):
            return True

    return False

def get_dynamic_multi_answer_settings(question, base_intent, requested_top_n, dynamic_config):
    # Multi-answer/list questions need broader context than strict single-answer facts.
    # Example: "Who are..." should not be limited like "Who killed...".
    if not has_multi_answer_cue(question):
        return None

    multi_answer_top_n = safe_int(
        dynamic_config.get("multi_answer_top_n"),
        SINGLE_FACT_TOP_N,
    )

    effective_top_n = max(
        safe_int(requested_top_n, SINGLE_FACT_TOP_N),
        multi_answer_top_n,
        SINGLE_FACT_TOP_N,
    )

    if effective_top_n < 1:
        effective_top_n = SINGLE_FACT_TOP_N

    return {
        "intent": f"multi_answer_{base_intent}",
        "top_n": effective_top_n,
        # Multi-answer/list questions should keep the selected evidence chunks.
        # Neighbor expansion can push out other relevant reranked chunks, so default is False.
        "enable_neighbor_expansion": safe_bool(
            dynamic_config.get("multi_answer_neighbor_expansion"),
            False,
        ),
        "selection_strategy": str(
            dynamic_config.get("multi_answer_selection_strategy", "rerank_first")
        ).strip().lower() or "rerank_first",
        "strict": False,
        "multi_answer": True,
    }

def get_dynamic_single_fact_settings(question, requested_top_n=None):
    # Compute effective single_fact settings per question intent.
    # settings.py keeps global defaults; JSON controls strict/dynamic behavior.
    intent, intent_config = get_answer_intent_config(question)

    dynamic_config = ANSWER_EVIDENCE_CONFIG.get("dynamic_single_fact", {})

    if not isinstance(dynamic_config, dict):
        dynamic_config = {}

    strict_intent_values = get_config_list(dynamic_config, "strict_intents")
    strict_intents = set(strict_intent_values)
    strict_intents_normalized = set(
        normalize_text(value)
        for value in strict_intent_values
    )

    requested_top_n = safe_int(requested_top_n, None)

    if requested_top_n is None:
        requested_top_n = SINGLE_FACT_TOP_N

    multi_answer_settings = get_dynamic_multi_answer_settings(
        question=question,
        base_intent=intent,
        requested_top_n=requested_top_n,
        dynamic_config=dynamic_config,
    )

    if multi_answer_settings:
        return multi_answer_settings

    configured_top_n = safe_int(intent_config.get("top_n"), None)

    if configured_top_n is None:
        if intent in strict_intents or normalize_text(intent) in strict_intents_normalized:
            configured_top_n = safe_int(dynamic_config.get("strict_top_n"), 2)
        elif normalize_text(intent) == normalize_text(dynamic_config.get("procedure_intent", "procedure")):
            configured_top_n = safe_int(dynamic_config.get("procedure_top_n"), requested_top_n)
        else:
            configured_top_n = requested_top_n

    if intent in strict_intents or normalize_text(intent) in strict_intents_normalized:
        effective_top_n = min(requested_top_n, configured_top_n)
    elif normalize_text(intent) == normalize_text(dynamic_config.get("procedure_intent", "procedure")):
        effective_top_n = max(requested_top_n, configured_top_n)
    else:
        effective_top_n = configured_top_n

    if effective_top_n < 1:
        effective_top_n = 1

    if "neighbor_expansion" in intent_config:
        effective_neighbor_expansion = safe_bool(
            intent_config.get("neighbor_expansion"),
            ENABLE_NEIGHBOR_EXPANSION,
        )
    elif intent in strict_intents or normalize_text(intent) in strict_intents_normalized:
        effective_neighbor_expansion = safe_bool(
            dynamic_config.get("strict_neighbor_expansion"),
            False,
        )
    elif normalize_text(intent) == normalize_text(dynamic_config.get("procedure_intent", "procedure")):
        effective_neighbor_expansion = safe_bool(
            dynamic_config.get("procedure_neighbor_expansion"),
            ENABLE_NEIGHBOR_EXPANSION,
        )
    else:
        # Default single-fact questions should stay focused.
        # Expansion is only enabled when JSON explicitly asks for it.
        effective_neighbor_expansion = safe_bool(
            dynamic_config.get("single_fact_neighbor_expansion"),
            False,
        )

    selection_strategy = str(intent_config.get("selection_strategy", "") or "").strip().lower()

    if not selection_strategy:
        if intent in strict_intents or normalize_text(intent) in strict_intents_normalized:
            selection_strategy = "primary_evidence"
        else:
            selection_strategy = "rerank_first"

    return {
        "intent": intent,
        "top_n": effective_top_n,
        "enable_neighbor_expansion": effective_neighbor_expansion,
        "selection_strategy": selection_strategy,
        "strict": intent in strict_intents or normalize_text(intent) in strict_intents_normalized,
        "multi_answer": False,
    }


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
]
