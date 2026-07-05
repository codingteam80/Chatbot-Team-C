from retrieval.context.mode_detection import (
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
)



# ============================================================
# 4A. ANSWER-INTENT AND DYNAMIC SINGLE-FACT SETTINGS
# ============================================================

def load_answer_evidence_config(config_path=DEFAULT_CONTEXT_CONFIG_PATH):
    # Load answer-intent rules from JSON.
    # Python stays generic; editable question/evidence terms stay in JSON.
    raw_config = read_context_json(config_path=config_path)

    if not isinstance(raw_config, dict):
        return {}

    answer_config = raw_config.get("answer_evidence", {})

    if not isinstance(answer_config, dict):
        return {}

    return answer_config


ANSWER_EVIDENCE_CONFIG = load_answer_evidence_config()


def get_answer_intent_configs():
    answer_config = ANSWER_EVIDENCE_CONFIG

    if not isinstance(answer_config, dict):
        return {}, []

    intents = answer_config.get("intents", {})

    if not isinstance(intents, dict):
        intents = {}

    intent_order = answer_config.get("intent_order", [])

    if not isinstance(intent_order, list) or not intent_order:
        intent_order = list(intents.keys())

    return intents, intent_order


def get_config_list(config, key):
    values = config.get(key, [])

    if isinstance(values, str):
        values = [values]

    if not isinstance(values, list):
        return []

    cleaned_values = []

    for value in values:
        value = str(value or "").strip()

        if value:
            cleaned_values.append(value)

    return cleaned_values


def config_phrase_matches(normalized_question, phrase):
    phrase = normalize_text(phrase)

    if not phrase:
        return False

    padded_question = f" {normalize_text(normalized_question)} "
    padded_phrase = f" {phrase} "

    return padded_phrase in padded_question


def detect_answer_intent(question):
    # Detect the type of answer needed by the question using JSON config.
    # This is not domain/file-specific. It only detects answer shape.
    intents, intent_order = get_answer_intent_configs()
    normalized_question = normalize_text(question)
    question_tokens = set(normalized_question.split())

    for intent_name in intent_order:
        intent_config = intents.get(intent_name, {})

        if not isinstance(intent_config, dict):
            continue

        question_regex_patterns = get_config_list(intent_config, "question_regex")

        for pattern in question_regex_patterns:
            try:
                if re.search(str(pattern), normalized_question, flags=re.IGNORECASE):
                    return intent_name
            except re.error:
                continue

        triggers = []
        triggers.extend(get_config_list(intent_config, "question_terms"))
        triggers.extend(get_config_list(intent_config, "question_phrases"))

        for trigger in triggers:
            normalized_trigger = normalize_text(trigger)

            if not normalized_trigger:
                continue

            if " " in normalized_trigger:
                if config_phrase_matches(normalized_question, normalized_trigger):
                    return intent_name
            elif normalized_trigger in question_tokens:
                return intent_name

    default_intent = ANSWER_EVIDENCE_CONFIG.get("default_intent", "general_fact")

    if not isinstance(default_intent, str) or not default_intent.strip():
        default_intent = "general_fact"

    return normalize_text(default_intent) or "general_fact"


def get_answer_intent_config(question):
    intents, _ = get_answer_intent_configs()
    intent = detect_answer_intent(question)
    intent_config = intents.get(intent, {})

    if not isinstance(intent_config, dict):
        intent_config = {}

    return intent, intent_config


def safe_int(value, default_value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default_value


def safe_float(value, default_value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default_value


def safe_bool(value, default_value):
    if isinstance(value, bool):
        return value

    if value is None:
        return default_value

    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def clamp_int(value, default_value, minimum=None, maximum=None):
    # Safe integer clamp for dynamic mode settings.
    value = safe_int(value, default_value)

    if minimum is not None:
        value = max(value, minimum)

    if maximum is not None:
        value = min(value, maximum)

    return value


def get_dynamic_retrieval_settings(question):
    from retrieval.context.selection_parts.cross_doc_helpers import should_force_single_fact_mode

    # Optional helper for the retrieval/rerank caller.
    # Context filter only receives reranked docs, so upstream code can call this
    # before semantic/BM25/hybrid/rerank to avoid reranking too many chunks.
    mode = detect_question_mode(question)

    if should_force_single_fact_mode(question, mode):
        mode = "single_fact"

    settings_by_mode = {
        "single_fact": {
            "semantic_k": 6,
            "bm25_k": 6,
            "hybrid_final_k": 8,
            "rerank_input_k": 6,
            "rerank_top_n": 6,
            "final_top_n": 5,
            "max_context_chars": 6500,
            "neighbor_expansion": False,
        },
        "list_answer": {
            "semantic_k": 10,
            "bm25_k": 10,
            "hybrid_final_k": 12,
            "rerank_input_k": 10,
            "rerank_top_n": 10,
            "final_top_n": 5,
            "max_context_chars": 6500,
            "neighbor_expansion": False,
        },
        "cross_doc": {
            "semantic_k": 10,
            "bm25_k": 10,
            "hybrid_final_k": 12,
            "rerank_input_k": 12,
            "rerank_top_n": 12,
            "final_top_n": 4,
            "max_context_chars": 6500,
            "neighbor_expansion": False,
        },
        "comparison": {
            "semantic_k": 10,
            "bm25_k": 10,
            "hybrid_final_k": 12,
            "rerank_input_k": 12,
            "rerank_top_n": 12,
            "final_top_n": 4,
            "max_context_chars": 6500,
            "neighbor_expansion": False,
        },
        "negative": {
            "semantic_k": 8,
            "bm25_k": 8,
            "hybrid_final_k": 10,
            "rerank_input_k": 8,
            "rerank_top_n": 8,
            "final_top_n": 5,
            "max_context_chars": 6500,
            "neighbor_expansion": False,
        },
        "false_premise": {
            "semantic_k": 8,
            "bm25_k": 8,
            "hybrid_final_k": 10,
            "rerank_input_k": 8,
            "rerank_top_n": 8,
            "final_top_n": 2,
            "max_context_chars": 4000,
            "neighbor_expansion": False,
        },
    }

    settings = dict(settings_by_mode.get(mode, settings_by_mode["single_fact"]))
    settings["mode"] = mode
    return settings


def get_dynamic_context_policy(question, mode, top_n=None, max_chars=MAX_CONTEXT_CHARS, max_per_source=MAX_PER_SOURCE):
    # Final-context policy. top_n is a maximum, not a target to force-fill.
    retrieval_settings = get_dynamic_retrieval_settings(question)

    default_top_n = retrieval_settings.get("final_top_n", SINGLE_FACT_TOP_N)
    requested_top_n = safe_int(top_n, None)

    if requested_top_n is None:
        effective_top_n = default_top_n
    else:
        effective_top_n = min(requested_top_n, default_top_n)

    effective_top_n = clamp_int(effective_top_n, default_top_n, minimum=1, maximum=12)

    default_max_chars = retrieval_settings.get("max_context_chars", MAX_CONTEXT_CHARS)
    effective_max_chars = min(
        clamp_int(max_chars, MAX_CONTEXT_CHARS, minimum=1000),
        clamp_int(default_max_chars, MAX_CONTEXT_CHARS, minimum=1000),
    )

    if mode == "list_answer":
        effective_max_per_source = None
    elif mode in {"cross_doc", "comparison"}:
        effective_max_per_source = 2
    else:
        effective_max_per_source = 2

    if max_per_source is not None and effective_max_per_source is not None:
        effective_max_per_source = min(
            clamp_int(max_per_source, MAX_PER_SOURCE, minimum=1),
            effective_max_per_source,
        )

    return {
        "mode": mode,
        "top_n": effective_top_n,
        "max_chars": effective_max_chars,
        "max_per_source": effective_max_per_source,
        "neighbor_expansion": bool(retrieval_settings.get("neighbor_expansion", False)),
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
]
