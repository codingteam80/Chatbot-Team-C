import json
import re
import unicodedata
from pathlib import Path


DEFAULT_QUERY_CONFIG_PATH = Path("config") / "query_expansion_config.json"


def normalize_config_key(value):
    # Same normalization for config words and query words.
    text = str(value or "").lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = re.sub(r"[^a-z0-9\u3040-\u30ff\u3400-\u9fff*]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def read_query_config(config_path=DEFAULT_QUERY_CONFIG_PATH):
    # Shared JSON config reader. Empty dict means safe generic behavior.
    try:
        config_path = Path(config_path)
        if not config_path.exists():
            return {}
        return json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return {}


def normalize_config_list(values, normalize=True):
    # Convert JSON string/list to a unique list.
    if isinstance(values, str):
        values = [values]

    if not isinstance(values, list):
        return []

    cleaned = []
    for value in values:
        value = normalize_config_key(value) if normalize else str(value or "").strip()
        if value and value not in cleaned:
            cleaned.append(value)

    return cleaned


def normalize_config_set(values):
    return set(normalize_config_list(values, normalize=True))


def normalize_config_patterns(values):
    # Regex patterns must stay as-is; only remove blanks and duplicates.
    return normalize_config_list(values, normalize=False)


def get_nested(data, *keys, default=None):
    current = data
    for key in keys:
        if not isinstance(current, dict):
            return {} if default is None else default
        current = current.get(key, {} if default is None else default)
    return current


def as_int(value, default_value, minimum=None, maximum=None):
    try:
        number = int(value)
    except (TypeError, ValueError):
        number = int(default_value)

    if minimum is not None:
        number = max(number, minimum)
    if maximum is not None:
        number = min(number, maximum)
    return number


def as_bool(value, default_value=False):
    if isinstance(value, bool):
        return value
    if value is None:
        return default_value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def load_stopwords(config_path=DEFAULT_QUERY_CONFIG_PATH):
    config = read_query_config(config_path)
    return normalize_config_set(config.get("stopwords", []))


def load_mode_detection_config(config_path=DEFAULT_QUERY_CONFIG_PATH):
    config = read_query_config(config_path)
    mode_config = get_nested(config, "mode_detection")

    return {
        "cross_doc_phrases": normalize_config_list(mode_config.get("cross_doc_phrases", [])),
        "comparison_phrases": normalize_config_list(mode_config.get("comparison_phrases", [])),
        "negative_phrases": normalize_config_list(mode_config.get("negative_phrases", [])),
        "false_premise_phrases": normalize_config_list(mode_config.get("false_premise_phrases", [])),
        "false_premise_patterns": normalize_config_patterns(mode_config.get("false_premise_patterns", [])),
        "list_phrases": normalize_config_list(mode_config.get("list_phrases", [])),
        "list_patterns": normalize_config_patterns(mode_config.get("list_patterns", [])),
    }


def load_context_terms(config_path=DEFAULT_QUERY_CONFIG_PATH):
    config = read_query_config(config_path)
    context_config = get_nested(config, "context_filter")
    stopwords = load_stopwords(config_path)

    question_weak_tokens = stopwords.union(
        normalize_config_set(context_config.get("question_weak_tokens", []))
    )

    source_weak_tokens = normalize_config_set(
        context_config.get("source_weak_tokens", [])
    )

    source_anchor_weak_tokens = question_weak_tokens.union(
        normalize_config_set(context_config.get("source_anchor_weak_tokens", []))
    )

    low_value = get_nested(context_config, "low_value")

    return {
        "stopwords": stopwords,
        "source_weak_tokens": source_weak_tokens,
        "question_weak_tokens": question_weak_tokens,
        "source_anchor_weak_tokens": source_anchor_weak_tokens,
        "cross_doc_terms": normalize_config_set(context_config.get("cross_doc_terms", [])),
        "reference_markers": normalize_config_set(low_value.get("reference_markers", [])),
        "low_value_section_phrases": normalize_config_set(low_value.get("low_value_section_phrases", [])),
        "low_value_body_phrases": normalize_config_set(low_value.get("low_value_body_phrases", [])),
    }


def load_retrieval_settings_by_mode(config_path=DEFAULT_QUERY_CONFIG_PATH):
    config = read_query_config(config_path)
    settings = get_nested(config, "context_filter", "retrieval_settings_by_mode")

    if isinstance(settings, dict) and settings:
        return settings

    # Numeric fallback only. Word lists/patterns still come from JSON.
    return {
        "single_fact": {
            "semantic_k": 6,
            "bm25_k": 6,
            "hybrid_final_k": 8,
            "rerank_input_k": 6,
            "rerank_top_n": 6,
            "final_top_n": 3,
            "max_context_chars": 6000,
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
            "final_top_n": 3,
            "max_context_chars": 5000,
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


def load_query_type_patterns(config_path=DEFAULT_QUERY_CONFIG_PATH):
    config = read_query_config(config_path)
    detector_config = get_nested(config, "query_type_detector")

    return {
        "direct_fact_patterns": normalize_config_patterns(detector_config.get("direct_fact_patterns", [])),
        "explanation_patterns": normalize_config_patterns(detector_config.get("explanation_patterns", [])),
        "comparison_patterns": normalize_config_patterns(detector_config.get("comparison_patterns", [])),
        "relationship_patterns": normalize_config_patterns(detector_config.get("relationship_patterns", [])),
        "multi_part_patterns": normalize_config_patterns(detector_config.get("multi_part_patterns", [])),
        "assumption_risk_patterns": normalize_config_patterns(detector_config.get("assumption_risk_patterns", [])),
    }


def load_query_analyzer_config(config_path=DEFAULT_QUERY_CONFIG_PATH):
    config = read_query_config(config_path)
    analyzer_config = get_nested(config, "query_analyzer")

    return {
        "category_keywords": get_nested(config, "category_keywords", default={}),
        "doc_type_keywords": get_nested(config, "doc_type_keywords", default={}),
        "source_hint_patterns": normalize_config_patterns(analyzer_config.get("source_hint_patterns", [])),
        "source_hint_cleanup_patterns": normalize_config_patterns(analyzer_config.get("source_hint_cleanup_patterns", [])),
        "language_hints": get_nested(analyzer_config, "language_hints", default={}),
    }
