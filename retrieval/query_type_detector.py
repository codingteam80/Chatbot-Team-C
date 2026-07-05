import json
import re
from pathlib import Path


# Intent-based detector.
# This is not based on topic, file name, history terms, company terms, or sample questions.
# Editable cue lists and regex patterns live in config/query_expansion_config.json.

DEFAULT_QUERY_CONFIG_PATH = Path("config") / "query_expansion_config.json"


def read_query_config(config_path=DEFAULT_QUERY_CONFIG_PATH):
    try:
        config_path = Path(config_path)
        if not config_path.exists():
            return {}
        return json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return {}


def normalize_config_patterns(values):
    # Keep regex patterns from JSON as-is.
    if isinstance(values, str):
        values = [values]
    if not isinstance(values, list):
        return []

    patterns = []
    for value in values:
        value = str(value or "").strip()
        if value and value not in patterns:
            patterns.append(value)
    return patterns


def normalize_config_phrases(values):
    # Keep phrase cues in JSON, but normalize them like queries.
    if isinstance(values, str):
        values = [values]
    if not isinstance(values, list):
        return []

    phrases = []
    for value in values:
        value = str(value or "").strip().lower()
        value = " ".join(value.split())
        if value and value not in phrases:
            phrases.append(value)
    return phrases


def load_detector_config():
    raw_config = read_query_config()
    detector_config = raw_config.get("query_type_detector", {}) if isinstance(raw_config, dict) else {}
    return detector_config if isinstance(detector_config, dict) else {}


def load_json_list_config():
    # List/enumeration cues stay under mode_detection so query_analyzer and detector share them.
    raw_config = read_query_config()
    mode_config = raw_config.get("mode_detection", {}) if isinstance(raw_config, dict) else {}
    if not isinstance(mode_config, dict):
        mode_config = {}

    return {
        "phrases": normalize_config_phrases(mode_config.get("list_phrases", [])),
        "patterns": normalize_config_patterns(mode_config.get("list_patterns", [])),
    }


def get_patterns(name):
    return normalize_config_patterns(load_detector_config().get(name, []))


JSON_LIST_QUESTION_CONFIG = load_json_list_config()
JSON_LIST_QUESTION_PHRASES = JSON_LIST_QUESTION_CONFIG["phrases"]
JSON_LIST_QUESTION_PATTERNS = JSON_LIST_QUESTION_CONFIG["patterns"]

DIRECT_FACT_PATTERNS = get_patterns("direct_fact_patterns")
EXPLANATION_PATTERNS = get_patterns("explanation_patterns")
COMPARISON_PATTERNS = get_patterns("comparison_patterns")
RELATIONSHIP_PATTERNS = get_patterns("relationship_patterns")
MULTI_PART_PATTERNS = get_patterns("multi_part_patterns")
ASSUMPTION_RISK_PATTERNS = get_patterns("assumption_risk_patterns")


def normalize_query_text(question):
    text = str(question or "").strip().lower()
    text = " ".join(text.split())
    return text


def phrase_exists(normalized_query, phrase):
    normalized_query = f" {normalize_query_text(normalized_query)} "
    phrase = normalize_query_text(phrase)
    if not phrase:
        return False
    return f" {phrase} " in normalized_query


def matches_any_phrase(text, phrases):
    return any(phrase_exists(text, phrase) for phrase in phrases or [])


def count_words(text):
    return len(re.findall(r"[a-z0-9]+", str(text or "").lower()))


def matches_any(text, patterns):
    for pattern in patterns or []:
        try:
            if re.search(pattern, text):
                return True
        except re.error:
            continue
    return False


def is_list_question(question):
    text = normalize_query_text(question)
    if JSON_LIST_QUESTION_PHRASES and matches_any_phrase(text, JSON_LIST_QUESTION_PHRASES):
        return True
    if JSON_LIST_QUESTION_PATTERNS and matches_any(text, JSON_LIST_QUESTION_PATTERNS):
        return True
    return False


def is_explanation_question(question):
    text = normalize_query_text(question)
    return matches_any(text, EXPLANATION_PATTERNS)


def is_comparison_question(question):
    text = normalize_query_text(question)
    return matches_any(text, COMPARISON_PATTERNS)


def is_relationship_question(question):
    text = normalize_query_text(question)
    return matches_any(text, RELATIONSHIP_PATTERNS)


def is_multi_part_question(question):
    text = normalize_query_text(question)
    return matches_any(text, MULTI_PART_PATTERNS)


def is_assumption_risk_question(question):
    text = normalize_query_text(question)
    return matches_any(text, ASSUMPTION_RISK_PATTERNS)


def is_broad_question(question):
    return (
        is_list_question(question)
        or is_explanation_question(question)
        or is_comparison_question(question)
        or is_relationship_question(question)
        or is_multi_part_question(question)
        or is_assumption_risk_question(question)
    )


def is_direct_fact_question(question):
    text = normalize_query_text(question)
    if not text:
        return False
    if is_broad_question(text):
        return False
    return matches_any(text, DIRECT_FACT_PATTERNS)


def should_require_rerank_proximity(question, auto_enabled=True, default_value=False):
    text = normalize_query_text(question)
    if not text:
        return bool(default_value)
    if not auto_enabled:
        return bool(default_value)
    if is_broad_question(text):
        return False
    if is_direct_fact_question(text):
        return True
    if count_words(text) <= 6:
        return True
    return False


def should_expand_neighbors_for_question(question):
    return (
        is_list_question(question)
        or is_multi_part_question(question)
        or is_relationship_question(question)
        or is_assumption_risk_question(question)
    )


def get_context_top_n(question, base_top_n=3):
    base_top_n = max(int(base_top_n or 3), 1)
    if is_list_question(question):
        return max(base_top_n, 5)
    if is_comparison_question(question) or is_relationship_question(question):
        return max(base_top_n, 5)
    if is_multi_part_question(question):
        return max(base_top_n, 5)
    if is_assumption_risk_question(question):
        return max(base_top_n, 4)
    if is_explanation_question(question):
        return max(base_top_n, 4)
    return base_top_n


def get_query_type_label(question):
    text = normalize_query_text(question)
    if not text:
        return "empty"
    if is_list_question(text):
        return "list"
    if is_comparison_question(text):
        return "comparison"
    if is_relationship_question(text):
        return "relationship"
    if is_multi_part_question(text):
        return "multi_part"
    if is_assumption_risk_question(text):
        return "assumption_risk"
    if is_explanation_question(text):
        return "explanation"
    if is_direct_fact_question(text):
        return "direct_fact"
    if count_words(text) <= 6:
        return "short_fact_like"
    return "general"


def get_query_profile(question, base_top_n=3, auto_proximity=True, default_proximity=False):
    label = get_query_type_label(question)
    context_top_n = get_context_top_n(question, base_top_n=base_top_n)
    return {
        "label": label,
        "require_proximity": should_require_rerank_proximity(
            question,
            auto_enabled=auto_proximity,
            default_value=default_proximity,
        ),
        "context_top_n": context_top_n,
        "rerank_top_n": context_top_n,
        "expand_neighbors": should_expand_neighbors_for_question(question),
    }
