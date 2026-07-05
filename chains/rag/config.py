import html
import json
import re
import unicodedata
from pathlib import Path

from config.settings import MAX_CONTEXT_CHARS, MAX_PER_SOURCE, NO_ANSWER_TEXT, PREVIEW_CHARS
from llm.prompt_builder import build_rag_prompt
from retrieval.context_filter import select_final_context_docs


EMPTY_VALUES = {"", "none", "nan", "null"}
SCORE_KEYS = ["semantic_distance", "hybrid_score", "metadata_boosted_score", "rerank_score"]



DEFAULT_QUERY_CONFIG_PATH = Path("config") / "query_expansion_config.json"


def read_query_config(config_path=DEFAULT_QUERY_CONFIG_PATH):
    # Read shared JSON config. Editable words and patterns live in JSON, not Python.
    try:
        config_path = Path(config_path)

        if not config_path.exists():
            return {}

        return json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return {}


def normalize_config_list(values):
    if isinstance(values, str):
        values = [values]

    if not isinstance(values, list):
        return []

    cleaned = []

    for value in values:
        value = str(value or "").strip()

        if value and value not in cleaned:
            cleaned.append(value)

    return cleaned


def normalize_config_key(value):
    value = str(value or "").lower()
    value = unicodedata.normalize("NFKD", value)
    value = "".join(char for char in value if not unicodedata.combining(char))
    value = re.sub(r"[^a-z0-9\u3040-\u30ff\u3400-\u9fff]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def normalize_config_set(values):
    return set(normalize_config_key(value) for value in normalize_config_list(values) if normalize_config_key(value))


def config_int(config, key, default_value):
    try:
        return int(config.get(key, default_value))
    except (TypeError, ValueError):
        return default_value


def config_bool(config, key, default_value=False):
    value = config.get(key, default_value)

    if isinstance(value, bool):
        return value

    if value is None:
        return default_value

    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def get_nested_dict(data, *keys):
    current = data

    for key in keys:
        if not isinstance(current, dict):
            return {}

        current = current.get(key, {})

    return current if isinstance(current, dict) else {}


def load_candidate_checklist_config():
    raw_config = read_query_config()

    if not isinstance(raw_config, dict):
        raw_config = {}

    checklist_config = raw_config.get("candidate_checklist", {})

    if not isinstance(checklist_config, dict):
        checklist_config = {}

    mode_config = get_nested_dict(raw_config, "mode_detection")
    list_intent = get_nested_dict(raw_config, "answer_evidence", "intents", "list_answer")

    list_question_patterns = normalize_config_list(
        checklist_config.get("list_question_patterns")
        or mode_config.get("list_patterns")
        or list_intent.get("question_patterns")
    )

    signal_terms = normalize_config_list(
        checklist_config.get("signal_terms")
        or list_intent.get("evidence_terms")
    )

    return {
        "enabled": config_bool(checklist_config, "enabled", True),
        "max_items": config_int(checklist_config, "max_items", 24),
        "evidence_chars": config_int(checklist_config, "evidence_chars", 160),
        "include_evidence": config_bool(checklist_config, "include_evidence", True),
        "apply_local_non_target_filter": config_bool(checklist_config, "apply_local_non_target_filter", False),
        "list_question_patterns": list_question_patterns,
        "who_question_patterns": normalize_config_list(checklist_config.get("who_question_patterns")),
        "stopwords": normalize_config_set(checklist_config.get("stopwords") or raw_config.get("stopwords")),
        "signal_terms": set(normalize_config_key(term) for term in signal_terms if normalize_config_key(term)),
        "weak_candidate_starts": normalize_config_set(checklist_config.get("weak_candidate_starts")),
        "weak_candidate_words": normalize_config_set(checklist_config.get("weak_candidate_words")),
        "who_mode_weak_entity_words": normalize_config_set(checklist_config.get("who_mode_weak_entity_words")),
        "who_mode_noise_starts": normalize_config_set(checklist_config.get("who_mode_noise_starts")),
        "target_type_question_terms": normalize_config_set(checklist_config.get("target_type_question_terms")),
        "obvious_non_target_words": normalize_config_set(checklist_config.get("obvious_non_target_words")),
        "background_actor_patterns": normalize_config_list(checklist_config.get("background_actor_patterns")),
        "relation_local_terms": set(normalize_config_key(term) for term in normalize_config_list(checklist_config.get("relation_local_terms")) if normalize_config_key(term)),
        "role_name_terms": normalize_config_list(checklist_config.get("role_name_terms")),
        "sentence_start_skip_words": normalize_config_set(checklist_config.get("sentence_start_skip_words")),
        "internal_weak_words": normalize_config_set(checklist_config.get("internal_weak_words")),
        "short_weak_values": normalize_config_set(checklist_config.get("short_weak_values")),
        "checklist_header": str(checklist_config.get("header", "LIST COVERAGE HINT:")).strip() or "LIST COVERAGE HINT:",
        "checklist_intro_lines": normalize_config_list(checklist_config.get("instructions")),
        "short_answer_rules": normalize_config_list(checklist_config.get("short_answer_rules")),
        "fallback_on_truncation": config_bool(checklist_config, "fallback_on_truncation", True),
        "fallback_max_items": config_int(checklist_config, "fallback_max_items", 12),
    }


CANDIDATE_CHECKLIST_CONFIG = load_candidate_checklist_config()
MAX_CANDIDATE_CHECKLIST_ITEMS = CANDIDATE_CHECKLIST_CONFIG["max_items"]
CANDIDATE_EVIDENCE_CHARS = CANDIDATE_CHECKLIST_CONFIG["evidence_chars"]
CANDIDATE_CHECKLIST_ENABLED = CANDIDATE_CHECKLIST_CONFIG["enabled"]
CANDIDATE_CHECKLIST_INCLUDE_EVIDENCE = CANDIDATE_CHECKLIST_CONFIG["include_evidence"]
APPLY_LOCAL_NON_TARGET_FILTER = CANDIDATE_CHECKLIST_CONFIG["apply_local_non_target_filter"]
LIST_QUESTION_PATTERNS = CANDIDATE_CHECKLIST_CONFIG["list_question_patterns"]
WHO_QUESTION_PATTERNS = CANDIDATE_CHECKLIST_CONFIG["who_question_patterns"]
GENERIC_STOPWORDS = CANDIDATE_CHECKLIST_CONFIG["stopwords"]
GENERIC_LIST_SIGNAL_TERMS = CANDIDATE_CHECKLIST_CONFIG["signal_terms"]
WEAK_CANDIDATE_STARTS = CANDIDATE_CHECKLIST_CONFIG["weak_candidate_starts"]
WEAK_CANDIDATE_WORDS = CANDIDATE_CHECKLIST_CONFIG["weak_candidate_words"]
WHO_MODE_WEAK_ENTITY_WORDS = CANDIDATE_CHECKLIST_CONFIG["who_mode_weak_entity_words"]
WHO_MODE_NOISE_STARTS = CANDIDATE_CHECKLIST_CONFIG["who_mode_noise_starts"]
TARGET_TYPE_QUESTION_TERMS = CANDIDATE_CHECKLIST_CONFIG["target_type_question_terms"]
OBVIOUS_NON_TARGET_WORDS = CANDIDATE_CHECKLIST_CONFIG["obvious_non_target_words"]
BACKGROUND_ACTOR_PATTERNS = CANDIDATE_CHECKLIST_CONFIG["background_actor_patterns"]
RELATION_LOCAL_TERMS = CANDIDATE_CHECKLIST_CONFIG["relation_local_terms"]
ROLE_NAME_TERMS = CANDIDATE_CHECKLIST_CONFIG["role_name_terms"]
SENTENCE_START_SKIP_WORDS = CANDIDATE_CHECKLIST_CONFIG["sentence_start_skip_words"]
INTERNAL_WEAK_WORDS = CANDIDATE_CHECKLIST_CONFIG["internal_weak_words"]
SHORT_WEAK_VALUES = CANDIDATE_CHECKLIST_CONFIG["short_weak_values"]
CANDIDATE_CHECKLIST_HEADER = CANDIDATE_CHECKLIST_CONFIG["checklist_header"]
CANDIDATE_CHECKLIST_INSTRUCTIONS = CANDIDATE_CHECKLIST_CONFIG["checklist_intro_lines"]
CANDIDATE_SHORT_ANSWER_RULES = CANDIDATE_CHECKLIST_CONFIG["short_answer_rules"]
CANDIDATE_FALLBACK_ON_TRUNCATION = CANDIDATE_CHECKLIST_CONFIG["fallback_on_truncation"]
CANDIDATE_FALLBACK_MAX_ITEMS = CANDIDATE_CHECKLIST_CONFIG["fallback_max_items"]
# ============================================================
# LIGHTWEIGHT LIST CANDIDATE CHECKLIST
# ============================================================
# Purpose:
# - Help list/enumeration questions cover more items from the final context.
# - No second LLM call.
# - No INCLUDE/SKIP validator.
# - No fixed expected count.
# - Generic only: works for people, roles, systems, documents, rules, and requirements.
# ============================================================


# Public names exported by this compatibility/refactor module.
__all__ = [
    'html',
    'json',
    're',
    'unicodedata',
    'Path',
    'MAX_CONTEXT_CHARS',
    'MAX_PER_SOURCE',
    'NO_ANSWER_TEXT',
    'PREVIEW_CHARS',
    'build_rag_prompt',
    'select_final_context_docs',
    'EMPTY_VALUES',
    'SCORE_KEYS',
    'DEFAULT_QUERY_CONFIG_PATH',
    'read_query_config',
    'normalize_config_list',
    'normalize_config_key',
    'normalize_config_set',
    'config_int',
    'config_bool',
    'get_nested_dict',
    'load_candidate_checklist_config',
    'CANDIDATE_CHECKLIST_CONFIG',
    'MAX_CANDIDATE_CHECKLIST_ITEMS',
    'CANDIDATE_EVIDENCE_CHARS',
    'CANDIDATE_CHECKLIST_ENABLED',
    'CANDIDATE_CHECKLIST_INCLUDE_EVIDENCE',
    'APPLY_LOCAL_NON_TARGET_FILTER',
    'LIST_QUESTION_PATTERNS',
    'WHO_QUESTION_PATTERNS',
    'GENERIC_STOPWORDS',
    'GENERIC_LIST_SIGNAL_TERMS',
    'WEAK_CANDIDATE_STARTS',
    'WEAK_CANDIDATE_WORDS',
    'WHO_MODE_WEAK_ENTITY_WORDS',
    'WHO_MODE_NOISE_STARTS',
    'TARGET_TYPE_QUESTION_TERMS',
    'OBVIOUS_NON_TARGET_WORDS',
    'BACKGROUND_ACTOR_PATTERNS',
    'RELATION_LOCAL_TERMS',
    'ROLE_NAME_TERMS',
    'SENTENCE_START_SKIP_WORDS',
    'INTERNAL_WEAK_WORDS',
    'SHORT_WEAK_VALUES',
    'CANDIDATE_CHECKLIST_HEADER',
    'CANDIDATE_CHECKLIST_INSTRUCTIONS',
    'CANDIDATE_SHORT_ANSWER_RULES',
    'CANDIDATE_FALLBACK_ON_TRUNCATION',
    'CANDIDATE_FALLBACK_MAX_ITEMS',
]
