import json
import re
from pathlib import Path


JAPANESE_TEXT_PATTERN = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff]")
DEFAULT_QUERY_CONFIG_PATH = Path("config") / "query_expansion_config.json"


def read_query_config(config_path=DEFAULT_QUERY_CONFIG_PATH):
    # Read shared config. Editable terms/patterns live in JSON, not Python.
    try:
        config_path = Path(config_path)
        if not config_path.exists():
            return {}
        return json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return {}


def get_query_analyzer_config():
    raw_config = read_query_config()
    analyzer_config = raw_config.get("query_analyzer", {}) if isinstance(raw_config, dict) else {}
    return analyzer_config if isinstance(analyzer_config, dict) else {}


def normalize_text(text):
    # Lowercase and simple spaces for easier matching.
    text = str(text or "").lower()
    text = text.replace("_", " ").replace("-", " ")
    text = re.sub(r"[^a-z0-9\u3040-\u30ff\u3400-\u9fff]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_config_list(values):
    if isinstance(values, str):
        values = [values]
    if not isinstance(values, list):
        return []

    normalized_values = []
    for value in values:
        value = normalize_text(value)
        if value and value not in normalized_values:
            normalized_values.append(value)
    return normalized_values


def normalize_config_patterns(values):
    # Convert JSON wildcard pattern strings to normalized unique list.
    # Keep "*" so patterns like "why did * become" still work.
    if isinstance(values, str):
        values = [values]
    if not isinstance(values, list):
        return []

    normalized_values = []
    for value in values:
        value = normalize_wildcard_pattern(value)
        if value and value not in normalized_values:
            normalized_values.append(value)
    return normalized_values


def normalize_regex_patterns(values):
    # Keep regex patterns from JSON as-is, but remove empty/duplicate entries.
    if isinstance(values, str):
        values = [values]
    if not isinstance(values, list):
        return []

    normalized_values = []
    for value in values:
        value = str(value or "").strip()
        if value and value not in normalized_values:
            normalized_values.append(value)
    return normalized_values


def normalize_keyword_map(keyword_map):
    if not isinstance(keyword_map, dict):
        return {}

    cleaned_map = {}
    for label, keywords in keyword_map.items():
        label = str(label or "").strip()
        if not label:
            continue

        cleaned_keywords = normalize_config_list(keywords)
        if cleaned_keywords:
            cleaned_map[label] = cleaned_keywords
    return cleaned_map


def load_keyword_map(config_key):
    raw_config = read_query_config()
    keyword_map = raw_config.get(config_key, {}) if isinstance(raw_config, dict) else {}
    return normalize_keyword_map(keyword_map)


def get_category_keywords():
    # Editable category terms live in config/query_expansion_config.json.
    return load_keyword_map("category_keywords")


def get_doc_type_keywords():
    # Editable document-type terms live in config/query_expansion_config.json.
    return load_keyword_map("doc_type_keywords")


def get_query_stopwords():
    analyzer_config = get_query_analyzer_config()
    stopwords = analyzer_config.get("stopwords", [])
    return set(normalize_config_list(stopwords))


def contains_keyword(text, keyword):
    # Match phrase or token.
    text = normalize_text(text)
    keyword = normalize_text(keyword)

    if not text or not keyword:
        return False

    if " " in keyword:
        return keyword in text

    return keyword in text.split()


def detect_first_label(query, keyword_map):
    # Return first matching label from a keyword map.
    for label, keywords in keyword_map.items():
        for keyword in keywords:
            if contains_keyword(query, keyword):
                return label
    return ""


def detect_language_hint(query):
    # Japanese character detection stays in code; editable language hint words are in JSON.
    if JAPANESE_TEXT_PATTERN.search(str(query or "")):
        return "ja"

    normalized_query = normalize_text(query)
    tokens = set(normalized_query.split())
    language_hints = get_query_analyzer_config().get("language_hints", {})

    if not isinstance(language_hints, dict):
        return ""

    for language_code, words in language_hints.items():
        for word in normalize_config_list(words):
            if word and (word in tokens or word in normalized_query):
                return str(language_code or "").strip()

    return ""


def load_mode_detection_config():
    # Mode detection is config-driven.
    raw_config = read_query_config()
    mode_config = raw_config.get("mode_detection", {}) if isinstance(raw_config, dict) else {}

    if not isinstance(mode_config, dict):
        mode_config = {}

    return {
        "cross_doc_phrases": normalize_config_list(mode_config.get("cross_doc_phrases", [])),
        "comparison_phrases": normalize_config_list(mode_config.get("comparison_phrases", [])),
        "negative_phrases": normalize_config_list(mode_config.get("negative_phrases", [])),
        "false_premise_phrases": normalize_config_list(mode_config.get("false_premise_phrases", [])),
        "false_premise_patterns": normalize_config_patterns(mode_config.get("false_premise_patterns", [])),
        "list_phrases": normalize_config_list(mode_config.get("list_phrases", [])),
        "list_patterns": normalize_regex_patterns(mode_config.get("list_patterns", [])),
    }


def phrase_exists(normalized_query, phrase):
    # Exact phrase boundary match.
    normalized_query = f" {normalize_text(normalized_query)} "
    phrase = normalize_text(phrase)

    if not phrase:
        return False

    return f" {phrase} " in normalized_query


def normalize_wildcard_pattern(pattern):
    # Keep "*" as a wildcard token, then normalize the rest like normal query text.
    pattern = str(pattern or "").lower()
    pattern = pattern.replace("_", " ").replace("-", " ")
    pattern = re.sub(r"[^a-z0-9\u3040-\u30ff\u3400-\u9fff*]+", " ", pattern)
    return re.sub(r"\s+", " ", pattern).strip()


def wildcard_pattern_exists(normalized_query, pattern, max_wildcard_tokens=8):
    # Supports config patterns like "why did * become" or "bakit * naging".
    normalized_query = normalize_text(normalized_query)
    pattern = normalize_wildcard_pattern(pattern)

    if not normalized_query or not pattern:
        return False

    if "*" not in pattern:
        return phrase_exists(normalized_query, pattern)

    tokens = pattern.split()
    regex_parts = []

    for index, token in enumerate(tokens):
        if token == "*":
            regex_parts.append(r"(?:\s+\S+){0," + str(max_wildcard_tokens) + r"}")
            continue

        escaped_token = re.escape(token)

        if index > 0 and tokens[index - 1] != "*":
            regex_parts.append(r"\s+")

        if index > 0 and tokens[index - 1] == "*":
            regex_parts.append(r"\s*")

        regex_parts.append(escaped_token)

    regex = r"\b" + "".join(regex_parts) + r"\b"
    return re.search(regex, normalized_query) is not None


def has_any_phrase(normalized_query, phrases):
    for phrase in phrases:
        if phrase_exists(normalized_query, phrase):
            return True
    return False


def has_any_wildcard_pattern(normalized_query, patterns):
    for pattern in patterns:
        if wildcard_pattern_exists(normalized_query, pattern):
            return True
    return False


def has_any_regex_pattern(normalized_query, patterns):
    normalized_query = normalize_text(normalized_query)

    for pattern in patterns:
        try:
            if re.search(pattern, normalized_query, flags=re.IGNORECASE):
                return True
        except re.error:
            continue
    return False


def detect_mode(query):
    # single_fact = direct answer about one main subject.
    # list_enumeration = question expects multiple items from one or more chunks/pages.
    # cross_doc = compare/connect/relationship BETWEEN two or more subjects.
    # false_premise = premise-bearing question that needs stricter context safety.
    normalized_query = normalize_text(query)
    mode_config = load_mode_detection_config()

    if (
        has_any_phrase(normalized_query, mode_config["false_premise_phrases"])
        or has_any_wildcard_pattern(normalized_query, mode_config["false_premise_patterns"])
    ):
        return "false_premise"

    if has_any_phrase(normalized_query, mode_config["negative_phrases"]):
        return "negative"

    has_list_shape = (
        has_any_phrase(normalized_query, mode_config["list_phrases"])
        or has_any_regex_pattern(normalized_query, mode_config["list_patterns"])
    )

    if has_list_shape:
        return "list_enumeration"

    if has_any_phrase(normalized_query, mode_config["comparison_phrases"]):
        return "comparison"

    if has_any_phrase(normalized_query, mode_config["cross_doc_phrases"]):
        return "cross_doc"

    return "single_fact"


def extract_important_terms(query):
    # Important terms for debug/source hints. Stopwords are JSON-driven.
    tokens = normalize_text(query).split()
    stopwords = get_query_stopwords()
    terms = []

    for token in tokens:
        if token in stopwords:
            continue
        if len(token) <= 1:
            continue
        if token not in terms:
            terms.append(token)
    return terms


def regex_first_group(pattern, text):
    try:
        match = re.search(pattern, text, flags=re.IGNORECASE)
    except re.error:
        return ""

    if not match:
        return ""

    try:
        return match.group(1)
    except IndexError:
        return ""


def extract_source_hint(query):
    # Source/file hint patterns are editable in JSON.
    query_text = str(query or "")
    analyzer_config = get_query_analyzer_config()
    patterns = analyzer_config.get("source_hint_patterns", [])
    cleanup_patterns = analyzer_config.get("source_hint_cleanup_patterns", [])

    if isinstance(patterns, str):
        patterns = [patterns]
    if isinstance(cleanup_patterns, str):
        cleanup_patterns = [cleanup_patterns]

    for pattern in patterns or []:
        value = regex_first_group(pattern, query_text)
        value = normalize_text(value)
        if not value:
            continue

        for cleanup_pattern in cleanup_patterns or []:
            try:
                value = re.sub(cleanup_pattern, "", value, flags=re.IGNORECASE).strip()
            except re.error:
                continue

        if value:
            return value
    return ""


def analyze_query(query, debug=False):
    # Main function called before retrieval.
    query = str(query or "").strip()
    important_terms = extract_important_terms(query)

    result = {
        "original_query": query,
        "normalized_query": normalize_text(query),
        "mode": detect_mode(query),
        "category": detect_first_label(query, get_category_keywords()),
        "doc_type": detect_first_label(query, get_doc_type_keywords()),
        "language": detect_language_hint(query),
        "source_hint": extract_source_hint(query),
        "important_terms": important_terms,
    }

    result["source_keywords"] = []
    if result["source_hint"]:
        result["source_keywords"] = result["source_hint"].split()
    else:
        result["source_keywords"] = important_terms[:6]

    if debug:
        print("[QUERY ANALYZER]", result, flush=True)

    return result


def build_metadata_filter(query_info):
    # Optional helper for using a Chroma where filter later.
    query_info = dict(query_info or {})
    filters = {}

    for key in ["category", "doc_type", "language"]:
        value = str(query_info.get(key, "") or "").strip()
        if value:
            filters[key] = value
    return filters


def build_query_info(query, debug=False):
    # Alias for readability in test files.
    return analyze_query(query, debug=debug)
