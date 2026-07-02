import json
import re
from pathlib import Path


JAPANESE_TEXT_PATTERN = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff]")

CATEGORY_KEYWORDS = {
    "coding": [
        "coding",
        "code",
        "program",
        "programming",
        "developer",
        "code review",
        "design review",
        "secure coding",
        "misra",
        "release",
        "コード",
        "レビュー",
        "設計",
        "リリース",
    ],
    "security": [
        "security",
        "secure",
        "password",
        "access",
        "authentication",
        "authorization",
        "vulnerability",
        "セキュリティ",
        "パスワード",
        "認証",
    ],
    "incident": [
        "incident",
        "issue",
        "bug",
        "defect",
        "escalation",
        "root cause",
        "corrective action",
        "障害",
        "インシデント",
        "報告",
    ],
    "it": [
        "it",
        "acceptable use",
        "device",
        "network",
        "email",
        "internet",
        "情報システム",
        "ネットワーク",
        "メール",
    ],
    "hr": [
        "hr",
        "leave",
        "attendance",
        "absence",
        "holiday",
        "overtime",
        "employee",
        "勤怠",
        "休暇",
        "社員",
    ],
}

DOC_TYPE_KEYWORDS = {
    "sop": ["sop", "standard operating procedure", "procedure", "process", "手順", "標準手順"],
    "policy": ["policy", "policies", "規程", "ポリシー"],
    "manual": ["manual", "handbook", "マニュアル"],
    "guideline": ["guideline", "guide", "rule", "rules", "standard", "misra", "ガイドライン", "ルール"],
    "checklist": ["checklist", "check list", "チェックリスト"],
    "report": ["report", "summary", "報告", "レポート"],
    "article": ["article", "wikipedia"],
}

DEFAULT_QUERY_CONFIG_PATH = Path("config") / "query_expansion_config.json"


STOPWORDS = {
    "a",
    "an",
    "ang",
    "ano",
    "are",
    "at",
    "ba",
    "be",
    "can",
    "do",
    "does",
    "for",
    "from",
    "how",
    "i",
    "in",
    "is",
    "it",
    "ng",
    "on",
    "or",
    "paano",
    "para",
    "sa",
    "the",
    "to",
    "what",
    "when",
    "where",
    "who",
    "why",
    "with",
    "yung",
}


def normalize_text(text):
    # Lowercase + simple spaces para madali mag-match.
    text = str(text or "").lower()
    text = text.replace("_", " ").replace("-", " ")
    text = re.sub(r"[^a-z0-9\u3040-\u30ff\u3400-\u9fff]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


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
    # Detect user language preference/hint.
    normalized_query = normalize_text(query)

    if JAPANESE_TEXT_PATTERN.search(str(query or "")):
        return "ja"

    if any(word in normalized_query.split() for word in ["japanese", "nihongo"]):
        return "ja"

    if "日本語" in str(query or ""):
        return "ja"

    if any(word in normalized_query.split() for word in ["english", "ingles", "en"]):
        return "en"

    return ""


def read_query_config(config_path=DEFAULT_QUERY_CONFIG_PATH):
    # Basahin ang shared config.
    # Kapag wala ang config, empty dict para safe fallback to single_fact.
    try:
        config_path = Path(config_path)

        if not config_path.exists():
            return {}

        return json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return {}


def normalize_config_list(values):
    # Convert JSON string/list values to normalized unique list.
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


def load_keyword_map(config_key, fallback_map):
    # Keyword maps are configurable in JSON. Python constants are only safe fallback.
    raw_config = read_query_config()
    keyword_map = raw_config.get(config_key, {}) if isinstance(raw_config, dict) else {}

    if not isinstance(keyword_map, dict):
        return fallback_map

    cleaned_map = {}

    for label, keywords in keyword_map.items():
        label = str(label or "").strip()

        if not label:
            continue

        cleaned_keywords = normalize_config_list(keywords)

        if cleaned_keywords:
            cleaned_map[label] = cleaned_keywords

    return cleaned_map or fallback_map


def get_category_keywords():
    return load_keyword_map("category_keywords", CATEGORY_KEYWORDS)


def get_doc_type_keywords():
    return load_keyword_map("doc_type_keywords", DOC_TYPE_KEYWORDS)


def load_mode_detection_config():
    # Mode detection is config-driven.
    # Avoid hardcoded sample-specific terms in Python.
    raw_config = read_query_config()
    mode_config = raw_config.get("mode_detection", {})

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
    # This prevents a standalone broad word from forcing cross_doc.
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
    # Supports config patterns like:
    # - "why did * become"
    # - "bakit * naging"
    #
    # "*" means up to max_wildcard_tokens words between fixed parts.
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
    # This uses phrase/pattern rules from config/query_expansion_config.json.
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

    # List/enumeration question shape has priority over broad relation words.
    # Example: "Which systems are connected to X?" asks for items, not a cross-doc essay.
    if has_list_shape:
        return "list_enumeration"

    if has_any_phrase(normalized_query, mode_config["comparison_phrases"]):
        return "comparison"

    if has_any_phrase(normalized_query, mode_config["cross_doc_phrases"]):
        return "cross_doc"

    return "single_fact"


def extract_important_terms(query):
    # Important terms para sa debug/source hint.
    tokens = normalize_text(query).split()
    terms = []

    for token in tokens:
        if token in STOPWORDS:
            continue

        if len(token) <= 1:
            continue

        if token not in terms:
            terms.append(token)

    return terms


def extract_source_hint(query):
    # Simple source/file hint detection.
    # Example: "from Code Review SOP" -> "code review sop".
    query_text = str(query or "")
    patterns = [
        r"(?:from|inside|in|within|source|file|document|doc)\s+([A-Za-z0-9_ .()\-/]+)",
        r"(?:sa|mula sa|galing sa)\s+([A-Za-z0-9_ .()\-/]+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, query_text, flags=re.IGNORECASE)
        if match:
            value = normalize_text(match.group(1))
            value = re.sub(r"\b(about|regarding|na|ng|ang|the)\b.*$", "", value).strip()
            if value:
                return value

    return ""


def analyze_query(query, debug=False):
    # Main function na tatawagin bago retrieval.
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

    # Source keywords are used by metadata_booster.
    # Hindi ito hard filter, hint lang.
    result["source_keywords"] = []

    if result["source_hint"]:
        result["source_keywords"] = result["source_hint"].split()
    else:
        result["source_keywords"] = important_terms[:6]

    if debug:
        print("[QUERY ANALYZER]", result, flush=True)

    return result


def build_metadata_filter(query_info):
    # Optional helper kung gusto mong gumamit ng Chroma where filter later.
    # Sa current recommended flow, metadata_boost muna instead of hard filter.
    query_info = dict(query_info or {})
    filters = {}

    for key in ["category", "doc_type", "language"]:
        value = str(query_info.get(key, "") or "").strip()
        if value:
            filters[key] = value

    return filters


def build_query_info(query, debug=False):
    # Alias para readable sa test files.
    return analyze_query(query, debug=debug)
