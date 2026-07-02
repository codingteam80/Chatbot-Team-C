import json
import re
from pathlib import Path


# Intent-based detector.
# Hindi ito naka-base sa topic, file name, history terms, company terms, or sample questions.
# Ang tinitingnan lang nito ay form ng tanong:
# direct fact, list/plural, broad explanation, comparison, relationship, multi-part, or assumption-risk.


DEFAULT_QUERY_CONFIG_PATH = Path("config") / "query_expansion_config.json"


def read_query_config(config_path=DEFAULT_QUERY_CONFIG_PATH):
    # Read shared JSON config. If missing/invalid, use safe in-code fallback patterns.
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
    # normalize_query_text is defined later, so use the same simple normalization here.
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


def load_json_list_config():
    # List/enumeration cues are configurable in query_expansion_config.json.
    # Do not add list words in Python; put them under mode_detection.list_phrases/list_patterns.
    raw_config = read_query_config()
    mode_config = raw_config.get("mode_detection", {}) if isinstance(raw_config, dict) else {}

    if not isinstance(mode_config, dict):
        mode_config = {}

    return {
        "phrases": normalize_config_phrases(mode_config.get("list_phrases", [])),
        "patterns": normalize_config_patterns(mode_config.get("list_patterns", [])),
    }


JSON_LIST_QUESTION_CONFIG = load_json_list_config()
JSON_LIST_QUESTION_PHRASES = JSON_LIST_QUESTION_CONFIG["phrases"]
JSON_LIST_QUESTION_PATTERNS = JSON_LIST_QUESTION_CONFIG["patterns"]


def phrase_exists(normalized_query, phrase):
    normalized_query = f" {normalize_query_text(normalized_query)} "
    phrase = normalize_query_text(phrase)

    if not phrase:
        return False

    return f" {phrase} " in normalized_query


def matches_any_phrase(text, phrases):
    return any(phrase_exists(text, phrase) for phrase in phrases or [])


DIRECT_FACT_PATTERNS = [
    # English direct fact forms.
    r"^who\s+(is|was|were)\b",
    r"^who\s+\w+\b",
    r"^when\b",
    r"^where\b",
    r"^which\b",
    r"^what\s+(is|was|were)\b",
    r"^(is|are|was|were|did|does|do|can|could|should|must)\b",

    # Filipino / Tagalog direct fact forms.
    r"^sino\b",
    r"^kailan\b",
    r"^saan\b",
    r"^alin\b",
    r"^ano\s+ang\b",
    r"^(ay|ba)\b",
]


# List/enumeration patterns are intentionally loaded from JSON.
# Keep this empty so tuning happens in config/query_expansion_config.json.
LIST_QUESTION_PATTERNS = []


EXPLANATION_PATTERNS = [
    # English explanation forms.
    r"^why\b",
    r"^how\b",
    r"\b(explain|describe|summarize|overview|purpose|reason|cause|effect|impact|role)\b",

    # Filipino / Tagalog explanation forms.
    r"\b(bakit|paano|ipaliwanag|ilarawan|ibuod|layunin|dahilan|sanhi|epekto|papel)\b",
]


COMPARISON_PATTERNS = [
    # English comparison/contrast forms.
    r"\b(compare|difference|different|distinguish|versus)\b",
    r"\bvs\.?\b",

    # Filipino / Tagalog comparison forms.
    r"\b(ihambing|pagkakaiba|kaibahan)\b",
]


RELATIONSHIP_PATTERNS = [
    # English relationship/synthesis forms.
    r"\b(relationship|connection|connect|related|relate|link|association)\b",
    r"\bbetween\b.+\band\b",

    # Filipino / Tagalog relationship/synthesis forms.
    r"\b(kaugnayan|nauugnay|ugnay|koneksyon)\b",
]


MULTI_PART_PATTERNS = [
    # Hindi lahat ng "and/or" ay multi-part.
    # Multi-part lang kapag may second question intent after connector.
    r"\b(and|or)\s+(what|how|why|when|where|who|which|is|are|was|were|does|do|did|should|must|can)\b",
    r"\b(at|o)\s+(ano|paano|bakit|kailan|saan|sino|alin)\b",
    r";",
]


ASSUMPTION_RISK_PATTERNS = [
    # Generic false-premise risk forms.
    # Example: "Why did X become Y?" or "How did X get classified as Y?"
    # Hindi ibig sabihin false agad; ibig sabihin huwag masyadong strict sa proximity
    # at dapat payagan ang correction kapag hindi supported ang assumption.
    r"^(why|how)\s+(did|does|do|was|were|is|are)\b",
    r"\b(become|became|serve as|served as|appointed as|assigned as|classified as|treated as|considered as)\b",
    r"\b(naging|itinuring|itinalaga|nakilala bilang)\b",
]


def normalize_query_text(question):
    # Normalize query para stable ang pattern matching.
    text = str(question or "").strip().lower()
    text = " ".join(text.split())
    return text


def count_words(text):
    # Bilangin ang words para matukoy kung short direct lookup siya.
    return len(re.findall(r"[a-z0-9]+", str(text or "").lower()))


def matches_any(text, patterns):
    # Generic regex matcher para isang function lang ang gamit.
    return any(re.search(pattern, text) for pattern in patterns)


def is_list_question(question):
    # List/plural questions need more context because answers may span chunks/pages.
    # All list cues come from config/query_expansion_config.json.
    text = normalize_query_text(question)

    if JSON_LIST_QUESTION_PHRASES and matches_any_phrase(text, JSON_LIST_QUESTION_PHRASES):
        return True

    if JSON_LIST_QUESTION_PATTERNS and matches_any(text, JSON_LIST_QUESTION_PATTERNS):
        return True

    return False


def is_explanation_question(question):
    # Explanation questions usually have distributed evidence.
    text = normalize_query_text(question)
    return matches_any(text, EXPLANATION_PATTERNS)


def is_comparison_question(question):
    # Comparison questions usually need multiple chunks or sources.
    text = normalize_query_text(question)
    return matches_any(text, COMPARISON_PATTERNS)


def is_relationship_question(question):
    # Relationship questions usually synthesize multiple facts.
    text = normalize_query_text(question)
    return matches_any(text, RELATIONSHIP_PATTERNS)


def is_multi_part_question(question):
    # Multi-part questions need more than one evidence span.
    text = normalize_query_text(question)
    return matches_any(text, MULTI_PART_PATTERNS)


def is_assumption_risk_question(question):
    # Questions with built-in assumptions should allow correction instead of forced fallback.
    text = normalize_query_text(question)
    return matches_any(text, ASSUMPTION_RISK_PATTERNS)


def is_broad_question(question):
    # Broad means proximity should usually be off.
    return (
        is_list_question(question)
        or is_explanation_question(question)
        or is_comparison_question(question)
        or is_relationship_question(question)
        or is_multi_part_question(question)
        or is_assumption_risk_question(question)
    )


def is_direct_fact_question(question):
    # Direct fact means a close span can help.
    text = normalize_query_text(question)

    if not text:
        return False

    if is_broad_question(text):
        return False

    return matches_any(text, DIRECT_FACT_PATTERNS)


def should_require_rerank_proximity(question, auto_enabled=True, default_value=False):
    # Auto mode:
    # direct fact -> proximity ON
    # list/broad/explain/compare/multi-part/assumption-risk -> proximity OFF
    text = normalize_query_text(question)

    if not text:
        return bool(default_value)

    if not auto_enabled:
        return bool(default_value)

    if is_broad_question(text):
        return False

    if is_direct_fact_question(text):
        return True

    # Short factual-looking questions are usually safe for proximity.
    if count_words(text) <= 6:
        return True

    return False


def should_expand_neighbors_for_question(question):
    # Expand adjacent chunks only when the answer likely continues across chunk/page boundaries.
    # This is generic and useful for manuals, SOPs, policies, and history docs.
    return (
        is_list_question(question)
        or is_multi_part_question(question)
        or is_relationship_question(question)
        or is_assumption_risk_question(question)
    )


def get_context_top_n(question, base_top_n=3):
    # Dynamic context size.
    # Direct facts can stay small; list/broad questions need more evidence.
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
    # Label para sa debug logs lang.
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


def get_query_profile(
    question,
    base_top_n=3,
    auto_proximity=True,
    default_proximity=False,
):
    # One place para kunin lahat ng query behavior.
    # Para hindi kumalat sa chatbot.py ang maraming if/else.
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
