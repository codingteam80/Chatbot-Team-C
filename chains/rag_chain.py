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



def load_value_only_question_config():
    raw_config = read_query_config()
    guard_config = raw_config.get("value_only_question", {})

    if not isinstance(guard_config, dict):
        guard_config = {}

    short_value_config = get_nested_dict(raw_config, "value_only_question", "short_value_answer")
    language_markers = get_nested_dict(raw_config, "language_markers")
    messages = guard_config.get("tautology_messages") or guard_config.get("clarification_messages") or {}

    if not isinstance(messages, dict):
        messages = {}

    return {
        "enabled": config_bool(guard_config, "enabled", True),
        "max_value_words": config_int(
            guard_config,
            "max_value_words",
            config_int(short_value_config, "max_words", 6),
        ),
        "question_prefix_patterns": normalize_config_list(guard_config.get("question_prefix_patterns")),
        "value_patterns": normalize_config_list(
            guard_config.get("value_patterns") or short_value_config.get("patterns")
        ),
        "tagalog_markers": set(normalize_config_list(language_markers.get("tagalog_markers"))),
        "tautology_messages": messages,
    }


VALUE_ONLY_QUESTION_CONFIG = load_value_only_question_config()


def word_count(text):
    return len(re.findall(r"\S+", normalize_space(text)))


def extract_value_only_question_value(question):
    # Extract the value from generic questions like "What is <date/value>?".
    if not VALUE_ONLY_QUESTION_CONFIG.get("enabled", True):
        return ""

    text = normalize_space(question)

    if not text:
        return ""

    for pattern in VALUE_ONLY_QUESTION_CONFIG.get("question_prefix_patterns", []):
        try:
            match = re.search(pattern, text, flags=re.IGNORECASE)
        except re.error:
            continue

        if not match:
            continue

        if match.groups():
            return normalize_space(match.group(1)).strip(" ?.!,:;\"'")

    return ""


def looks_like_configured_short_value(value):
    # The actual value patterns live in config/query_expansion_config.json.
    value = normalize_space(value).strip(" ?.!,:;\"'")

    if not value:
        return False

    max_words = VALUE_ONLY_QUESTION_CONFIG.get("max_value_words", 6)

    if word_count(value) > max_words:
        return False

    for pattern in VALUE_ONLY_QUESTION_CONFIG.get("value_patterns", []):
        try:
            if re.search(pattern, value, flags=re.IGNORECASE):
                return True
        except re.error:
            continue

    return False


def is_value_only_question(question):
    value = extract_value_only_question_value(question)
    return bool(value and looks_like_configured_short_value(value))


def is_tagalog_text(text):
    tokens = set(re.findall(r"[a-zA-ZÀ-ÿ']+", str(text or "").lower()))
    return bool(tokens.intersection(VALUE_ONLY_QUESTION_CONFIG.get("tagalog_markers", set())))


def get_value_only_tautology_clarification(question):
    messages = VALUE_ONLY_QUESTION_CONFIG.get("tautology_messages", {}) or {}

    if is_tagalog_text(question):
        return messages.get("Tagalog") or "Anong tao, event, policy, o topic ang tinutukoy mo?"

    return messages.get("English") or "Which person, event, policy, or topic do you mean?"


def is_tautological_value_answer(question, answer):
    # Reject answers that only echo the value from a value-only question.
    value = extract_value_only_question_value(question)

    if not value or not looks_like_configured_short_value(value):
        return False

    answer_key = normalize_config_key(answer)
    value_key = normalize_config_key(value)

    if not answer_key or not value_key:
        return False

    if answer_key == value_key:
        return True

    connector_words = {"is", "was", "are", "were", "ang", "ay", "yung", "means", "meaning"}
    answer_words = answer_key.split()
    value_words = set(value_key.split())

    if answer_words and all(word in value_words or word in connector_words for word in answer_words):
        return True

    return False
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


def normalize_space(text):
    return " ".join(str(text or "").split()).strip()


def normalize_candidate_key(text):
    text = str(text or "").lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = re.sub(r"[^a-z0-9\u3040-\u30ff\u3400-\u9fff]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def is_list_question(question):
    question_key = normalize_candidate_key(question)

    if not question_key:
        return False

    for pattern in LIST_QUESTION_PATTERNS:
        if re.search(pattern, question_key, flags=re.IGNORECASE):
            return True

    return False


def is_who_list_question(question):
    question_key = normalize_candidate_key(question)

    for pattern in WHO_QUESTION_PATTERNS:
        if re.search(pattern, question_key, flags=re.IGNORECASE):
            return True

    return False


def get_useful_question_terms(question):
    terms = []

    for token in normalize_candidate_key(question).split():
        if token in GENERIC_STOPWORDS:
            continue

        if len(token) <= 1:
            continue

        if token not in terms:
            terms.append(token)

    return terms


def strip_retrieval_metadata(text):
    # Some chunks include a synthetic metadata prefix before the real excerpt.
    # Remove only the prefix so metadata names/places do not become candidates.
    text = normalize_space(text)

    if text.lower().startswith("retrieval context:"):
        match = re.search(r"\blanguage\s*:\s*[a-z]{2}\s+", text, flags=re.IGNORECASE)

        if match:
            return text[match.end():].strip()

    return text


def get_doc_text(doc):
    return strip_retrieval_metadata(getattr(doc, "page_content", "") or "")


def split_candidate_sentences(text):
    text = strip_retrieval_metadata(text)
    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return []

    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", text)
        if sentence.strip()
    ]


def sentence_has_list_or_relation_signal(sentence, question_terms=None):
    sentence_key = normalize_candidate_key(sentence)

    if not sentence_key:
        return False

    for term in GENERIC_LIST_SIGNAL_TERMS:
        if normalize_candidate_key(term) in sentence_key:
            return True

    # Semicolons and colon often mark list-like spans.
    if ";" in sentence or ":" in sentence:
        return True

    # A comma-heavy sentence with multiple capitalized entities is often an enumeration.
    if sentence.count(",") >= 2 and len(re.findall(r"\b[A-ZÀ-ÖØ-Þ][A-Za-zÀ-ÖØ-öø-ÿ.'-]+", sentence)) >= 2:
        return True

    for term in question_terms or []:
        if term in sentence_key:
            return True

    return False


def is_target_type_question(question):
    question_words = set(normalize_candidate_key(question).split())
    return bool(question_words & TARGET_TYPE_QUESTION_TERMS)


def get_local_candidate_window(sentence, start_index, end_index, radius=90):
    left = max(0, start_index - radius)
    right = min(len(sentence), end_index + radius)
    return sentence[left:right]


def make_evidence_snippet(text, candidate=""):
    # Keep a short human-readable clue beside a checklist item.
    # This is not a source citation; it only reminds the LLM why the candidate was extracted.
    text = normalize_space(text)

    if not text:
        return ""

    # Prefer the local clause around the candidate instead of the whole sentence.
    if candidate:
        candidate_index = text.lower().find(str(candidate).lower())

        if candidate_index >= 0:
            start = max(0, candidate_index - 70)
            end = min(len(text), candidate_index + len(candidate) + 90)
            text = text[start:end].strip(" ,;:-")

    if len(text) <= CANDIDATE_EVIDENCE_CHARS:
        return text

    return text[:CANDIDATE_EVIDENCE_CHARS].rstrip(" ,;:-") + "..."


def has_local_relation_signal(text):
    text_key = normalize_candidate_key(text)

    if not text_key:
        return False

    for term in RELATION_LOCAL_TERMS:
        term_key = normalize_candidate_key(term)
        if term_key and term_key in text_key:
            return True

    return False


def is_enumeration_sentence(sentence):
    # Generic list-like sentence: multiple separators and named entities.
    # This catches paragraphs such as "items have been identified: A; B; C"
    # without requiring any domain-specific names.
    named_count = len(re.findall(r"\b[A-ZÀ-ÖØ-Þ][A-Za-zÀ-ÖØ-öø-ÿ.'-]*(?:\s+[A-ZÀ-ÖØ-Þ][A-Za-zÀ-ÖØ-öø-ÿ.'-]*)+", sentence))
    return (";" in sentence or sentence.count(",") >= 2) and named_count >= 2


def clean_candidate_text(candidate):
    candidate = html.unescape(str(candidate or ""))
    candidate = re.sub(r"\[[^\]]*\]", " ", candidate)
    candidate = re.sub(r"\([^)]{120,}\)", " ", candidate)
    candidate = re.sub(r"\s+", " ", candidate).strip(" .,:;|-–—\t\r\n")
    candidate = re.sub(r"'s$", "", candidate).strip()

    if len(candidate) < 2:
        return ""

    if len(candidate.split()) > 8:
        return ""

    return candidate


def is_weak_candidate(candidate):
    candidate = clean_candidate_text(candidate)
    key = normalize_candidate_key(candidate)

    if not key:
        return True

    words = key.split()

    if not words:
        return True

    if key.isdigit() or len(key) <= 1:
        return True

    if words[0] in WEAK_CANDIDATE_STARTS:
        return True

    if any(word in WEAK_CANDIDATE_WORDS for word in words):
        return True

    # Avoid fragments with pronouns; these are usually sentence pieces, not list items.
    if set(words) & GENERIC_STOPWORDS and len(words) == 1:
        return True

    if key in SHORT_WEAK_VALUES:
        return True

    return False


def is_background_actor(candidate, local_window):
    # Avoid people mentioned as authors/researchers/reporters/background actors,
    # not as the requested answer item. This is generic syntax filtering.
    candidate_key = normalize_candidate_key(candidate)
    window_key = normalize_candidate_key(local_window)

    if not candidate_key or not window_key:
        return False

    escaped_candidate = re.escape(candidate_key)

    for pattern in BACKGROUND_ACTOR_PATTERNS:
        compiled = pattern.format(candidate=escaped_candidate)
        if re.search(compiled, window_key, flags=re.IGNORECASE):
            return True

    return False


def is_weak_who_candidate(candidate, target_type_mode=False, local_window=""):
    key = normalize_candidate_key(candidate)
    word_list = key.split()
    words = set(word_list)

    if not words:
        return True

    if words & WHO_MODE_WEAK_ENTITY_WORDS:
        return True

    if word_list and word_list[0] in WHO_MODE_NOISE_STARTS:
        return True

    # Avoid internal preposition fragments from headings/titles such as
    # "Name In Location" or "Book Of Something".
    if INTERNAL_WEAK_WORDS and any(word in INTERNAL_WEAK_WORDS for word in word_list[1:-1]):
        return True

    # Avoid obvious possessive/title fragments that often come from book titles or places.
    if key.endswith(" s") or " s " in key:
        return True

    if is_background_actor(candidate, local_window):
        return True

    if APPLY_LOCAL_NON_TARGET_FILTER and target_type_mode and (words & OBVIOUS_NON_TARGET_WORDS):
        return True

    return False


def add_candidate(
    candidates,
    candidate,
    reason="",
    score=0,
    who_mode=False,
    target_type_mode=False,
    local_window="",
    evidence="",
):
    candidate = clean_candidate_text(candidate)

    if not candidate or is_weak_candidate(candidate):
        return

    if who_mode and is_weak_who_candidate(candidate, target_type_mode=target_type_mode, local_window=local_window):
        return

    key = normalize_candidate_key(candidate)
    evidence = make_evidence_snippet(evidence or local_window, candidate=candidate)

    if key not in candidates:
        candidates[key] = {
            "candidate": candidate,
            "reason": reason,
            "score": score,
            "evidence": evidence,
        }
        return

    current_score = candidates[key].get("score", 0)

    if score > current_score:
        candidates[key]["score"] = score
        candidates[key]["reason"] = reason

    # Keep the most informative clue.
    current_evidence = candidates[key].get("evidence", "")

    if evidence and (not current_evidence or len(evidence) > len(current_evidence)):
        candidates[key]["evidence"] = evidence

def extract_bullet_candidates(text, candidates):
    for line in str(text or "").splitlines():
        line = normalize_space(line)
        match = re.match(r"^(?:[-*•]|\d+[.)])\s+(.+)$", line)

        if not match:
            continue

        item_text = re.split(r"\s+[-–—:]\s+|:", match.group(1), maxsplit=1)[0]
        add_candidate(candidates, item_text, reason="bullet", score=20, evidence=line)


def extract_named_candidates(sentence, candidates, score=0, who_mode=False, target_type_mode=False):
    # Extract two-or-more-word proper nouns. This avoids most sentence-start single-word noise.
    pattern = r"\b[A-ZÀ-ÖØ-Þ][A-Za-zÀ-ÖØ-öø-ÿ.'-]*(?:\s+[A-ZÀ-ÖØ-Þ][A-Za-zÀ-ÖØ-öø-ÿ.'-]*){1,5}\b"
    enum_sentence = is_enumeration_sentence(sentence)

    for match in re.finditer(pattern, sentence):
        candidate = match.group(0)
        candidate_key = normalize_candidate_key(candidate)

        if candidate_key.endswith(" s"):
            candidate = candidate[:-2]

        # Avoid candidates that are just a capitalized phrase at the beginning of a sentence.
        if match.start() == 0 and normalize_candidate_key(candidate.split()[0]) in SENTENCE_START_SKIP_WORDS:
            continue

        local_window = get_local_candidate_window(sentence, match.start(), match.end())
        local_relation = has_local_relation_signal(local_window)
        candidate_score = score

        if local_relation:
            candidate_score += 8

        if enum_sentence:
            candidate_score += 4

        # For who/person questions, avoid adding every name from a long noisy sentence.
        # Keep candidates that are in a list/enumeration sentence or locally connected
        # to a relationship/action cue. This stays generic and avoids hardcoded names.
        if who_mode and not enum_sentence and not local_relation:
            continue

        add_candidate(
            candidates,
            candidate,
            reason="named_entity",
            score=candidate_score,
            who_mode=who_mode,
            target_type_mode=target_type_mode,
            local_window=local_window,
            evidence=sentence,
        )


def extract_role_name_candidates(sentence, candidates, score=0, who_mode=False, target_type_mode=False):
    # Generic support for one-word names after person-role words.
    # The role words are loaded from config/query_expansion_config.json.
    if not ROLE_NAME_TERMS:
        return

    escaped_terms = [re.escape(term) for term in ROLE_NAME_TERMS if str(term or "").strip()]

    if not escaped_terms:
        return

    role_pattern = rf"\b(?:{'|'.join(escaped_terms)})\b(?:\s+named)?\s*,?\s+([A-ZÀ-ÖØ-Þ][A-Za-zÀ-ÖØ-öø-ÿ.'-]{{2,}})\b"

    for match in re.finditer(role_pattern, sentence):
        candidate = match.group(1)
        local_window = get_local_candidate_window(sentence, match.start(1), match.end(1))

        add_candidate(
            candidates,
            candidate,
            reason="role_name",
            score=score + 5,
            who_mode=who_mode,
            target_type_mode=target_type_mode,
            local_window=local_window,
            evidence=sentence,
        )


def extract_delimited_candidates(sentence, candidates, score=0):
    # Use only on signal/list-like sentences. This keeps extraction generic but less noisy.
    for piece in re.split(r";|,|\band\b|\bor\b", sentence):
        piece = clean_candidate_text(piece)

        if not piece:
            continue

        if 1 <= len(piece.split()) <= 8:
            add_candidate(candidates, piece, reason="delimited_item", score=score, evidence=sentence)


def extract_quoted_candidates(sentence, candidates, score=0):
    for match in re.finditer(r"[\"'“”‘’]([^\"'“”‘’]{2,80})[\"'“”‘’]", sentence):
        add_candidate(candidates, match.group(1), reason="quoted_term", score=score, evidence=sentence)


def extract_list_candidates(context_docs, question=""):
    # Build a soft coverage checklist from final context only.
    # The LLM still decides what is supported and relevant in the normal answer.
    if not is_list_question(question):
        return []

    candidates = {}
    question_terms = get_useful_question_terms(question)
    who_mode = is_who_list_question(question)
    target_type_mode = is_target_type_question(question)

    for doc in context_docs or []:
        text = get_doc_text(doc)

        if not text:
            continue

        extract_bullet_candidates(text, candidates)

        for sentence in split_candidate_sentences(text):
            has_signal = sentence_has_list_or_relation_signal(sentence, question_terms=question_terms)

            if not has_signal:
                continue

            # Score first, cap later. This prevents early noisy chunks from hiding
            # later list-like sentences that contain more supported items.
            score = 10

            if ";" in sentence:
                score += 3

            if sentence.count(",") >= 2:
                score += 2

            if who_mode:
                extract_named_candidates(sentence, candidates, score=score, who_mode=True, target_type_mode=target_type_mode)
                extract_role_name_candidates(sentence, candidates, score=score, who_mode=True, target_type_mode=target_type_mode)
            else:
                extract_delimited_candidates(sentence, candidates, score=score)
                extract_quoted_candidates(sentence, candidates, score=score)
                extract_named_candidates(sentence, candidates, score=score, target_type_mode=target_type_mode)

    candidate_rows = list(candidates.values())

    priority = {
        "bullet": 0,
        "named_entity": 1 if who_mode else 3,
        "delimited_item": 2,
        "quoted_term": 4,
    }

    candidate_rows.sort(
        key=lambda item: (
            -int(item.get("score", 0) or 0),
            priority.get(item.get("reason", ""), 9),
            normalize_candidate_key(item.get("candidate", "")),
        )
    )

    # If a single-word candidate is already represented inside a stronger
    # multi-word candidate, drop the single-word duplicate to reduce noise.
    multiword_token_sets = []

    for item in candidate_rows:
        words = normalize_candidate_key(item.get("candidate", "")).split()
        if len(words) > 1:
            multiword_token_sets.append(set(words))

    deduped_rows = []

    for item in candidate_rows:
        words = normalize_candidate_key(item.get("candidate", "")).split()

        if len(words) == 1 and any(words[0] in token_set for token_set in multiword_token_sets):
            continue

        deduped_rows.append(item)

    return deduped_rows[:MAX_CANDIDATE_CHECKLIST_ITEMS]


def build_candidate_checklist_block(question, context_docs):
    if not CANDIDATE_CHECKLIST_ENABLED:
        return ""

    candidates = extract_list_candidates(context_docs=context_docs, question=question)

    if not candidates:
        return ""

    lines = [CANDIDATE_CHECKLIST_HEADER]

    for instruction in CANDIDATE_CHECKLIST_INSTRUCTIONS:
        lines.append(f"- {instruction}")

    for rule in CANDIDATE_SHORT_ANSWER_RULES:
        lines.append(f"- {rule}")

    lines.append("Possible candidates to review from all excerpts:" if CANDIDATE_CHECKLIST_INCLUDE_EVIDENCE else "Possible candidates to review from all excerpts:")

    for item in candidates:
        evidence = normalize_space(item.get("evidence", ""))

        if CANDIDATE_CHECKLIST_INCLUDE_EVIDENCE and evidence:
            lines.append(f"- {item['candidate']} — {evidence}")
        else:
            lines.append(f"- {item['candidate']}")

    return "\n".join(lines)


def insert_candidate_checklist_into_prompt(prompt, question, context_docs):
    # Insert the checklist before the final-answer instruction block.
    # Never append it after "FINAL ANSWER:" because the model treats anything there as answer text.
    checklist = build_candidate_checklist_block(question, context_docs)

    if not checklist:
        return prompt

    prompt = str(prompt or "")

    # Use the LAST marker, not the first, because some prompt templates mention
    # final-answer wording inside rules before the real answer area.
    marker_positions = []

    for marker in ["\nFINAL ANSWER CHECK:", "\nFINAL ANSWER:"]:
        position = prompt.rfind(marker)

        if position >= 0:
            marker_positions.append((position, marker))

    if marker_positions:
        insert_at, marker = min(marker_positions, key=lambda item: item[0])

        # When both FINAL ANSWER CHECK and FINAL ANSWER exist, insert before the
        # check block. Otherwise insert before the answer marker.
        check_position = prompt.rfind("\nFINAL ANSWER CHECK:")

        if check_position >= 0:
            insert_at = check_position

        before = prompt[:insert_at].rstrip()
        after = prompt[insert_at:]
        return before + "\n\n" + checklist + "\n" + after

    return prompt.rstrip() + "\n\n" + checklist + "\n"


def extract_text(response):
    # Kunin ang text mula sa LangChain response or streaming chunk.
    if response is None:
        return ""

    if hasattr(response, "content"):
        return str(response.content or "")

    return str(response or "")


def clean_generated_answer(answer, question=""):
    # Tanggalin common local LLM artifacts.
    answer = str(answer or "").strip()
    question = str(question or "").strip()

    if not answer:
        return ""

    if question and answer.lower().startswith(question.lower()):
        answer = answer[len(question):].strip()

    prefixes = [
        "ANSWER:",
        "Answer:",
        "FINAL ANSWER:",
        "Final answer:",
        "SAGOT:",
        "Sagot:",
        "Response:",
    ]

    changed = True

    while changed:
        changed = False

        for prefix in prefixes:
            if answer.startswith(prefix):
                answer = answer[len(prefix):].strip()
                changed = True

    split_patterns = [
        r"\n\s*\*\*Answer:\*\*\s*",
        r"\n\s*Answer:\s*",
        r"\n\s*Final answer:\s*",
        r"\n\s*FINAL ANSWER:\s*",
    ]

    for pattern in split_patterns:
        parts = re.split(pattern, answer, maxsplit=1, flags=re.IGNORECASE)

        if len(parts) == 2 and parts[0].strip() and parts[1].strip():
            answer = parts[1].strip()
            break

    return answer.strip()


def safe_answer(raw_answer, question=""):
    # Fallback kapag blank ang LLM output.
    raw_text = extract_text(raw_answer).strip()
    cleaned_text = clean_generated_answer(raw_text, question).strip()

    if cleaned_text:
        return cleaned_text

    if raw_text:
        return raw_text

    return NO_ANSWER_TEXT



def looks_like_truncated_answer(answer):
    # Generic truncation detector for local LLM outputs.
    # This does not use domain-specific words.
    text = str(answer or "").rstrip()

    if not text:
        return False

    if text.endswith(("-", "*", "**", ":", ",", ";", "(", "[", "{", "—", "–")):
        return True

    if text.count("**") % 2 != 0:
        return True

    if text.count("(") > text.count(")"):
        return True

    if text.count("[") > text.count("]"):
        return True

    lines = [line.strip() for line in text.splitlines() if line.strip()]

    if not lines:
        return False

    last_line = lines[-1]

    if re.match(r"^[-*•]\s+", last_line) and not re.search(r"[.!?)]$", last_line):
        return True

    return False


def candidate_name_only(candidate):
    candidate = clean_candidate_text(candidate)
    candidate = re.sub(r"\s+[—–-]\s+.*$", "", candidate).strip()
    candidate = re.sub(r"\s*[:;,.]\s*$", "", candidate).strip()
    return candidate


def build_candidate_fallback_answer(question, context_docs):
    # Last-resort compact answer when the LLM output is visibly cut off.
    # It uses the same JSON-configurable candidate extractor and does not call the LLM again.
    if not CANDIDATE_FALLBACK_ON_TRUNCATION:
        return ""

    if not is_list_question(question):
        return ""

    candidates = extract_list_candidates(context_docs=context_docs, question=question)

    if not candidates:
        return ""

    lines = []
    seen = set()

    for item in candidates:
        name = candidate_name_only(item.get("candidate", ""))
        key = normalize_candidate_key(name)

        if not name or not key or key in seen:
            continue

        seen.add(key)
        lines.append(f"- **{name}**")

        if len(lines) >= max(1, CANDIDATE_FALLBACK_MAX_ITEMS):
            break

    return "\n".join(lines).strip()


def get_candidate_fallback_names(question, context_docs):
    # Reuse the same generic candidate extractor used for list fallback answers.
    # This is not domain-specific; it only checks if a list answer missed many
    # candidates that were already extracted from the final context.
    if not CANDIDATE_FALLBACK_ON_TRUNCATION:
        return []

    if not is_list_question(question):
        return []

    candidates = extract_list_candidates(context_docs=context_docs, question=question)
    names = []
    seen = set()

    for item in candidates:
        name = candidate_name_only(item.get("candidate", ""))
        key = normalize_candidate_key(name)

        if not name or not key or key in seen:
            continue

        seen.add(key)
        names.append(name)

        if len(names) >= max(1, CANDIDATE_FALLBACK_MAX_ITEMS):
            break

    return names


def get_answer_list_item_scope(answer):
    # When an answer has bullets/numbered items plus a note, evaluate list coverage
    # from the actual list items only. A note may mention omitted candidates while
    # saying they are not part of the answer, so counting the whole paragraph can
    # hide under-coverage.
    item_lines = []

    for line in str(answer or "").splitlines():
        cleaned = line.strip()

        if re.match(r"^(?:[-*•]|\d+[.)])\s+", cleaned):
            item_lines.append(cleaned)

    if item_lines:
        return "\n".join(item_lines)

    return str(answer or "")


def count_answer_list_items(answer):
    count = 0

    for line in str(answer or "").splitlines():
        cleaned = line.strip()

        if re.match(r"^(?:[-*•]|\d+[.)])\s+", cleaned):
            count += 1

    return count


def count_candidate_mentions(answer, candidates):
    answer_key = normalize_candidate_key(answer)

    if not answer_key:
        return 0

    answer_words = set(answer_key.split())
    count = 0

    for candidate in candidates or []:
        candidate_key = normalize_candidate_key(candidate)

        if not candidate_key:
            continue

        # Exact normalized phrase match first. This catches names and policy terms.
        if candidate_key in answer_key:
            count += 1
            continue

        # For multi-word candidates, allow a conservative token overlap match.
        candidate_words = [word for word in candidate_key.split() if len(word) > 2]

        if len(candidate_words) >= 2 and sum(1 for word in candidate_words if word in answer_words) >= min(2, len(candidate_words)):
            count += 1

    return count


def repair_undercovered_list_answer(answer, question, context_docs):
    # If the question asks for a list and the LLM only returned a small subset,
    # replace it with the deterministic candidate list extracted from final context.
    # This prevents small local models from over-summarizing list questions.
    if not is_list_question(question):
        return answer

    candidates = get_candidate_fallback_names(question, context_docs)

    if len(candidates) < 4:
        return answer

    candidate_count = len(candidates)
    list_item_count = count_answer_list_items(answer)
    answer_scope = get_answer_list_item_scope(answer)
    mentioned = count_candidate_mentions(answer_scope, candidates)

    # For explicit list questions, require near-complete coverage from the final-context
    # candidate set. The older half-coverage threshold allowed answers with only a
    # small subset of supported items, for example 4 names out of 9.
    if candidate_count <= CANDIDATE_FALLBACK_MAX_ITEMS:
        required = candidate_count
    else:
        required = max(3, (candidate_count * 85 + 99) // 100)

    # If the model wrote bullets, evaluate only those bullets as the actual answer.
    # A later note must not hide missing list items.
    if list_item_count and mentioned >= required and list_item_count >= required:
        return answer

    if not list_item_count and mentioned >= required:
        return answer

    fallback_answer = build_candidate_fallback_answer(question, context_docs)

    if fallback_answer:
        return fallback_answer

    return answer


def repair_truncated_list_answer(answer, question, context_docs):
    if not looks_like_truncated_answer(answer):
        return answer

    fallback_answer = build_candidate_fallback_answer(question, context_docs)

    if fallback_answer:
        return fallback_answer

    return answer


def apply_retry_instructions(prompt, correction_retry=False, completion_retry=False, **_ignored):
    # Add focused retry instructions only for fallback/truncation.
    # Important: no list coverage validator here, so list questions do not trigger a second LLM call.
    if not correction_retry and not completion_retry:
        return prompt

    lines = [
        "",
        "RETRY INSTRUCTIONS:",
        "You are retrying because the previous answer was incomplete or not usable.",
    ]

    if completion_retry:
        lines.extend([
            "- The previous answer may have been cut off. Rewrite the answer from the beginning.",
            "- Keep the answer concise but complete.",
            "- Finish the final sentence completely with proper punctuation.",
        ])

    if correction_retry:
        lines.extend([
            "- The previous answer may have used the fallback even though relevant excerpts exist.",
            "- Answer only the directly supported part or give a brief directly supported correction.",
            "- If the question has an unsupported premise, correct the premise briefly and do not explain why it happened.",
        ])

    lines.extend([
        "- Do not add source labels, evidence labels, or extra sections.",
        "- Return only one final answer body.",
    ])

    return str(prompt or "").rstrip() + "\n" + "\n".join(lines)


def prepare_context_docs(
    question,
    docs,
    semantic_docs=None,
    bm25_docs=None,
    debug=False,
    all_chunks=None,
):
    # Ito lang ang final context filter bago gumawa ng prompt.
    # chatbot.py must pass reranked docs here, not already-filtered final docs.
    # all_chunks is needed for neighbor chunk expansion.
    return select_final_context_docs(
        reranked_docs=docs or [],
        question=question,
        semantic_docs=semantic_docs or [],
        bm25_docs=bm25_docs or [],
        all_chunks=all_chunks or [],
        top_n=None,
        max_chars=MAX_CONTEXT_CHARS,
        max_per_source=MAX_PER_SOURCE,
        debug=debug,
    )


def build_prompt_from_context(question, context_docs, chat_history=""):
    # Gumawa ng prompt from already selected final context docs.
    # For list/enumeration questions, add a soft candidate checklist into the same prompt.
    # This keeps generation to one LLM call and avoids strict validator/retry behavior.
    prompt = build_rag_prompt(
        question=question,
        docs=context_docs or [],
        chat_history=chat_history,
    )

    return insert_candidate_checklist_into_prompt(
        prompt=prompt,
        question=question,
        context_docs=context_docs or [],
    )


def build_prompt_with_context(
    question,
    docs,
    semantic_docs=None,
    bm25_docs=None,
    chat_history="",
    debug=False,
    all_chunks=None,
):
    # Return both prompt and exact context docs used inside the prompt.
    context_docs = prepare_context_docs(
        question=question,
        docs=docs,
        semantic_docs=semantic_docs,
        bm25_docs=bm25_docs,
        debug=debug,
        all_chunks=all_chunks,
    )

    prompt = build_prompt_from_context(
        question=question,
        context_docs=context_docs,
        chat_history=chat_history,
    )

    return {
        "prompt": prompt,
        "context_docs": context_docs,
    }


def build_prompt(
    question,
    docs,
    semantic_docs=None,
    bm25_docs=None,
    chat_history="",
    debug=False,
    all_chunks=None,
):
    # Backward-compatible helper. Returns only the prompt.
    result = build_prompt_with_context(
        question=question,
        docs=docs,
        semantic_docs=semantic_docs,
        bm25_docs=bm25_docs,
        chat_history=chat_history,
        debug=debug,
        all_chunks=all_chunks,
    )

    return result["prompt"]



def answer_normalize_key(text):
    text = str(text or "").lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = re.sub(r"[^a-z0-9\u3040-\u30ff\u3400-\u9fff]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def is_filipino_or_tagalog_question(question):
    question_key = answer_normalize_key(question)
    cues = {
        "kailan",
        "sino",
        "ano",
        "alin",
        "bakit",
        "paano",
        "ipinagdiriwang",
        "ginugunita",
        "pilipinas",
        "kalayaan",
    }
    return bool(set(question_key.split()) & cues)


def get_doc_metadata_value(doc, key, default=""):
    metadata = dict(getattr(doc, "metadata", {}) or {})
    value = metadata.get(key, default)
    return str(value or "").strip()


def has_context_mode(context_docs, mode_name):
    mode_name = str(mode_name or "").strip().lower()

    for doc in context_docs or []:
        mode = get_doc_metadata_value(doc, "context_mode").lower()
        if mode == mode_name:
            return True

    return False


def collect_evidence_snippets(context_docs, limit=2):
    snippets = []

    for doc in context_docs or []:
        metadata = dict(getattr(doc, "metadata", {}) or {})
        snippet = str(metadata.get("evidence_snippet") or "").strip()

        if not snippet:
            snippet = str(getattr(doc, "page_content", "") or "").strip()

        snippet = re.sub(r"\s+", " ", snippet).strip()

        if not snippet:
            continue

        if len(snippet) > 260:
            snippet = snippet[:260].rstrip(" ,;:-") + "..."

        if snippet not in snippets:
            snippets.append(snippet)

        if len(snippets) >= limit:
            break

    return snippets


def clean_guard_sentence(text):
    text = re.sub(r"\s+", " ", str(text or "")).strip(" ;")
    text = re.sub(r"\.{2,}$", ".", text)
    text = text.rstrip(" ;")

    if text and not text.endswith((".", "?", "!")):
        text += "."

    return text


def join_supported_snippets(snippets, filipino=False):
    cleaned = []

    for snippet in snippets or []:
        sentence = clean_guard_sentence(snippet)

        if sentence and sentence not in cleaned:
            cleaned.append(sentence)

    if not cleaned:
        return ""

    if len(cleaned) == 1:
        return cleaned[0]

    connector = " Dagdag pa, " if filipino else " Also, "
    return cleaned[0] + connector + " ".join(cleaned[1:])


def build_false_premise_answer(question, context_docs):
    snippets = collect_evidence_snippets(context_docs, limit=2)
    filipino = is_filipino_or_tagalog_question(question)
    supported_text = join_supported_snippets(snippets, filipino=filipino)

    if filipino:
        if supported_text:
            return "Hindi direktang sinusuportahan ng nakuha na teksto ang premise ng tanong. Ang suportadong impormasyon: " + supported_text
        return "Hindi direktang sinusuportahan ng nakuha na teksto ang premise ng tanong."

    if supported_text:
        return "The premise of the question is not directly supported. Supported information: " + supported_text

    return "The premise of the question is not directly supported."


def strip_retrieval_prefix_for_answer(text):
    text = re.sub(r"\s+", " ", str(text or "")).strip()

    if text.lower().startswith("retrieval context:"):
        match = re.search(r"\blanguage\s*:\s*[a-z]{2}\s+", text, flags=re.IGNORECASE)
        if match:
            return text[match.end():].strip()

    return text


def get_combined_context_text(context_docs):
    parts = []

    for doc in context_docs or []:
        metadata = dict(getattr(doc, "metadata", {}) or {})
        snippet = str(metadata.get("evidence_snippet") or "").strip()
        body = strip_retrieval_prefix_for_answer(getattr(doc, "page_content", "") or "")
        text = " ".join(part for part in [snippet, body] if part)

        if text:
            parts.append(text)

    return re.sub(r"\s+", " ", " ".join(parts)).strip()


def looks_like_date_question(question):
    question_key = answer_normalize_key(question)
    cues = {
        "when",
        "date",
        "birthday",
        "kailan",
        "ipinagdiriwang",
        "ginugunita",
        "deadline",
        "schedule",
        "expiry",
        "expiration",
    }
    return bool(set(question_key.split()) & cues)


def get_answer_intents(context_docs):
    intents = set()

    for doc in context_docs or []:
        metadata = dict(getattr(doc, "metadata", {}) or {})
        for key in ["dynamic_answer_intent", "answer_intent"]:
            value = str(metadata.get(key) or "").strip().lower()
            if value:
                intents.add(value)

    return intents


def extract_date_and_significance(context_docs):
    text = get_combined_context_text(context_docs)

    if not text:
        return "", ""

    date_match = re.search(r"\bDate\s+([A-Z][a-z]+\s+\d{1,2}(?:,\s*\d{4})?)\b", text)

    if not date_match:
        date_match = re.search(r"\bobserved\s+(?:annually\s+)?on\s+([A-Z][a-z]+\s+\d{1,2}(?:,\s*\d{4})?)\b", text, flags=re.IGNORECASE)

    if not date_match:
        date_match = re.search(r"\b([A-Z][a-z]+\s+\d{1,2},\s*\d{4})\b", text)

    date_value = date_match.group(1).strip() if date_match else ""

    significance = ""
    sig_match = re.search(
        r"\bSignificance\s+(.+?)(?:\bDate\b|\bNext time\b|\bFrequency\b|\bRelated to\b|$)",
        text,
        flags=re.IGNORECASE,
    )

    if sig_match:
        significance = re.sub(r"\s+", " ", sig_match.group(1)).strip(" .,:;-")
    else:
        commem_match = re.search(r"\bcommemorating\s+(.+?)(?:\.|;|$)", text, flags=re.IGNORECASE)
        if commem_match:
            significance = re.sub(r"\s+", " ", commem_match.group(1)).strip(" .,:;-")

    return date_value, significance


def simple_filipino_significance(text):
    text = re.sub(r"\s+", " ", str(text or "")).strip(" .")

    # Keep this generic: for any independence/commemoration phrasing, avoid
    # awkward literal translations of adjectival country names such as
    # "Philippine". The question already names the subject, so "nito" is safer.
    patterns = [
        r"declaring\s+(.+?)\s+independence\s+from\s+(.+)$",
        r"the\s+declaration\s+of\s+(.+?)\s+independence\s+from\s+(.+)$",
        r"(.+?)\s+declared\s+.+?\s+independence\s+from\s+(.+)$",
    ]

    for pattern in patterns:
        match = re.match(pattern, text, flags=re.IGNORECASE)
        if match:
            source = match.group(2).strip(" .")
            return f"deklarasyon ng kalayaan nito mula sa {source}"

    return text


def asks_for_celebration_or_significance(question):
    question_key = answer_normalize_key(question)
    cues = {
        "celebrate",
        "celebrated",
        "observed",
        "holiday",
        "commemorate",
        "commemorates",
        "significance",
        "ipinagdiriwang",
        "ginugunita",
        "araw",
        "kalayaan",
    }
    return bool(set(question_key.split()) & cues)


def build_date_fact_answer(question, context_docs):
    date_value, significance = extract_date_and_significance(context_docs)

    if not date_value:
        return ""

    filipino = is_filipino_or_tagalog_question(question)

    if filipino:
        if significance:
            return f"Ipinagdiriwang ito tuwing {date_value}. Ginugunita nito ang {simple_filipino_significance(significance)}."
        return f"Ipinagdiriwang ito tuwing {date_value}."

    if significance:
        return f"It is observed on {date_value}. It commemorates {significance}."

    return date_value


def apply_grounded_answer_guards(answer, question, context_docs):
    # Deterministic safety layer after the LLM.
    # It uses only final context docs and their system-generated metadata.
    if is_tautological_value_answer(question, answer):
        return get_value_only_tautology_clarification(question)

    if has_context_mode(context_docs, "false_premise"):
        return build_false_premise_answer(question, context_docs)

    intents = get_answer_intents(context_docs)

    if "date_fact" in intents or looks_like_date_question(question):
        date_value, significance = extract_date_and_significance(context_docs)
        answer_key = answer_normalize_key(answer)
        date_key = answer_normalize_key(date_value)

        # For holiday/observance/significance questions, prefer deterministic
        # extraction from the final context. This prevents small local models from
        # inventing years, garbling Filipino, or saying the opposite event while
        # still using only final retrieved evidence.
        if date_value and significance and asks_for_celebration_or_significance(question):
            repaired = build_date_fact_answer(question, context_docs)
            if repaired:
                return repaired

        if date_value and date_key and date_key not in answer_key:
            repaired = build_date_fact_answer(question, context_docs)
            if repaired:
                return repaired

    return answer


def generate_answer_with_context(
    question,
    docs,
    llm,
    semantic_docs=None,
    bm25_docs=None,
    chat_history="",
    debug=False,
    strict_assumption_check=True,
    correction_retry=False,
    completion_retry=False,
    list_coverage_retry=False,
    previous_answer="",
    all_chunks=None,
):
    # Main non-streaming answer function for UI and tests.
    # list_coverage_retry and previous_answer are accepted only for backward compatibility.
    # They are intentionally ignored to avoid extra LLM calls and overly strict list behavior.
    prompt_result = build_prompt_with_context(
        question=question,
        docs=docs,
        semantic_docs=semantic_docs,
        bm25_docs=bm25_docs,
        chat_history=chat_history,
        debug=debug,
        all_chunks=all_chunks,
    )

    prompt = apply_retry_instructions(
        prompt_result["prompt"],
        correction_retry=correction_retry,
        completion_retry=completion_retry,
    )
    context_docs = prompt_result["context_docs"]

    if debug:
        print("\n" + "=" * 60)
        print("PROMPT SENT TO LLM")
        print("=" * 60)
        print(prompt[:5000])

    response = llm.invoke(prompt)
    answer = safe_answer(response, question=question)
    answer = repair_truncated_list_answer(answer, question=question, context_docs=context_docs)
    answer = repair_undercovered_list_answer(answer, question=question, context_docs=context_docs)
    answer = apply_grounded_answer_guards(answer, question=question, context_docs=context_docs)
    answer = clean_generated_answer(answer=answer, question=question)

    return {
        "answer": answer,
        "context_docs": context_docs,
        "prompt": prompt,
    }


def generate_answer(
    question,
    docs,
    llm,
    semantic_docs=None,
    bm25_docs=None,
    chat_history="",
    debug=False,
    strict_assumption_check=True,
    correction_retry=False,
    completion_retry=False,
    list_coverage_retry=False,
    previous_answer="",
    all_chunks=None,
):
    # Backward-compatible answer function. Returns answer text only.
    result = generate_answer_with_context(
        question=question,
        docs=docs,
        llm=llm,
        semantic_docs=semantic_docs,
        bm25_docs=bm25_docs,
        chat_history=chat_history,
        debug=debug,
        strict_assumption_check=strict_assumption_check,
        correction_retry=correction_retry,
        completion_retry=completion_retry,
        list_coverage_retry=list_coverage_retry,
        previous_answer=previous_answer,
        all_chunks=all_chunks,
    )

    return result["answer"]


def stream_answer(
    question,
    docs,
    llm,
    semantic_docs=None,
    bm25_docs=None,
    chat_history="",
    debug=False,
    all_chunks=None,
):
    # Streaming RAG answer.
    # Uses the same context preparation as generate_answer().
    prompt_result = build_prompt_with_context(
        question=question,
        docs=docs,
        semantic_docs=semantic_docs,
        bm25_docs=bm25_docs,
        chat_history=chat_history,
        debug=debug,
        all_chunks=all_chunks,
    )

    prompt = prompt_result["prompt"]

    if debug:
        print("\n" + "=" * 60)
        print("PROMPT SENT TO LLM")
        print("=" * 60)
        print(prompt[:5000])

    emitted_text = False

    for chunk in llm.stream(prompt):
        text = extract_text(chunk)

        if text.strip():
            emitted_text = True

        yield text

    if not emitted_text:
        yield NO_ANSWER_TEXT


def clean_preview_text(text, limit=PREVIEW_CHARS):
    # Linisin source preview para sa UI/report.
    text = html.unescape(str(text or ""))
    text = re.sub(r"<[^>]+>", " ", text)
    text = " ".join(text.split())

    if len(text) <= limit:
        return text

    return text[:limit].rstrip() + "..."


def safe_metadata_value(value, default_value="N/A"):
    # Safe display value.
    if value is None:
        return default_value

    text = str(value).strip()

    if not text or text.lower() in EMPTY_VALUES:
        return default_value

    return text


def get_source_display_name(raw_source):
    # File stem para readable sa UI.
    raw_source = safe_metadata_value(raw_source, "Unknown source")

    if raw_source == "Unknown source":
        return raw_source

    return Path(raw_source).stem


def get_source_file_name(raw_source):
    # File name with extension.
    raw_source = safe_metadata_value(raw_source, "Unknown source")

    if raw_source == "Unknown source":
        return raw_source

    return Path(raw_source).name


def get_doc_key(raw_source, page, preview, metadata):
    # Stable key para hindi maulit ang same source card.
    chunk_id = metadata.get("chunk_id") or metadata.get("chunk_index")

    if chunk_id is not None:
        return (raw_source, page, chunk_id)

    return (raw_source, page, preview[:120])


def get_sources(docs, preview_limit=PREVIEW_CHARS):
    # Convert exact final context docs into UI source cards.
    sources = []
    seen = set()

    for doc in docs or []:
        metadata = dict(getattr(doc, "metadata", {}) or {})
        raw_source = safe_metadata_value(metadata.get("source") or metadata.get("file_name"), "Unknown source")
        page = safe_metadata_value(metadata.get("page"), "N/A")
        preview = clean_preview_text(getattr(doc, "page_content", ""), limit=preview_limit)
        item_key = get_doc_key(raw_source, page, preview, metadata)

        if item_key in seen:
            continue

        seen.add(item_key)

        item = {
            "source": get_source_display_name(raw_source),
            "title": get_source_display_name(raw_source),
            "file_name": get_source_file_name(raw_source),
            "source_path": raw_source,
            "file_path": raw_source,
            "path": raw_source,
            "page": page,
            "preview": preview,
            "metadata": metadata,
        }

        for score_key in SCORE_KEYS:
            if score_key in metadata:
                item[score_key] = metadata[score_key]

        sources.append(item)

    return sources
