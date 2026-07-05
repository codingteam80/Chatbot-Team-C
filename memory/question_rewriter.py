import json
import os
import re
from pathlib import Path

from config.settings import CLARIFY_PREFIX


CONFIG_FILE_NAME = "query_expansion_config.json"
CONFIG_SECTION = "question_rewriter"


def normalize_space(text):
    # Collapse whitespace for predictable matching.
    return re.sub(r"\s+", " ", str(text or "")).strip()


def get_config_paths():
    # Search common project locations without depending on the current folder.
    paths = []
    env_path = normalize_space(os.getenv("QUERY_EXPANSION_CONFIG_PATH", ""))

    if env_path:
        paths.append(Path(env_path))

    file_path = Path(__file__).resolve()
    paths.extend([
        Path.cwd() / CONFIG_FILE_NAME,
        Path.cwd() / "config" / CONFIG_FILE_NAME,
        file_path.with_name(CONFIG_FILE_NAME),
        file_path.parent.parent / CONFIG_FILE_NAME,
        file_path.parent.parent / "config" / CONFIG_FILE_NAME,
    ])

    unique_paths = []
    seen = set()

    for path in paths:
        key = str(path)
        if key in seen:
            continue

        seen.add(key)
        unique_paths.append(path)

    return unique_paths


def load_rewriter_config():
    # Load rewrite rules from JSON so patterns stay outside Python code.
    for path in get_config_paths():
        if not path.is_file():
            continue

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue

        config = data.get(CONFIG_SECTION, {})
        if isinstance(config, dict):
            return config

    return {}


REWRITER_CONFIG = load_rewriter_config()


def get_config_list(key):
    # Return a clean list from the JSON section.
    value = REWRITER_CONFIG.get(key, [])

    if not isinstance(value, list):
        return []

    return [normalize_space(item) for item in value if normalize_space(item)]


def get_config_dict(key):
    # Return a dictionary from the JSON section.
    value = REWRITER_CONFIG.get(key, {})
    return value if isinstance(value, dict) else {}


def get_config_int(key, default):
    # Read numeric thresholds from JSON with a safe fallback.
    try:
        return int(REWRITER_CONFIG.get(key, default))
    except (TypeError, ValueError):
        return default


def get_config_bool(key, default=False):
    # Read boolean switches from JSON with a safe fallback.
    value = REWRITER_CONFIG.get(key, default)
    return value if isinstance(value, bool) else default


def get_config_text(key, default=""):
    # Read a text pattern from JSON.
    value = REWRITER_CONFIG.get(key, default)
    return value if isinstance(value, str) else default


REFERENCE_WORDS = set(get_config_list("reference_words"))
OBJECT_REFERENCE_WORDS = set(get_config_list("object_reference_words"))
QUESTION_STARTS = set(get_config_list("question_starts"))
QUESTION_WORDS = set(get_config_list("question_words"))
QUESTION_PREFIX_PATTERNS = get_config_list("standalone_topic_patterns")
TOPIC_EXTRACTION_PATTERNS = get_config_list("topic_extraction_patterns")
TRAILING_FILLERS = get_config_list("trailing_fillers")
CONTEXTUAL_REFERENCE_PATTERNS = get_config_list("contextual_reference_patterns")
CONTEXTUAL_FOLLOWUP_PATTERNS = get_config_list("contextual_followup_patterns")
VAGUE_FOLLOWUP_REWRITES = get_config_dict("vague_followup_rewrites")
SOURCE_SUFFIX_PATTERN = get_config_text("source_suffix_pattern")
TOPIC_PREFIX_PATTERN = get_config_text("topic_prefix_pattern")
DEMONSTRATIVE_NOUN_PATTERN = get_config_text("demonstrative_noun_pattern")
DEMONSTRATIVE_NOUN_TEMPLATE = get_config_text("demonstrative_noun_rewrite_template", "{topic}")
TOKEN_PATTERN = get_config_text("token_pattern", r"[\w'’-]+")
MIN_STANDALONE_MEANINGFUL_TERMS = get_config_int("min_standalone_meaningful_terms", 2)
MIN_TOPIC_MEANINGFUL_TERMS = get_config_int("min_topic_meaningful_terms", 1)
USE_SOURCE_FALLBACK_TOPIC = get_config_bool("use_source_fallback_topic", False)


# Backward-compatible name for older imports.
SHORT_FOLLOWUP_STARTS = QUESTION_STARTS


def compile_word_group(words):
    # Build a safe regex group from JSON words.
    clean_words = [re.escape(word) for word in words if normalize_space(word)]

    if not clean_words:
        return ""

    return "(?:" + "|".join(sorted(clean_words, key=len, reverse=True)) + ")"


REFERENCE_PATTERN = compile_word_group(REFERENCE_WORDS)
OBJECT_REFERENCE_PATTERN = compile_word_group(OBJECT_REFERENCE_WORDS or REFERENCE_WORDS)
QUESTION_START_PATTERN = compile_word_group(QUESTION_STARTS)


def ensure_question_mark(text):
    # Keep retrieval queries question-like without changing meaning.
    text = normalize_space(text).rstrip(". ")

    if text and not text.endswith("?"):
        text += "?"

    return text


def is_clarification(text):
    # Detect internal clarification marker.
    return str(text or "").strip().upper().startswith(str(CLARIFY_PREFIX).upper())


def safe_regex_search(pattern, text):
    # Ignore invalid JSON regex entries instead of breaking the app.
    try:
        return re.search(pattern, text, flags=re.I)
    except re.error:
        return None


def safe_regex_match(pattern, text):
    # Ignore invalid JSON regex entries instead of breaking the app.
    try:
        return re.match(pattern, text, flags=re.I)
    except re.error:
        return None


def get_tokens(text):
    # Tokenize text using the JSON token pattern.
    try:
        return [token.lower() for token in re.findall(TOKEN_PATTERN, normalize_space(text))]
    except re.error:
        return [token.lower() for token in normalize_space(text).split()]


def count_meaningful_terms(text):
    # Count topic-bearing terms, not total words.
    tokens = get_tokens(text)
    return len([token for token in tokens if token not in QUESTION_WORDS and token not in REFERENCE_WORDS])


def clean_topic(text):
    # Remove common wrappers before using text as a topic.
    topic = normalize_space(text).strip(" ?.!,:;\"'")

    if SOURCE_SUFFIX_PATTERN:
        topic = re.sub(SOURCE_SUFFIX_PATTERN, "", topic, flags=re.I)

    for pattern in TRAILING_FILLERS:
        topic = re.sub(pattern, "", topic, flags=re.I).strip()

    if TOPIC_PREFIX_PATTERN:
        topic = re.sub(TOPIC_PREFIX_PATTERN, "", topic, flags=re.I).strip()

    return topic.strip(" ?.!,:;\"'")


def contains_reference_word(question):
    # Detect contextual references without treating every short question as a follow-up.
    text = normalize_space(question).lower().rstrip("?")

    if not text:
        return False

    for pattern in CONTEXTUAL_REFERENCE_PATTERNS:
        if safe_regex_search(pattern, text):
            return True

    if REFERENCE_PATTERN and re.search(rf"^{REFERENCE_PATTERN}\b", text, flags=re.I):
        return True

    if QUESTION_START_PATTERN and REFERENCE_PATTERN:
        if re.search(rf"^{QUESTION_START_PATTERN}\s+{REFERENCE_PATTERN}\b", text, flags=re.I):
            return True

        if re.search(rf"^{QUESTION_START_PATTERN}\s+\w+\s+{REFERENCE_PATTERN}\b", text, flags=re.I):
            return True

    if OBJECT_REFERENCE_PATTERN:
        if re.search(rf"\b{OBJECT_REFERENCE_PATTERN}\s*$", text, flags=re.I):
            return True

    return False


def looks_like_contextual_followup(question):
    # Detect vague or explicitly contextual follow-ups of any length.
    text = normalize_space(question).lower().rstrip("?")

    if not text:
        return False

    if contains_reference_word(text):
        return True

    if text in {normalize_space(key).lower().rstrip("?") for key in VAGUE_FOLLOWUP_REWRITES}:
        return True

    for pattern in CONTEXTUAL_FOLLOWUP_PATTERNS:
        if safe_regex_search(pattern, text):
            return True

    return False


def looks_like_short_followup(question):
    # Compatibility wrapper. Detection is no longer based on word count.
    return looks_like_contextual_followup(question)


def extract_topic_with_patterns(question, patterns):
    # Use JSON patterns to pull a topic from common question forms.
    question = normalize_space(question)
    lowered = question.lower().rstrip("?")

    for pattern in patterns:
        match = safe_regex_match(pattern, lowered)

        if not match or not match.groups():
            continue

        start, end = match.span(1)
        topic = clean_topic(question[start:end])

        if topic and count_meaningful_terms(topic) >= MIN_TOPIC_MEANINGFUL_TERMS:
            return topic

    return ""


def has_clear_standalone_topic(question):
    # A standalone question has its own topic and no contextual reference.
    question = normalize_space(question)

    if not question or contains_reference_word(question):
        return False

    topic = extract_topic_with_patterns(question, QUESTION_PREFIX_PATTERNS)
    if topic:
        return True

    return count_meaningful_terms(question) >= MIN_STANDALONE_MEANINGFUL_TERMS


def is_standalone_question(question):
    # Public helper for compatibility.
    return has_clear_standalone_topic(question)


def is_follow_up_question(question):
    # Public helper used by chains.chatbot.
    if not normalize_space(question):
        return False

    if has_clear_standalone_topic(question):
        return False

    return looks_like_contextual_followup(question)


def extract_topic_from_question(question):
    # Pull a likely topic from a previous clear user question.
    question = clean_topic(question)

    if not question or contains_reference_word(question):
        return ""

    topic = extract_topic_with_patterns(question, TOPIC_EXTRACTION_PATTERNS)
    if topic:
        return topic

    topic = extract_topic_with_patterns(question, QUESTION_PREFIX_PATTERNS)
    if topic:
        return topic

    if count_meaningful_terms(question) >= MIN_TOPIC_MEANINGFUL_TERMS:
        return question

    return ""


def get_history_value(line):
    # Extract value from a memory line.
    line = normalize_space(line)

    if ":" not in line:
        return "", ""

    label, value = line.split(":", 1)
    return label.strip().lower(), clean_topic(value)


def get_latest_topic(chat_history, current_question=None):
    # Prefer the latest clear user topic. Source fallback is disabled by default.
    current_key = normalize_space(current_question).lower()
    fallback_source = ""

    for raw_line in reversed(str(chat_history or "").splitlines()):
        label, value = get_history_value(raw_line)

        if not value or value.lower() == current_key:
            continue

        if label == "source":
            if USE_SOURCE_FALLBACK_TOPIC and not fallback_source:
                fallback_source = value
            continue

        if label != "user":
            continue

        if is_follow_up_question(value):
            continue

        topic = extract_topic_from_question(value)
        if topic:
            return topic

    return fallback_source


def cleanup_rewrite(text):
    # Small grammar cleanup for common follow-up rewrites.
    text = normalize_space(text)
    text = re.sub(r"\bdid\s+(.+?)\s+died\b", r"did \1 die", text, flags=re.I)
    text = re.sub(r"\bwhen\s+did\s+(.+?)\s+died\b", r"when did \1 die", text, flags=re.I)
    return text


def replace_demonstrative_noun(question, topic):
    # Rewrite phrases like "that policy" without attaching the noun to the topic name.
    if not DEMONSTRATIVE_NOUN_PATTERN:
        return question

    def replacement(match):
        noun = match.group(2) if len(match.groups()) >= 2 else "topic"
        return render_template(DEMONSTRATIVE_NOUN_TEMPLATE, topic).replace("{noun}", noun)

    try:
        return re.sub(DEMONSTRATIVE_NOUN_PATTERN, replacement, question, count=1, flags=re.I)
    except re.error:
        return question


def replace_reference(question, topic):
    # Replace the first clear reference with the latest topic.
    rewritten = replace_demonstrative_noun(question, topic)

    if rewritten != question:
        return cleanup_rewrite(rewritten)

    if not REFERENCE_PATTERN:
        return question

    def replacement(match):
        word = match.group(0).lower()

        if word in {"its", "his", "their"}:
            return f"{topic}'s"

        return topic

    rewritten = re.sub(
        rf"\b{REFERENCE_PATTERN}\b",
        replacement,
        question,
        count=1,
        flags=re.I,
    )

    return cleanup_rewrite(rewritten)


def render_template(template, topic):
    # Render JSON rewrite templates.
    return normalize_space(str(template or "").replace("{topic}", topic))


def deterministic_followup_rewrite(question, topic):
    # Resolve contextual follow-ups without an extra LLM call.
    question = normalize_space(question)
    topic = clean_topic(topic)

    if not question or not topic:
        return ""

    q = question.lower().rstrip("?")
    template = VAGUE_FOLLOWUP_REWRITES.get(q)

    if template:
        return ensure_question_mark(render_template(template, topic))

    if contains_reference_word(question):
        return ensure_question_mark(replace_reference(question, topic))

    if looks_like_contextual_followup(question):
        return ensure_question_mark(f"{question.rstrip('?')} about {topic}")

    return ""


def rewrite_question(question, chat_history, llm=None):
    # Rewrite only contextual follow-up questions.
    question = normalize_space(question)
    chat_history = str(chat_history or "").strip()

    if not question or not chat_history:
        return question

    if is_standalone_question(question):
        return ensure_question_mark(question)

    if not is_follow_up_question(question):
        return question

    latest_topic = get_latest_topic(chat_history, current_question=question)
    rewritten = deterministic_followup_rewrite(question, latest_topic)

    if rewritten:
        return rewritten

    return f"{CLARIFY_PREFIX} Which person, item, or topic do you mean?"
