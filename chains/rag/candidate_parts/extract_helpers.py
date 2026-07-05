from chains.rag.config import (
    html,
    json,
    re,
    unicodedata,
    Path,
    MAX_CONTEXT_CHARS,
    MAX_PER_SOURCE,
    NO_ANSWER_TEXT,
    PREVIEW_CHARS,
    build_rag_prompt,
    select_final_context_docs,
    EMPTY_VALUES,
    SCORE_KEYS,
    DEFAULT_QUERY_CONFIG_PATH,
    read_query_config,
    normalize_config_list,
    normalize_config_key,
    normalize_config_set,
    config_int,
    config_bool,
    get_nested_dict,
    load_candidate_checklist_config,
    CANDIDATE_CHECKLIST_CONFIG,
    MAX_CANDIDATE_CHECKLIST_ITEMS,
    CANDIDATE_EVIDENCE_CHARS,
    CANDIDATE_CHECKLIST_ENABLED,
    CANDIDATE_CHECKLIST_INCLUDE_EVIDENCE,
    APPLY_LOCAL_NON_TARGET_FILTER,
    LIST_QUESTION_PATTERNS,
    WHO_QUESTION_PATTERNS,
    GENERIC_STOPWORDS,
    GENERIC_LIST_SIGNAL_TERMS,
    WEAK_CANDIDATE_STARTS,
    WEAK_CANDIDATE_WORDS,
    WHO_MODE_WEAK_ENTITY_WORDS,
    WHO_MODE_NOISE_STARTS,
    TARGET_TYPE_QUESTION_TERMS,
    OBVIOUS_NON_TARGET_WORDS,
    BACKGROUND_ACTOR_PATTERNS,
    RELATION_LOCAL_TERMS,
    ROLE_NAME_TERMS,
    SENTENCE_START_SKIP_WORDS,
    INTERNAL_WEAK_WORDS,
    SHORT_WEAK_VALUES,
    CANDIDATE_CHECKLIST_HEADER,
    CANDIDATE_CHECKLIST_INSTRUCTIONS,
    CANDIDATE_SHORT_ANSWER_RULES,
    CANDIDATE_FALLBACK_ON_TRUNCATION,
    CANDIDATE_FALLBACK_MAX_ITEMS,
)

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
    'normalize_space',
    'normalize_candidate_key',
    'is_list_question',
    'is_who_list_question',
    'get_useful_question_terms',
    'strip_retrieval_metadata',
    'get_doc_text',
    'split_candidate_sentences',
    'sentence_has_list_or_relation_signal',
    'is_target_type_question',
    'get_local_candidate_window',
    'make_evidence_snippet',
    'has_local_relation_signal',
    'is_enumeration_sentence',
    'clean_candidate_text',
    'is_weak_candidate',
    'is_background_actor',
    'is_weak_who_candidate',
]
