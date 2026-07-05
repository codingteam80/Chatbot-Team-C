from chains.rag.repair_parts.retry_basics import (
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
    normalize_space,
    normalize_candidate_key,
    is_list_question,
    is_who_list_question,
    get_useful_question_terms,
    strip_retrieval_metadata,
    get_doc_text,
    split_candidate_sentences,
    sentence_has_list_or_relation_signal,
    is_target_type_question,
    get_local_candidate_window,
    make_evidence_snippet,
    has_local_relation_signal,
    is_enumeration_sentence,
    clean_candidate_text,
    is_weak_candidate,
    is_background_actor,
    is_weak_who_candidate,
    add_candidate,
    extract_bullet_candidates,
    extract_named_candidates,
    extract_role_name_candidates,
    extract_delimited_candidates,
    extract_quoted_candidates,
    extract_list_candidates,
    build_candidate_checklist_block,
    insert_candidate_checklist_into_prompt,
    extract_text,
    clean_generated_answer,
    safe_answer,
    looks_like_truncated_answer,
    candidate_name_only,
    build_candidate_fallback_answer,
    repair_truncated_list_answer,
    normalize_retry_key,
    is_no_answer_response,
    get_context_modes,
    is_false_premise_context,
    answer_rejects_false_premise,
    build_empty_answer_retry_question,
    build_no_answer_retry_question,
    build_false_premise_retry_question,
    invoke_retry_answer,
)



# ============================================================
# SAFE MERGED ANSWER REPAIRS
# ============================================================
# Conservative merge from the previous working retry layer:
# - no broad Tagalog/date rewrite
# - no global answer rewriting
# - only touches death actor failures and false-premise cleanup
# - uses candidate docs already retrieved/reranked, no hardcoded names/dates
# ============================================================

DEATH_ACTION_TERMS = {
    "kill", "killed", "killing", "death", "died", "dead",
    "assassinated", "assassination", "executed", "execution",
}

GROUP_ACTOR_WORDS = [
    "army", "forces", "force", "troops", "soldiers", "fighters",
    "men", "followers", "supporters", "group",
]

QUESTION_START_WORDS = {
    "who", "what", "when", "where", "why", "how", "did", "does",
    "do", "was", "were", "is", "are", "the", "a", "an",
}


def retry_key_contains(text, phrase):
    return normalize_retry_key(phrase) in normalize_retry_key(text)


def answer_starts_with_false_premise_rejection(answer):
    return normalize_retry_key(answer).startswith("the premise is not supported")


def clean_answer_labels_only(answer):
    # Remove only explicit answer labels. Do not rewrite normal answers globally.
    text = str(answer or "").strip()

    if not text:
        return ""

    label_pattern = r"(?is)\b(?:final\s+answer|answer|sagot)\s*:\s*"
    matches = list(re.finditer(label_pattern, text))

    if matches:
        last = matches[-1]
        if last.start() > 0:
            text = text[last.end():].strip()

    return clean_generated_answer(text).strip()


MONTH_NAME_PATTERN = (
    r"\b("
    r"January|February|March|April|May|June|July|August|September|October|November|December"
    r")\s+(\d{1,2}),?\s+(\d{4})\b"
)


def is_effective_empty_answer(answer):
    # Treat blank output and label-only output such as "Final Answer:" as empty.
    text = str(answer or "").strip()

    if not text:
        return True

    cleaned = clean_answer_labels_only(text)

    if not cleaned:
        return True

    if not normalize_retry_key(cleaned):
        return True

    return False


def extract_date_query_text(question):
    match = re.search(MONTH_NAME_PATTERN, str(question or ""), flags=re.IGNORECASE)

    if not match:
        return ""

    month = match.group(1)
    day = str(int(match.group(2)))
    year = match.group(3)
    return f"{month.title()} {day}, {year}"


def is_short_date_query(question):
    question_key = normalize_retry_key(question)

    if not question_key:
        return False

    if len(question_key.split()) > 6:
        return False

    return bool(extract_date_query_text(question))


def doc_subject_from_metadata(doc):
    metadata = dict(getattr(doc, "metadata", {}) or {})
    title = str(metadata.get("title") or metadata.get("file_stem") or metadata.get("file_name") or "").strip()

    if not title:
        return ""

    title = re.sub(r"\s*-\s*Wikipedia\s*$", "", title, flags=re.IGNORECASE)
    title = re.sub(r"\.[a-z0-9]+$", "", title, flags=re.IGNORECASE)
    title = normalize_space(title)

    return title


def sentence_has_date_terms(sentence, date_text):
    sentence_key = normalize_retry_key(sentence)
    date_key = normalize_retry_key(date_text)

    if not sentence_key or not date_key:
        return False

    return all(term in sentence_key for term in date_key.split())


def sentence_is_birth_date_evidence(sentence):
    sentence_key = normalize_retry_key(sentence)
    birth_terms = {"born", "birth", "birthday"}

    return bool(set(sentence_key.split()) & birth_terms)


def build_short_date_context_answer(question, context_docs):
    # Deterministic fallback only for short date queries that got blank output.
    # This does not rewrite normal answers and does not affect list answers.
    if not is_short_date_query(question):
        return ""

    date_text = extract_date_query_text(question)

    if not date_text:
        return ""

    best_sentence = ""
    best_subject = ""

    for doc in context_docs or []:
        subject = doc_subject_from_metadata(doc)

        for sentence in split_candidate_sentences(get_doc_text(doc)):
            if not sentence_has_date_terms(sentence, date_text):
                continue

            if sentence_is_birth_date_evidence(sentence):
                best_sentence = sentence
                best_subject = subject
                break

            if not best_sentence:
                best_sentence = sentence
                best_subject = subject

        if best_sentence and sentence_is_birth_date_evidence(best_sentence):
            break

    if not best_sentence:
        return ""

    if sentence_is_birth_date_evidence(best_sentence) and best_subject:
        return f"{date_text} was the birth date of {best_subject}."

    return best_sentence.strip()


def build_false_premise_statement_from_question(question):
    text = str(question or "").strip().rstrip("?.!")

    patterns = [
        r"^\s*why\s+did\s+(.+)$",
        r"^\s*how\s+did\s+(.+)$",
        r"^\s*why\s+was\s+(.+)$",
        r"^\s*how\s+was\s+(.+)$",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)

        if match:
            claim = match.group(1).strip()
            claim = re.sub(r"\bbecome\b", "became", claim, count=1, flags=re.IGNORECASE)
            return claim[0].upper() + claim[1:] if claim else ""

    return text


def build_false_premise_safe_answer(question):
    claim = build_false_premise_statement_from_question(question)

    if claim:
        return f"The premise is not supported. It is not directly supported that {claim}."

    return "The premise is not supported."


def repair_false_premise_answer(answer, question, false_premise_mode):
    if not false_premise_mode:
        return answer

    answer_text = clean_answer_labels_only(answer)

    if answer_starts_with_false_premise_rejection(answer_text):
        return answer_text

    if answer_rejects_false_premise(answer_text):
        return build_false_premise_safe_answer(question)

    return answer_text


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
    'add_candidate',
    'extract_bullet_candidates',
    'extract_named_candidates',
    'extract_role_name_candidates',
    'extract_delimited_candidates',
    'extract_quoted_candidates',
    'extract_list_candidates',
    'build_candidate_checklist_block',
    'insert_candidate_checklist_into_prompt',
    'extract_text',
    'clean_generated_answer',
    'safe_answer',
    'looks_like_truncated_answer',
    'candidate_name_only',
    'build_candidate_fallback_answer',
    'repair_truncated_list_answer',
    'normalize_retry_key',
    'is_no_answer_response',
    'get_context_modes',
    'is_false_premise_context',
    'answer_rejects_false_premise',
    'build_empty_answer_retry_question',
    'build_no_answer_retry_question',
    'build_false_premise_retry_question',
    'invoke_retry_answer',
    'DEATH_ACTION_TERMS',
    'GROUP_ACTOR_WORDS',
    'QUESTION_START_WORDS',
    'retry_key_contains',
    'answer_starts_with_false_premise_rejection',
    'clean_answer_labels_only',
    'MONTH_NAME_PATTERN',
    'is_effective_empty_answer',
    'extract_date_query_text',
    'is_short_date_query',
    'doc_subject_from_metadata',
    'sentence_has_date_terms',
    'sentence_is_birth_date_evidence',
    'build_short_date_context_answer',
    'build_false_premise_statement_from_question',
    'build_false_premise_safe_answer',
    'repair_false_premise_answer',
]
