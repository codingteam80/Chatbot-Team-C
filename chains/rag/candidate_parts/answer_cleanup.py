from chains.rag.candidate_parts.prompt_block import (
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
)



def extract_text(response):
    # Get the text from the LangChain response or streaming chunk.
    if response is None:
        return ""

    if hasattr(response, "content"):
        return str(response.content or "")

    return str(response or "")


def clean_generated_answer(answer, question=""):
    # Remove common local LLM artifacts.
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

    return answer.strip()


def safe_answer(raw_answer, question=""):
    # Fallback when the LLM output is blank.
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


def repair_truncated_list_answer(answer, question, context_docs):
    if not looks_like_truncated_answer(answer):
        return answer

    fallback_answer = build_candidate_fallback_answer(question, context_docs)

    if fallback_answer:
        return fallback_answer

    return answer


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
]
