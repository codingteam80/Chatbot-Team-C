from chains.rag.candidates import (
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
)


# ============================================================
# GENERIC POST-ANSWER RETRY GUARDS
# ============================================================
# These guards run after the first LLM answer only when there is an obvious
# generation failure. They do not change retrieval, context filtering, or
# working list answers.
# ============================================================


def load_post_answer_retry_config():
    # Retry phrase lists are stored in JSON so Python stays generic.
    raw_config = read_query_config()

    if not isinstance(raw_config, dict):
        return {
            "no_answer_fallback_phrases": [],
            "false_premise_rejection_phrases": [],
        }

    retry_config = raw_config.get("post_answer_retry", {})

    if not isinstance(retry_config, dict):
        retry_config = {}

    return {
        "no_answer_fallback_phrases": normalize_config_list(retry_config.get("no_answer_fallback_phrases", [])),
        "false_premise_rejection_phrases": normalize_config_list(retry_config.get("false_premise_rejection_phrases", [])),
    }


POST_ANSWER_RETRY_CONFIG = load_post_answer_retry_config()


def normalize_retry_key(text):
    text = str(text or "").lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = re.sub(r"[^a-z0-9\u3040-\u30ff\u3400-\u9fff]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def is_no_answer_response(answer):
    answer_key = normalize_retry_key(answer)

    if not answer_key:
        return False

    fallback_key = normalize_retry_key(NO_ANSWER_TEXT)

    if fallback_key and answer_key == fallback_key:
        return True

    fallback_phrases = POST_ANSWER_RETRY_CONFIG.get("no_answer_fallback_phrases", [])

    return any(normalize_retry_key(phrase) in answer_key for phrase in fallback_phrases)


def get_context_modes(context_docs):
    modes = set()

    for doc in context_docs or []:
        metadata = dict(getattr(doc, "metadata", {}) or {})

        for key in ["context_mode", "dynamic_context_policy_mode", "answer_intent"]:
            value = normalize_retry_key(metadata.get(key, ""))

            if value:
                modes.add(value)

    return modes


def is_false_premise_context(context_docs):
    modes = get_context_modes(context_docs)
    return "false premise" in modes or "false_premise" in modes


def answer_rejects_false_premise(answer):
    answer_key = normalize_retry_key(answer)

    if not answer_key:
        return False

    rejection_phrases = POST_ANSWER_RETRY_CONFIG.get("false_premise_rejection_phrases", [])

    return any(normalize_retry_key(phrase) in answer_key for phrase in rejection_phrases)


def build_empty_answer_retry_question(question):
    return (
        "Identify what the user's query refers to using the provided excerpts. "
        "If the query is a date, name, acronym, code, title, or short keyword, "
        "state the directly associated person, event, document, policy, deadline, status, or fact. "
        "Return one concise answer. "
        f"Query: {question}"
    )


def build_no_answer_retry_question(question):
    return (
        "Answer the question using any relevant information in the provided excerpts. "
        "If the excerpts only partially answer the question, give only the supported part carefully. "
        "Use the no-answer fallback only when none of the excerpts are relevant. "
        f"Question: {question}"
    )


def build_false_premise_retry_question(question):
    return (
        "The question may contain an unsupported premise. "
        "First check whether the exact role, title, action, status, date, cause, or relationship is supported. "
        "If the exact premise is not supported, start exactly with: The premise is not supported. "
        "Do not answer why or how an unsupported premise happened. "
        "Do not replace the questioned role, title, action, status, date, or relationship with a different related one. "
        "After rejecting the premise, give only the corrected supported fact if available. "
        f"Question: {question}"
    )


def invoke_retry_answer(llm, retry_question, context_docs, chat_history="", debug=False, retry_reason=""):
    from chains.rag.context_prepare import build_prompt_from_context

    retry_prompt = build_prompt_from_context(
        question=retry_question,
        context_docs=context_docs,
        chat_history=chat_history,
    )

    if debug:
        print("\n" + "=" * 60)
        print(f"POST-ANSWER RETRY: {retry_reason}")
        print("=" * 60)
        print(retry_prompt[:5000])

    retry_response = llm.invoke(retry_prompt)
    retry_answer = safe_answer(retry_response, question=retry_question)
    retry_answer = repair_truncated_list_answer(
        retry_answer,
        question=retry_question,
        context_docs=context_docs,
    )

    return retry_answer, retry_prompt


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
]
