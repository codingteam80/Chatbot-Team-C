from chains.rag.repair_parts.death_actor import (
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
    DEATH_ACTION_TERMS,
    GROUP_ACTOR_WORDS,
    QUESTION_START_WORDS,
    retry_key_contains,
    answer_starts_with_false_premise_rejection,
    clean_answer_labels_only,
    MONTH_NAME_PATTERN,
    is_effective_empty_answer,
    extract_date_query_text,
    is_short_date_query,
    doc_subject_from_metadata,
    sentence_has_date_terms,
    sentence_is_birth_date_evidence,
    build_short_date_context_answer,
    build_false_premise_statement_from_question,
    build_false_premise_safe_answer,
    repair_false_premise_answer,
    get_all_candidate_docs,
    question_contains_death_action,
    is_death_actor_question,
    extract_question_action_target,
    extract_yes_no_actor,
    extract_capitalized_terms,
    target_terms_from_question,
    sentence_mentions_target,
    sentence_mentions_death_action,
    extract_group_actor_phrase,
    select_death_actor_evidence_sentence,
    build_death_actor_supported_answer,
)



def apply_generic_post_answer_retry(answer, question, context_docs, llm, chat_history="", debug=False, candidate_docs=None):
    # Do not touch answers when there is no final context.
    if not context_docs:
        return answer, ""

    answer_text = str(answer or "").strip()
    false_premise_mode = is_false_premise_context(context_docs)
    all_candidate_docs = get_all_candidate_docs(context_docs, candidate_docs)

    # Safe deterministic repair only for short date queries that returned blank or label-only output.
    date_fallback_answer = build_short_date_context_answer(question, context_docs)

    if date_fallback_answer and (is_effective_empty_answer(answer_text) or is_no_answer_response(answer_text)):
        return date_fallback_answer, ""

    # Safe deterministic repair only for death/actor questions that already failed.
    # This uses retrieved candidate docs so the final-context intro chunk does not hide
    # a better ranked evidence chunk from the same retrieval result.
    if is_death_actor_question(question):
        actor_answer = build_death_actor_supported_answer(question, all_candidate_docs)

        if actor_answer and (
            is_no_answer_response(answer_text)
            or answer_starts_with_false_premise_rejection(answer_text)
            or retry_key_contains(answer_text, "does not specify who")
        ):
            return actor_answer, ""

    # If false-premise mode already rejected the premise but formatted it badly,
    # clean it without a second LLM call. Normal questions are not affected.
    if false_premise_mode and answer_rejects_false_premise(answer_text):
        return repair_false_premise_answer(answer_text, question, false_premise_mode), ""

    retry_reason = ""
    retry_question = ""

    if false_premise_mode and not answer_rejects_false_premise(answer_text):
        retry_reason = "false_premise_not_rejected"
        retry_question = build_false_premise_retry_question(question)
    elif is_effective_empty_answer(answer_text):
        retry_reason = "empty_answer_with_context"
        retry_question = build_empty_answer_retry_question(question)
    elif is_no_answer_response(answer_text):
        retry_reason = "no_answer_with_context"
        retry_question = build_no_answer_retry_question(question)

    if not retry_question:
        return answer, ""

    retry_answer, retry_prompt = invoke_retry_answer(
        llm=llm,
        retry_question=retry_question,
        context_docs=context_docs,
        chat_history=chat_history,
        debug=debug,
        retry_reason=retry_reason,
    )

    retry_answer_text = str(retry_answer or "").strip()

    if date_fallback_answer and (is_effective_empty_answer(retry_answer_text) or is_no_answer_response(retry_answer_text)):
        return date_fallback_answer, retry_prompt

    # Re-check death/actor repair after retry.
    if is_death_actor_question(question):
        actor_answer = build_death_actor_supported_answer(question, all_candidate_docs)

        if actor_answer and (
            is_no_answer_response(retry_answer_text)
            or answer_starts_with_false_premise_rejection(retry_answer_text)
            or retry_key_contains(retry_answer_text, "does not specify who")
        ):
            return actor_answer, retry_prompt

    if retry_reason == "false_premise_not_rejected":
        if retry_answer_text and answer_rejects_false_premise(retry_answer_text):
            return repair_false_premise_answer(retry_answer_text, question, false_premise_mode), retry_prompt

        # Last-resort safe answer only for documents already marked false_premise.
        return build_false_premise_safe_answer(question), retry_prompt

    if retry_answer_text and not is_no_answer_response(retry_answer_text):
        return retry_answer_text, retry_prompt

    return answer, retry_prompt


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
    'get_all_candidate_docs',
    'question_contains_death_action',
    'is_death_actor_question',
    'extract_question_action_target',
    'extract_yes_no_actor',
    'extract_capitalized_terms',
    'target_terms_from_question',
    'sentence_mentions_target',
    'sentence_mentions_death_action',
    'extract_group_actor_phrase',
    'select_death_actor_evidence_sentence',
    'build_death_actor_supported_answer',
    'apply_generic_post_answer_retry',
    'apply_retry_instructions',
]
