from chains.rag.context_prepare import (
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
    apply_generic_post_answer_retry,
    apply_retry_instructions,
    prepare_context_docs,
    build_prompt_from_context,
    build_prompt_with_context,
    build_prompt,
)



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

    # Hard lock for short date queries such as "June 19, 1861".
    # If the final context already contains the direct date evidence, do not ask
    # the LLM anymore because small local models can randomly output blank/label-only text.
    # This only affects short explicit date queries and leaves normal questions/list answers unchanged.
    pre_llm_date_answer = build_short_date_context_answer(question, context_docs)

    if pre_llm_date_answer:
        return {
            "answer": pre_llm_date_answer,
            "context_docs": context_docs,
            "prompt": prompt,
            "retry_prompt": "",
        }

    if debug:
        print("\n" + "=" * 60)
        print("PROMPT SENT TO LLM")
        print("=" * 60)
        print(prompt[:5000])

    response = llm.invoke(prompt)
    answer = safe_answer(response, question=question)
    answer = repair_truncated_list_answer(answer, question=question, context_docs=context_docs)

    retry_prompt = ""

    if not correction_retry and not completion_retry:
        answer, retry_prompt = apply_generic_post_answer_retry(
            answer=answer,
            question=question,
            context_docs=context_docs,
            llm=llm,
            chat_history=chat_history,
            debug=debug,
            candidate_docs=docs,
        )

    return {
        "answer": answer,
        "context_docs": context_docs,
        "prompt": prompt,
        "retry_prompt": retry_prompt,
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
    context_docs = prompt_result["context_docs"]

    # Same hard lock for streaming UI path.
    pre_llm_date_answer = build_short_date_context_answer(question, context_docs)

    if pre_llm_date_answer:
        yield pre_llm_date_answer
        return

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
    'prepare_context_docs',
    'build_prompt_from_context',
    'build_prompt_with_context',
    'build_prompt',
    'generate_answer_with_context',
    'generate_answer',
    'stream_answer',
]
