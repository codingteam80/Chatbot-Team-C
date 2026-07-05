from chains.rag.generation import (
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
    generate_answer_with_context,
    generate_answer,
    stream_answer,
)



def clean_preview_text(text, limit=PREVIEW_CHARS):
    # Clean the source preview for the UI/report.
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
    # File stem for a readable UI label.
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
    # Stable key to avoid repeating the same source card.
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
    'clean_preview_text',
    'safe_metadata_value',
    'get_source_display_name',
    'get_source_file_name',
    'get_doc_key',
    'get_sources',
]
