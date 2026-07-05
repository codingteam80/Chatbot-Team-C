from chains.rag.repair_parts.false_premise import (
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
)



def get_all_candidate_docs(context_docs, candidate_docs=None):
    docs = []
    seen = set()

    for doc in list(context_docs or []) + list(candidate_docs or []):
        metadata = dict(getattr(doc, "metadata", {}) or {})
        key = (
            metadata.get("source") or metadata.get("file_name") or "",
            metadata.get("page"),
            metadata.get("chunk_id") or metadata.get("chunk_index"),
            id(doc),
        )

        if key in seen:
            continue

        seen.add(key)
        docs.append(doc)

    return docs


def question_contains_death_action(question):
    words = set(normalize_retry_key(question).split())
    return bool(words & DEATH_ACTION_TERMS)


def is_death_actor_question(question):
    question_key = normalize_retry_key(question)

    if not question_key or not question_contains_death_action(question_key):
        return False

    return question_key.startswith(("who ", "did ", "was ", "were ", "which person", "which group"))


def extract_question_action_target(question):
    question_text = str(question or "").strip()
    action_pattern = (
        r"\b(kill(?:ed|ing)?|died|death|dead|assassinated|assassination|"
        r"executed|execution)\b"
    )

    match = re.search(action_pattern + r"\s+(.+?)(?:[?.!]|$)", question_text, flags=re.IGNORECASE)

    if not match:
        return "", ""

    action = match.group(1)
    target = match.group(2).strip(" ?.!,:;")
    target = re.sub(r"\b(in|during|after|before|at|on|from)\b.*$", "", target, flags=re.IGNORECASE).strip()
    return action, target


def extract_yes_no_actor(question):
    question_text = str(question or "").strip()
    pattern = (
        r"^\s*(?:did|does|do|was|were|is|are)\s+(.+?)\s+"
        r"(kill(?:ed|ing)?|assassinate(?:d)?|execute(?:d)?)\b"
    )

    match = re.search(pattern, question_text, flags=re.IGNORECASE)

    if not match:
        return ""

    actor = match.group(1).strip(" ?.!,:;")

    if normalize_retry_key(actor) in QUESTION_START_WORDS:
        return ""

    return actor


def extract_capitalized_terms(text):
    pattern = r"\b[A-ZÀ-ÖØ-Þ][A-Za-zÀ-ÖØ-öø-ÿ.'-]*(?:\s+[A-ZÀ-ÖØ-Þ][A-Za-zÀ-ÖØ-öø-ÿ.'-]*){0,5}\b"
    terms = []

    for match in re.finditer(pattern, str(text or "")):
        value = match.group(0).strip()
        key = normalize_retry_key(value)

        if not key or key in QUESTION_START_WORDS:
            continue

        if value not in terms:
            terms.append(value)

    return terms


def target_terms_from_question(question):
    _, target = extract_question_action_target(question)

    if target:
        terms = [target]
        terms.extend(extract_capitalized_terms(target))

        for word in re.split(r"\s+", target):
            word = word.strip(" ?.!,:;")
            if len(word) > 2 and normalize_retry_key(word) not in QUESTION_START_WORDS:
                terms.append(word)

        deduped = []
        seen = set()

        for term in terms:
            key = normalize_retry_key(term)
            compact_key = key.replace(" ", "")

            if not key or compact_key in seen:
                continue

            seen.add(compact_key)
            deduped.append(term)

        return deduped

    return extract_capitalized_terms(question)


def sentence_mentions_target(sentence, target_terms):
    sentence_key = normalize_retry_key(sentence)
    compact_sentence_key = sentence_key.replace(" ", "")

    for target in target_terms or []:
        target_key = normalize_retry_key(target)
        compact_target_key = target_key.replace(" ", "")

        if not target_key:
            continue

        if all(word in sentence_key for word in target_key.split()):
            return True

        if compact_target_key and compact_target_key in compact_sentence_key:
            return True

    return False


def sentence_mentions_death_action(sentence):
    sentence_key = normalize_retry_key(sentence)
    return bool(set(sentence_key.split()) & DEATH_ACTION_TERMS)


def extract_group_actor_phrase(sentence):
    actor_name_pattern = r"([A-ZÀ-ÖØ-Þ][A-Za-zÀ-ÖØ-öø-ÿ.'-]*(?:\s+[A-ZÀ-ÖØ-Þ][A-Za-zÀ-ÖØ-öø-ÿ.'-]*){0,4})"
    group_words = "|".join(re.escape(word) for word in GROUP_ACTOR_WORDS)

    patterns = [
        rf"\b(?:against\s+the\s+|against\s+)?({group_words})\s+(?:of|under)\s+{actor_name_pattern}(?:,\s*([^,.;]{{1,80}}))?",
        rf"\b({group_words})\s+led\s+by\s+{actor_name_pattern}(?:,\s*([^,.;]{{1,80}}))?",
    ]

    for pattern in patterns:
        match = re.search(pattern, sentence, flags=re.IGNORECASE)

        if not match:
            continue

        group_word = match.group(1).strip()
        actor_name = match.group(2).strip(" ,.;:-")
        descriptor = (match.group(3) or "").strip(" ,.;:-")

        if descriptor:
            return f"{group_word} of {actor_name}, {descriptor}"

        return f"{group_word} of {actor_name}"

    return ""


def select_death_actor_evidence_sentence(question, candidate_docs):
    if not is_death_actor_question(question):
        return "", ""

    target_terms = target_terms_from_question(question)

    if not target_terms:
        return "", ""

    best_sentence = ""
    best_group = ""
    best_score = -1

    for doc in candidate_docs or []:
        for sentence in split_candidate_sentences(get_doc_text(doc)):
            if not sentence_mentions_target(sentence, target_terms):
                continue

            if not sentence_mentions_death_action(sentence):
                continue

            group_phrase = extract_group_actor_phrase(sentence)
            score = 10

            if group_phrase:
                score += 10

            if "battle" in normalize_retry_key(sentence):
                score += 3

            if score > best_score:
                best_score = score
                best_sentence = sentence
                best_group = group_phrase

    return best_sentence, best_group


def build_death_actor_supported_answer(question, candidate_docs):
    sentence, group_phrase = select_death_actor_evidence_sentence(question, candidate_docs)

    if not sentence:
        return ""

    _, target = extract_question_action_target(question)
    target = target or (target_terms_from_question(question) or ["the person"])[0]
    alleged_actor = extract_yes_no_actor(question)

    if group_phrase:
        if alleged_actor:
            return (
                f"It is not stated that {alleged_actor} personally killed {target}. "
                f"{target} was killed after losing a battle against the {group_phrase}."
            )

        return (
            f"{target} was killed after losing a battle against the {group_phrase}. "
            "It does not state that one specific person personally killed him."
        )

    return sentence


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
]
