from chains.rag.candidate_parts.extract_helpers import (
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
)



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
]
