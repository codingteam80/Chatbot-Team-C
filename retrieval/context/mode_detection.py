from retrieval.context.neighbor_expansion import (
    json,
    re,
    unicodedata,
    defaultdict,
    copy,
    Path,
    MAX_CONTEXT_CHARS,
    MAX_PER_SOURCE,
    SINGLE_FACT_TOP_N,
    CROSS_DOC_TOP_N,
    COMPARISON_TOP_N,
    NEGATIVE_TOP_N,
    FALSE_PREMISE_TOP_N,
    ENABLE_NEIGHBOR_EXPANSION,
    NEIGHBOR_WINDOW,
    analyze_query,
    DEFAULT_CONTEXT_CONFIG_PATH,
    normalize_text,
    normalize_config_list,
    read_context_json,
    load_context_filter_terms,
    CONTEXT_FILTER_TERMS,
    SOURCE_WEAK_TOKENS,
    QUESTION_WEAK_TOKENS,
    SOURCE_ANCHOR_WEAK_TOKENS,
    CROSS_DOC_TERMS,
    load_low_value_filter_config,
    LOW_VALUE_FILTER_CONFIG,
    LOW_VALUE_REFERENCE_MARKERS,
    LOW_VALUE_SECTION_PHRASES,
    LOW_VALUE_BODY_PHRASES,
    get_metadata,
    get_source_name,
    get_source_key,
    get_document_key,
    remove_duplicate_docs,
    get_chunk_number,
    find_chunk_position,
    clone_neighbor_doc,
    get_page_number,
    clone_neighbor_doc_once,
    expand_neighbor_chunks,
    get_context_score,
    get_original_rank,
    get_question_terms,
    get_question_anchor_terms,
    sort_docs_by_rerank_score,
)



# ============================================================
# 2. QUESTION MODE DETECTION
# ============================================================

# CROSS_DOC_TERMS is loaded from config/query_expansion_config.json.
# See context_filter.cross_doc_terms in the JSON config.
def detect_question_mode(question):
    # Detect final context mode.
    # list_answer is separate from single_fact because it needs multiple evidence chunks.
    from retrieval.context.selection_parts.cross_doc_helpers import (
        has_any_phrase,
        has_any_wildcard_or_regex,
        has_explicit_cross_doc_cue,
        load_mode_detection_terms,
    )
    from retrieval.context.scoring_parts.config_rules import has_multi_answer_cue

    detected_mode = "single_fact"
    mode_terms = load_mode_detection_terms()

    if analyze_query is not None:
        query_info = analyze_query(question)
        mode = normalize_text(query_info.get("mode", ""))

        if mode in mode_terms["cross_doc_modes"]:
            detected_mode = "cross_doc"
        elif mode in mode_terms["comparison_modes"]:
            detected_mode = "comparison"
        elif mode in mode_terms["negative_modes"]:
            detected_mode = "negative"
        elif mode in mode_terms["false_premise_modes"]:
            detected_mode = "false_premise"
        elif mode in mode_terms["list_modes"]:
            detected_mode = "list_answer"
        else:
            detected_mode = "single_fact"
    else:
        normalized_question = normalize_text(question)
        mode_terms = load_mode_detection_terms()

        if (
            has_any_phrase(normalized_question, mode_terms["false_premise_phrases"])
            or has_any_wildcard_or_regex(normalized_question, mode_terms["false_premise_patterns"])
        ):
            detected_mode = "false_premise"
        elif has_any_phrase(normalized_question, mode_terms["comparison_phrases"]):
            detected_mode = "comparison"
        elif has_explicit_cross_doc_cue(question):
            detected_mode = "cross_doc"
        elif has_multi_answer_cue(question):
            detected_mode = "list_answer"
        else:
            detected_mode = "single_fact"

    # Generic question-shape override.
    # This catches list-style questions even when query_analyzer returns single_fact.
    if detected_mode == "single_fact" and has_multi_answer_cue(question):
        detected_mode = "list_answer"

    if detected_mode in {"cross_doc", "comparison"} and not has_explicit_cross_doc_cue(question):
        return "single_fact"

    return detected_mode

def limit_context_docs(docs, max_chars=MAX_CONTEXT_CHARS, max_per_source=None):
    # Limit the total characters in the final context.
    selected_docs = []
    total_chars = 0
    source_counts = defaultdict(int)

    for doc in docs or []:
        source_key = get_source_key(doc)

        if max_per_source is not None and source_counts[source_key] >= max_per_source:
            continue

        text = str(getattr(doc, "page_content", "") or "")
        text_length = len(text)

        if total_chars + text_length > max_chars:
            if not selected_docs:
                selected_docs.append(doc)

            break

        selected_docs.append(doc)
        total_chars += text_length
        source_counts[source_key] += 1

    return selected_docs


# ============================================================
# 4. SINGLE FACT HELPERS
# ============================================================

def count_source_question_matches(question_terms, source_key):
    # Count how many question terms matched the source name.
    source_tokens = set(normalize_text(source_key).split())
    match_count = 0

    for term in question_terms:
        if term in source_tokens:
            match_count += 1

    return match_count


def source_key_has_exact_question_phrase(source_key, normalized_question):
    # Exact phrase check for obvious source/entity questions.
    # Example: "what is code review sop" -> source_key "code review sop".
    source_key = normalize_text(source_key)
    normalized_question = normalize_text(normalized_question)

    if not source_key or not normalized_question:
        return False

    source_tokens = source_key.split()

    # Single-token sources like "katipunan" should match as a whole word.
    if len(source_tokens) == 1:
        pattern = rf"\b{re.escape(source_key)}\b"
        return re.search(pattern, normalized_question) is not None

    pattern = rf"\b{re.escape(source_key)}\b"
    return re.search(pattern, normalized_question) is not None


def is_safe_single_token_source_anchor(source_key, matched_terms):
    # Allow one-token title anchors like "Katipunan" or "MISRA".
    # Do not allow generic one-token anchors from descriptive words.
    source_tokens = normalize_text(source_key).split()

    if len(source_tokens) != 1:
        return False

    if len(matched_terms) != 1:
        return False

    token = source_tokens[0]

    if token in SOURCE_ANCHOR_WEAK_TOKENS:
        return False

    if token.isdigit():
        return False

    return token == matched_terms[0]


def is_strong_source_anchor(source_key, question_terms, normalized_question):
    # Generic rule:
    # 1. Exact source phrase in the question is strong.
    # 2. Two or more useful source-title terms matched is strong.
    # 3. One-token source titles can anchor if that exact token is in the question.
    #
    # This prevents descriptive questions like:
    # "secret group ... armed revolution ... 1896"
    # from wrongly anchoring to "Philippine Revolution" just because of "revolution".
    source_tokens = normalize_text(source_key).split()
    question_term_set = set(question_terms or [])

    matched_terms = []

    for token in source_tokens:
        if token in SOURCE_ANCHOR_WEAK_TOKENS:
            continue

        if token.isdigit():
            continue

        if token in question_term_set:
            matched_terms.append(token)

    exact_phrase_match = source_key_has_exact_question_phrase(
        source_key=source_key,
        normalized_question=normalized_question,
    )

    if exact_phrase_match:
        return True, len(matched_terms), "exact_source_phrase_match"

    if len(matched_terms) >= 2:
        return True, len(matched_terms), "multi_term_source_match"

    if is_safe_single_token_source_anchor(source_key, matched_terms):
        return True, len(matched_terms), "single_token_source_match"

    return False, len(matched_terms), "weak_or_descriptive_source_match"


def choose_single_fact_anchor_source(docs, question):
    # Select the anchor source only when the source/entity match is strong.
    # A single generic title word such as "revolution" is not enough.
    question_terms = get_question_anchor_terms(question)
    normalized_question = normalize_text(question)

    if not question_terms:
        return None, "no_question_source_match"

    best_source_key = None
    best_match_count = 0
    best_original_rank = 999999
    best_rerank_score = -999999.0
    best_reason = "no_question_source_match"

    for doc in docs:
        source_key = get_source_key(doc)
        is_strong, match_count, reason = is_strong_source_anchor(
            source_key=source_key,
            question_terms=question_terms,
            normalized_question=normalized_question,
        )
        original_rank = get_original_rank(doc)
        rerank_score = get_context_score(doc)

        if not is_strong:
            continue

        if match_count > best_match_count:
            best_source_key = source_key
            best_match_count = match_count
            best_original_rank = original_rank
            best_rerank_score = rerank_score
            best_reason = reason
            continue

        if match_count == best_match_count:
            if rerank_score > best_rerank_score:
                best_source_key = source_key
                best_original_rank = original_rank
                best_rerank_score = rerank_score
                best_reason = reason
                continue

            if rerank_score == best_rerank_score and original_rank < best_original_rank:
                best_source_key = source_key
                best_original_rank = original_rank
                best_rerank_score = rerank_score
                best_reason = reason

    if best_source_key and best_match_count > 0:
        return best_source_key, best_reason

    return None, "no_question_source_match"

def apply_confident_filter_inside_anchor_source(
    docs,
    high_confidence_score=3.0,
    max_score_drop=3.0,
):
    # Apply confidence filter only within selected anchor source.
    #
    # Important:
    # - This is not a global filter.
    # - It does not remove other sources before the mode/anchor is known.
    # - It is used only after choosing an anchor source.
    #
    # Example:
    # anchor source = "emilio aguinaldo"
    # best score = 6.32
    # cutoff = 3.32
    # keep only same-source docs with score >= 3.32
    if not docs:
        return docs

    scores = [get_context_score(doc) for doc in docs]
    best_score = max(scores)

    if best_score < high_confidence_score:
        return docs

    cutoff_score = best_score - max_score_drop
    filtered_docs = []

    for doc in docs:
        score = get_context_score(doc)

        if score >= cutoff_score:
            filtered_docs.append(doc)

    if filtered_docs:
        return filtered_docs

    return docs



def build_rank_map(docs):
    # Create a lookup for exact chunk ranks from retriever results.
    rank_map = {}

    for rank, doc in enumerate(docs or [], start=1):
        rank_map[get_document_key(doc)] = rank

    return rank_map


def get_rank_from_metadata(doc, keys):
    # Get the retrieval rank saved in metadata.
    metadata = get_metadata(doc)

    for key in keys:
        value = metadata.get(key)

        if value is None:
            continue

        try:
            return int(value)
        except (TypeError, ValueError):
            continue

    return None


def get_retrieval_rank(doc, rank_map, metadata_keys):
    # Priority 1: exact rank map from original semantic/BM25 results.
    # Priority 2: metadata rank saved by hybrid_retriever/RRF.
    doc_key = get_document_key(doc)

    if doc_key in rank_map:
        return rank_map[doc_key]

    return get_rank_from_metadata(doc, metadata_keys)


def get_retrieval_agreement_score(doc, semantic_rank_map, bm25_rank_map):
    # Generic direct-evidence signal.
    # The score is higher when both semantic and BM25 retrieval found the same chunk.
    semantic_rank = get_retrieval_rank(
        doc=doc,
        rank_map=semantic_rank_map,
        metadata_keys=("semantic_rank", "retrieval_rank_0"),
    )
    bm25_rank = get_retrieval_rank(
        doc=doc,
        rank_map=bm25_rank_map,
        metadata_keys=("bm25_rank", "retrieval_rank_1"),
    )

    score = 0.0

    if semantic_rank is not None:
        score += 2.0 / max(semantic_rank, 1)

    if bm25_rank is not None:
        score += 2.0 / max(bm25_rank, 1)

    if semantic_rank is not None and bm25_rank is not None:
        score += 4.0

    if semantic_rank == 1 and bm25_rank == 1:
        score += 4.0

    return score, semantic_rank, bm25_rank


def get_intro_chunk_score(doc):
    # Small bonus for intro/main chunks.
    # Useful for identity/direct fact questions, but not hardcoded to a source.
    metadata = get_metadata(doc)

    page = str(metadata.get("page", "")).strip().lower()
    chunk_id = str(metadata.get("chunk_id") or metadata.get("chunk_index") or "").strip().lower()

    score = 0.0

    if page in {"0", "1"}:
        score += 1.0

    if chunk_id in {"0", "1"} or "chunk_0" in chunk_id or "chunk_1" in chunk_id:
        score += 1.0

    return score


def get_direct_window_score(question_terms, doc, window_size=80, step_size=20):
    # More generic than plain keyword count.
    # When useful question terms are close together in the chunk, it is more likely to be direct context.
    text = normalize_text(getattr(doc, "page_content", "") or "")
    words = text.split()

    if not words or not question_terms:
        return 0.0

    best_window_matches = 0

    for start in range(0, len(words), step_size):
        window_text = " ".join(words[start:start + window_size])
        window_matches = 0

        for term in question_terms:
            if term in window_text:
                window_matches += 1

        if window_matches > best_window_matches:
            best_window_matches = window_matches

    return float(best_window_matches)


def has_retrieval_signal(docs, semantic_docs=None, bm25_docs=None):
    # Use primary evidence scoring only when semantic/BM25 ranks exist.
    if semantic_docs or bm25_docs:
        return True

    for doc in docs or []:
        metadata = get_metadata(doc)

        if any(key in metadata for key in ("semantic_rank", "bm25_rank", "retrieval_rank_0", "retrieval_rank_1")):
            return True

    return False


# Public names exported by this compatibility/refactor module.
__all__ = [
    'json',
    're',
    'unicodedata',
    'defaultdict',
    'copy',
    'Path',
    'MAX_CONTEXT_CHARS',
    'MAX_PER_SOURCE',
    'SINGLE_FACT_TOP_N',
    'CROSS_DOC_TOP_N',
    'COMPARISON_TOP_N',
    'NEGATIVE_TOP_N',
    'FALSE_PREMISE_TOP_N',
    'ENABLE_NEIGHBOR_EXPANSION',
    'NEIGHBOR_WINDOW',
    'analyze_query',
    'DEFAULT_CONTEXT_CONFIG_PATH',
    'normalize_text',
    'normalize_config_list',
    'read_context_json',
    'load_context_filter_terms',
    'CONTEXT_FILTER_TERMS',
    'SOURCE_WEAK_TOKENS',
    'QUESTION_WEAK_TOKENS',
    'SOURCE_ANCHOR_WEAK_TOKENS',
    'CROSS_DOC_TERMS',
    'load_low_value_filter_config',
    'LOW_VALUE_FILTER_CONFIG',
    'LOW_VALUE_REFERENCE_MARKERS',
    'LOW_VALUE_SECTION_PHRASES',
    'LOW_VALUE_BODY_PHRASES',
    'get_metadata',
    'get_source_name',
    'get_source_key',
    'get_document_key',
    'remove_duplicate_docs',
    'get_chunk_number',
    'find_chunk_position',
    'clone_neighbor_doc',
    'get_page_number',
    'clone_neighbor_doc_once',
    'expand_neighbor_chunks',
    'get_context_score',
    'get_original_rank',
    'get_question_terms',
    'get_question_anchor_terms',
    'sort_docs_by_rerank_score',
    'detect_question_mode',
    'limit_context_docs',
    'count_source_question_matches',
    'source_key_has_exact_question_phrase',
    'is_safe_single_token_source_anchor',
    'is_strong_source_anchor',
    'choose_single_fact_anchor_source',
    'apply_confident_filter_inside_anchor_source',
    'build_rank_map',
    'get_rank_from_metadata',
    'get_retrieval_rank',
    'get_retrieval_agreement_score',
    'get_intro_chunk_score',
    'get_direct_window_score',
    'has_retrieval_signal',
]
