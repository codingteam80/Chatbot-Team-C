from retrieval.reranker_parts.features import (
    hashlib,
    re,
    cmp_to_key,
    FlagReranker,
    RERANK_BATCH_SIZE,
    RERANK_MAX_CHARS,
    RERANK_MAX_LENGTH,
    RERANK_TOP_N,
    RERANK_USE_FP16,
    RERANKER_MODEL_NAME,
    RERANK_TIE_MARGIN,
    QUERY_EXPANSION_CONFIG_PATH,
    get_config_stopwords,
    normalize_query_text,
    get_question_stopwords,
    load_reranker,
    normalize_scores,
    trim_document_text,
    strip_retrieval_context_prefix,
    get_document_label,
    get_document_dedup_key,
    deduplicate_documents,
    tokenize,
    stem_simple,
    get_query_terms,
    get_min_evidence_matches,
    get_max_allowed_span,
    word_matches_term,
    get_term_positions,
    get_best_span,
    get_evidence_info,
    add_evidence_metadata,
    passes_evidence_check,
)



def compute_rerank_scores(
    reranker,
    pairs,
    batch_size=RERANK_BATCH_SIZE,
    max_length=RERANK_MAX_LENGTH,
):
    # Compute the score of each query-document pair.
    # Use a fallback because supported parameters may differ by FlagEmbedding version.
    if reranker is None:
        raise ValueError("No reranker received. Load it first using load_reranker().")

    try:
        return reranker.compute_score(
            pairs,
            batch_size=batch_size,
            max_length=max_length,
        )
    except TypeError:
        pass

    try:
        return reranker.compute_score(
            pairs,
            batch_size=batch_size,
        )
    except TypeError:
        pass

    return reranker.compute_score(pairs)


def apply_confident_top_score_filter(
    items,
    high_confidence_score=3.0,
    max_score_drop=3.0,
    debug=False,
):
    # When the best score is very high, there is a clear winner.
    # Do not force distant low-score docs into the results.
    # Warning: this is global if used here.
    # Preferred flow: keep this OFF here, then filter inside context_filter.py.
    #
    # Example:
    # best score = 6.32
    # next score = -0.49
    # gap = 6.81
    # Result: keep only the clear winner.
    if not items:
        return items

    best_score = items[0][1]

    if best_score < high_confidence_score:
        return items

    cutoff_score = best_score - max_score_drop
    filtered_items = []

    for item in items:
        score = item[1]

        if score >= cutoff_score:
            filtered_items.append(item)

    if debug:
        removed_count = len(items) - len(filtered_items)
        print(
            f"[RERANKER] Confident top-score filter: "
            f"best={best_score:.4f}, cutoff={cutoff_score:.4f}, removed={removed_count}",
            flush=True,
        )

    return filtered_items



def compare_rerank_items(left_item, right_item, tie_margin=RERANK_TIE_MARGIN):
    # Custom sorting rule for reranker results.
    #
    # Normal rule:
    # - Higher reranker score wins.
    #
    # Tie rule:
    # - When scores are very close, use the original rank.
    # - Original rank usually comes from the RRF order.
    # - Smaller original_rank = earlier in retrieval/RRF.
    #
    # Example:
    # Bonifacio score = 4.7390, original_rank = 3
    # Emilio score    = 4.6641, original_rank = 1
    # gap = 0.0749
    #
    # If tie_margin = 0.15, tie sila.
    # Since original_rank 1 is Emilio, Emilio should come first.
    left_doc, left_score, left_original_rank = left_item
    right_doc, right_score, right_original_rank = right_item

    score_gap = abs(left_score - right_score)

    if score_gap <= tie_margin:
        if left_original_rank < right_original_rank:
            return -1

        if left_original_rank > right_original_rank:
            return 1

        return 0

    if left_score > right_score:
        return -1

    if left_score < right_score:
        return 1

    return 0


def sort_rerank_items_with_tie_breaker(
    documents,
    scores,
    tie_margin=RERANK_TIE_MARGIN,
    use_tie_breaker=True,
):
    # Convert docs + scores into sortable items.
    #
    # item format:
    # (doc, score, original_rank)
    #
    # original_rank means the position before reranker sorting.
    # Since input documents usually come from RRF, this is the RRF order.
    items = []

    for original_rank, (doc, score) in enumerate(zip(documents, scores), start=1):
        items.append((doc, float(score), original_rank))

    if not use_tie_breaker:
        return sorted(
            items,
            key=lambda item: item[1],
            reverse=True,
        )

    return sorted(
        items,
        key=cmp_to_key(
            lambda left_item, right_item: compare_rerank_items(
                left_item=left_item,
                right_item=right_item,
                tie_margin=tie_margin,
            )
        ),
    )


# Public names exported by this compatibility/refactor module.
__all__ = [
    'hashlib',
    're',
    'cmp_to_key',
    'FlagReranker',
    'RERANK_BATCH_SIZE',
    'RERANK_MAX_CHARS',
    'RERANK_MAX_LENGTH',
    'RERANK_TOP_N',
    'RERANK_USE_FP16',
    'RERANKER_MODEL_NAME',
    'RERANK_TIE_MARGIN',
    'QUERY_EXPANSION_CONFIG_PATH',
    'get_config_stopwords',
    'normalize_query_text',
    'get_question_stopwords',
    'load_reranker',
    'normalize_scores',
    'trim_document_text',
    'strip_retrieval_context_prefix',
    'get_document_label',
    'get_document_dedup_key',
    'deduplicate_documents',
    'tokenize',
    'stem_simple',
    'get_query_terms',
    'get_min_evidence_matches',
    'get_max_allowed_span',
    'word_matches_term',
    'get_term_positions',
    'get_best_span',
    'get_evidence_info',
    'add_evidence_metadata',
    'passes_evidence_check',
    'compute_rerank_scores',
    'apply_confident_top_score_filter',
    'compare_rerank_items',
    'sort_rerank_items_with_tie_breaker',
]
