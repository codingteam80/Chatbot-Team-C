from retrieval.reranker_parts.scoring import (
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
    compute_rerank_scores,
    apply_confident_top_score_filter,
    compare_rerank_items,
    sort_rerank_items_with_tie_breaker,
)



def rerank_documents(
    query,
    documents,
    reranker,
    top_n=RERANK_TOP_N,
    min_score=None,
    show_scores=False,
    return_scores=False,
    max_chars=RERANK_MAX_CHARS,
    batch_size=RERANK_BATCH_SIZE,
    max_length=RERANK_MAX_LENGTH,
    debug=False,
    use_evidence_check=True,
    evidence_min_matches=None,
    require_proximity=True,
    filter_by_evidence=False,
    use_confident_top_score_filter=False,
    use_original_rank_tie_breaker=True,
    rerank_tie_margin=RERANK_TIE_MARGIN,
    high_confidence_score=3.0,
    max_score_drop=3.0,
    fallback_if_empty=True,
    deduplicate=True,
):
    # Recommended flow:
    # 1. Compute reranker score.
    # 2. Sort by reranker score highest to lowest.
    # 3. If scores are almost tied, keep original RRF order.
    # 4. Use evidence/proximity as metadata by default, not as a hard gate.
    # 5. Do not use global confident filtering by default.
    #    Final confidence cleanup is handled by context_filter.py
    #    after mode and anchor source are known.
    #
    # Important:
    # Evidence/proximity should NOT outrank reranker score.
    # By default, they only mark weak chunks in metadata.
    # Set filter_by_evidence=True only when you intentionally want hard filtering.
    query = str(query or "").strip()

    if not query:
        if debug:
            print("[RERANKER] Empty query. Returning no results.", flush=True)
        return []

    if not documents:
        if debug:
            print("[RERANKER] No documents received. Returning no results.", flush=True)
        return []

    if reranker is None:
        raise ValueError("No reranker received. Load it first using load_reranker().")

    if deduplicate:
        documents = deduplicate_documents(documents, debug=debug)

    if not documents:
        if debug:
            print("[RERANKER] No documents left after dedup. Returning no results.", flush=True)
        return []

    pairs = []

    for doc in documents:
        text = trim_document_text(getattr(doc, "page_content", ""), max_chars=max_chars)
        pairs.append([query, text])

    if debug:
        print(f"[RERANKER] Query: {query}", flush=True)
        print(f"[RERANKER] Candidate docs: {len(documents)}", flush=True)
        print(f"[RERANKER] top_n={top_n}, max_chars={max_chars}, batch_size={batch_size}", flush=True)
        print(f"[RERANKER] use_evidence_check={use_evidence_check}", flush=True)
        print(f"[RERANKER] require_proximity={require_proximity}", flush=True)
        print(f"[RERANKER] filter_by_evidence={filter_by_evidence}", flush=True)
        print(f"[RERANKER] use_confident_top_score_filter={use_confident_top_score_filter}", flush=True)
        print(f"[RERANKER] use_original_rank_tie_breaker={use_original_rank_tie_breaker}", flush=True)
        print(f"[RERANKER] rerank_tie_margin={rerank_tie_margin}", flush=True)
        print(f"[RERANKER] deduplicate={deduplicate}", flush=True)

    scores = compute_rerank_scores(
        reranker=reranker,
        pairs=pairs,
        batch_size=batch_size,
        max_length=max_length,
    )

    scores = normalize_scores(scores)

    ranked_items = sort_rerank_items_with_tie_breaker(
        documents=documents,
        scores=scores,
        tie_margin=rerank_tie_margin,
        use_tie_breaker=use_original_rank_tie_breaker,
    )

    query_terms = get_query_terms(query)
    required_matches = evidence_min_matches

    if required_matches is None:
        required_matches = get_min_evidence_matches(query_terms)

    if debug and use_evidence_check:
        print(f"[RERANKER] Query terms: {query_terms}", flush=True)
        print(f"[RERANKER] Required evidence matches: {required_matches}", flush=True)

    passed_items = []
    failed_count = 0

    for rerank_rank, (doc, score, original_rank) in enumerate(ranked_items, start=1):
        if min_score is not None and score < min_score:
            failed_count += 1
            continue

        doc.metadata = dict(doc.metadata or {})
        doc.metadata["rerank_score"] = float(score)
        doc.metadata["rerank_rank"] = rerank_rank
        doc.metadata["rerank_original_rank"] = original_rank

        if use_evidence_check:
            evidence = add_evidence_metadata(
                query,
                doc,
                check_proximity=require_proximity,
            )

            match_count = evidence["match_count"]
            best_span = evidence["best_span"]
            proximity_ok = evidence["proximity_ok"]

            passed_evidence = match_count >= required_matches

            if require_proximity:
                passed_evidence = passed_evidence and proximity_ok
        else:
            match_count = 0
            best_span = None
            proximity_ok = True
            passed_evidence = True

        doc.metadata["rerank_evidence_passed"] = bool(passed_evidence)
        doc.metadata["rerank_filter_by_evidence"] = bool(filter_by_evidence)

        if passed_evidence or not filter_by_evidence:
            # Keep original score order.
            # Do not sort by proximity.
            # Evidence/proximity is metadata by default, not a hard drop.
            passed_items.append((doc, float(score)))
        else:
            failed_count += 1

        if debug or show_scores:
            evidence_text = ""

            if use_evidence_check:
                evidence_text = (
                    f" | evidence={match_count}/{required_matches}"
                    f" | span={best_span}"
                    f" | proximity_checked={doc.metadata.get('evidence_proximity_checked')}"
                    f" | proximity_ok={proximity_ok}"
                    f" | matched={doc.metadata.get('evidence_matched_terms')}"
                )

            print(
                f"[RERANKER] Rank {rerank_rank}"
                f" | score={float(score):.4f}"
                f" | original_rank={original_rank}"
                f"{evidence_text}"
                f" | {get_document_label(doc)}",
                flush=True,
            )

    if use_confident_top_score_filter:
        passed_items = apply_confident_top_score_filter(
            passed_items,
            high_confidence_score=high_confidence_score,
            max_score_drop=max_score_drop,
            debug=debug,
        )

    results = passed_items[:top_n]

    if not results and fallback_if_empty:
        if debug:
            print("[RERANKER] No docs passed evidence check. Fallback to top reranked docs.", flush=True)

        for doc, score, original_rank in ranked_items[:top_n]:
            doc.metadata = dict(doc.metadata or {})
            doc.metadata["rerank_score"] = float(score)
            doc.metadata["rerank_original_rank"] = original_rank
            results.append((doc, float(score)))

    if debug:
        print(f"[RERANKER] Kept docs before top_n: {len(passed_items)}", flush=True)
        print(f"[RERANKER] Dropped by min_score/evidence filter: {failed_count}", flush=True)
        print(f"[RERANKER] Final docs: {len(results)}", flush=True)

    if return_scores:
        return results

    return [doc for doc, _ in results]


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
    'rerank_documents',
]
