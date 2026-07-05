try:
    from config.settings import (
        BM25_K,
        BM25_WEIGHT,
        ENABLE_METADATA_BOOST,
        HYBRID_FINAL_K,
        RRF_K,
        SEMANTIC_K,
        SEMANTIC_WEIGHT,
        USE_E5_PREFIX,
    )
except ImportError:
    SEMANTIC_K = 9
    BM25_K = 9
    HYBRID_FINAL_K = 11
    RRF_K = 60
    SEMANTIC_WEIGHT = 1.0
    BM25_WEIGHT = 1.0
    ENABLE_METADATA_BOOST = True
    USE_E5_PREFIX = True

from retrieval.bm25_retriever import bm25_search
from retrieval.metadata_booster import apply_metadata_boost
from retrieval.query_analyzer import analyze_query
from retrieval.semantic_retriever import semantic_search_with_scores


def get_document_key(doc):
    # Unique key to avoid repeating the same chunk.
    metadata = dict(getattr(doc, "metadata", {}) or {})
    source = metadata.get("source") or metadata.get("file_name") or ""
    page = metadata.get("page", "")
    chunk_id = metadata.get("chunk_id") or metadata.get("chunk_index") or ""

    if chunk_id != "":
        return (str(source), str(page), str(chunk_id))

    preview = str(getattr(doc, "page_content", "") or "")[:250]
    return (str(source), str(page), preview)


def get_document_label(doc):
    # Human-readable label for debugging.
    metadata = dict(getattr(doc, "metadata", {}) or {})
    source = metadata.get("file_name") or metadata.get("source") or "Unknown source"
    page = metadata.get("page", "N/A")
    chunk_id = metadata.get("chunk_id") or metadata.get("chunk_index") or "Unknown chunk"
    return f"{source} | page={page} | {chunk_id}"


def remove_duplicates(docs, debug=False):
    # Keep the first copy of each chunk.
    unique_docs = []
    seen_keys = set()

    for doc in docs or []:
        key = get_document_key(doc)

        if key in seen_keys:
            continue

        seen_keys.add(key)
        unique_docs.append(doc)

    if debug:
        original_count = len(docs or [])
        print(
            f"[HYBRID] Dedup: {original_count} -> {len(unique_docs)} docs, "
            f"removed={original_count - len(unique_docs)}",
            flush=True,
        )

    return unique_docs


def tag_retrieval_ranks(docs, rank_name, retrieval_rank_key):
    # Save original rank from a retriever.
    for rank, doc in enumerate(docs or [], start=1):
        doc.metadata = dict(getattr(doc, "metadata", {}) or {})
        doc.metadata[rank_name] = rank
        doc.metadata[retrieval_rank_key] = rank

    return docs


def reciprocal_rank_fusion(ranked_lists, weights=None, rrf_k=RRF_K, final_k=HYBRID_FINAL_K, debug=False):
    # RRF = combine ranks from semantic and BM25.
    if not ranked_lists:
        return []

    if weights is None:
        weights = [1.0] * len(ranked_lists)

    scores_by_key = {}
    docs_by_key = {}

    for list_index, docs in enumerate(ranked_lists):
        weight = weights[list_index] if list_index < len(weights) else 1.0

        for rank, doc in enumerate(docs or [], start=1):
            key = get_document_key(doc)

            if key not in docs_by_key:
                docs_by_key[key] = doc

            stored_doc = docs_by_key[key]
            stored_doc.metadata = dict(getattr(stored_doc, "metadata", {}) or {})

            score = float(weight) / (float(rrf_k) + rank)
            scores_by_key[key] = scores_by_key.get(key, 0.0) + score

            stored_doc.metadata[f"retrieval_rank_{list_index}"] = rank

            if list_index == 0:
                stored_doc.metadata["semantic_rank"] = rank
            elif list_index == 1:
                stored_doc.metadata["bm25_rank"] = rank

    ranked_keys = sorted(scores_by_key, key=scores_by_key.get, reverse=True)
    fused_docs = []

    for final_rank, key in enumerate(ranked_keys[:final_k], start=1):
        doc = docs_by_key[key]
        doc.metadata = dict(getattr(doc, "metadata", {}) or {})
        doc.metadata["hybrid_score"] = float(scores_by_key[key])
        doc.metadata["hybrid_rank"] = final_rank
        doc.metadata["rerank_original_rank"] = final_rank
        fused_docs.append(doc)

    if debug:
        print(f"[HYBRID] RRF candidates: {len(scores_by_key)}", flush=True)
        print(f"[HYBRID] Final fused docs: {len(fused_docs)}", flush=True)

        for index, doc in enumerate(fused_docs[:5], start=1):
            metadata = dict(getattr(doc, "metadata", {}) or {})
            print(
                f"[HYBRID] Top {index}: "
                f"score={metadata.get('hybrid_score')} | {get_document_label(doc)}",
                flush=True,
            )

    return fused_docs


def hybrid_search(
    query,
    vectorstore,
    bm25_retriever,
    semantic_k=SEMANTIC_K,
    bm25_k=BM25_K,
    final_k=HYBRID_FINAL_K,
    use_rrf=True,
    use_metadata_boost=ENABLE_METADATA_BOOST,
    use_e5_prefix=USE_E5_PREFIX,
    semantic_weight=SEMANTIC_WEIGHT,
    bm25_weight=BM25_WEIGHT,
    debug=False,
    return_details=False,
):
    # Full retrieval flow:
    # 1. Analyze query hints.
    # 2. Semantic search.
    # 3. BM25 search.
    # 4. RRF combine.
    # 5. Optional metadata boost.
    query = str(query or "").strip()

    if not query:
        if debug:
            print("[HYBRID] Empty query. Returning no results.", flush=True)
        return {"semantic_docs": [], "bm25_docs": [], "hybrid_docs": [], "query_info": {}} if return_details else []

    if vectorstore is None:
        raise ValueError("No vectorstore received. Load Chroma first.")

    if bm25_retriever is None:
        raise ValueError("No BM25 retriever received. Create BM25 retriever first.")

    query_info = analyze_query(query, debug=debug)

    semantic_docs = semantic_search_with_scores(
        vectorstore=vectorstore,
        query=query,
        k=semantic_k,
        use_e5_prefix=use_e5_prefix,
        debug=debug,
    )

    bm25_docs = bm25_search(
        bm25_retriever=bm25_retriever,
        query=query,
        k=bm25_k,
        debug=debug,
    )

    semantic_docs = tag_retrieval_ranks(semantic_docs, "semantic_rank", "retrieval_rank_0")
    bm25_docs = tag_retrieval_ranks(bm25_docs, "bm25_rank", "retrieval_rank_1")

    if use_rrf:
        hybrid_docs = reciprocal_rank_fusion(
            ranked_lists=[semantic_docs, bm25_docs],
            weights=[semantic_weight, bm25_weight],
            rrf_k=RRF_K,
            final_k=final_k,
            debug=debug,
        )
    else:
        hybrid_docs = remove_duplicates(semantic_docs + bm25_docs, debug=debug)[:final_k]

        for index, doc in enumerate(hybrid_docs, start=1):
            doc.metadata = dict(getattr(doc, "metadata", {}) or {})
            doc.metadata["hybrid_rank"] = index
            doc.metadata["rerank_original_rank"] = index

    if use_metadata_boost:
        hybrid_docs = apply_metadata_boost(
            docs=hybrid_docs,
            query_info=query_info,
            base_score_key="hybrid_score",
            debug=debug,
        )[:final_k]

        for index, doc in enumerate(hybrid_docs, start=1):
            doc.metadata = dict(getattr(doc, "metadata", {}) or {})
            doc.metadata["metadata_boost_rank"] = index
            doc.metadata["rerank_original_rank"] = index

    if return_details:
        return {
            "query_info": query_info,
            "semantic_docs": semantic_docs,
            "bm25_docs": bm25_docs,
            "hybrid_docs": hybrid_docs,
        }

    return hybrid_docs
