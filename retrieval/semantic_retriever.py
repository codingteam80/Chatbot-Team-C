try:
    from config.settings import MMR_FETCH_K, MMR_LAMBDA, SEMANTIC_K, USE_E5_PREFIX
except ImportError:
    SEMANTIC_K = 9
    USE_E5_PREFIX = True
    MMR_FETCH_K = 20
    MMR_LAMBDA = 0.5


# Semantic retrieval module.
# Purpose:
# - Get meaning-based chunks from Chroma/vectorstore.
# - Add "query:" only for semantic/vector search when USE_E5_PREFIX=True.
# - Do not affect BM25, reranker, or final LLM context.
# - Save clear rank and score metadata for hybrid/RRF/debug reports.


def format_semantic_query(query, use_e5_prefix=USE_E5_PREFIX):
    # E5 needs "query:" only when embedding the user query for vector search.
    query = str(query or "").strip()

    if not query:
        return ""

    if use_e5_prefix and not query.lower().startswith("query:"):
        return f"query: {query}"

    return query


def get_raw_query(query):
    # Raw query for debugging, BM25, reranker, and context-related stages.
    query = str(query or "").strip()

    if query.lower().startswith("query:"):
        return query.split(":", 1)[1].strip()

    return query


def get_doc_metadata(doc):
    # Safe metadata getter.
    return dict(getattr(doc, "metadata", {}) or {})


def get_doc_label(doc):
    # Human-readable debug label.
    metadata = get_doc_metadata(doc)
    source = metadata.get("file_name") or metadata.get("source") or "Unknown source"
    page = metadata.get("page", "N/A")
    chunk_id = metadata.get("chunk_id") or metadata.get("chunk_index") or "Unknown chunk"
    return f"{source} | page={page} | {chunk_id}"


def distance_to_similarity(distance):
    # Chroma returns distance where lower is better.
    # Convert it to a simple higher-is-better score for debug/optional filters.
    try:
        distance = float(distance)
    except (TypeError, ValueError):
        return 0.0

    if distance < 0:
        distance = 0.0

    return 1.0 / (1.0 + distance)


def attach_semantic_metadata(doc, rank, distance=None, search_type="similarity"):
    # Save all semantic metadata needed by hybrid/RRF/context filter.
    doc.metadata = dict(getattr(doc, "metadata", {}) or {})
    doc.metadata["semantic_rank"] = int(rank)
    doc.metadata["retrieval_rank_0"] = int(rank)
    doc.metadata["semantic_search_type"] = search_type

    if distance is not None:
        distance = float(distance)
        doc.metadata["semantic_distance"] = distance
        doc.metadata["semantic_similarity_score"] = distance_to_similarity(distance)

        # Backward compatibility:
        # Older modules may read semantic_score.
        # Keep it as distance because older reports already treated it as Chroma distance.
        doc.metadata["semantic_score"] = distance

    return doc


class SemanticRetriever:
    # Small wrapper so the .invoke() path also gets the E5 query prefix.
    # Use this when another chain needs a retriever object.

    def __init__(self, vectorstore, k=SEMANTIC_K, search_type="similarity", score_threshold=None, use_e5_prefix=USE_E5_PREFIX):
        if vectorstore is None:
            raise ValueError("No vectorstore received. Load Chroma first.")

        self.vectorstore = vectorstore
        self.k = k
        self.search_type = search_type
        self.score_threshold = score_threshold
        self.use_e5_prefix = use_e5_prefix

    def invoke(self, query, config=None, **kwargs):
        # Raw input here, but the semantic formatted query is passed to the vectorstore.
        if self.search_type == "mmr":
            return mmr_search(
                vectorstore=self.vectorstore,
                query=query,
                k=self.k,
                use_e5_prefix=self.use_e5_prefix,
                debug=False,
            )

        return semantic_search(
            vectorstore=self.vectorstore,
            query=query,
            k=self.k,
            use_e5_prefix=self.use_e5_prefix,
            debug=False,
        )

    def get_relevant_documents(self, query):
        # Compatibility in older LangChain code.
        return self.invoke(query)


def get_semantic_retriever(vectorstore, k=SEMANTIC_K, search_type="similarity", score_threshold=None, use_e5_prefix=USE_E5_PREFIX):
    # Retriever wrapper for chains that need .invoke().
    # Difference from vectorstore.as_retriever(): this formats E5 semantic queries safely.
    return SemanticRetriever(
        vectorstore=vectorstore,
        k=k,
        search_type=search_type,
        score_threshold=score_threshold,
        use_e5_prefix=use_e5_prefix,
    )


def semantic_search_with_scores(vectorstore, query, k=SEMANTIC_K, use_e5_prefix=USE_E5_PREFIX, debug=False):
    # Semantic search with Chroma distance score.
    # Chroma distance: lower means more similar.
    if vectorstore is None:
        raise ValueError("No vectorstore received. Load Chroma first.")

    raw_query = get_raw_query(query)
    formatted_query = format_semantic_query(raw_query, use_e5_prefix=use_e5_prefix)

    if not formatted_query:
        if debug:
            print("[SEMANTIC] Empty query. Returning no results.", flush=True)
        return []

    raw_results = vectorstore.similarity_search_with_score(formatted_query, k=k)
    docs = []

    for rank, item in enumerate(raw_results, start=1):
        doc, distance = item
        doc = attach_semantic_metadata(
            doc=doc,
            rank=rank,
            distance=distance,
            search_type="similarity",
        )
        docs.append(doc)

    if debug:
        print(f"[SEMANTIC] Raw query      : {raw_query}", flush=True)
        print(f"[SEMANTIC] Vector query   : {formatted_query}", flush=True)
        print(f"[SEMANTIC] Results        : {len(docs)}", flush=True)

        for index, doc in enumerate(docs[:5], start=1):
            metadata = get_doc_metadata(doc)
            print(
                f"[SEMANTIC] Top {index}: "
                f"distance={metadata.get('semantic_distance')} | "
                f"similarity={metadata.get('semantic_similarity_score')} | "
                f"{get_doc_label(doc)}",
                flush=True,
            )

    return docs


def semantic_search(vectorstore, query, k=SEMANTIC_K, use_e5_prefix=USE_E5_PREFIX, debug=False):
    # Semantic search returning documents only, but still with rank/score metadata.
    return semantic_search_with_scores(
        vectorstore=vectorstore,
        query=query,
        k=k,
        use_e5_prefix=use_e5_prefix,
        debug=debug,
    )


def mmr_search(
    vectorstore,
    query,
    k=SEMANTIC_K,
    fetch_k=MMR_FETCH_K,
    lambda_mult=MMR_LAMBDA,
    use_e5_prefix=USE_E5_PREFIX,
    debug=False,
):
    # MMR search = semantic search with diversity.
    # Useful for broad/cross-doc queries, but normal semantic_search is safer for direct facts.
    if vectorstore is None:
        raise ValueError("No vectorstore received. Load Chroma first.")

    raw_query = get_raw_query(query)
    formatted_query = format_semantic_query(raw_query, use_e5_prefix=use_e5_prefix)

    if not formatted_query:
        if debug:
            print("[SEMANTIC-MMR] Empty query. Returning no results.", flush=True)
        return []

    docs = vectorstore.max_marginal_relevance_search(
        formatted_query,
        k=k,
        fetch_k=fetch_k,
        lambda_mult=lambda_mult,
    )

    for rank, doc in enumerate(docs, start=1):
        attach_semantic_metadata(
            doc=doc,
            rank=rank,
            distance=None,
            search_type="mmr",
        )

    if debug:
        print(f"[SEMANTIC-MMR] Raw query    : {raw_query}", flush=True)
        print(f"[SEMANTIC-MMR] Vector query : {formatted_query}", flush=True)
        print(f"[SEMANTIC-MMR] Results      : {len(docs)}", flush=True)

        for index, doc in enumerate(docs[:5], start=1):
            print(f"[SEMANTIC-MMR] Top {index}: {get_doc_label(doc)}", flush=True)

    return docs
