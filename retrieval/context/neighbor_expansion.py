from retrieval.context.normalization import (
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
)



# ============================================================
# 1B. NEIGHBOR CHUNK EXPANSION
# ============================================================

def get_chunk_number(doc):
    # Get the numeric chunk index when available in metadata.
    metadata = get_metadata(doc)

    for key in ("chunk_index", "chunk_id", "chunk"):
        value = metadata.get(key)

        if value is None:
            continue

        numbers = re.findall(r"\d+", str(value))

        if numbers:
            return int(numbers[-1])

    return None


def find_chunk_position(target_doc, all_chunks):
    # Find the exact chunk position in the full chunk list.
    target_key = get_document_key(target_doc)

    for index, chunk in enumerate(all_chunks or []):
        if get_document_key(chunk) == target_key:
            return index

    # Fallback when the key is not exact but the source and chunk number match.
    target_source = get_source_key(target_doc)
    target_chunk_number = get_chunk_number(target_doc)

    if target_chunk_number is None:
        return None

    for index, chunk in enumerate(all_chunks or []):
        if get_source_key(chunk) != target_source:
            continue

        if get_chunk_number(chunk) == target_chunk_number:
            return index

    return None


def clone_neighbor_doc(doc, base_doc, offset):
    # Create a copy to avoid polluting cached all_chunks metadata.
    try:
        neighbor_doc = doc.copy(deep=True)
    except Exception:
        neighbor_doc = copy(doc)

    metadata = dict(getattr(neighbor_doc, "metadata", {}) or {})
    base_metadata = get_metadata(base_doc)

    metadata["neighbor_expanded"] = True
    metadata["neighbor_offset"] = offset
    metadata["neighbor_from_source"] = get_source_name(base_doc)
    metadata["neighbor_from_page"] = base_metadata.get("page")
    metadata["neighbor_from_chunk"] = (
        base_metadata.get("chunk_id")
        or base_metadata.get("chunk_index")
        or ""
    )

    neighbor_doc.metadata = metadata
    return neighbor_doc



def get_page_number(doc):
    # Get the numeric page when available in metadata.
    metadata = get_metadata(doc)

    for key in ("page", "page_number", "page_index"):
        value = metadata.get(key)

        if value is None:
            continue

        numbers = re.findall(r"\d+", str(value))

        if numbers:
            return int(numbers[-1])

    return None


def clone_neighbor_doc_once(doc, base_doc, offset, reason):
    neighbor_doc = clone_neighbor_doc(
        doc=doc,
        base_doc=base_doc,
        offset=offset,
    )
    neighbor_doc.metadata = dict(getattr(neighbor_doc, "metadata", {}) or {})
    neighbor_doc.metadata["neighbor_expand_reason"] = reason
    return neighbor_doc


def expand_neighbor_chunks(selected_docs, all_chunks, window=1):
    # Add neighboring chunks from the same source to the selected docs.
    # Important: current chunk first, then the NEXT chunk/page before the previous one.
    # Reason: many PDF sections continue on the next page.
    selected_docs = list(selected_docs or [])
    all_chunks = list(all_chunks or [])

    try:
        window = int(window)
    except (TypeError, ValueError):
        window = 0

    if not selected_docs or not all_chunks or window <= 0:
        return selected_docs

    expanded_docs = []
    seen_keys = set()

    def add_doc(doc):
        doc_key = get_document_key(doc)

        if doc_key in seen_keys:
            return False

        seen_keys.add(doc_key)
        expanded_docs.append(doc)
        return True

    def add_neighbor(neighbor_doc, base_doc, offset, reason):
        if get_document_key(neighbor_doc) == get_document_key(base_doc):
            return add_doc(base_doc)

        return add_doc(
            clone_neighbor_doc_once(
                doc=neighbor_doc,
                base_doc=base_doc,
                offset=offset,
                reason=reason,
            )
        )

    def iter_same_source_page(source_key, target_page):
        for chunk in all_chunks:
            if get_source_key(chunk) != source_key:
                continue

            chunk_page = get_page_number(chunk)

            if chunk_page == target_page:
                yield chunk

    for selected_doc in selected_docs:
        selected_source = get_source_key(selected_doc)
        selected_page = get_page_number(selected_doc)
        selected_position = find_chunk_position(selected_doc, all_chunks)

        # Always keep the selected answer-bearing chunk first.
        add_doc(selected_doc)

        # 1. Add next chunks first. This catches section continuations.
        if selected_position is not None:
            for step in range(1, window + 1):
                neighbor_position = selected_position + step

                if neighbor_position >= len(all_chunks):
                    break

                neighbor_doc = all_chunks[neighbor_position]

                if get_source_key(neighbor_doc) != selected_source:
                    continue

                add_neighbor(
                    neighbor_doc=neighbor_doc,
                    base_doc=selected_doc,
                    offset=step,
                    reason="next_chunk_neighbor",
                )

        # 2. Add next pages first. This fixes PDF page continuation cases.
        if selected_page is not None:
            for step in range(1, window + 1):
                target_page = selected_page + step

                for neighbor_doc in iter_same_source_page(selected_source, target_page):
                    add_neighbor(
                        neighbor_doc=neighbor_doc,
                        base_doc=selected_doc,
                        offset=step,
                        reason="next_page_neighbor",
                    )

        # 3. Add previous chunks/pages only after next neighbors.
        # If MAX_CONTEXT_CHARS cuts the context, next continuation is protected first.
        if selected_position is not None:
            for step in range(1, window + 1):
                neighbor_position = selected_position - step

                if neighbor_position < 0:
                    break

                neighbor_doc = all_chunks[neighbor_position]

                if get_source_key(neighbor_doc) != selected_source:
                    continue

                add_neighbor(
                    neighbor_doc=neighbor_doc,
                    base_doc=selected_doc,
                    offset=-step,
                    reason="previous_chunk_neighbor",
                )

        if selected_page is not None:
            for step in range(1, window + 1):
                target_page = selected_page - step

                for neighbor_doc in iter_same_source_page(selected_source, target_page):
                    add_neighbor(
                        neighbor_doc=neighbor_doc,
                        base_doc=selected_doc,
                        offset=-step,
                        reason="previous_page_neighbor",
                    )

    return expanded_docs

def get_context_score(doc):
    # Get the score for debug prints and local confidence filtering.
    metadata = get_metadata(doc)

    for key in ("rerank_score", "score", "hybrid_score", "semantic_score"):
        value = metadata.get(key)

        if value is None:
            continue

        try:
            return float(value)
        except (TypeError, ValueError):
            continue

    return 0.0


def get_original_rank(doc):
    # This usually comes from the RRF order before reranking.
    # Smaller value = earlier in retrieval/RRF.
    metadata = get_metadata(doc)
    value = metadata.get("rerank_original_rank")

    try:
        return int(value)
    except (TypeError, ValueError):
        return 999999


def get_question_terms(question):
    # Get useful terms from the question.
    raw_tokens = normalize_text(question).split()
    terms = []

    for token in raw_tokens:
        if token in QUESTION_WEAK_TOKENS:
            continue

        if len(token) <= 1:
            continue

        if token not in terms:
            terms.append(token)

    return terms


def get_question_anchor_terms(question):
    # Get only strong terms that are safe to use as source/entity anchors.
    # Example:
    # - "Who is the first Philippine president?" -> no anchor terms.
    # - "Who was Apolinario Mabini?" -> apolinario, mabini.
    # - "What is Katipunan?" -> katipunan.
    raw_tokens = normalize_text(question).split()
    terms = []

    for token in raw_tokens:
        if token in SOURCE_ANCHOR_WEAK_TOKENS:
            continue

        if len(token) <= 1:
            continue

        if token not in terms:
            terms.append(token)

    return terms


def sort_docs_by_rerank_score(docs):
    # Higher rerank_score means better relevance ranking.
    # Tie-breaker: smaller original rank is better.
    return sorted(
        docs or [],
        key=lambda doc: (
            get_context_score(doc),
            -get_original_rank(doc),
        ),
        reverse=True,
    )


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
]
