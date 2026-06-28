from pathlib import Path


UNKNOWN_SOURCE_KEY = "unknown_source"


def normalize_source_key(value):
    # Gawing stable ang source name kahit Windows slash, Linux slash, o ibang extension.
    text = str(value or "").strip().replace("\\", "/")

    if not text:
        return UNKNOWN_SOURCE_KEY

    name = Path(text).name
    stem = Path(name).stem or name
    stem = stem.replace("–", "-").replace("—", "-")
    return stem.lower().strip() or UNKNOWN_SOURCE_KEY


def get_source_key(doc):
    # Source/file ang basis ng diversity, hindi history-specific keywords.
    metadata = getattr(doc, "metadata", None) or {}
    return normalize_source_key(metadata.get("source"))


def get_document_key(doc):
    # Stable key para hindi maulit ang parehong chunk mula sa multiple retrieval queries.
    metadata = getattr(doc, "metadata", None) or {}
    source_key = get_source_key(doc)
    page = metadata.get("page", "")
    chunk_id = metadata.get("chunk_id") or metadata.get("chunk_index")

    if chunk_id is not None:
        return (source_key, str(page), str(chunk_id))

    # Fallback kapag walang chunk_id/chunk_index metadata.
    preview = str(getattr(doc, "page_content", "") or "")[:250]
    return (source_key, str(page), preview)


def add_unique_document(selected_docs, seen_keys, doc):
    # Idagdag lang ang document kung hindi pa naidagdag ang same chunk.
    doc_key = get_document_key(doc)

    if doc_key in seen_keys:
        return False

    seen_keys.add(doc_key)
    selected_docs.append(doc)
    return True


def interleave_document_groups(document_groups, max_docs):
    # Round-robin merge para may chance ang bawat retrieval query na magbigay ng candidate.
    # Example: query1 result1, query2 result1, query3 result1, then query1 result2, etc.
    groups = [list(group or []) for group in document_groups or [] if group]

    if not groups:
        return []

    max_docs = max(1, int(max_docs or 1))
    max_group_length = max(len(group) for group in groups)
    selected_docs = []
    seen_keys = set()

    for rank_index in range(max_group_length):
        for group in groups:
            if rank_index >= len(group):
                continue

            add_unique_document(selected_docs, seen_keys, group[rank_index])

            if len(selected_docs) >= max_docs:
                return selected_docs

    return selected_docs


def limit_documents_per_source(docs, max_docs, max_per_source):
    # Limitahan ang sobrang daming chunks mula sa iisang file/source.
    # Generic ito para sa SOP/manual/policy docs, hindi lang history docs.
    max_docs = max(1, int(max_docs or 1))
    max_per_source = max(1, int(max_per_source or 1))

    selected_docs = []
    overflow_docs = []
    seen_keys = set()
    source_counts = {}

    for doc in docs or []:
        doc_key = get_document_key(doc)

        if doc_key in seen_keys:
            continue

        seen_keys.add(doc_key)
        source_key = get_source_key(doc)
        current_count = source_counts.get(source_key, 0)

        if current_count < max_per_source:
            selected_docs.append(doc)
            source_counts[source_key] = current_count + 1
        else:
            overflow_docs.append(doc)

        if len(selected_docs) >= max_docs:
            return selected_docs

    # Fallback: kung kulang ang selected docs, punuin gamit overflow para hindi mabitin ang context.
    for doc in overflow_docs:
        selected_docs.append(doc)

        if len(selected_docs) >= max_docs:
            break

    return selected_docs[:max_docs]


def balance_candidates(document_groups, max_docs=15, max_per_source=2, enabled=True):
    # Main pre-rerank balancer.
    # Step 1: interleave candidates from each retrieval query.
    # Step 2: avoid too many chunks from the same source before rerank.
    if not enabled:
        return interleave_document_groups(
            document_groups=document_groups,
            max_docs=max_docs,
        )

    interleaved_docs = interleave_document_groups(
        document_groups=document_groups,
        max_docs=max_docs * 2,
    )

    return limit_documents_per_source(
        docs=interleaved_docs,
        max_docs=max_docs,
        max_per_source=max_per_source,
    )


def balance_reranked_documents(docs, max_docs=4, max_per_source=2, enabled=True):
    # Optional post-rerank balance bago context filter.
    # Ginagamit ito para hindi laging top chunks ng iisang source ang final context.
    if not enabled:
        return list(docs or [])[:max_docs]

    return limit_documents_per_source(
        docs=docs,
        max_docs=max_docs,
        max_per_source=max_per_source,
    )

def get_group_document_keys(group):
    # Keys ng isang retrieval-query group para ma-map sa reranked docs.
    return {get_document_key(doc) for doc in group or []}


def count_group_sources(group):
    # Bilangin kung aling source ang dominant sa isang retrieval-query group.
    source_counts = {}

    for doc in group or []:
        source_key = get_source_key(doc)
        source_counts[source_key] = source_counts.get(source_key, 0) + 1

    return source_counts


def get_rank_position_map(ranked_docs):
    # Mas mababang number = mas mataas ang rerank position.
    rank_map = {}

    for index, doc in enumerate(ranked_docs or []):
        rank_map[get_document_key(doc)] = index

    return rank_map


def find_best_group_doc(group, ranked_docs, selected_source_keys=None):
    # Piliin ang representative doc ng retrieval-query group.
    # Priority:
    # 1. Source na hindi pa represented sa final context.
    # 2. Source na mas dominant sa group.
    # 3. Mas mataas na rerank position.
    selected_source_keys = set(selected_source_keys or [])
    group_docs = list(group or [])

    if not group_docs:
        return None

    group_keys = get_group_document_keys(group_docs)
    source_counts = count_group_sources(group_docs)
    rank_map = get_rank_position_map(ranked_docs)

    candidates = []

    for doc in group_docs:
        doc_key = get_document_key(doc)
        source_key = get_source_key(doc)
        rank_position = rank_map.get(doc_key, len(rank_map) + len(candidates))
        in_reranked_pool = doc_key in group_keys and doc_key in rank_map
        new_source_score = 1 if source_key not in selected_source_keys else 0
        source_count = source_counts.get(source_key, 0)

        candidates.append((
            new_source_score,
            source_count,
            1 if in_reranked_pool else 0,
            -rank_position,
            doc,
        ))

    candidates.sort(key=lambda item: item[:4], reverse=True)
    return candidates[0][-1]


def balance_reranked_documents_with_query_coverage(
    docs,
    query_doc_groups=None,
    max_docs=4,
    max_per_source=2,
    enabled=True,
):
    # Final-context balancer para sa multi-query/cross-document questions.
    # Goal: kapag may several retrieval queries, may representative doc bawat query group.
    # Hindi ito naka-base sa topic names; naka-base ito sa retrieval group coverage.
    if not enabled:
        return list(docs or [])[:max_docs]

    groups = [list(group or []) for group in query_doc_groups or [] if group]

    if len(groups) <= 1:
        return balance_reranked_documents(
            docs=docs,
            max_docs=max_docs,
            max_per_source=max_per_source,
            enabled=enabled,
        )

    max_docs = max(1, int(max_docs or 1))
    ranked_docs = list(docs or [])
    selected_docs = []
    seen_keys = set()

    selected_source_keys = set()

    for group in groups:
        representative_doc = find_best_group_doc(
            group=group,
            ranked_docs=ranked_docs,
            selected_source_keys=selected_source_keys,
        )

        if representative_doc is None:
            continue

        added = add_unique_document(
            selected_docs=selected_docs,
            seen_keys=seen_keys,
            doc=representative_doc,
        )

        if added:
            selected_source_keys.add(get_source_key(representative_doc))

        if len(selected_docs) >= max_docs:
            return selected_docs[:max_docs]

    for doc in limit_documents_per_source(
        docs=ranked_docs,
        max_docs=max_docs * 2,
        max_per_source=max_per_source,
    ):
        add_unique_document(
            selected_docs=selected_docs,
            seen_keys=seen_keys,
            doc=doc,
        )

        if len(selected_docs) >= max_docs:
            break

    return selected_docs[:max_docs]


