import re
import unicodedata
from copy import copy
from pathlib import Path

from retrieval.context_config import load_context_terms


CONTEXT_TERMS = load_context_terms()
SOURCE_WEAK_TOKENS = CONTEXT_TERMS["source_weak_tokens"]
QUESTION_WEAK_TOKENS = CONTEXT_TERMS["question_weak_tokens"]
SOURCE_ANCHOR_WEAK_TOKENS = CONTEXT_TERMS["source_anchor_weak_tokens"]
REFERENCE_MARKERS = CONTEXT_TERMS["reference_markers"]
LOW_VALUE_SECTION_PHRASES = CONTEXT_TERMS["low_value_section_phrases"]
LOW_VALUE_BODY_PHRASES = CONTEXT_TERMS["low_value_body_phrases"]


def normalize_text(text):
    # Normalize accents/punctuation so Andres and Andrés can match.
    text = str(text or "").lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = re.sub(r"[^a-z0-9\u3040-\u30ff\u3400-\u9fff]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def get_metadata(doc):
    return dict(getattr(doc, "metadata", {}) or {})


def get_source_name(doc):
    metadata = get_metadata(doc)
    return metadata.get("file_name") or metadata.get("source") or metadata.get("title") or "unknown source"


def get_source_key(doc):
    source_name = Path(str(get_source_name(doc))).stem
    tokens = normalize_text(source_name).split()
    useful_tokens = [token for token in tokens if token not in SOURCE_WEAK_TOKENS]
    return " ".join(useful_tokens) or normalize_text(source_name) or "unknown source"


def get_document_key(doc):
    metadata = get_metadata(doc)
    source = metadata.get("source") or metadata.get("file_name") or ""
    page = metadata.get("page", "")
    chunk_id = metadata.get("chunk_id") or metadata.get("chunk_index") or ""

    if chunk_id != "":
        return (str(source), str(page), str(chunk_id))

    preview = str(getattr(doc, "page_content", "") or "")[:250]
    return (str(source), str(page), preview)


def add_unique_doc(result, seen, doc):
    key = get_document_key(doc)
    if key in seen:
        return False
    seen.add(key)
    result.append(doc)
    return True


def remove_duplicate_docs(docs):
    result = []
    seen = set()
    for doc in docs or []:
        add_unique_doc(result, seen, doc)
    return result


def get_chunk_number(doc):
    metadata = get_metadata(doc)
    for key in ("chunk_index", "chunk_id", "chunk"):
        numbers = re.findall(r"\d+", str(metadata.get(key, "")))
        if numbers:
            return int(numbers[-1])
    return None


def find_chunk_position(target_doc, all_chunks):
    target_key = get_document_key(target_doc)
    for index, chunk in enumerate(all_chunks or []):
        if get_document_key(chunk) == target_key:
            return index

    target_source = get_source_key(target_doc)
    target_chunk_number = get_chunk_number(target_doc)
    if target_chunk_number is None:
        return None

    for index, chunk in enumerate(all_chunks or []):
        if get_source_key(chunk) == target_source and get_chunk_number(chunk) == target_chunk_number:
            return index

    return None


def clone_neighbor_doc(doc, base_doc, offset):
    try:
        neighbor_doc = doc.copy(deep=True)
    except Exception:
        neighbor_doc = copy(doc)

    metadata = get_metadata(neighbor_doc)
    base_metadata = get_metadata(base_doc)
    metadata["neighbor_expanded"] = True
    metadata["neighbor_offset"] = offset
    metadata["neighbor_from_source"] = get_source_name(base_doc)
    metadata["neighbor_from_page"] = base_metadata.get("page")
    metadata["neighbor_from_chunk"] = base_metadata.get("chunk_id") or base_metadata.get("chunk_index") or ""
    neighbor_doc.metadata = metadata
    return neighbor_doc


def expand_neighbor_chunks(selected_docs, all_chunks, window=1):
    selected_docs = list(selected_docs or [])
    all_chunks = list(all_chunks or [])

    try:
        window = int(window)
    except (TypeError, ValueError):
        window = 0

    if not selected_docs or not all_chunks or window <= 0:
        return selected_docs

    expanded = []
    seen = set()

    for selected_doc in selected_docs:
        add_unique_doc(expanded, seen, selected_doc)
        selected_source = get_source_key(selected_doc)
        selected_position = find_chunk_position(selected_doc, all_chunks)

        if selected_position is None:
            continue

        for offset in range(1, window + 1):
            for neighbor_position in (selected_position + offset, selected_position - offset):
                if neighbor_position < 0 or neighbor_position >= len(all_chunks):
                    continue

                neighbor_doc = all_chunks[neighbor_position]
                if get_source_key(neighbor_doc) != selected_source:
                    continue

                add_unique_doc(
                    expanded,
                    seen,
                    clone_neighbor_doc(neighbor_doc, selected_doc, neighbor_position - selected_position),
                )

    return expanded


def get_numeric_metadata(doc, keys, default_value=0.0):
    metadata = get_metadata(doc)
    for key in keys:
        value = metadata.get(key)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return default_value


def get_context_score(doc):
    # Prefer reranker score, then hybrid/metadata/semantic fallback.
    return get_numeric_metadata(
        doc,
        ["rerank_score", "metadata_boosted_score", "hybrid_score", "semantic_similarity"],
        default_value=0.0,
    )


def get_original_rank(doc):
    metadata = get_metadata(doc)
    for key in ("rerank_rank", "hybrid_rank", "semantic_rank", "bm25_rank", "retrieval_rank_0"):
        value = metadata.get(key)
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return 999999


def sort_docs_by_relevance(docs):
    return sorted(
        list(docs or []),
        key=lambda doc: (-get_context_score(doc), get_original_rank(doc)),
    )


def get_question_terms(question, weak_tokens=None, min_length=2):
    weak_tokens = weak_tokens if weak_tokens is not None else QUESTION_WEAK_TOKENS
    terms = []
    for token in normalize_text(question).split():
        if token in weak_tokens:
            continue
        if len(token) < min_length:
            continue
        if token not in terms:
            terms.append(token)
    return terms


def count_matches(text, terms):
    tokens = set(normalize_text(text).split())
    return sum(1 for term in terms or [] if term in tokens)


def get_doc_text(doc, include_metadata=True):
    body = str(getattr(doc, "page_content", "") or "")
    if not include_metadata:
        return body

    metadata = get_metadata(doc)
    metadata_text = " ".join(
        str(metadata.get(key, "") or "")
        for key in ("file_name", "source", "title", "section", "category", "doc_type")
    )
    return f"{metadata_text}\n{body}"


def is_low_value_doc(doc):
    metadata = get_metadata(doc)
    section = normalize_text(metadata.get("section") or metadata.get("title") or "")
    body = normalize_text(getattr(doc, "page_content", "") or "")

    if any(marker in body for marker in REFERENCE_MARKERS):
        return True

    if any(phrase and phrase in section for phrase in LOW_VALUE_SECTION_PHRASES):
        return True

    if any(phrase and phrase in body for phrase in LOW_VALUE_BODY_PHRASES):
        return True

    return False


def filter_low_value_docs(docs, min_keep=1):
    docs = list(docs or [])
    if len(docs) <= min_keep:
        return docs

    kept = [doc for doc in docs if not is_low_value_doc(doc)]
    return kept if len(kept) >= min_keep else docs[:min_keep]


def limit_context_docs(docs, max_chars=6000, max_per_source=None):
    selected = []
    source_counts = {}
    total_chars = 0

    for doc in docs or []:
        source_key = get_source_key(doc)
        if max_per_source is not None and source_counts.get(source_key, 0) >= max_per_source:
            continue

        text = str(getattr(doc, "page_content", "") or "")
        next_total = total_chars + len(text)
        if selected and next_total > int(max_chars or 6000):
            break

        selected.append(doc)
        source_counts[source_key] = source_counts.get(source_key, 0) + 1
        total_chars = next_total

    return selected


def annotate_docs(docs, mode, reason=""):
    for index, doc in enumerate(docs or [], start=1):
        metadata = get_metadata(doc)
        metadata["context_mode"] = mode
        metadata["context_rank"] = index
        if reason:
            metadata["context_selection_reason"] = reason
        doc.metadata = metadata
    return docs
