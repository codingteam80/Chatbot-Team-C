import re

try:
    from config.settings import (
        METADATA_BOOST_CATEGORY,
        METADATA_BOOST_DOC_TYPE,
        METADATA_BOOST_LANGUAGE,
        METADATA_BOOST_MAX,
        METADATA_BOOST_SECTION_TERM,
        METADATA_BOOST_SOURCE_HINT,
        METADATA_BOOST_TITLE_TERM,
    )
except ImportError:
    # Safe fallback kapag wala pa sa settings.py.
    METADATA_BOOST_CATEGORY = 0.0
    METADATA_BOOST_DOC_TYPE = 0.0
    METADATA_BOOST_LANGUAGE = 0.0
    METADATA_BOOST_SOURCE_HINT = 0.0020
    METADATA_BOOST_TITLE_TERM = 0.0015
    METADATA_BOOST_SECTION_TERM = 0.0015
    METADATA_BOOST_MAX = 0.0025


METADATA_BOOST_WEIGHTS = {
    "category": METADATA_BOOST_CATEGORY,
    "doc_type": METADATA_BOOST_DOC_TYPE,
    "language": METADATA_BOOST_LANGUAGE,
    "source_hint": METADATA_BOOST_SOURCE_HINT,
    "title_term": METADATA_BOOST_TITLE_TERM,
    "section_term": METADATA_BOOST_SECTION_TERM,
}


def normalize_text(text):
    # Simple lowercase text normalization para sa metadata matching.
    return " ".join(re.findall(r"[a-z0-9]+", str(text or "").lower()))


def get_metadata(doc):
    # Safe metadata getter.
    return dict(getattr(doc, "metadata", {}) or {})


def get_query_terms(query_info):
    # Gumamit ng important_terms/source_keywords mula query_analyzer kung meron.
    terms = []

    for key in ["important_terms", "source_keywords"]:
        values = query_info.get(key) or []

        if isinstance(values, str):
            values = [values]

        for value in values:
            normalized = normalize_text(value)
            if normalized and normalized not in terms:
                terms.append(normalized)

    return terms


def has_term_match(metadata_value, query_terms):
    # True kapag may query term na tumama sa title/section/source text.
    metadata_text = normalize_text(metadata_value)

    if not metadata_text:
        return False

    for term in query_terms:
        if not term:
            continue

        if term in metadata_text or metadata_text in term:
            return True

    return False


def compute_metadata_boost(doc, query_info):
    # Metadata boost should be a small tie-breaker only.
    # Huwag gawing mas malakas kaysa RRF score.
    metadata = get_metadata(doc)
    boost = 0.0
    reasons = []

    category_hint = normalize_text(query_info.get("category"))
    doc_type_hint = normalize_text(query_info.get("doc_type"))
    language_hint = normalize_text(query_info.get("language"))
    source_hint = normalize_text(query_info.get("source_hint"))
    query_terms = get_query_terms(query_info)

    category = normalize_text(metadata.get("category"))
    doc_type = normalize_text(metadata.get("doc_type"))
    language = normalize_text(metadata.get("language"))
    source = normalize_text(metadata.get("source") or metadata.get("file_name"))
    title = metadata.get("title") or ""
    section = metadata.get("section") or ""

    if category_hint and category_hint == category:
        boost += METADATA_BOOST_WEIGHTS["category"]
        reasons.append("category_match")

    if doc_type_hint and doc_type_hint == doc_type:
        boost += METADATA_BOOST_WEIGHTS["doc_type"]
        reasons.append("doc_type_match")

    if language_hint and language_hint == language:
        boost += METADATA_BOOST_WEIGHTS["language"]
        reasons.append("language_match")

    if source_hint and source_hint in source:
        boost += METADATA_BOOST_WEIGHTS["source_hint"]
        reasons.append("source_hint_match")

    if has_term_match(title, query_terms):
        boost += METADATA_BOOST_WEIGHTS["title_term"]
        reasons.append("title_term_match")

    if has_term_match(section, query_terms):
        boost += METADATA_BOOST_WEIGHTS["section_term"]
        reasons.append("section_term_match")

    boost = min(float(boost), float(METADATA_BOOST_MAX))

    return boost, reasons


def apply_metadata_boost(docs, query_info, base_score_key="hybrid_score", debug=False):
    # Apply small metadata boost after RRF.
    # Sort by boosted score, then keep original order as tie-breaker.
    boosted_items = []

    for original_rank, doc in enumerate(docs or [], start=1):
        metadata = get_metadata(doc)
        base_score = float(metadata.get(base_score_key, 0.0) or 0.0)
        boost_score, reasons = compute_metadata_boost(doc, query_info or {})
        boosted_score = base_score + boost_score

        doc.metadata = dict(metadata)
        doc.metadata["metadata_boost_score"] = float(boost_score)
        doc.metadata["metadata_boosted_score"] = float(boosted_score)
        doc.metadata["metadata_boost_reasons"] = ", ".join(reasons)

        boosted_items.append((doc, boosted_score, original_rank))

    boosted_items.sort(key=lambda item: (-item[1], item[2]))
    boosted_docs = [doc for doc, _, _ in boosted_items]

    if debug:
        print("[METADATA_BOOST] Applied metadata boost.", flush=True)
        print(f"[METADATA_BOOST] Max boost: {METADATA_BOOST_MAX}", flush=True)

        for index, doc in enumerate(boosted_docs[:5], start=1):
            metadata = get_metadata(doc)
            print(
                f"[METADATA_BOOST] Top {index}: "
                f"base={metadata.get(base_score_key)} | "
                f"boost={metadata.get('metadata_boost_score')} | "
                f"boosted={metadata.get('metadata_boosted_score')} | "
                f"reasons={metadata.get('metadata_boost_reasons')}",
                flush=True,
            )

    return boosted_docs
