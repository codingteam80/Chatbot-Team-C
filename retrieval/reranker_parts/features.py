from retrieval.reranker_parts.model import (
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
)



def tokenize(text):
    # Simple lowercase tokenizer.
    # Uses the same normalization style as query_expander.py when available.
    normalized_text = normalize_query_text(text)
    return re.findall(r"[a-z0-9]+", normalized_text)


def stem_simple(word):
    # Simple stemmer for matching:
    # killed/killing/kills -> kill
    # Philippines -> philippine
    word = str(word or "").lower().strip()

    irregular = {
        "1st": "first",
        "2nd": "second",
        "3rd": "third",
    }

    if word in irregular:
        return irregular[word]

    if len(word) > 5 and word.endswith("ing"):
        return word[:-3]

    if len(word) > 4 and word.endswith("ied"):
        return word[:-3] + "y"

    if len(word) > 4 and word.endswith("ed"):
        return word[:-2]

    if len(word) > 4 and word.endswith("es"):
        return word[:-2]

    if len(word) > 3 and word.endswith("s"):
        return word[:-1]

    return word


def get_query_terms(query):
    # Get important words from the query.
    # Stopwords come from the JSON config through query_expander.py.
    raw_terms = tokenize(query)
    stopwords = get_question_stopwords()

    terms = []

    for term in raw_terms:
        if term in stopwords:
            continue

        stemmed = stem_simple(term)

        if stemmed and stemmed not in terms:
            terms.append(stemmed)

    return terms


def get_min_evidence_matches(query_terms):
    # Minimum matched terms to avoid being too strict.
    if not query_terms:
        return 0

    if len(query_terms) <= 2:
        return len(query_terms)

    return 2


def get_max_allowed_span(query_terms):
    # Max word distance between important terms.
    # When important terms are close together, it is more likely to be a direct answer.
    if len(query_terms) <= 1:
        return None

    if len(query_terms) == 2:
        return 12

    return 18


def word_matches_term(word, term):
    # Generic matching.
    # Exact/stem match first.
    # Then use fallback substring matching for names like lapulapu/lapu.
    word = stem_simple(word)
    term = stem_simple(term)

    if not word or not term:
        return False

    if word == term:
        return True

    if term in word:
        return True

    if word in term and len(word) >= 4:
        return True

    return False


def get_term_positions(text, query_terms):
    # Find where each important term appears in the chunk.
    words = tokenize(text)
    stemmed_words = [stem_simple(word) for word in words]

    positions_by_term = {}

    for term in query_terms:
        positions = []

        for index, word in enumerate(stemmed_words):
            if word_matches_term(word, term):
                positions.append(index)

        positions_by_term[term] = positions

    return positions_by_term


def get_best_span(positions_by_term, matched_terms):
    # Measure how close the matched terms are.
    # Use a sliding window to avoid slowing down on broad/multi-part questions.
    # The old backtracking approach can explode when there are many repeated terms in a chunk.
    if len(matched_terms) <= 1:
        return 0

    all_positions = []

    for term in matched_terms:
        positions = positions_by_term.get(term, [])

        if not positions:
            return None

        for position in positions[:20]:
            all_positions.append((position, term))

    all_positions.sort(key=lambda item: item[0])

    best_span = None
    left = 0
    term_counts = {}
    covered_terms = 0
    required_terms = len(matched_terms)

    for right_pos, right_term in all_positions:
        if term_counts.get(right_term, 0) == 0:
            covered_terms += 1

        term_counts[right_term] = term_counts.get(right_term, 0) + 1

        while covered_terms == required_terms and left < len(all_positions):
            left_pos, left_term = all_positions[left]
            span = right_pos - left_pos

            if best_span is None or span < best_span:
                best_span = span

            term_counts[left_term] -= 1

            if term_counts[left_term] == 0:
                covered_terms -= 1

            left += 1

    return best_span


def get_evidence_info(query, doc, check_proximity=True):
    # Evidence check:
    # 1. Count how many important query terms are in the chunk.
    # 2. The proximity/span check is optional to avoid slowing down on broad questions.
    query_terms = get_query_terms(query)
    text = strip_retrieval_context_prefix(getattr(doc, "page_content", ""))

    positions_by_term = get_term_positions(text, query_terms)
    matched_terms = []

    for term in query_terms:
        positions = positions_by_term.get(term, [])

        if positions:
            matched_terms.append(term)

    max_allowed_span = get_max_allowed_span(query_terms)

    if check_proximity:
        best_span = get_best_span(positions_by_term, matched_terms)

        if best_span is None:
            proximity_ok = False
        elif max_allowed_span is None:
            proximity_ok = True
        else:
            proximity_ok = best_span <= max_allowed_span
    else:
        # When it is not a direct fact question, term coverage is enough.
        # Do not compute the best span to avoid excessive slowdown.
        best_span = None
        proximity_ok = True

    return {
        "query_terms": query_terms,
        "matched_terms": matched_terms,
        "match_count": len(matched_terms),
        "best_span": best_span,
        "max_allowed_span": max_allowed_span,
        "proximity_ok": proximity_ok,
        "proximity_checked": bool(check_proximity),
    }


def add_evidence_metadata(query, doc, check_proximity=True):
    # Save evidence details in metadata so they can be seen in reports/debugging.
    evidence = get_evidence_info(query, doc, check_proximity=check_proximity)

    doc.metadata = dict(doc.metadata or {})
    doc.metadata["evidence_query_terms"] = ", ".join(evidence["query_terms"])
    doc.metadata["evidence_matched_terms"] = ", ".join(evidence["matched_terms"])
    doc.metadata["evidence_match_count"] = int(evidence["match_count"])
    doc.metadata["evidence_best_span"] = evidence["best_span"]
    doc.metadata["evidence_max_allowed_span"] = evidence["max_allowed_span"]
    doc.metadata["evidence_proximity_ok"] = bool(evidence["proximity_ok"])
    doc.metadata["evidence_proximity_checked"] = bool(evidence["proximity_checked"])

    return evidence


def passes_evidence_check(query, doc, min_matches=None, require_proximity=True):
    # True when there are enough important query terms and they are close enough.
    evidence = add_evidence_metadata(query, doc, check_proximity=require_proximity)

    query_terms = evidence["query_terms"]

    if min_matches is None:
        min_matches = get_min_evidence_matches(query_terms)

    if min_matches <= 0:
        return True

    enough_matches = evidence["match_count"] >= min_matches

    if not require_proximity:
        return enough_matches

    return enough_matches and evidence["proximity_ok"]


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
]
