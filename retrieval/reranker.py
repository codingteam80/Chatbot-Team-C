import hashlib
import re
from functools import cmp_to_key

from FlagEmbedding import FlagReranker

from config.settings import (
    RERANK_BATCH_SIZE,
    RERANK_MAX_CHARS,
    RERANK_MAX_LENGTH,
    RERANK_TOP_N,
    RERANK_USE_FP16,
    RERANKER_MODEL_NAME,
)

try:
    from config.settings import RERANK_TIE_MARGIN
except ImportError:
    # Kapag wala pa sa settings.py, ito ang default.
    # 0.15 means: kapag score gap ay 0.15 or lower, considered tie.
    RERANK_TIE_MARGIN = 0.15


try:
    from retrieval.query_expander import (
        DEFAULT_CONFIG_PATH as QUERY_EXPANSION_CONFIG_PATH,
        get_config_stopwords,
        normalize_text as normalize_query_text,
    )
except ImportError:
    QUERY_EXPANSION_CONFIG_PATH = None

    def get_config_stopwords(config_path=None):
        # Empty fallback kapag hindi pa nailalagay ang query_expander.py.
        return set()

    def normalize_query_text(text):
        # Minimal fallback normalize.
        return " ".join(re.findall(r"[a-z0-9]+", str(text or "").lower()))


def get_question_stopwords():
    # Stopwords are loaded from config/query_expansion_config.json.
    # Walang hardcoded stopword list dito sa reranker.py.
    return set(get_config_stopwords(QUERY_EXPANSION_CONFIG_PATH))


def load_reranker(model_name=RERANKER_MODEL_NAME, use_fp16=RERANK_USE_FP16, debug=False):
    # I-load ang reranker model.
    # Reranker = second judge na pipili ng pinaka relevant chunks.
    if debug:
        print(f"[RERANKER] Loading model: {model_name}", flush=True)
        print(f"[RERANKER] use_fp16={use_fp16}", flush=True)

    reranker = FlagReranker(model_name, use_fp16=use_fp16)

    if debug:
        print("[RERANKER] Model loaded.", flush=True)

    return reranker


def normalize_scores(scores):
    # Gawing normal Python list ang scores para madaling i-sort.
    if isinstance(scores, (int, float)):
        return [float(scores)]

    if hasattr(scores, "tolist"):
        return scores.tolist()

    return list(scores)


def trim_document_text(text, max_chars=RERANK_MAX_CHARS):
    # Putulin ang chunk text bago ipasa sa reranker.
    # Purpose: mas mabilis at iwas sobrang habang input.
    clean_text = " ".join(str(text or "").split())

    if len(clean_text) <= max_chars:
        return clean_text

    return clean_text[:max_chars].rstrip()




def strip_retrieval_context_prefix(text):
    # Alisin ang metadata prefix kapag evidence check ang gagamit.
    # Reason: metadata helps retrieval/reranking, pero evidence should come from actual chunk content.
    text = str(text or "")

    if not text.startswith("Retrieval context:"):
        return text

    parts = text.split("\n\n", 1)

    if len(parts) == 2:
        return parts[1]

    return text

def get_document_label(doc):
    # Human-readable label para sa debug output.
    metadata = dict(getattr(doc, "metadata", {}) or {})

    source = metadata.get("file_name") or metadata.get("source") or "Unknown source"
    page = metadata.get("page", "N/A")
    section = metadata.get("section") or "Unknown section"
    category = metadata.get("category") or "Unknown category"
    doc_type = metadata.get("doc_type") or "Unknown type"
    chunk_id = metadata.get("chunk_id") or metadata.get("chunk_index") or "Unknown chunk"

    return f"{source} | page={page} | section={section} | {category}/{doc_type} | {chunk_id}"


def get_document_dedup_key(doc):
    # Gumawa ng stable key para matanggal ang duplicate chunks bago rerank.
    # Priority 1: source + chunk_id/chunk_index kapag meron.
    # Priority 2: source + page + exact normalized text hash.
    # Priority 3: text hash lang kapag kulang ang metadata.
    metadata = dict(getattr(doc, "metadata", {}) or {})

    source = (
        metadata.get("source")
        or metadata.get("file_name")
        or metadata.get("file")
        or ""
    )
    chunk_id = (
        metadata.get("chunk_id")
        or metadata.get("chunk_index")
        or metadata.get("id")
        or ""
    )
    page = metadata.get("page", "")

    source = str(source or "").strip().lower()
    chunk_id = str(chunk_id or "").strip().lower()
    page = str(page or "").strip().lower()

    if source and chunk_id:
        return ("chunk", source, chunk_id)

    text = strip_retrieval_context_prefix(getattr(doc, "page_content", ""))
    normalized_text = " ".join(str(text or "").lower().split())

    if normalized_text:
        text_hash = hashlib.md5(
            normalized_text.encode("utf-8", errors="ignore")
        ).hexdigest()

        if source or page:
            return ("text_with_location", source, page, text_hash)

        return ("text", text_hash)

    # Fallback para hindi aksidenteng ma-merge ang empty docs na walang metadata.
    return ("object", id(doc))


def deduplicate_documents(documents, debug=False):
    # Tanggalin ang duplicate docs bago gumawa ng query-document pairs.
    # Mas mabilis ang rerank dahil hindi na paulit-ulit sinuscore ang parehong chunk.
    # First occurrence ang kinikeep dahil usually RRF/original retrieval order na ito.
    unique_docs = []
    seen_keys = {}
    duplicate_count = 0

    for input_rank, doc in enumerate(documents, start=1):
        key = get_document_dedup_key(doc)

        if key in seen_keys:
            duplicate_count += 1
            kept_doc = seen_keys[key]
            kept_doc.metadata = dict(getattr(kept_doc, "metadata", {}) or {})
            kept_doc.metadata["rerank_duplicate_count"] = int(
                kept_doc.metadata.get("rerank_duplicate_count", 1)
            ) + 1
            continue

        doc.metadata = dict(getattr(doc, "metadata", {}) or {})
        doc.metadata["rerank_first_input_rank"] = input_rank
        doc.metadata["rerank_duplicate_count"] = 1

        seen_keys[key] = doc
        unique_docs.append(doc)

    if debug:
        print(
            f"[RERANKER] Dedup before rerank: "
            f"input={len(documents)}, unique={len(unique_docs)}, removed={duplicate_count}",
            flush=True,
        )

    return unique_docs


def tokenize(text):
    # Simple lowercase tokenizer.
    # Uses the same normalization style as query_expander.py when available.
    normalized_text = normalize_query_text(text)
    return re.findall(r"[a-z0-9]+", normalized_text)


def stem_simple(word):
    # Simple stemmer para mag-match ang:
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
    # Kunin ang important words mula sa query.
    # Stopwords galing sa JSON config via query_expander.py.
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
    # Minimum matched terms para hindi sobrang strict.
    if not query_terms:
        return 0

    if len(query_terms) <= 2:
        return len(query_terms)

    return 2


def get_max_allowed_span(query_terms):
    # Max word distance ng important terms.
    # Kapag magkakalapit ang important terms, mas likely na direct answer.
    if len(query_terms) <= 1:
        return None

    if len(query_terms) == 2:
        return 12

    return 18


def word_matches_term(word, term):
    # Generic matching.
    # Exact/stem match muna.
    # Tapos fallback substring para sa names like lapulapu/lapu.
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
    # Hanapin saan lumabas ang bawat important term sa chunk.
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
    # Sukatin kung gaano kalapit ang matched terms.
    # Sliding-window ang gamit para hindi bumagal sa broad/multi-part questions.
    # Old backtracking approach can explode kapag maraming repeated terms sa chunk.
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
    # 1. Count kung ilang important query terms ang nasa chunk.
    # 2. Optional lang ang proximity/span check para hindi bumagal sa broad questions.
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
        # Kapag hindi direct fact question, enough na ang term coverage.
        # Hindi na kukuwentahin ang best span para iwas sobrang bagal.
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
    # Save evidence details sa metadata para makita sa report/debug.
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
    # True kapag may enough important query terms at close enough sila.
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


def compute_rerank_scores(
    reranker,
    pairs,
    batch_size=RERANK_BATCH_SIZE,
    max_length=RERANK_MAX_LENGTH,
):
    # Compute score ng bawat query-document pair.
    # May fallback dahil iba-iba minsan ang supported parameters ng FlagEmbedding version.
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
    # Kapag sobrang taas ng best score, ibig sabihin may clear winner.
    # Huwag nang pilitin isama ang malalayong mababang score.
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
    # Custom sorting rule para sa reranker results.
    #
    # Normal rule:
    # - Higher reranker score wins.
    #
    # Tie rule:
    # - Kapag sobrang dikit ng scores, gamitin ang original rank.
    # - Original rank usually galing sa RRF order.
    # - Mas maliit original_rank = mas nauna sa retrieval/RRF.
    #
    # Example:
    # Bonifacio score = 4.7390, original_rank = 3
    # Emilio score    = 4.6641, original_rank = 1
    # gap = 0.0749
    #
    # If tie_margin = 0.15, tie sila.
    # Since original_rank 1 si Emilio, Emilio dapat mauna.
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
    # original_rank means position bago reranker sorting.
    # Since input documents usually galing RRF, ito ang RRF order.
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
