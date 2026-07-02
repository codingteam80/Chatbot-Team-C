import json
import re
import unicodedata
from collections import defaultdict
from copy import copy
from pathlib import Path

try:
    from config.settings import (
        MAX_CONTEXT_CHARS,
        MAX_PER_SOURCE,
        SINGLE_FACT_TOP_N,
        CROSS_DOC_TOP_N,
        COMPARISON_TOP_N,
        NEGATIVE_TOP_N,
        FALSE_PREMISE_TOP_N,
        ENABLE_NEIGHBOR_EXPANSION,
        NEIGHBOR_WINDOW,
    )
except ImportError:
    MAX_CONTEXT_CHARS = 6000
    MAX_PER_SOURCE = 2
    SINGLE_FACT_TOP_N = 3
    CROSS_DOC_TOP_N = 5
    COMPARISON_TOP_N = 4
    NEGATIVE_TOP_N = 2
    FALSE_PREMISE_TOP_N = 2
    ENABLE_NEIGHBOR_EXPANSION = True
    NEIGHBOR_WINDOW = 1

try:
    from retrieval.query_analyzer import analyze_query
except ImportError:
    analyze_query = None


# ============================================================
# MODE-AWARE CONTEXT FILTER
# ============================================================
# Purpose:
# - Piliin kung anong chunks ang ipapasa sa LLM.
# - Beginner-friendly para madaling i-debug.
# - Word lists are loaded from config/query_expansion_config.json.
#
# Flow:
# 1. Detect mode:
#    - single_fact
#    - cross_doc
#
# 2. If cross_doc:
#    - detect anchor source from the question when possible
#    - put anchor source first, then supporting sources
#    - do not force weak chunks just for diversity
#
# 3. If single_fact:
#    - check if question has entity/source match
#    - if yes, anchor to matching source
#    - if no, anchor to top reranked source
#
# 4. Apply confident filter only inside the selected anchor source.
#
# Expected pipeline:
# semantic + BM25 -> RRF -> reranker -> context_filter -> LLM
# ============================================================


# ============================================================
# 1. BASIC HELPERS
# ============================================================

DEFAULT_CONTEXT_CONFIG_PATH = Path("config") / "query_expansion_config.json"


def normalize_text(text):
    # Gawing lowercase at tanggalin punctuation.
    # Normalize accents para ang "Andrés" at "Andres" ay mag-match.
    text = str(text or "").lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = re.sub(r"[^a-z0-9\u3040-\u30ff\u3400-\u9fff]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_config_list(values):
    # Gawing set ng normalized tokens ang JSON list.
    # Lahat ng editable words galing sa JSON, hindi sa Python code.
    if isinstance(values, str):
        values = [values]

    if not isinstance(values, list):
        return set()

    normalized_values = set()

    for value in values:
        value = normalize_text(value)

        if value:
            normalized_values.add(value)

    return normalized_values


def read_context_json(config_path=DEFAULT_CONTEXT_CONFIG_PATH):
    # Basahin ang shared query/context config.
    # Kapag wala or invalid, empty config ang gagamitin.
    try:
        config_path = Path(config_path)

        if not config_path.exists():
            return {}

        return json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return {}


def load_context_filter_terms(config_path=DEFAULT_CONTEXT_CONFIG_PATH):
    # Lahat ng terms dito ay configurable sa JSON.
    # Python logic lang ito, walang domain-specific word list.
    raw_config = read_context_json(config_path=config_path)

    if not isinstance(raw_config, dict):
        raw_config = {}

    context_config = raw_config.get("context_filter", {})

    if not isinstance(context_config, dict):
        context_config = {}

    stopwords = normalize_config_list(raw_config.get("stopwords", []))
    question_weak_tokens = stopwords.union(
        normalize_config_list(context_config.get("question_weak_tokens", []))
    )

    source_weak_tokens = normalize_config_list(
        context_config.get("source_weak_tokens", [])
    )

    source_anchor_weak_tokens = question_weak_tokens.union(
        normalize_config_list(context_config.get("source_anchor_weak_tokens", []))
    )

    cross_doc_terms = normalize_config_list(
        context_config.get("cross_doc_terms", [])
    )

    return {
        "source_weak_tokens": source_weak_tokens,
        "question_weak_tokens": question_weak_tokens,
        "source_anchor_weak_tokens": source_anchor_weak_tokens,
        "cross_doc_terms": cross_doc_terms,
    }


CONTEXT_FILTER_TERMS = load_context_filter_terms()
SOURCE_WEAK_TOKENS = CONTEXT_FILTER_TERMS["source_weak_tokens"]
QUESTION_WEAK_TOKENS = CONTEXT_FILTER_TERMS["question_weak_tokens"]
SOURCE_ANCHOR_WEAK_TOKENS = CONTEXT_FILTER_TERMS["source_anchor_weak_tokens"]
CROSS_DOC_TERMS = CONTEXT_FILTER_TERMS["cross_doc_terms"]


def load_low_value_filter_config(config_path=DEFAULT_CONTEXT_CONFIG_PATH):
    # Low-value/reference/infobox-like chunk rules are configurable in JSON.
    raw_config = read_context_json(config_path=config_path)
    context_config = raw_config.get("context_filter", {}) if isinstance(raw_config, dict) else {}

    if not isinstance(context_config, dict):
        context_config = {}

    low_value_config = context_config.get("low_value", {})

    if not isinstance(low_value_config, dict):
        low_value_config = {}

    return {
        "reference_markers": normalize_config_list(low_value_config.get("reference_markers", [])),
        "low_value_section_phrases": normalize_config_list(low_value_config.get("low_value_section_phrases", [])),
        "low_value_body_phrases": normalize_config_list(low_value_config.get("low_value_body_phrases", [])),
    }


LOW_VALUE_FILTER_CONFIG = load_low_value_filter_config()
LOW_VALUE_REFERENCE_MARKERS = LOW_VALUE_FILTER_CONFIG["reference_markers"]
LOW_VALUE_SECTION_PHRASES = LOW_VALUE_FILTER_CONFIG["low_value_section_phrases"]
LOW_VALUE_BODY_PHRASES = LOW_VALUE_FILTER_CONFIG["low_value_body_phrases"]


def get_metadata(doc):
    # Safe metadata getter.
    return dict(getattr(doc, "metadata", {}) or {})


def get_source_name(doc):
    # Kunin ang pinaka-readable na source name.
    metadata = get_metadata(doc)

    return (
        metadata.get("file_name")
        or metadata.get("source")
        or metadata.get("title")
        or "unknown source"
    )


def get_source_key(doc):
    # Gumawa ng stable source key para malaman kung same document/source.
    # Example:
    # "data/Apolinario Mabini - Wikipedia.pdf"
    # -> "apolinario mabini"
    source_name = Path(str(get_source_name(doc))).stem
    tokens = normalize_text(source_name).split()

    useful_tokens = []

    for token in tokens:
        if token in SOURCE_WEAK_TOKENS:
            continue

        useful_tokens.append(token)

    if useful_tokens:
        return " ".join(useful_tokens)

    return normalize_text(source_name) or "unknown source"


def get_document_key(doc):
    # Gumawa ng stable key para hindi maulit ang same chunk.
    metadata = get_metadata(doc)

    source = metadata.get("source") or metadata.get("file_name") or ""
    page = metadata.get("page", "")
    chunk_id = metadata.get("chunk_id") or metadata.get("chunk_index") or ""

    if chunk_id != "":
        return (str(source), str(page), str(chunk_id))

    preview = str(getattr(doc, "page_content", "") or "")[:250]
    return (str(source), str(page), preview)


def remove_duplicate_docs(docs):
    # Tanggalin duplicate chunks habang pinapanatili ang current order.
    unique_docs = []
    seen_keys = set()

    for doc in docs or []:
        doc_key = get_document_key(doc)

        if doc_key in seen_keys:
            continue

        seen_keys.add(doc_key)
        unique_docs.append(doc)

    return unique_docs


# ============================================================
# 1B. NEIGHBOR CHUNK EXPANSION
# ============================================================

def get_chunk_number(doc):
    # Kunin ang numeric chunk index kung available sa metadata.
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
    # Hanapin ang exact chunk position sa full chunk list.
    target_key = get_document_key(target_doc)

    for index, chunk in enumerate(all_chunks or []):
        if get_document_key(chunk) == target_key:
            return index

    # Fallback kapag hindi exact ang key pero same source at same chunk number.
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
    # Gumawa ng copy para hindi madumihan ang cached all_chunks metadata.
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
    # Kunin ang numeric page kung available sa metadata.
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
    # Dagdagan ang selected docs ng neighbor chunks mula sa same source.
    # Important: current chunk muna, then NEXT chunk/page muna bago previous.
    # Reason: maraming PDF sections ay continuation sa next page.
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
    # Kunin score para sa debug print at local confident filter.
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
    # Usually galing ito sa RRF order bago mag-rerank.
    # Mas maliit = mas nauna sa retrieval/RRF.
    metadata = get_metadata(doc)
    value = metadata.get("rerank_original_rank")

    try:
        return int(value)
    except (TypeError, ValueError):
        return 999999


def get_question_terms(question):
    # Kunin useful terms mula sa question.
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
    # Kunin lang ang strong terms na safe gamitin pang source/entity anchor.
    # Example:
    # - "Who is the first Philippine president?" -> walang anchor terms.
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


# ============================================================
# 2. QUESTION MODE DETECTION
# ============================================================

# CROSS_DOC_TERMS is loaded from config/query_expansion_config.json.
# See context_filter.cross_doc_terms in the JSON config.
def detect_question_mode(question):
    # Detect final context mode.
    # list_answer is separate from single_fact because it needs multiple evidence chunks.
    detected_mode = "single_fact"
    mode_terms = load_mode_detection_terms()

    if analyze_query is not None:
        query_info = analyze_query(question)
        mode = normalize_text(query_info.get("mode", ""))

        if mode in mode_terms["cross_doc_modes"]:
            detected_mode = "cross_doc"
        elif mode in mode_terms["comparison_modes"]:
            detected_mode = "comparison"
        elif mode in mode_terms["negative_modes"]:
            detected_mode = "negative"
        elif mode in mode_terms["false_premise_modes"]:
            detected_mode = "false_premise"
        elif mode in mode_terms["list_modes"]:
            detected_mode = "list_answer"
        else:
            detected_mode = "single_fact"
    else:
        normalized_question = normalize_text(question)
        mode_terms = load_mode_detection_terms()

        if (
            has_any_phrase(normalized_question, mode_terms["false_premise_phrases"])
            or has_any_wildcard_or_regex(normalized_question, mode_terms["false_premise_patterns"])
        ):
            detected_mode = "false_premise"
        elif has_any_phrase(normalized_question, mode_terms["comparison_phrases"]):
            detected_mode = "comparison"
        elif has_explicit_cross_doc_cue(question):
            detected_mode = "cross_doc"
        elif has_multi_answer_cue(question):
            detected_mode = "list_answer"
        else:
            detected_mode = "single_fact"

    # Generic question-shape override.
    # This catches list-style questions even when query_analyzer returns single_fact.
    if detected_mode == "single_fact" and has_multi_answer_cue(question):
        detected_mode = "list_answer"

    if detected_mode in {"cross_doc", "comparison"} and not has_explicit_cross_doc_cue(question):
        return "single_fact"

    return detected_mode

def limit_context_docs(docs, max_chars=MAX_CONTEXT_CHARS, max_per_source=None):
    # Limitahan ang total characters ng final context.
    selected_docs = []
    total_chars = 0
    source_counts = defaultdict(int)

    for doc in docs or []:
        source_key = get_source_key(doc)

        if max_per_source is not None and source_counts[source_key] >= max_per_source:
            continue

        text = str(getattr(doc, "page_content", "") or "")
        text_length = len(text)

        if total_chars + text_length > max_chars:
            if not selected_docs:
                selected_docs.append(doc)

            break

        selected_docs.append(doc)
        total_chars += text_length
        source_counts[source_key] += 1

    return selected_docs


# ============================================================
# 4. SINGLE FACT HELPERS
# ============================================================

def count_source_question_matches(question_terms, source_key):
    # Bilangin kung ilang question terms ang tumama sa source name.
    source_tokens = set(normalize_text(source_key).split())
    match_count = 0

    for term in question_terms:
        if term in source_tokens:
            match_count += 1

    return match_count


def source_key_has_exact_question_phrase(source_key, normalized_question):
    # Exact phrase check para sa obvious source/entity questions.
    # Example: "what is code review sop" -> source_key "code review sop".
    source_key = normalize_text(source_key)
    normalized_question = normalize_text(normalized_question)

    if not source_key or not normalized_question:
        return False

    source_tokens = source_key.split()

    # Single-token sources like "katipunan" should match as a whole word.
    if len(source_tokens) == 1:
        pattern = rf"\b{re.escape(source_key)}\b"
        return re.search(pattern, normalized_question) is not None

    pattern = rf"\b{re.escape(source_key)}\b"
    return re.search(pattern, normalized_question) is not None


def is_safe_single_token_source_anchor(source_key, matched_terms):
    # Allow one-token title anchors like "Katipunan" or "MISRA".
    # Do not allow generic one-token anchors from descriptive words.
    source_tokens = normalize_text(source_key).split()

    if len(source_tokens) != 1:
        return False

    if len(matched_terms) != 1:
        return False

    token = source_tokens[0]

    if token in SOURCE_ANCHOR_WEAK_TOKENS:
        return False

    if token.isdigit():
        return False

    return token == matched_terms[0]


def is_strong_source_anchor(source_key, question_terms, normalized_question):
    # Generic rule:
    # 1. Exact source phrase in the question is strong.
    # 2. Two or more useful source-title terms matched is strong.
    # 3. One-token source titles can anchor if that exact token is in the question.
    #
    # This prevents descriptive questions like:
    # "secret group ... armed revolution ... 1896"
    # from wrongly anchoring to "Philippine Revolution" just because of "revolution".
    source_tokens = normalize_text(source_key).split()
    question_term_set = set(question_terms or [])

    matched_terms = []

    for token in source_tokens:
        if token in SOURCE_ANCHOR_WEAK_TOKENS:
            continue

        if token.isdigit():
            continue

        if token in question_term_set:
            matched_terms.append(token)

    exact_phrase_match = source_key_has_exact_question_phrase(
        source_key=source_key,
        normalized_question=normalized_question,
    )

    if exact_phrase_match:
        return True, len(matched_terms), "exact_source_phrase_match"

    if len(matched_terms) >= 2:
        return True, len(matched_terms), "multi_term_source_match"

    if is_safe_single_token_source_anchor(source_key, matched_terms):
        return True, len(matched_terms), "single_token_source_match"

    return False, len(matched_terms), "weak_or_descriptive_source_match"


def choose_single_fact_anchor_source(docs, question):
    # Piliin ang anchor source only kapag strong ang source/entity match.
    # Hindi sapat ang isang generic title word tulad ng "revolution".
    question_terms = get_question_anchor_terms(question)
    normalized_question = normalize_text(question)

    if not question_terms:
        return None, "no_question_source_match"

    best_source_key = None
    best_match_count = 0
    best_original_rank = 999999
    best_rerank_score = -999999.0
    best_reason = "no_question_source_match"

    for doc in docs:
        source_key = get_source_key(doc)
        is_strong, match_count, reason = is_strong_source_anchor(
            source_key=source_key,
            question_terms=question_terms,
            normalized_question=normalized_question,
        )
        original_rank = get_original_rank(doc)
        rerank_score = get_context_score(doc)

        if not is_strong:
            continue

        if match_count > best_match_count:
            best_source_key = source_key
            best_match_count = match_count
            best_original_rank = original_rank
            best_rerank_score = rerank_score
            best_reason = reason
            continue

        if match_count == best_match_count:
            if rerank_score > best_rerank_score:
                best_source_key = source_key
                best_original_rank = original_rank
                best_rerank_score = rerank_score
                best_reason = reason
                continue

            if rerank_score == best_rerank_score and original_rank < best_original_rank:
                best_source_key = source_key
                best_original_rank = original_rank
                best_rerank_score = rerank_score
                best_reason = reason

    if best_source_key and best_match_count > 0:
        return best_source_key, best_reason

    return None, "no_question_source_match"

def apply_confident_filter_inside_anchor_source(
    docs,
    high_confidence_score=3.0,
    max_score_drop=3.0,
):
    # Apply confidence filter only within selected anchor source.
    #
    # Important:
    # - Hindi ito global filter.
    # - Hindi nito tatanggalin ang ibang source bago malaman ang mode/anchor.
    # - Ginagamit lang ito after pumili ng anchor source.
    #
    # Example:
    # anchor source = "emilio aguinaldo"
    # best score = 6.32
    # cutoff = 3.32
    # keep only same-source docs with score >= 3.32
    if not docs:
        return docs

    scores = [get_context_score(doc) for doc in docs]
    best_score = max(scores)

    if best_score < high_confidence_score:
        return docs

    cutoff_score = best_score - max_score_drop
    filtered_docs = []

    for doc in docs:
        score = get_context_score(doc)

        if score >= cutoff_score:
            filtered_docs.append(doc)

    if filtered_docs:
        return filtered_docs

    return docs



def build_rank_map(docs):
    # Gumawa ng lookup ng exact chunk rank mula sa retriever results.
    rank_map = {}

    for rank, doc in enumerate(docs or [], start=1):
        rank_map[get_document_key(doc)] = rank

    return rank_map


def get_rank_from_metadata(doc, keys):
    # Kunin ang retrieval rank na nai-save sa metadata.
    metadata = get_metadata(doc)

    for key in keys:
        value = metadata.get(key)

        if value is None:
            continue

        try:
            return int(value)
        except (TypeError, ValueError):
            continue

    return None


def get_retrieval_rank(doc, rank_map, metadata_keys):
    # Priority 1: exact rank map from original semantic/BM25 results.
    # Priority 2: metadata rank saved by hybrid_retriever/RRF.
    doc_key = get_document_key(doc)

    if doc_key in rank_map:
        return rank_map[doc_key]

    return get_rank_from_metadata(doc, metadata_keys)


def get_retrieval_agreement_score(doc, semantic_rank_map, bm25_rank_map):
    # Generic direct-evidence signal.
    # Mataas ang score kapag parehong semantic at BM25 nakita ang same chunk.
    semantic_rank = get_retrieval_rank(
        doc=doc,
        rank_map=semantic_rank_map,
        metadata_keys=("semantic_rank", "retrieval_rank_0"),
    )
    bm25_rank = get_retrieval_rank(
        doc=doc,
        rank_map=bm25_rank_map,
        metadata_keys=("bm25_rank", "retrieval_rank_1"),
    )

    score = 0.0

    if semantic_rank is not None:
        score += 2.0 / max(semantic_rank, 1)

    if bm25_rank is not None:
        score += 2.0 / max(bm25_rank, 1)

    if semantic_rank is not None and bm25_rank is not None:
        score += 4.0

    if semantic_rank == 1 and bm25_rank == 1:
        score += 4.0

    return score, semantic_rank, bm25_rank


def get_intro_chunk_score(doc):
    # Small bonus kapag intro/main chunk.
    # Useful for identity/direct fact questions, pero hindi hardcoded sa source.
    metadata = get_metadata(doc)

    page = str(metadata.get("page", "")).strip().lower()
    chunk_id = str(metadata.get("chunk_id") or metadata.get("chunk_index") or "").strip().lower()

    score = 0.0

    if page in {"0", "1"}:
        score += 1.0

    if chunk_id in {"0", "1"} or "chunk_0" in chunk_id or "chunk_1" in chunk_id:
        score += 1.0

    return score


def get_direct_window_score(question_terms, doc, window_size=80, step_size=20):
    # Mas generic kaysa plain keyword count.
    # Kapag useful question terms ay magkakalapit sa chunk, mas likely direct context.
    text = normalize_text(getattr(doc, "page_content", "") or "")
    words = text.split()

    if not words or not question_terms:
        return 0.0

    best_window_matches = 0

    for start in range(0, len(words), step_size):
        window_text = " ".join(words[start:start + window_size])
        window_matches = 0

        for term in question_terms:
            if term in window_text:
                window_matches += 1

        if window_matches > best_window_matches:
            best_window_matches = window_matches

    return float(best_window_matches)


def has_retrieval_signal(docs, semantic_docs=None, bm25_docs=None):
    # Gamitin lang primary evidence scoring kapag may semantic/BM25 ranks.
    if semantic_docs or bm25_docs:
        return True

    for doc in docs or []:
        metadata = get_metadata(doc)

        if any(key in metadata for key in ("semantic_rank", "bm25_rank", "retrieval_rank_0", "retrieval_rank_1")):
            return True

    return False


# ============================================================
# 4A. ANSWER-INTENT AND DYNAMIC SINGLE-FACT SETTINGS
# ============================================================

def load_answer_evidence_config(config_path=DEFAULT_CONTEXT_CONFIG_PATH):
    # Load answer-intent rules from JSON.
    # Python stays generic; editable question/evidence terms stay in JSON.
    raw_config = read_context_json(config_path=config_path)

    if not isinstance(raw_config, dict):
        return {}

    answer_config = raw_config.get("answer_evidence", {})

    if not isinstance(answer_config, dict):
        return {}

    return answer_config


ANSWER_EVIDENCE_CONFIG = load_answer_evidence_config()


def get_answer_intent_configs():
    answer_config = ANSWER_EVIDENCE_CONFIG

    if not isinstance(answer_config, dict):
        return {}, []

    intents = answer_config.get("intents", {})

    if not isinstance(intents, dict):
        intents = {}

    intent_order = answer_config.get("intent_order", [])

    if not isinstance(intent_order, list) or not intent_order:
        intent_order = list(intents.keys())

    return intents, intent_order


def get_config_list(config, key):
    values = config.get(key, [])

    if isinstance(values, str):
        values = [values]

    if not isinstance(values, list):
        return []

    cleaned_values = []

    for value in values:
        value = str(value or "").strip()

        if value:
            cleaned_values.append(value)

    return cleaned_values


def config_phrase_matches(normalized_question, phrase):
    phrase = normalize_text(phrase)

    if not phrase:
        return False

    padded_question = f" {normalize_text(normalized_question)} "
    padded_phrase = f" {phrase} "

    return padded_phrase in padded_question


def detect_answer_intent(question):
    # Detect the type of answer needed by the question using JSON config.
    # This is not domain/file-specific. It only detects answer shape.
    intents, intent_order = get_answer_intent_configs()
    normalized_question = normalize_text(question)
    question_tokens = set(normalized_question.split())

    for intent_name in intent_order:
        intent_config = intents.get(intent_name, {})

        if not isinstance(intent_config, dict):
            continue

        triggers = []
        triggers.extend(get_config_list(intent_config, "question_terms"))
        triggers.extend(get_config_list(intent_config, "question_phrases"))

        for trigger in triggers:
            normalized_trigger = normalize_text(trigger)

            if not normalized_trigger:
                continue

            if " " in normalized_trigger:
                if config_phrase_matches(normalized_question, normalized_trigger):
                    return intent_name
            elif normalized_trigger in question_tokens:
                return intent_name

    default_intent = ANSWER_EVIDENCE_CONFIG.get("default_intent", "general_fact")

    if not isinstance(default_intent, str) or not default_intent.strip():
        default_intent = "general_fact"

    return normalize_text(default_intent) or "general_fact"


def get_answer_intent_config(question):
    intents, _ = get_answer_intent_configs()
    intent = detect_answer_intent(question)
    intent_config = intents.get(intent, {})

    if not isinstance(intent_config, dict):
        intent_config = {}

    return intent, intent_config


def safe_int(value, default_value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default_value


def safe_float(value, default_value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default_value


def safe_bool(value, default_value):
    if isinstance(value, bool):
        return value

    if value is None:
        return default_value

    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def clamp_int(value, default_value, minimum=None, maximum=None):
    # Safe integer clamp for dynamic mode settings.
    value = safe_int(value, default_value)

    if minimum is not None:
        value = max(value, minimum)

    if maximum is not None:
        value = min(value, maximum)

    return value


def get_dynamic_retrieval_settings(question):
    # Optional helper for the retrieval/rerank caller.
    # Context filter only receives reranked docs, so upstream code can call this
    # before semantic/BM25/hybrid/rerank to avoid reranking too many chunks.
    mode = detect_question_mode(question)

    if should_force_single_fact_mode(question, mode):
        mode = "single_fact"

    settings_by_mode = {
        "single_fact": {
            "semantic_k": 6,
            "bm25_k": 6,
            "hybrid_final_k": 8,
            "rerank_input_k": 6,
            "rerank_top_n": 6,
            "final_top_n": 5,
            "max_context_chars": 6500,
            "neighbor_expansion": False,
        },
        "list_answer": {
            "semantic_k": 10,
            "bm25_k": 10,
            "hybrid_final_k": 12,
            "rerank_input_k": 10,
            "rerank_top_n": 10,
            "final_top_n": 5,
            "max_context_chars": 6500,
            "neighbor_expansion": False,
        },
        "cross_doc": {
            "semantic_k": 10,
            "bm25_k": 10,
            "hybrid_final_k": 12,
            "rerank_input_k": 12,
            "rerank_top_n": 12,
            "final_top_n": 4,
            "max_context_chars": 6500,
            "neighbor_expansion": False,
        },
        "comparison": {
            "semantic_k": 10,
            "bm25_k": 10,
            "hybrid_final_k": 12,
            "rerank_input_k": 12,
            "rerank_top_n": 12,
            "final_top_n": 4,
            "max_context_chars": 6500,
            "neighbor_expansion": False,
        },
        "negative": {
            "semantic_k": 8,
            "bm25_k": 8,
            "hybrid_final_k": 10,
            "rerank_input_k": 8,
            "rerank_top_n": 8,
            "final_top_n": 5,
            "max_context_chars": 6500,
            "neighbor_expansion": False,
        },
        "false_premise": {
            "semantic_k": 8,
            "bm25_k": 8,
            "hybrid_final_k": 10,
            "rerank_input_k": 8,
            "rerank_top_n": 8,
            "final_top_n": 2,
            "max_context_chars": 4000,
            "neighbor_expansion": False,
        },
    }

    settings = dict(settings_by_mode.get(mode, settings_by_mode["single_fact"]))
    settings["mode"] = mode
    return settings


def get_dynamic_context_policy(question, mode, top_n=None, max_chars=MAX_CONTEXT_CHARS, max_per_source=MAX_PER_SOURCE):
    # Final-context policy. top_n is a maximum, not a target to force-fill.
    retrieval_settings = get_dynamic_retrieval_settings(question)

    default_top_n = retrieval_settings.get("final_top_n", SINGLE_FACT_TOP_N)
    requested_top_n = safe_int(top_n, None)

    if requested_top_n is None:
        effective_top_n = default_top_n
    else:
        effective_top_n = min(requested_top_n, default_top_n)

    effective_top_n = clamp_int(effective_top_n, default_top_n, minimum=1, maximum=12)

    default_max_chars = retrieval_settings.get("max_context_chars", MAX_CONTEXT_CHARS)
    effective_max_chars = min(
        clamp_int(max_chars, MAX_CONTEXT_CHARS, minimum=1000),
        clamp_int(default_max_chars, MAX_CONTEXT_CHARS, minimum=1000),
    )

    if mode == "list_answer":
        effective_max_per_source = None
    elif mode in {"cross_doc", "comparison"}:
        effective_max_per_source = 2
    else:
        effective_max_per_source = 2

    if max_per_source is not None and effective_max_per_source is not None:
        effective_max_per_source = min(
            clamp_int(max_per_source, MAX_PER_SOURCE, minimum=1),
            effective_max_per_source,
        )

    return {
        "mode": mode,
        "top_n": effective_top_n,
        "max_chars": effective_max_chars,
        "max_per_source": effective_max_per_source,
        "neighbor_expansion": bool(retrieval_settings.get("neighbor_expansion", False)),
    }


def config_phrase_in_text(normalized_text, phrases):
    normalized_text = f" {normalize_text(normalized_text)} "

    for phrase in phrases or []:
        phrase = normalize_text(phrase)

        if phrase and f" {phrase} " in normalized_text:
            return True

    return False


def is_reference_like_context_doc(doc):
    # Generic low-value chunk detector.
    # Reference/infobox-like markers come from query_expansion_config.json.
    metadata = get_metadata(doc)
    section = normalize_text(metadata.get("section", ""))
    title = normalize_text(metadata.get("title", ""))
    body = str(getattr(doc, "page_content", "") or "")
    normalized_body = normalize_text(body)
    combined_header = f" {title} {section} "

    if config_phrase_in_text(combined_header, LOW_VALUE_REFERENCE_MARKERS):
        return True

    if config_phrase_in_text(combined_header, LOW_VALUE_SECTION_PHRASES):
        return True

    if config_phrase_in_text(normalized_body, LOW_VALUE_BODY_PHRASES):
        return True

    marker_hits = 0

    for marker in LOW_VALUE_REFERENCE_MARKERS:
        if marker and marker in normalized_body:
            marker_hits += 1

    url_like_hits = len(re.findall(r"https?://|www\.|\bdoi\b|\bisbn\b", body, flags=re.IGNORECASE))

    if marker_hits >= 2 or url_like_hits >= 2:
        return True

    # Generic URL/query/citation noise detector.
    # This is not source-specific; it catches chunks that are mostly references,
    # copied URL query strings, archive links, or bibliography fragments.
    raw_combined_text = " ".join([
        str(metadata.get("section", "")),
        str(metadata.get("title", "")),
        body,
    ])

    has_url_query_noise = re.search(
        r"(&(?:dq|pg|id|q|source)=|books\.google|archive\.org|wayback|https?://|www\.)",
        raw_combined_text,
        flags=re.IGNORECASE,
    ) is not None

    # Match citation/list-reference fragments even when the date is written as
    # "(23 July 2020)" or when the section starts with a numbered reference.
    # One strong citation-looking section is enough because final cleanup still
    # protects chunks with configured direct answer evidence.
    citation_line_hits = len(re.findall(
        r"(?:^|\n|\s)\d{1,3}\.\s+[A-Z][^\n]{10,220}(?:\([^)]*\d{4}[^)]*\)|Retrieved|Archived|ISBN|doi|http|www\.)",
        raw_combined_text,
        flags=re.IGNORECASE,
    ))

    section_citation_like = re.search(
        r"^\s*\d{1,3}\.\s+.+(?:\([^)]*\d{4}[^)]*\)|Retrieved|Archived|ISBN|doi|http|www\.)",
        str(metadata.get("section", "")),
        flags=re.IGNORECASE,
    ) is not None

    if has_url_query_noise or section_citation_like or citation_line_hits >= 1:
        return True

    words = normalized_body.split()

    if len(words) < 25 and marker_hits > 0:
        return True

    return False


def is_low_value_context_doc(question, doc):
    # Keep answer-bearing chunks, drop weak/reference-like chunks.
    question_terms = get_question_terms(question)
    body_text = get_doc_body_text(doc)
    body_match_count = count_question_term_coverage(question_terms, body_text)
    direct_score = get_direct_window_score(question_terms, doc)
    answer_score = get_answer_pattern_score(question, question_terms, doc)

    if answer_score > 0:
        return False

    # Drop reference/infobox-like chunks before generic term-count rescue.
    # This prevents metadata/status tables from passing only because they repeat the topic words.
    if is_reference_like_context_doc(doc):
        return True

    if direct_score >= 2:
        return False

    if body_match_count >= max(2, min(3, len(question_terms))):
        return False

    body_words = normalize_text(body_text).split()

    if len(body_words) < 20 and body_match_count == 0:
        return True

    return False


def filter_low_value_context_docs(question, docs, min_keep=1):
    # Filter weak chunks but never return empty when candidates exist.
    docs = remove_duplicate_docs(docs)

    if not docs:
        return []

    kept_docs = []

    for doc in docs:
        if not is_low_value_context_doc(question, doc):
            kept_docs.append(doc)

    if kept_docs:
        return kept_docs

    return docs[:max(1, min_keep)]


def has_list_answer_evidence(question, doc):
    # Body-focused evidence check for list questions.
    # Metadata/title matches alone should not make a chunk useful.
    question_terms = get_question_terms(question)
    body_text = get_doc_body_text(doc)
    body_match_count = count_question_term_coverage(question_terms, body_text)
    direct_score = get_direct_window_score(question_terms, doc)
    answer_score = get_answer_pattern_score(question, question_terms, doc)

    return answer_score > 0 or direct_score >= 2 or body_match_count >= 2


def has_any_regex(normalized_question, patterns):
    # Regex patterns are read from JSON config.
    for pattern in patterns or []:
        try:
            if re.search(str(pattern), normalized_question, flags=re.IGNORECASE):
                return True
        except re.error:
            continue

    return False


def has_multi_answer_cue(question):
    # Detect questions that expect a list, not one exact actor/date/value.
    # Editable list cues stay in query_expansion_config.json.
    normalized_question = normalize_text(question)

    if not normalized_question:
        return False

    mode_terms = load_mode_detection_terms()
    dynamic_config = ANSWER_EVIDENCE_CONFIG.get("dynamic_single_fact", {})

    if not isinstance(dynamic_config, dict):
        dynamic_config = {}

    list_phrases = []
    list_phrases.extend(mode_terms.get("list_phrases", []))
    list_phrases.extend(get_config_list(dynamic_config, "multi_answer_phrases"))

    if has_any_phrase(normalized_question, list_phrases):
        return True

    list_patterns = []
    list_patterns.extend(mode_terms.get("list_patterns", []))
    list_patterns.extend(get_config_list(dynamic_config, "multi_answer_patterns"))

    if has_any_regex(normalized_question, list_patterns):
        return True

    # Optional generic plural rule.
    # The blocked helper words are configurable in JSON using non_list_after_what.
    enable_plural_what_rule = safe_bool(
        dynamic_config.get("enable_plural_what_rule"),
        True,
    )

    if not enable_plural_what_rule:
        return False

    tokens = normalized_question.split()

    if len(tokens) >= 2 and tokens[0] == "what":
        second_token = tokens[1]
        non_list_after_what = set(
            normalize_text(value)
            for value in get_config_list(dynamic_config, "non_list_after_what")
        )

        if second_token not in non_list_after_what and second_token.endswith("s"):
            return True

    return False

def get_dynamic_multi_answer_settings(question, base_intent, requested_top_n, dynamic_config):
    # Multi-answer/list questions need broader context than strict single-answer facts.
    # Example: "Who are..." should not be limited like "Who killed...".
    if not has_multi_answer_cue(question):
        return None

    multi_answer_top_n = safe_int(
        dynamic_config.get("multi_answer_top_n"),
        SINGLE_FACT_TOP_N,
    )

    effective_top_n = max(
        safe_int(requested_top_n, SINGLE_FACT_TOP_N),
        multi_answer_top_n,
        SINGLE_FACT_TOP_N,
    )

    if effective_top_n < 1:
        effective_top_n = SINGLE_FACT_TOP_N

    return {
        "intent": f"multi_answer_{base_intent}",
        "top_n": effective_top_n,
        # Multi-answer/list questions should keep the selected evidence chunks.
        # Neighbor expansion can push out other relevant reranked chunks, so default is False.
        "enable_neighbor_expansion": safe_bool(
            dynamic_config.get("multi_answer_neighbor_expansion"),
            False,
        ),
        "selection_strategy": str(
            dynamic_config.get("multi_answer_selection_strategy", "rerank_first")
        ).strip().lower() or "rerank_first",
        "strict": False,
        "multi_answer": True,
    }

def get_dynamic_single_fact_settings(question, requested_top_n=None):
    # Compute effective single_fact settings per question intent.
    # settings.py keeps global defaults; JSON controls strict/dynamic behavior.
    intent, intent_config = get_answer_intent_config(question)

    dynamic_config = ANSWER_EVIDENCE_CONFIG.get("dynamic_single_fact", {})

    if not isinstance(dynamic_config, dict):
        dynamic_config = {}

    strict_intent_values = get_config_list(dynamic_config, "strict_intents")
    strict_intents = set(strict_intent_values)
    strict_intents_normalized = set(
        normalize_text(value)
        for value in strict_intent_values
    )

    requested_top_n = safe_int(requested_top_n, None)

    if requested_top_n is None:
        requested_top_n = SINGLE_FACT_TOP_N

    multi_answer_settings = get_dynamic_multi_answer_settings(
        question=question,
        base_intent=intent,
        requested_top_n=requested_top_n,
        dynamic_config=dynamic_config,
    )

    if multi_answer_settings:
        return multi_answer_settings

    configured_top_n = safe_int(intent_config.get("top_n"), None)

    if configured_top_n is None:
        if intent in strict_intents or normalize_text(intent) in strict_intents_normalized:
            configured_top_n = safe_int(dynamic_config.get("strict_top_n"), 2)
        elif normalize_text(intent) == normalize_text(dynamic_config.get("procedure_intent", "procedure")):
            configured_top_n = safe_int(dynamic_config.get("procedure_top_n"), requested_top_n)
        else:
            configured_top_n = requested_top_n

    if intent in strict_intents or normalize_text(intent) in strict_intents_normalized:
        effective_top_n = min(requested_top_n, configured_top_n)
    elif normalize_text(intent) == normalize_text(dynamic_config.get("procedure_intent", "procedure")):
        effective_top_n = max(requested_top_n, configured_top_n)
    else:
        effective_top_n = configured_top_n

    if effective_top_n < 1:
        effective_top_n = 1

    if "neighbor_expansion" in intent_config:
        effective_neighbor_expansion = safe_bool(
            intent_config.get("neighbor_expansion"),
            ENABLE_NEIGHBOR_EXPANSION,
        )
    elif intent in strict_intents or normalize_text(intent) in strict_intents_normalized:
        effective_neighbor_expansion = safe_bool(
            dynamic_config.get("strict_neighbor_expansion"),
            False,
        )
    elif normalize_text(intent) == normalize_text(dynamic_config.get("procedure_intent", "procedure")):
        effective_neighbor_expansion = safe_bool(
            dynamic_config.get("procedure_neighbor_expansion"),
            ENABLE_NEIGHBOR_EXPANSION,
        )
    else:
        # Default single-fact questions should stay focused.
        # Expansion is only enabled when JSON explicitly asks for it.
        effective_neighbor_expansion = safe_bool(
            dynamic_config.get("single_fact_neighbor_expansion"),
            False,
        )

    selection_strategy = str(intent_config.get("selection_strategy", "") or "").strip().lower()

    if not selection_strategy:
        if intent in strict_intents or normalize_text(intent) in strict_intents_normalized:
            selection_strategy = "primary_evidence"
        else:
            selection_strategy = "rerank_first"

    return {
        "intent": intent,
        "top_n": effective_top_n,
        "enable_neighbor_expansion": effective_neighbor_expansion,
        "selection_strategy": selection_strategy,
        "strict": intent in strict_intents or normalize_text(intent) in strict_intents_normalized,
        "multi_answer": False,
    }


def get_doc_body_text(doc):
    return str(getattr(doc, "page_content", "") or "")


def get_doc_full_text(doc):
    metadata = get_metadata(doc)

    return " ".join(
        [
            str(metadata.get("title", "")),
            str(metadata.get("section", "")),
            str(metadata.get("file_name", "")),
            str(metadata.get("source", "")),
            get_doc_body_text(doc),
        ]
    )


def count_question_term_coverage(question_terms, text):
    normalized_text = normalize_text(text)
    match_count = 0

    for term in question_terms or []:
        if term in normalized_text:
            match_count += 1

    return match_count


def config_term_matches(text, term):
    normalized_text = f" {normalize_text(text)} "
    normalized_term = normalize_text(term)

    if not normalized_term:
        return False

    return f" {normalized_term} " in normalized_text


def has_any_evidence_term(text, terms):
    for term in terms:
        if config_term_matches(text, term):
            return True

    return False


def has_any_evidence_regex(text, patterns):
    for pattern in patterns:
        try:
            if re.search(pattern, text, flags=re.IGNORECASE):
                return True
        except re.error:
            continue

    return False


def get_answer_pattern_score(question, question_terms, doc):
    # Score whether a chunk contains the expected answer shape for the question.
    # The terms and regex patterns are loaded from JSON, not hardcoded per file/topic.
    _, intent_config = get_answer_intent_config(question)

    if not intent_config:
        return 0.0

    body_text = get_doc_body_text(doc)
    full_text = get_doc_full_text(doc)

    evidence_terms = get_config_list(intent_config, "evidence_terms")
    evidence_regex = get_config_list(intent_config, "evidence_regex")

    has_evidence = False

    if evidence_terms:
        has_evidence = has_any_evidence_term(body_text, evidence_terms)

    if not has_evidence and evidence_regex:
        has_evidence = has_any_evidence_regex(body_text, evidence_regex)

    question_match_count = count_question_term_coverage(question_terms, full_text)
    min_question_matches = safe_int(
        intent_config.get("min_question_term_matches"),
        1,
    )

    required_evidence_ok = True
    required_rules = intent_config.get("required_evidence_terms_by_question_terms", [])

    if not isinstance(required_rules, list):
        required_rules = []

    normalized_question = normalize_text(question)

    for rule in required_rules:
        if not isinstance(rule, dict):
            continue

        rule_question_terms = get_config_list(rule, "question_terms")
        rule_question_phrases = get_config_list(rule, "question_phrases")
        rule_evidence_terms = get_config_list(rule, "evidence_terms")
        rule_evidence_regex = get_config_list(rule, "evidence_regex")
        rule_exclude_terms = get_config_list(rule, "exclude_terms")
        rule_exclude_regex = get_config_list(rule, "exclude_regex")

        question_rule_matched = False

        for term in rule_question_terms:
            normalized_term = normalize_text(term)

            if normalized_term and normalized_term in set(normalized_question.split()):
                question_rule_matched = True
                break

        if not question_rule_matched:
            for phrase in rule_question_phrases:
                if config_phrase_matches(normalized_question, phrase):
                    question_rule_matched = True
                    break

        if question_rule_matched:
            if rule_exclude_terms and has_any_evidence_term(full_text, rule_exclude_terms):
                required_evidence_ok = False
                doc.metadata = dict(getattr(doc, "metadata", {}) or {})
                doc.metadata["answer_evidence_excluded"] = True
                doc.metadata["answer_evidence_exclusion_source"] = "required_rule_exclude_terms"
                break

            if rule_exclude_regex and has_any_evidence_regex(full_text, rule_exclude_regex):
                required_evidence_ok = False
                doc.metadata = dict(getattr(doc, "metadata", {}) or {})
                doc.metadata["answer_evidence_excluded"] = True
                doc.metadata["answer_evidence_exclusion_source"] = "required_rule_exclude_regex"
                break

            required_term_ok = True
            required_regex_ok = True

            if rule_evidence_terms:
                required_term_ok = has_any_evidence_term(body_text, rule_evidence_terms)

            if rule_evidence_regex:
                required_regex_ok = has_any_evidence_regex(body_text, rule_evidence_regex)

            if (rule_evidence_terms or rule_evidence_regex) and not (required_term_ok or required_regex_ok):
                required_evidence_ok = False
                break

    score = 0.0

    if has_evidence and required_evidence_ok and question_match_count >= min_question_matches:
        score += safe_float(intent_config.get("evidence_match_score"), 5.0)
    elif has_evidence and required_evidence_ok:
        score += safe_float(intent_config.get("weak_evidence_match_score"), 1.0)
    elif evidence_terms or evidence_regex:
        score -= safe_float(intent_config.get("missing_evidence_penalty"), 0.0)

    if not required_evidence_ok:
        score -= safe_float(intent_config.get("required_evidence_missing_penalty"), 6.0)

    term_match_score = safe_float(intent_config.get("question_term_match_score"), 0.5)
    max_term_bonus = safe_int(intent_config.get("max_question_term_bonus"), 3)
    score += min(question_match_count, max_term_bonus) * term_match_score

    # Store generic evidence flags so final selection can distinguish
    # true answer evidence from loose question-term overlap.
    doc.metadata = dict(getattr(doc, "metadata", {}) or {})
    doc.metadata["answer_evidence_has_evidence"] = bool(has_evidence)
    doc.metadata["answer_evidence_required_ok"] = bool(required_evidence_ok)
    doc.metadata["answer_evidence_question_match_count"] = int(question_match_count)

    return score



def get_top1_agreement_doc(semantic_docs=None, bm25_docs=None):
    # Strong generic signal: semantic top 1 and BM25 top 1 are the exact same chunk.
    # Kapag nangyari ito sa single_fact question, usually direct-answer chunk iyon.
    if not semantic_docs or not bm25_docs:
        return None

    semantic_top = semantic_docs[0]
    bm25_top = bm25_docs[0]

    if get_document_key(semantic_top) != get_document_key(bm25_top):
        return None

    return semantic_top


def score_primary_evidence_doc(question, question_terms, doc, semantic_rank_map, bm25_rank_map):
    # One place lang ang formula para consistent.
    retrieval_score, semantic_rank, bm25_rank = get_retrieval_agreement_score(
        doc=doc,
        semantic_rank_map=semantic_rank_map,
        bm25_rank_map=bm25_rank_map,
    )
    direct_window_score = get_direct_window_score(
        question_terms=question_terms,
        doc=doc,
    )
    answer_pattern_score = get_answer_pattern_score(
        question=question,
        question_terms=question_terms,
        doc=doc,
    )
    intro_chunk_score = get_intro_chunk_score(doc)
    rerank_score = get_context_score(doc)

    # Evidence-first scoring:
    # - answer_pattern_score checks if the chunk has the expected answer shape
    #   for the question intent, using JSON-configured terms/regex.
    # - direct_window_score keeps question terms close to the evidence.
    # - retrieval/rerank are useful, but only as secondary signals.
    primary_evidence_score = (
        (answer_pattern_score * 4.0)
        + (direct_window_score * 3.0)
        + intro_chunk_score
        + (retrieval_score * 0.35)
        + (rerank_score * 0.05)
    )

    doc.metadata = dict(getattr(doc, "metadata", {}) or {})
    doc.metadata["semantic_rank_for_primary"] = semantic_rank
    doc.metadata["bm25_rank_for_primary"] = bm25_rank
    doc.metadata["retrieval_agreement_score"] = float(retrieval_score)
    doc.metadata["direct_window_score"] = float(direct_window_score)
    doc.metadata["answer_intent"] = detect_answer_intent(question)
    doc.metadata["answer_pattern_score"] = float(answer_pattern_score)
    doc.metadata["intro_chunk_score"] = float(intro_chunk_score)
    doc.metadata["primary_evidence_score"] = float(primary_evidence_score)
    doc.metadata["primary_evidence_score_mode"] = "answer_pattern_weighted_evidence"

    return doc


def sort_primary_evidence_docs(docs):
    return sorted(
        docs or [],
        key=lambda doc: (
            float(get_metadata(doc).get("primary_evidence_score", 0.0)),
            get_context_score(doc),
            -get_original_rank(doc),
        ),
        reverse=True,
    )


def select_primary_evidence_docs(
    question,
    reranked_docs,
    semantic_docs=None,
    bm25_docs=None,
    top_n=3,
):
    # Generic selector para sa single_fact questions.
    # Hindi ito naka-hardcode sa specific answer/source.
    # Goal: kapag maraming chunks pareho ang keyword match,
    # piliin ang may pinaka-direct evidence.
    base_docs = list(reranked_docs or [])

    if not has_retrieval_signal(base_docs, semantic_docs=semantic_docs, bm25_docs=bm25_docs):
        return []

    question_terms = get_question_terms(question)
    semantic_rank_map = build_rank_map(semantic_docs)
    bm25_rank_map = build_rank_map(bm25_docs)

    candidates = remove_duplicate_docs(
        list(reranked_docs or []) + list(semantic_docs or []) + list(bm25_docs or [])
    )

    agreed_top_doc = get_top1_agreement_doc(
        semantic_docs=semantic_docs,
        bm25_docs=bm25_docs,
    )

    # Strict primary-source mode:
    # Kapag exact same chunk ang semantic top 1 at BM25 top 1 sa single_fact,
    # huwag nang punuan ang context gamit ibang sources.
    # Mas safe na 1-2 direct chunks lang kaysa maraming related pero confusing chunks.
    if agreed_top_doc is not None:
        primary_source_key = get_source_key(agreed_top_doc)
        same_source_candidates = []

        for doc in candidates:
            if get_source_key(doc) == primary_source_key:
                same_source_candidates.append(doc)

        scored_same_source_docs = []

        for doc in same_source_candidates:
            scored_doc = score_primary_evidence_doc(
                question=question,
                question_terms=question_terms,
                doc=doc,
                semantic_rank_map=semantic_rank_map,
                bm25_rank_map=bm25_rank_map,
            )
            scored_doc.metadata["primary_source_lock"] = True
            scored_doc.metadata["primary_source_lock_reason"] = "semantic_bm25_top1_exact_chunk_agreement"
            scored_same_source_docs.append(scored_doc)

        scored_same_source_docs = sort_primary_evidence_docs(scored_same_source_docs)

        # Siguraduhin na ang exact agreed top chunk ay kasama kahit may tie/noise.
        agreed_key = get_document_key(agreed_top_doc)
        ordered_docs = []
        seen_keys = set()

        for doc in scored_same_source_docs:
            if get_document_key(doc) == agreed_key:
                ordered_docs.append(doc)
                seen_keys.add(agreed_key)
                break

        for doc in scored_same_source_docs:
            doc_key = get_document_key(doc)

            if doc_key in seen_keys:
                continue

            ordered_docs.append(doc)
            seen_keys.add(doc_key)

            if len(ordered_docs) >= top_n:
                break

        return ordered_docs[:top_n]

    scored_docs = []

    for doc in candidates:
        scored_doc = score_primary_evidence_doc(
            question=question,
            question_terms=question_terms,
            doc=doc,
            semantic_rank_map=semantic_rank_map,
            bm25_rank_map=bm25_rank_map,
        )
        scored_docs.append(scored_doc)

    scored_docs = sort_primary_evidence_docs(scored_docs)

    return scored_docs[:top_n]



# ============================================================
# 4B. FINAL CONTEXT QUALITY HELPERS
# ============================================================

def get_primary_evidence_score(doc):
    # Score created by score_primary_evidence_doc().
    metadata = get_metadata(doc)

    try:
        return float(metadata.get("primary_evidence_score", 0.0))
    except (TypeError, ValueError):
        return 0.0


def get_direct_score(doc):
    # Direct window score = ilang important question terms ang magkakalapit sa chunk.
    metadata = get_metadata(doc)

    try:
        return float(metadata.get("direct_window_score", 0.0))
    except (TypeError, ValueError):
        return 0.0

def get_answer_score(doc):
    # Answer pattern score = may expected answer shape ba ang chunk.
    metadata = get_metadata(doc)

    try:
        return float(metadata.get("answer_pattern_score", 0.0))
    except (TypeError, ValueError):
        return 0.0



def get_retrieval_score(doc):
    # Retrieval agreement score = semantic/BM25 agreement signal.
    metadata = get_metadata(doc)

    try:
        return float(metadata.get("retrieval_agreement_score", 0.0))
    except (TypeError, ValueError):
        return 0.0


def prepare_primary_candidates(question, candidates, semantic_docs=None, bm25_docs=None):
    # Score all available candidates for final context selection.
    # Important: candidates should include reranked + semantic + BM25 docs.
    question_terms = get_question_terms(question)
    semantic_rank_map = build_rank_map(semantic_docs)
    bm25_rank_map = build_rank_map(bm25_docs)

    scored_docs = []

    for doc in remove_duplicate_docs(candidates):
        scored_doc = score_primary_evidence_doc(
            question=question,
            question_terms=question_terms,
            doc=doc,
            semantic_rank_map=semantic_rank_map,
            bm25_rank_map=bm25_rank_map,
        )
        scored_docs.append(scored_doc)

    return sort_primary_evidence_docs(scored_docs)


def trim_weak_single_fact_support(docs, top_n=3, min_keep=1, strict=False):
    # Huwag piliting punuin ang top_n kung weak na ang support chunks.
    # In strict mode, rank 1 is not automatically trusted.
    # It must contain the answer shape required by the dynamic JSON intent.
    docs = list(docs or [])

    if not docs:
        return []

    best_score = get_primary_evidence_score(docs[0])
    best_direct = get_direct_score(docs[0])
    kept_docs = []

    for index, doc in enumerate(docs, start=1):
        if len(kept_docs) >= top_n:
            break

        primary_score = get_primary_evidence_score(doc)
        direct_score = get_direct_score(doc)
        answer_score = get_answer_score(doc)
        retrieval_score = get_retrieval_score(doc)
        rerank_score = get_context_score(doc)

        keep_reason = ""

        if strict:
            # Strict single-fact rule:
            # never keep a chunk only because it is rank 1.
            # The chunk must have the expected answer evidence from JSON.
            if answer_score <= 0:
                continue

            if direct_score > 0:
                keep_reason = "strict_answer_evidence"
            elif primary_score >= max(1.0, best_score * 0.35):
                keep_reason = "strict_primary_answer_evidence"
        else:
            if index == 1:
                keep_reason = "best_primary_evidence"
            elif answer_score > 0 and primary_score >= max(1.0, best_score * 0.35):
                keep_reason = "useful_answer_pattern_support"
            elif direct_score > 0 and answer_score >= 0 and primary_score >= max(1.0, best_score * 0.45):
                keep_reason = "useful_direct_support"
            elif retrieval_score >= 4.0 and answer_score > 0 and primary_score >= max(1.0, best_score * 0.35):
                keep_reason = "semantic_bm25_agreement_answer_support"
            elif best_direct <= 1 and direct_score > 0 and answer_score >= 0:
                # Kapag weak ang top 1 direct match, rescue docs lower in ranking that have direct terms.
                keep_reason = "rescued_lower_rank_direct_evidence"
            elif rerank_score > 0 and answer_score > 0 and primary_score >= max(1.0, best_score * 0.50):
                keep_reason = "positive_rerank_answer_support"

        if keep_reason:
            doc.metadata = dict(getattr(doc, "metadata", {}) or {})
            doc.metadata["context_keep_reason"] = keep_reason
            doc.metadata["strict_single_fact_filter"] = bool(strict)
            kept_docs.append(doc)

    if len(kept_docs) >= min_keep:
        return kept_docs[:top_n]

    if strict:
        return []

    return docs[:min_keep]



# ============================================================
# 4C. DYNAMIC RELATED-CHUNK SELECTION HELPERS
# ============================================================


def get_doc_body_without_retrieval_header(doc):
    # Some ingested chunks start with "Retrieval context: title: ... language: en".
    # For evidence snippets and body matching, prefer the actual body after that header.
    body = str(getattr(doc, "page_content", "") or "")

    if "Retrieval context:" not in body[:120]:
        return body

    language_match = re.search(r"\blanguage\s*:\s*[a-z]{2}\s+", body[:600], flags=re.IGNORECASE)

    if language_match:
        return body[language_match.end():].strip()

    # Fallback: many chunks place the real text after the last metadata pipe.
    header_end = body[:600].rfind("|")

    if header_end >= 0:
        return body[header_end + 1:].strip()

    return body


def split_evidence_sentences(text):
    # Generic sentence splitter for preview/evidence only.
    # It is intentionally simple and language-agnostic enough for English/Japanese chunks.
    text = str(text or "").replace("\r", "\n")
    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return []

    parts = re.split(r"(?<=[.!?。！？])\s+|\n+", text)
    cleaned_parts = []

    for part in parts:
        part = part.strip()

        if len(part) < 20:
            continue

        cleaned_parts.append(part)

    if cleaned_parts:
        return cleaned_parts

    return [text[:500].strip()]


def get_evidence_terms_for_question(question):
    _, intent_config = get_answer_intent_config(question)

    if not isinstance(intent_config, dict):
        return [], []

    return (
        get_config_list(intent_config, "evidence_terms"),
        get_config_list(intent_config, "evidence_regex"),
    )


def score_evidence_sentence(question, sentence):
    # Score a single sentence as an answer-bearing preview.
    # Uses only question terms and JSON-configured evidence terms/regex.
    question_terms = get_question_terms(question)
    normalized_sentence = normalize_text(sentence)
    score = 0.0

    for term in question_terms:
        if term and term in normalized_sentence:
            score += 1.0

    evidence_terms, evidence_regex = get_evidence_terms_for_question(question)

    for term in evidence_terms:
        if config_term_matches(sentence, term):
            score += 2.0

    for pattern in evidence_regex:
        try:
            if re.search(pattern, sentence, flags=re.IGNORECASE):
                score += 2.0
        except re.error:
            continue

    return score


def get_best_evidence_snippet(question, doc, max_chars=360):
    # Return the best answer-bearing sentence/snippet for report/source UI.
    # This fixes misleading section labels without hardcoding any topic.
    body = get_doc_body_without_retrieval_header(doc)
    sentences = split_evidence_sentences(body)

    if not sentences:
        return ""

    best_sentence = max(
        sentences,
        key=lambda sentence: score_evidence_sentence(question, sentence),
    )

    best_sentence = re.sub(r"\s+", " ", best_sentence).strip()

    if len(best_sentence) > max_chars:
        return best_sentence[:max_chars].rstrip() + "..."

    return best_sentence


def annotate_evidence_snippet(question, doc):
    # Store a better preview for downstream reports/source drawer.
    snippet = get_best_evidence_snippet(question, doc)

    if snippet:
        doc.metadata = dict(getattr(doc, "metadata", {}) or {})
        doc.metadata["evidence_snippet"] = snippet

    return doc


def has_configured_final_answer_evidence(question, doc):
    # Strong protection for reference-like chunks.
    # If a chunk looks like bibliography/URL noise, keep it only when JSON
    # answer-evidence rules found the expected answer shape in the body.
    question_terms = get_question_terms(question)
    answer_score = get_answer_pattern_score(
        question=question,
        question_terms=question_terms,
        doc=doc,
    )
    metadata = get_metadata(doc)

    has_configured_evidence = bool(metadata.get("answer_evidence_has_evidence"))
    required_evidence_ok = metadata.get("answer_evidence_required_ok", True) is not False
    evidence_excluded = bool(metadata.get("answer_evidence_excluded"))

    return bool(
        has_configured_evidence
        and required_evidence_ok
        and not evidence_excluded
        and answer_score > 0
    )


def has_direct_final_answer_evidence(question, doc):
    # Generic final evidence check for non-reference chunks.
    # Reference-like chunks are handled more strictly by
    # has_configured_final_answer_evidence().
    if has_configured_final_answer_evidence(question, doc):
        return True

    question_terms = get_question_terms(question)
    body_text = get_doc_body_without_retrieval_header(doc)
    body_match_count = count_question_term_coverage(question_terms, body_text)
    direct_score = get_direct_window_score(question_terms, doc)

    required_body_matches = max(2, min(4, len(question_terms)))

    if direct_score >= 4 and body_match_count >= required_body_matches:
        return True

    return False


def clean_final_context_docs(question, docs, mode="single_fact", min_keep=1):
    # Last gate before the LLM.
    # Remove reference/URL/bibliography-like chunks unless they contain
    # direct answer evidence. top_n remains a maximum, not a fill target.
    docs = remove_duplicate_docs(docs)

    if not docs:
        return []

    cleaned_docs = []
    removed_count = 0

    for doc in docs:
        reference_like = is_reference_like_context_doc(doc)

        # Reference-like chunks are allowed only when they contain configured
        # direct answer evidence. This prevents bibliography/URL chunks from
        # passing because of loose question-term proximity.
        if reference_like and not has_configured_final_answer_evidence(question, doc):
            removed_count += 1
            continue

        if not reference_like and not has_direct_final_answer_evidence(question, doc):
            # Do not over-filter all modes; this is only a last-pass cleanup.
            # Non-reference chunks may still be useful support, so keep them.
            pass

        doc = annotate_evidence_snippet(question, doc)
        doc.metadata = dict(getattr(doc, "metadata", {}) or {})
        doc.metadata["final_low_value_cleanup"] = "kept"
        doc.metadata["final_low_value_cleanup_mode"] = mode
        cleaned_docs.append(doc)

    if cleaned_docs:
        for doc in cleaned_docs:
            doc.metadata = dict(getattr(doc, "metadata", {}) or {})
            doc.metadata["final_low_value_removed_count"] = removed_count

        return cleaned_docs

    non_reference_fallback_docs = [doc for doc in docs if not is_reference_like_context_doc(doc)]
    fallback_source_docs = non_reference_fallback_docs or docs
    fallback_docs = fallback_source_docs[:max(1, min_keep)]

    for doc in fallback_docs:
        doc = annotate_evidence_snippet(question, doc)
        doc.metadata = dict(getattr(doc, "metadata", {}) or {})
        doc.metadata["final_low_value_cleanup"] = "fallback_kept_to_avoid_empty_context"
        doc.metadata["final_low_value_removed_count"] = removed_count

    return fallback_docs


def get_rerank_rank(doc):
    metadata = get_metadata(doc)

    for key in ("rerank_rank", "rerank_original_rank"):
        value = metadata.get(key)

        if value is None:
            continue

        try:
            return int(value)
        except (TypeError, ValueError):
            continue

    return 999999


def get_body_question_match_count(question_terms, doc):
    return count_question_term_coverage(
        question_terms=question_terms,
        text=get_doc_body_without_retrieval_header(doc),
    )


def get_full_question_match_count(question_terms, doc):
    return count_question_term_coverage(
        question_terms=question_terms,
        text=get_doc_full_text(doc),
    )


def score_related_candidate(question, doc, semantic_rank_map=None, bm25_rank_map=None, mode="single_fact"):
    # Score whether a chunk is actually related/good enough for final context.
    # This is generic: no file names, no sample questions, no domain-specific code.
    normalized_question = normalize_text(question)
    metadata = get_metadata(doc)

    # Cache per exact question + mode during one run. This avoids re-scoring the
    # same chunk several times when the selector does cleanup passes.
    if (
        metadata.get("related_candidate_question") == normalized_question
        and metadata.get("related_candidate_mode") == mode
        and "related_candidate_score" in metadata
    ):
        return doc

    question_terms = get_question_terms(question)

    if semantic_rank_map is None:
        semantic_rank_map = {}

    if bm25_rank_map is None:
        bm25_rank_map = {}

    doc = score_primary_evidence_doc(
        question=question,
        question_terms=question_terms,
        doc=doc,
        semantic_rank_map=semantic_rank_map,
        bm25_rank_map=bm25_rank_map,
    )

    metadata = get_metadata(doc)
    answer_score = float(metadata.get("answer_pattern_score", 0.0))
    direct_score = float(metadata.get("direct_window_score", 0.0))
    retrieval_score = float(metadata.get("retrieval_agreement_score", 0.0))
    primary_score = float(metadata.get("primary_evidence_score", 0.0))
    rerank_score = get_context_score(doc)
    body_match_count = get_body_question_match_count(question_terms, doc)
    full_match_count = get_full_question_match_count(question_terms, doc)
    has_evidence = bool(metadata.get("answer_evidence_has_evidence", False))
    required_ok = bool(metadata.get("answer_evidence_required_ok", True))

    # Rerank can be negative, especially for small local models/rerankers.
    # Do not punish negative rerank too much; use it only as a small bonus when positive.
    positive_rerank_bonus = max(0.0, rerank_score) * 0.25

    # True evidence is stronger than loose term overlap. This prevents reference
    # chunks or generic same-source chunks from filling the final max top_n.
    evidence_bonus = 3.0 if (has_evidence and required_ok) else 0.0
    missing_required_penalty = -4.0 if not required_ok else 0.0

    related_score = (
        (answer_score * 1.5)
        + evidence_bonus
        + missing_required_penalty
        + (direct_score * 1.50)
        + (body_match_count * 1.50)
        + (full_match_count * 0.20)
        + (retrieval_score * 0.50)
        + positive_rerank_bonus
        + (primary_score * 0.10)
    )

    doc.metadata = dict(getattr(doc, "metadata", {}) or {})
    doc.metadata["related_body_match_count"] = int(body_match_count)
    doc.metadata["related_full_match_count"] = int(full_match_count)
    doc.metadata["related_candidate_score"] = float(related_score)
    doc.metadata["related_candidate_mode"] = mode
    doc.metadata["related_candidate_question"] = normalized_question
    doc.metadata["related_has_answer_evidence"] = bool(has_evidence and required_ok)

    # Evidence snippets are added only after final selection. Doing it here for
    # every candidate made final filtering slow.
    return doc


def sort_related_candidates(docs):
    return sorted(
        docs or [],
        key=lambda doc: (
            float(get_metadata(doc).get("related_candidate_score", 0.0)),
            float(get_metadata(doc).get("answer_pattern_score", 0.0)),
            float(get_metadata(doc).get("direct_window_score", 0.0)),
            float(get_metadata(doc).get("retrieval_agreement_score", 0.0)),
            get_context_score(doc),
            -get_original_rank(doc),
        ),
        reverse=True,
    )


def is_good_related_candidate(question, doc, mode="single_fact", rank=1, best_related_score=0.0):
    # Decide if a chunk is good enough for final context.
    # top_n is only a max; this function prevents weak chunks from filling the max.
    metadata = get_metadata(doc)
    question_terms = get_question_terms(question)
    term_count = max(1, len(question_terms))

    answer_score = float(metadata.get("answer_pattern_score", 0.0))
    direct_score = float(metadata.get("direct_window_score", 0.0))
    retrieval_score = float(metadata.get("retrieval_agreement_score", 0.0))
    related_score = float(metadata.get("related_candidate_score", 0.0))
    body_match_count = int(metadata.get("related_body_match_count", 0))
    full_match_count = int(metadata.get("related_full_match_count", 0))
    rerank_rank = get_rerank_rank(doc)
    has_evidence = bool(metadata.get("answer_evidence_has_evidence", False))
    required_ok = bool(metadata.get("answer_evidence_required_ok", True))

    # Strong answer evidence means the chunk has the expected evidence terms/regex
    # for the detected answer intent, not just repeated words from the question.
    strong_answer_evidence = has_evidence and required_ok and answer_score >= 2.0

    # References/metadata chunks can pass only when they truly contain answer evidence.
    if is_reference_like_context_doc(doc) and not strong_answer_evidence:
        return False

    # If a question-specific required evidence rule failed, do not keep the chunk.
    if not required_ok:
        return False

    if mode == "list_answer":
        direct_needed = min(2, term_count)
        body_needed = min(2, term_count)

        if strong_answer_evidence:
            return True

        if direct_score >= direct_needed and body_match_count >= 1:
            return True

        if body_match_count >= body_needed and retrieval_score >= 0.5:
            return True

        if rerank_rank <= 4 and body_match_count >= body_needed:
            return True

        if related_score >= max(3.0, best_related_score * 0.45):
            if body_match_count > 0 or direct_score > 0 or retrieval_score >= 1.0:
                return True

        return False

    # Single-fact questions need cleaner context. Max top_n can be 5, but only
    # chunks with direct answer evidence or strong term coverage should enter.
    body_needed = min(3, term_count)
    direct_needed = min(3, term_count)

    if strong_answer_evidence:
        return True

    if body_match_count >= body_needed and direct_score >= min(2, direct_needed):
        return True

    if body_match_count >= body_needed and retrieval_score >= 1.0:
        return True

    if rerank_rank <= 2 and body_match_count >= body_needed:
        return True

    # Last gentle rescue for weak retrieval cases: keep only the very best chunk,
    # and only if it has some body evidence.
    if rank == 1 and body_match_count >= max(1, min(2, term_count)):
        return True

    return False


def select_dynamic_related_docs(
    question,
    candidates,
    semantic_docs=None,
    bm25_docs=None,
    top_n=5,
    max_chars=MAX_CONTEXT_CHARS,
    mode="single_fact",
    min_keep=1,
    preserve_order_docs=None,
):
    # Main generic final-pool selector.
    # 1. Score the whole candidate pool.
    # 2. Count how many chunks are actually good/related.
    # 3. Use that count as the dynamic final size, capped by top_n/max_chars.
    #
    # This keeps the context clean for the LLM without being too strict.
    candidates = filter_low_value_context_docs(
        question=question,
        docs=remove_duplicate_docs(candidates),
        min_keep=min_keep,
    )

    if not candidates:
        return []

    semantic_rank_map = build_rank_map(semantic_docs)
    bm25_rank_map = build_rank_map(bm25_docs)

    scored_docs = []

    for doc in candidates:
        scored_docs.append(
            score_related_candidate(
                question=question,
                doc=doc,
                semantic_rank_map=semantic_rank_map,
                bm25_rank_map=bm25_rank_map,
                mode=mode,
            )
        )

    score_order = sort_related_candidates(scored_docs)
    best_related_score = float(get_metadata(score_order[0]).get("related_candidate_score", 0.0)) if score_order else 0.0
    good_keys = set()

    for rank, doc in enumerate(score_order, start=1):
        if is_good_related_candidate(
            question=question,
            doc=doc,
            mode=mode,
            rank=rank,
            best_related_score=best_related_score,
        ):
            good_keys.add(get_document_key(doc))

    if preserve_order_docs is None:
        ordered_docs = score_order
    else:
        # Preserve rerank/retrieval order when possible, but only for good docs.
        ordered_docs = []
        seen_keys = set()
        scored_by_key = {get_document_key(doc): doc for doc in scored_docs}

        for doc in preserve_order_docs:
            doc_key = get_document_key(doc)

            if doc_key in seen_keys:
                continue

            if doc_key in scored_by_key:
                ordered_docs.append(scored_by_key[doc_key])
                seen_keys.add(doc_key)

        for doc in score_order:
            doc_key = get_document_key(doc)

            if doc_key in seen_keys:
                continue

            ordered_docs.append(doc)
            seen_keys.add(doc_key)

    selected_docs = []
    seen_keys = set()
    total_chars = 0
    max_docs = max(1, safe_int(top_n, 5))

    for doc in ordered_docs:
        if len(selected_docs) >= max_docs:
            break

        doc_key = get_document_key(doc)

        if doc_key not in good_keys:
            continue

        text_length = len(str(getattr(doc, "page_content", "") or ""))

        if selected_docs and total_chars + text_length > max_chars:
            break

        if add_unique_doc(selected_docs, seen_keys, doc):
            total_chars += text_length

    if len(selected_docs) >= min_keep:
        selected_docs = selected_docs[:max_docs]
        for selected_doc in selected_docs:
            annotate_evidence_snippet(question, selected_doc)
        return selected_docs

    # Gentle fallback: if all candidates were weak, keep only the best non-reference candidate.
    # This avoids empty context but does not flood the LLM with noise.
    fallback_docs = []

    for doc in score_order:
        if is_reference_like_context_doc(doc) and get_answer_score(doc) <= 0:
            continue

        fallback_docs.append(doc)
        break

    if fallback_docs:
        for selected_doc in fallback_docs:
            annotate_evidence_snippet(question, selected_doc)
        return fallback_docs

    fallback = score_order[:max(1, min_keep)]
    for selected_doc in fallback:
        annotate_evidence_snippet(question, selected_doc)
    return fallback

def get_cross_doc_anchor_sources(question, docs):
    # For cross-doc questions, more than one source may be explicitly named.
    # Example: Treaty of Paris + Spanish-American War + Philippine-American War.
    question_terms = get_question_anchor_terms(question)
    normalized_question = normalize_text(question)
    anchors = []
    seen_sources = set()

    for doc in docs or []:
        source_key = get_source_key(doc)

        if source_key in seen_sources:
            continue

        is_strong, match_count, reason = is_strong_source_anchor(
            source_key=source_key,
            question_terms=question_terms,
            normalized_question=normalized_question,
        )

        if not is_strong:
            continue

        anchors.append(
            {
                "source_key": source_key,
                "match_count": match_count,
                "reason": reason,
                "score": get_context_score(doc),
                "original_rank": get_original_rank(doc),
            }
        )
        seen_sources.add(source_key)

    anchors.sort(
        key=lambda item: (
            item["match_count"],
            item["score"],
            -item["original_rank"],
        ),
        reverse=True,
    )

    return anchors


def is_useful_cross_doc_support(doc, selected_docs):
    # Cross-doc should prefer relevant chunks from different sources.
    # Do not fill context with completely weak chunks unless nothing else exists.
    direct_score = float(get_metadata(doc).get("cross_doc_direct_score", 0.0))
    retrieval_score = float(get_metadata(doc).get("cross_doc_retrieval_score", 0.0))
    rerank_score = float(get_metadata(doc).get("cross_doc_safe_rerank_score", 0.0))

    if direct_score > 0:
        return True

    if retrieval_score >= 1.0:
        return True

    if rerank_score > 0:
        return True

    # Allow first one if there is no selected doc yet.
    # Otherwise do not fill weak support.
    return not selected_docs


def load_mode_detection_terms(config_path=DEFAULT_CONTEXT_CONFIG_PATH):
    # Read mode detection phrases, regex patterns, and mode aliases from JSON config.
    # List/enumeration cues intentionally live in JSON, not Python.
    raw_config = read_context_json(config_path=config_path)
    mode_config = raw_config.get("mode_detection", {}) if isinstance(raw_config, dict) else {}

    if not isinstance(mode_config, dict):
        mode_config = {}

    mode_aliases = mode_config.get("mode_aliases", {})

    if not isinstance(mode_aliases, dict):
        mode_aliases = {}

    cross_doc_phrases = list(mode_config.get("cross_doc_phrases", []))
    comparison_phrases = list(mode_config.get("comparison_phrases", []))
    false_premise_phrases = list(mode_config.get("false_premise_phrases", []))
    false_premise_patterns = get_config_list(mode_config, "false_premise_patterns")
    list_phrases = list(mode_config.get("list_phrases", []))
    list_patterns = get_config_list(mode_config, "list_patterns")

    return {
        "cross_doc_phrases": normalize_config_list(cross_doc_phrases),
        "comparison_phrases": normalize_config_list(comparison_phrases),
        "false_premise_phrases": normalize_config_list(false_premise_phrases),
        "false_premise_patterns": false_premise_patterns,
        "list_phrases": normalize_config_list(list_phrases),
        "list_patterns": list_patterns,
        "cross_doc_modes": normalize_config_list(mode_aliases.get("cross_doc", ["cross_doc", "cross document"])),
        "comparison_modes": normalize_config_list(mode_aliases.get("comparison", ["comparison", "compare"])),
        "negative_modes": normalize_config_list(mode_aliases.get("negative", ["negative", "no_answer", "unsupported"])),
        "false_premise_modes": normalize_config_list(mode_aliases.get("false_premise", ["false_premise", "false premise"])),
        "list_modes": normalize_config_list(mode_aliases.get("list_answer", ["list_answer", "list_enumeration", "list", "enumeration"])),
    }


def phrase_exists(normalized_question, phrase):
    # Exact phrase boundary match.
    normalized_question = f" {normalize_text(normalized_question)} "
    phrase = normalize_text(phrase)

    if not phrase:
        return False

    return f" {phrase} " in normalized_question


def has_any_phrase(normalized_question, phrases):
    for phrase in phrases:
        if phrase_exists(normalized_question, phrase):
            return True

    return False


def has_wildcard_phrase_match(normalized_question, pattern):
    # Supports JSON wildcard patterns like "why did * become".
    # This keeps false-premise detection configurable without hardcoding sample topics.
    pattern = str(pattern or "").strip()

    if "*" not in pattern:
        return False

    parts = [normalize_text(part) for part in pattern.split("*")]
    parts = [part for part in parts if part]

    if not parts:
        return False

    cursor = 0

    for part in parts:
        index = normalized_question.find(part, cursor)

        if index < 0:
            return False

        cursor = index + len(part)

    return True


def has_any_wildcard_or_regex(normalized_question, patterns):
    # JSON false-premise patterns may be wildcard-style or regex-style.
    for pattern in patterns or []:
        pattern_text = str(pattern or "").strip()

        if not pattern_text:
            continue

        if "*" in pattern_text and has_wildcard_phrase_match(normalized_question, pattern_text):
            return True

        try:
            if re.search(pattern_text, normalized_question, flags=re.IGNORECASE):
                return True
        except re.error:
            continue

    return False


def has_explicit_cross_doc_cue(question):
    # Cross-doc must come from explicit compare/connect/between phrases.
    # Broad standalone tokens should not force cross-doc mode.
    normalized_question = normalize_text(question)
    mode_terms = load_mode_detection_terms()

    phrases = []
    phrases.extend(mode_terms["comparison_phrases"])
    phrases.extend(mode_terms["cross_doc_phrases"])

    return has_any_phrase(normalized_question, phrases)


def should_force_single_fact_mode(question, detected_mode):
    # Query analyzer may mark broad words as cross_doc.
    # For final context, cross-doc should require a strong comparison/connection cue.
    if detected_mode not in {"cross_doc", "comparison"}:
        return False

    return not has_explicit_cross_doc_cue(question)


def select_best_source_only_context(scored_docs, top_n=3, strict=False):
    # For single-fact questions, once the best answer-bearing source is found,
    # do not add other sources just to fill the context.
    # In strict mode, if the best source has no answer evidence, scan all candidates
    # for the best answer-bearing chunk instead of forcing the top source.
    scored_docs = list(scored_docs or [])

    if not scored_docs:
        return []

    best_doc = scored_docs[0]
    best_source_key = get_source_key(best_doc)

    same_source_docs = []

    for doc in scored_docs:
        if get_source_key(doc) == best_source_key:
            same_source_docs.append(doc)

    selected_docs = trim_weak_single_fact_support(
        docs=same_source_docs,
        top_n=top_n,
        min_keep=1,
        strict=strict,
    )

    if selected_docs or not strict:
        return selected_docs

    # Strict fallback: keep only chunks with JSON-based answer evidence.
    # This is still generic; it does not check file names, topics, or chunk IDs.
    return trim_weak_single_fact_support(
        docs=scored_docs,
        top_n=top_n,
        min_keep=1,
        strict=True,
    )

def has_positive_rerank_signal(docs, min_score=0.0):
    # Positive rerank score means the reranker found at least one usable direct candidate.
    # Kapag lahat negative, mas safe bumalik sa primary evidence scoring.
    for doc in docs or []:
        rerank_score = get_rerank_score(doc)

        if rerank_score is not None and rerank_score >= min_score:
            return True

    return False


def is_primary_evidence_protected_intent(intent):
    # These answer types need exact answer-shape evidence from JSON.
    # Example: birthdate/date questions must prefer chunks with "born on" style evidence,
    # not merely chunks that have a high rerank score.
    normalized_intent = normalize_text(intent)
    protected_markers = {
        "date",
        "deadline",
        "time",
    }

    for marker in protected_markers:
        if marker in normalized_intent:
            return True

    return False


def should_prefer_top_rerank_source(dynamic_settings, docs):
    # Dynamic rule by question type:
    # - list/multi-answer questions: use reranked evidence, not strict primary-evidence trimming
    # - exact date/definition-like questions: keep strict primary evidence
    # - other actor/fact/judgment questions: trust the top reranked source first,
    #   then keep only focused support from that same source.
    docs = list(docs or [])

    if not docs:
        return False

    intent = dynamic_settings.get("intent", "")

    if dynamic_settings.get("multi_answer"):
        return True

    if is_primary_evidence_protected_intent(intent):
        return False

    return True


def sort_docs_by_retrieval_priority(docs):
    # For focused single-answer support, prefer docs that retrieval/RRF already placed high.
    # This prevents low-rerank but high-evidence-looking chunks from replacing the actual source.
    return sorted(
        docs or [],
        key=lambda doc: (
            -get_original_rank(doc),
            get_context_score(doc),
        ),
        reverse=True,
    )


def select_top_rerank_source_context(
    reranked_docs,
    question,
    semantic_docs=None,
    bm25_docs=None,
    top_n=2,
    max_chars=MAX_CONTEXT_CHARS,
    multi_answer=False,
):
    # Use the top reranked source as the anchor, but do not force-fill to top_n.
    # The final size is based on how many chunks in the pool are actually related/good.
    docs = sort_docs_by_rerank_score(remove_duplicate_docs(reranked_docs))

    if not docs:
        return []

    top_doc = docs[0]
    top_source_key = get_source_key(top_doc)

    candidate_docs = remove_duplicate_docs(
        list(docs or []) + list(semantic_docs or []) + list(bm25_docs or [])
    )

    source_pool = [
        doc for doc in candidate_docs
        if get_source_key(doc) == top_source_key
    ]

    if not source_pool:
        source_pool = [top_doc]

    mode = "list_answer" if multi_answer else "single_fact"
    selected_docs = select_dynamic_related_docs(
        question=question,
        candidates=source_pool,
        semantic_docs=semantic_docs,
        bm25_docs=bm25_docs,
        top_n=top_n,
        max_chars=max_chars,
        mode=mode,
        min_keep=1,
        preserve_order_docs=docs,
    )

    if not selected_docs:
        selected_docs = [top_doc]

    keep_reason = "dynamic_related_top_source"
    filter_scope = (
        "list_answer_dynamic_related_top_source"
        if multi_answer
        else "single_fact_dynamic_related_top_source"
    )

    annotate_context_docs(
        docs=selected_docs,
        mode=mode,
        anchor_source_key=top_source_key,
        anchor_reason="dynamic_top_rerank_source_related_pool",
        filter_scope=filter_scope,
        keep_reason=keep_reason,
    )

    for doc in selected_docs:
        doc.metadata = dict(getattr(doc, "metadata", {}) or {})
        doc.metadata["dynamic_related_pool_size"] = len(source_pool)
        doc.metadata["dynamic_related_selected_count"] = len(selected_docs)

    return limit_context_docs(
        docs=selected_docs,
        max_chars=max_chars,
        max_per_source=None,
    )


def get_anchor_source_keys(question, docs):
    # Reuse the same strong source-anchor rules.
    # Kapag more than one source/entity ang nasa question, huwag mag-lock sa isang source lang.
    anchors = get_cross_doc_anchor_sources(question=question, docs=docs)
    return [anchor["source_key"] for anchor in anchors]


def add_unique_doc(selected_docs, seen_keys, doc):
    # Add one doc while preserving order and avoiding duplicates.
    doc_key = get_document_key(doc)

    if doc_key in seen_keys:
        return False

    seen_keys.add(doc_key)
    selected_docs.append(doc)
    return True


def annotate_context_docs(
    docs,
    mode,
    anchor_source_key,
    anchor_reason,
    filter_scope,
    keep_reason,
):
    # Save context-selection metadata for debug reports.
    for doc in docs or []:
        doc.metadata = dict(getattr(doc, "metadata", {}) or {})
        doc.metadata["context_mode"] = mode
        doc.metadata["context_anchor_source"] = anchor_source_key or "none"
        doc.metadata["context_anchor_reason"] = anchor_reason
        doc.metadata["context_confident_filter_scope"] = filter_scope

        if keep_reason and not doc.metadata.get("context_keep_reason"):
            doc.metadata["context_keep_reason"] = keep_reason

    return docs


def select_rerank_first_context(
    reranked_docs,
    question,
    semantic_docs=None,
    bm25_docs=None,
    top_n=3,
    max_chars=MAX_CONTEXT_CHARS,
):
    # Normal single_fact path.
    # top_n is a maximum only. Final docs are selected from the candidate pool
    # based on generic related/evidence signals.
    docs = remove_duplicate_docs(reranked_docs)
    docs = sort_docs_by_rerank_score(docs)

    candidate_docs = remove_duplicate_docs(
        list(docs or []) + list(semantic_docs or []) + list(bm25_docs or [])
    )
    candidate_docs = filter_low_value_context_docs(
        question=question,
        docs=candidate_docs,
        min_keep=1,
    )

    if not candidate_docs:
        return []

    anchor_source_key, anchor_reason = choose_single_fact_anchor_source(
        docs=candidate_docs,
        question=question,
    )
    anchor_source_keys = get_anchor_source_keys(question=question, docs=candidate_docs)

    if len(anchor_source_keys) >= 2:
        anchor_source_key = None
        anchor_reason = "multiple_source_anchors_no_single_source_lock"

    if anchor_source_key:
        pool = [doc for doc in candidate_docs if get_source_key(doc) == anchor_source_key]
        preserve_order_docs = [doc for doc in docs if get_source_key(doc) == anchor_source_key]
        filter_scope = "single_fact_dynamic_related_anchor_source"
    else:
        pool = candidate_docs
        preserve_order_docs = docs
        filter_scope = "single_fact_dynamic_related_no_source_lock"

    selected_docs = select_dynamic_related_docs(
        question=question,
        candidates=pool,
        semantic_docs=semantic_docs,
        bm25_docs=bm25_docs,
        top_n=top_n,
        max_chars=max_chars,
        mode="single_fact",
        min_keep=1,
        preserve_order_docs=preserve_order_docs,
    )

    if not selected_docs:
        selected_docs = candidate_docs[:1]
        anchor_reason = "fallback_top_available"
        filter_scope = "fallback_top_available"

    annotate_context_docs(
        docs=selected_docs,
        mode="single_fact",
        anchor_source_key=anchor_source_key or ", ".join(anchor_source_keys),
        anchor_reason=anchor_reason,
        filter_scope=filter_scope,
        keep_reason="dynamic_related_evidence",
    )

    return limit_context_docs(
        docs=selected_docs,
        max_chars=max_chars,
        max_per_source=None,
    )


def select_safety_context(
    reranked_docs,
    question,
    semantic_docs=None,
    bm25_docs=None,
    top_n=2,
    max_chars=MAX_CONTEXT_CHARS,
    mode="safety",
):
    # Strict path for negative / unsupported / false-premise questions.
    #
    # Rule:
    # - Small focused evidence only.
    # - No neighbor expansion.
    # - No rerank-first filling.
    # - Do not lock to a single source when multiple entities/sources are in the premise.
    # Build the candidate pool from reranked/semantic results first.
    # BM25 is still used for scoring agreement, but BM25-only chunks are not allowed
    # to enter the final list unless reranked/semantic retrieval already surfaced them.
    # This prevents keyword-only matches from filling list answers with weak chunks.
    candidates = remove_duplicate_docs(
        list(reranked_docs or []) + list(semantic_docs or [])
    )

    if not candidates:
        # Safe fallback only when rerank/semantic produced nothing.
        candidates = remove_duplicate_docs(list(bm25_docs or []))

    candidates = filter_low_value_context_docs(
        question=question,
        docs=candidates,
        min_keep=1,
    )

    if not candidates:
        return []

    scored_docs = prepare_primary_candidates(
        question=question,
        candidates=candidates,
        semantic_docs=semantic_docs,
        bm25_docs=bm25_docs,
    )

    selected_docs = trim_weak_single_fact_support(
        docs=scored_docs,
        top_n=top_n,
        min_keep=1,
    )

    anchor_source_keys = get_anchor_source_keys(question=question, docs=candidates)
    anchor_source_label = ", ".join(anchor_source_keys) if anchor_source_keys else "none"

    annotate_context_docs(
        docs=selected_docs,
        mode=mode,
        anchor_source_key=anchor_source_label,
        anchor_reason="strict_safety_primary_evidence",
        filter_scope="negative_false_premise_safety",
        keep_reason="strict_safety_evidence",
    )

    return limit_context_docs(
        docs=selected_docs,
        max_chars=max_chars,
        max_per_source=None,
    )



def get_dynamic_list_context_limit(question, docs, max_chars=MAX_CONTEXT_CHARS, top_n=None):
    # Dynamic list limit.
    # top_n is a maximum only; weak chunks are not added just to fill it.
    docs = list(docs or [])

    if not docs:
        return 0

    max_docs = safe_int(top_n, 5)

    if max_docs < 1:
        max_docs = 1

    useful_docs = []

    for doc in docs:
        if is_low_value_context_doc(question, doc):
            continue

        if has_list_answer_evidence(question, doc):
            useful_docs.append(doc)

    if not useful_docs:
        useful_docs = docs[:1]

    total_chars = 0
    dynamic_count = 0

    for doc in useful_docs:
        if dynamic_count >= max_docs:
            break

        text_length = len(str(getattr(doc, "page_content", "") or ""))

        if dynamic_count > 0 and total_chars + text_length > max_chars:
            break

        total_chars += text_length
        dynamic_count += 1

    return max(dynamic_count, 1)


def get_list_answer_doc_score(question, doc, semantic_rank_map, bm25_rank_map):
    # Dynamic score for list/list-style questions.
    # Generic: combines answer-shape, direct terms, retrieval agreement, and rerank.
    question_terms = get_question_terms(question)

    scored_doc = score_primary_evidence_doc(
        question=question,
        question_terms=question_terms,
        doc=doc,
        semantic_rank_map=semantic_rank_map,
        bm25_rank_map=bm25_rank_map,
    )

    metadata = get_metadata(scored_doc)
    rerank_score = get_context_score(scored_doc)
    primary_score = float(metadata.get("primary_evidence_score", 0.0))
    direct_score = float(metadata.get("direct_window_score", 0.0))
    answer_score = float(metadata.get("answer_pattern_score", 0.0))
    retrieval_score = float(metadata.get("retrieval_agreement_score", 0.0))

    list_score = (
        primary_score
        + direct_score
        + answer_score
        + retrieval_score
        + (rerank_score * 0.10)
    )

    scored_doc.metadata["list_answer_score"] = float(list_score)
    return scored_doc


def sort_list_answer_docs(docs):
    return sorted(
        docs or [],
        key=lambda doc: (
            float(get_metadata(doc).get("list_answer_score", 0.0)),
            get_context_score(doc),
            -get_original_rank(doc),
        ),
        reverse=True,
    )


def select_list_answer_context(
    reranked_docs,
    question,
    semantic_docs=None,
    bm25_docs=None,
    top_n=5,
    max_chars=MAX_CONTEXT_CHARS,
):
    # Dynamic selector for list/list-style questions.
    # The number of final chunks is based on how many related/good chunks exist
    # in the pool, capped by top_n. BM25-only chunks may be used only when they
    # pass the same generic evidence/relatedness checks.
    candidates = remove_duplicate_docs(
        list(reranked_docs or []) + list(semantic_docs or []) + list(bm25_docs or [])
    )
    candidates = filter_low_value_context_docs(
        question=question,
        docs=candidates,
        min_keep=1,
    )

    if not candidates:
        return []

    semantic_rank_map = build_rank_map(semantic_docs)
    bm25_rank_map = build_rank_map(bm25_docs)

    scored_docs = []

    for doc in candidates:
        scored_docs.append(
            get_list_answer_doc_score(
                question=question,
                doc=doc,
                semantic_rank_map=semantic_rank_map,
                bm25_rank_map=bm25_rank_map,
            )
        )
        score_related_candidate(
            question=question,
            doc=doc,
            semantic_rank_map=semantic_rank_map,
            bm25_rank_map=bm25_rank_map,
            mode="list_answer",
        )

    scored_docs = sort_list_answer_docs(scored_docs)

    source_scores = defaultdict(float)
    source_counts = defaultdict(int)

    for doc in scored_docs:
        source_key = get_source_key(doc)
        list_score = float(get_metadata(doc).get("list_answer_score", 0.0))
        related_score = float(get_metadata(doc).get("related_candidate_score", 0.0))
        score = list_score + related_score

        if score <= 0:
            continue

        source_scores[source_key] += score
        source_counts[source_key] += 1

    if source_scores:
        anchor_source_key = max(
            source_scores,
            key=lambda key: (source_scores[key], source_counts[key]),
        )
        source_pool = [
            doc for doc in scored_docs
            if get_source_key(doc) == anchor_source_key
        ]
        anchor_reason = "dynamic_list_answer_best_related_source_cluster"
    else:
        anchor_source_key = get_source_key(scored_docs[0])
        source_pool = [
            doc for doc in scored_docs
            if get_source_key(doc) == anchor_source_key
        ]
        anchor_reason = "dynamic_list_answer_fallback_top_source"

    preserve_order_docs = remove_duplicate_docs(list(reranked_docs or []) + list(semantic_docs or []) + list(bm25_docs or []))

    selected_docs = select_dynamic_related_docs(
        question=question,
        candidates=source_pool,
        semantic_docs=semantic_docs,
        bm25_docs=bm25_docs,
        top_n=top_n,
        max_chars=max_chars,
        mode="list_answer",
        min_keep=1,
        preserve_order_docs=preserve_order_docs,
    )

    if not selected_docs:
        selected_docs = scored_docs[:1]

    annotate_context_docs(
        docs=selected_docs,
        mode="list_answer",
        anchor_source_key=anchor_source_key,
        anchor_reason=anchor_reason,
        filter_scope="dynamic_list_answer_related_source_cluster",
        keep_reason="dynamic_list_answer_related_evidence",
    )

    for doc in selected_docs:
        doc.metadata = dict(getattr(doc, "metadata", {}) or {})
        doc.metadata["dynamic_multi_answer"] = True
        doc.metadata["dynamic_selection_strategy"] = "dynamic_related_list_answer"
        doc.metadata["dynamic_related_pool_size"] = len(source_pool)
        doc.metadata["dynamic_related_selected_count"] = len(selected_docs)

    return limit_context_docs(
        docs=selected_docs,
        max_chars=max_chars,
        max_per_source=None,
    )


def select_single_fact_context(
    reranked_docs,
    question,
    semantic_docs=None,
    bm25_docs=None,
    all_chunks=None,
    enable_neighbor_expansion=ENABLE_NEIGHBOR_EXPANSION,
    neighbor_window=NEIGHBOR_WINDOW,
    top_n=3,
    max_chars=MAX_CONTEXT_CHARS,
):
    # Dynamic direct factual question selection.
    #
    # The final behavior depends on answer shape:
    # - normal single-fact and list-style questions can keep up to the dynamic top_n
    # - list/multi-answer questions use rerank-first and no neighbor expansion
    # - false-premise questions stay strict through the separate false_premise mode
    #   so normal direct facts do not get over-trimmed.
    dynamic_settings = get_dynamic_single_fact_settings(
        question=question,
        requested_top_n=top_n,
    )

    intent = dynamic_settings["intent"]
    top_n = dynamic_settings["top_n"]
    selection_strategy = dynamic_settings["selection_strategy"]
    is_multi_answer = bool(dynamic_settings.get("multi_answer"))
    prefer_top_rerank_source = should_prefer_top_rerank_source(
        dynamic_settings=dynamic_settings,
        docs=reranked_docs,
    )

    use_neighbor_expansion = (
        enable_neighbor_expansion
        and dynamic_settings["enable_neighbor_expansion"]
        and not is_multi_answer
        and not prefer_top_rerank_source
    )

    docs = remove_duplicate_docs(reranked_docs)
    docs = filter_low_value_context_docs(
        question=question,
        docs=docs,
        min_keep=1,
    )
    docs = sort_docs_by_rerank_score(docs)

    semantic_docs = filter_low_value_context_docs(
        question=question,
        docs=semantic_docs,
        min_keep=1,
    ) if semantic_docs else semantic_docs
    bm25_docs = filter_low_value_context_docs(
        question=question,
        docs=bm25_docs,
        min_keep=1,
    ) if bm25_docs else bm25_docs

    if not docs and not semantic_docs and not bm25_docs:
        return []

    # Dynamic source-first path for actor/judgment/list-style facts.
    # This prevents primary-evidence scoring from replacing correct top reranked chunks
    # with generic chunks that only contain loose evidence terms.
    if prefer_top_rerank_source:
        # Use the dynamic top_n for both normal single-fact and multi-answer cases.
        # Only false_premise remains strict through its separate mode path.
        effective_top_n = top_n
        selected_docs = select_top_rerank_source_context(
            reranked_docs=docs,
            question=question,
            semantic_docs=semantic_docs,
            bm25_docs=bm25_docs,
            top_n=effective_top_n,
            max_chars=max_chars,
            multi_answer=is_multi_answer,
        )

    elif selection_strategy == "primary_evidence":
        scored_docs = prepare_primary_candidates(
            question=question,
            candidates=list(docs or []) + list(semantic_docs or []) + list(bm25_docs or []),
            semantic_docs=semantic_docs,
            bm25_docs=bm25_docs,
        )

        selected_docs = select_best_source_only_context(
            scored_docs=scored_docs,
            top_n=top_n,
            strict=dynamic_settings["strict"],
        )

        if not selected_docs and not dynamic_settings["strict"]:
            selected_docs = scored_docs[:1]

        annotate_context_docs(
            docs=selected_docs,
            mode="single_fact",
            anchor_source_key=get_source_key(selected_docs[0]) if selected_docs else "primary_evidence",
            anchor_reason=f"dynamic_{intent}_primary_evidence",
            filter_scope="dynamic_single_fact_primary_evidence",
            keep_reason="dynamic_primary_evidence",
        )

        selected_docs = limit_context_docs(
            docs=selected_docs,
            max_chars=max_chars,
            max_per_source=None,
        )

    elif has_positive_rerank_signal(docs):
        selected_docs = select_rerank_first_context(
            reranked_docs=docs,
            question=question,
            semantic_docs=semantic_docs,
            bm25_docs=bm25_docs,
            top_n=top_n,
            max_chars=max_chars,
        )

    else:
        scored_docs = prepare_primary_candidates(
            question=question,
            candidates=list(docs or []) + list(semantic_docs or []) + list(bm25_docs or []),
            semantic_docs=semantic_docs,
            bm25_docs=bm25_docs,
        )

        selected_docs = select_best_source_only_context(
            scored_docs=scored_docs,
            top_n=top_n,
            strict=dynamic_settings["strict"],
        )

        if not selected_docs and not dynamic_settings["strict"]:
            selected_docs = scored_docs[:1]

        annotate_context_docs(
            docs=selected_docs,
            mode="single_fact",
            anchor_source_key=get_source_key(selected_docs[0]) if selected_docs else "primary_evidence",
            anchor_reason=f"dynamic_{intent}_low_confidence_primary_evidence",
            filter_scope="dynamic_single_fact_low_confidence_primary_evidence",
            keep_reason="dynamic_low_confidence_primary_evidence",
        )

        selected_docs = limit_context_docs(
            docs=selected_docs,
            max_chars=max_chars,
            max_per_source=None,
        )

    # Final generic cleanup: top_n is a maximum, not a target.
    # Remove weak fill chunks that are not related enough to the question.
    selected_docs = select_dynamic_related_docs(
        question=question,
        candidates=selected_docs,
        semantic_docs=semantic_docs,
        bm25_docs=bm25_docs,
        top_n=top_n,
        max_chars=max_chars,
        mode="list_answer" if is_multi_answer else "single_fact",
        min_keep=1,
        preserve_order_docs=selected_docs,
    )

    if use_neighbor_expansion and all_chunks and neighbor_window > 0:
        selected_docs = expand_neighbor_chunks(
            selected_docs=selected_docs,
            all_chunks=all_chunks,
            window=neighbor_window,
        )

    for doc in selected_docs or []:
        doc.metadata = dict(getattr(doc, "metadata", {}) or {})
        doc.metadata["dynamic_answer_intent"] = intent
        doc.metadata["dynamic_single_fact_top_n"] = top_n
        doc.metadata["dynamic_neighbor_expansion"] = bool(use_neighbor_expansion)
        doc.metadata["dynamic_selection_strategy"] = selection_strategy
        doc.metadata["dynamic_strict_single_fact"] = bool(dynamic_settings["strict"])
        doc.metadata["dynamic_multi_answer"] = bool(is_multi_answer)
        doc.metadata["dynamic_prefer_top_rerank_source"] = bool(prefer_top_rerank_source)

    return limit_context_docs(
        docs=selected_docs,
        max_chars=max_chars,
        max_per_source=None,
    )

# ============================================================
# 5. CROSS DOCUMENT CONTEXT
# ============================================================


def get_rerank_score(doc):
    # Kunin lang ang real reranker score.
    # Important:
    # - Huwag ihalo dito ang Chroma distance score.
    # - Ang Chroma distance ay lower-is-better, pero rerank_score ay higher-is-better.
    metadata = get_metadata(doc)
    value = metadata.get("rerank_score")

    if value is None:
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None



def keep_useful_cross_doc_candidates(docs, min_rerank_score=0.0):
    # Tanggalin ang weak reranked chunks kapag may usable positive chunks naman.
    # Generic ito. Hindi naka-base sa source name, topic, or keyword.
    #
    # Bakit kailangan?
    # - Kapag may positive rerank_score, ibig sabihin may malinaw na useful chunks.
    # - Kung ganun, huwag isama ang negative rerank_score chunks just to fill top_n.
    # - Pero kapag lahat negative or walang rerank_score, fallback pa rin sa original docs.
    docs = list(docs or [])

    if not docs:
        return []

    has_positive_rerank = False

    for doc in docs:
        rerank_score = get_rerank_score(doc)

        if rerank_score is not None and rerank_score >= min_rerank_score:
            has_positive_rerank = True
            break

    if not has_positive_rerank:
        return docs

    kept_docs = []

    for doc in docs:
        rerank_score = get_rerank_score(doc)

        # Keep docs na hindi pa na-rerank dahil baka galing sila sa semantic/BM25
        # at useful as support candidate.
        if rerank_score is None:
            kept_docs.append(doc)
            continue

        if rerank_score >= min_rerank_score:
            kept_docs.append(doc)

    if kept_docs:
        return kept_docs

    return docs



def score_cross_doc_candidate(question_terms, doc, semantic_rank_map, bm25_rank_map):
    # Simple and stable candidate score for cross-doc.
    #
    # Priority:
    # 1. Real rerank_score kapag meron.
    # 2. Retrieval agreement kapag same chunk nakita ng semantic at BM25.
    # 3. Direct window score as small tie-breaker only.
    #
    # Hindi ito hardcoded sa specific topic/source.
    rerank_score = get_rerank_score(doc)

    if rerank_score is None:
        safe_rerank_score = 0.0
    else:
        safe_rerank_score = rerank_score

    retrieval_score, semantic_rank, bm25_rank = get_retrieval_agreement_score(
        doc=doc,
        semantic_rank_map=semantic_rank_map,
        bm25_rank_map=bm25_rank_map,
    )
    direct_score = get_direct_window_score(
        question_terms=question_terms,
        doc=doc,
    )

    doc.metadata = dict(getattr(doc, "metadata", {}) or {})
    doc.metadata["cross_doc_safe_rerank_score"] = float(safe_rerank_score)
    doc.metadata["cross_doc_retrieval_score"] = float(retrieval_score)
    doc.metadata["cross_doc_direct_score"] = float(direct_score)
    doc.metadata["semantic_rank_for_cross_doc"] = semantic_rank
    doc.metadata["bm25_rank_for_cross_doc"] = bm25_rank

    return doc



def sort_cross_doc_candidates(docs):
    # Sort by stable signals.
    # safe_rerank_score is first so negative reranked chunks do not beat useful chunks.
    return sorted(
        docs or [],
        key=lambda doc: (
            float(get_metadata(doc).get("cross_doc_safe_rerank_score", 0.0)),
            float(get_metadata(doc).get("cross_doc_retrieval_score", 0.0)),
            float(get_metadata(doc).get("cross_doc_direct_score", 0.0)),
            -get_original_rank(doc),
        ),
        reverse=True,
    )



def select_cross_doc_context(
    reranked_docs,
    question="",
    semantic_docs=None,
    bm25_docs=None,
    top_n=5,
    max_chars=MAX_CONTEXT_CHARS,
    max_per_source=MAX_PER_SOURCE,
):
    # Para sa cross-document/comparison questions.
    #
    # Final context rule:
    # - Do not source-first lock to top 1.
    # - Use multiple explicitly named sources when present.
    # - If explicit sources exist, do not add unrelated sources just to fill top_n.
    # - Fill only with useful support; top_n is maximum only.
    candidate_docs = remove_duplicate_docs(
        list(reranked_docs or []) + list(semantic_docs or []) + list(bm25_docs or [])
    )
    candidate_docs = filter_low_value_context_docs(
        question=question,
        docs=candidate_docs,
        min_keep=1,
    )

    if not candidate_docs:
        return []

    candidate_docs = keep_useful_cross_doc_candidates(candidate_docs)

    question_terms = get_question_terms(question)
    semantic_rank_map = build_rank_map(semantic_docs)
    bm25_rank_map = build_rank_map(bm25_docs)

    scored_docs = []

    for doc in candidate_docs:
        scored_doc = score_cross_doc_candidate(
            question_terms=question_terms,
            doc=doc,
            semantic_rank_map=semantic_rank_map,
            bm25_rank_map=bm25_rank_map,
        )
        scored_docs.append(scored_doc)

    scored_docs = sort_cross_doc_candidates(scored_docs)
    anchor_sources = get_cross_doc_anchor_sources(question=question, docs=scored_docs)
    anchor_source_keys = [item["source_key"] for item in anchor_sources]
    anchor_source_key_set = set(anchor_source_keys)
    anchor_reason = "multi_source_question_anchor" if anchor_sources else "no_explicit_source_anchor"

    selected_docs = []
    selected_doc_keys = set()
    source_counts = defaultdict(int)

    def add_doc(doc, filter_scope):
        doc_key = get_document_key(doc)

        if doc_key in selected_doc_keys:
            return False

        source_key = get_source_key(doc)

        if max_per_source is not None and source_counts[source_key] >= max_per_source:
            return False

        if not is_useful_cross_doc_support(doc, selected_docs):
            return False

        doc.metadata = dict(getattr(doc, "metadata", {}) or {})
        doc.metadata["context_mode"] = "cross_doc"
        doc.metadata["context_anchor_source"] = ", ".join(anchor_source_keys) if anchor_source_keys else "none"
        doc.metadata["context_anchor_reason"] = anchor_reason
        doc.metadata["context_confident_filter_scope"] = filter_scope

        selected_docs.append(doc)
        selected_doc_keys.add(doc_key)
        source_counts[source_key] += 1
        return True

    # Pass 1: one best chunk from every explicit source mentioned in the question.
    for anchor in anchor_sources:
        if len(selected_docs) >= top_n:
            break

        anchor_source_key = anchor["source_key"]

        for doc in scored_docs:
            if get_source_key(doc) != anchor_source_key:
                continue

            if add_doc(doc, "cross_doc_explicit_anchor_source"):
                break

    if anchor_source_keys:
        # Pass 2A: if explicit anchors exist, only add extra support from those same anchor sources.
        for doc in scored_docs:
            if len(selected_docs) >= top_n:
                break

            source_key = get_source_key(doc)

            if source_key not in anchor_source_key_set:
                continue

            add_doc(doc, "cross_doc_anchor_extra_support")
    else:
        # Pass 2B: no explicit anchors, allow one best chunk from each useful source.
        for doc in scored_docs:
            if len(selected_docs) >= top_n:
                break

            source_key = get_source_key(doc)

            if source_counts[source_key] > 0:
                continue

            add_doc(doc, "cross_doc_diversified_support")

        # Pass 3: optional extra support, still respecting max_per_source.
        for doc in scored_docs:
            if len(selected_docs) >= top_n:
                break

            add_doc(doc, "cross_doc_extra_useful_support")

    if not selected_docs:
        selected_docs = scored_docs[:min(top_n, len(scored_docs))]

        for doc in selected_docs:
            doc.metadata = dict(getattr(doc, "metadata", {}) or {})
            doc.metadata["context_mode"] = "cross_doc"
            doc.metadata["context_anchor_source"] = "fallback"
            doc.metadata["context_anchor_reason"] = "fallback_top_scored_docs"
            doc.metadata["context_confident_filter_scope"] = "fallback_top_scored_docs"

    final_docs = limit_context_docs(
        docs=selected_docs,
        max_chars=max_chars,
        max_per_source=max_per_source,
    )

    return final_docs

# ============================================================
# 6. MAIN FUNCTION
# ============================================================

def select_final_context_docs(
    reranked_docs,
    question,
    semantic_docs=None,
    bm25_docs=None,
    all_chunks=None,
    enable_neighbor_expansion=ENABLE_NEIGHBOR_EXPANSION,
    neighbor_window=NEIGHBOR_WINDOW,
    top_n=None,
    max_chars=MAX_CONTEXT_CHARS,
    max_per_source=MAX_PER_SOURCE,
    debug=True,
):
    # Main function after reranker.
    # Treatment depends on mode:
    # - single_fact: source/entity anchor or primary evidence fallback
    # - cross_doc/comparison: diverse supporting sources
    # - negative/false_premise: small evidence set only
    mode = detect_question_mode(question)

    if should_force_single_fact_mode(question, mode):
        mode = "single_fact"

    policy = get_dynamic_context_policy(
        question=question,
        mode=mode,
        top_n=top_n,
        max_chars=max_chars,
        max_per_source=max_per_source,
    )
    final_top_n = policy["top_n"]
    final_max_chars = policy["max_chars"]
    final_max_per_source = policy["max_per_source"]
    final_neighbor_expansion = bool(enable_neighbor_expansion and policy["neighbor_expansion"])

    if mode == "cross_doc":
        final_docs = select_cross_doc_context(
            reranked_docs=reranked_docs,
            question=question,
            semantic_docs=semantic_docs,
            bm25_docs=bm25_docs,
            top_n=final_top_n,
            max_chars=final_max_chars,
            max_per_source=final_max_per_source,
        )
    elif mode == "comparison":
        final_docs = select_cross_doc_context(
            reranked_docs=reranked_docs,
            question=question,
            semantic_docs=semantic_docs,
            bm25_docs=bm25_docs,
            top_n=final_top_n,
            max_chars=final_max_chars,
            max_per_source=final_max_per_source,
        )
    elif mode == "negative":
        final_docs = select_safety_context(
            reranked_docs=reranked_docs,
            question=question,
            semantic_docs=semantic_docs,
            bm25_docs=bm25_docs,
            top_n=final_top_n,
            max_chars=final_max_chars,
            mode="negative",
        )
    elif mode == "false_premise":
        final_docs = select_safety_context(
            reranked_docs=reranked_docs,
            question=question,
            semantic_docs=semantic_docs,
            bm25_docs=bm25_docs,
            top_n=final_top_n,
            max_chars=final_max_chars,
            mode="false_premise",
        )
    elif mode == "list_answer":
        final_docs = select_list_answer_context(
            reranked_docs=reranked_docs,
            question=question,
            semantic_docs=semantic_docs,
            bm25_docs=bm25_docs,
            top_n=final_top_n,
            max_chars=final_max_chars,
        )
    else:
        final_docs = select_single_fact_context(
            reranked_docs=reranked_docs,
            question=question,
            semantic_docs=semantic_docs,
            bm25_docs=bm25_docs,
            all_chunks=all_chunks,
            enable_neighbor_expansion=final_neighbor_expansion,
            neighbor_window=neighbor_window,
            top_n=final_top_n,
            max_chars=final_max_chars,
        )

    final_docs = clean_final_context_docs(
        question=question,
        docs=final_docs,
        mode=mode,
        min_keep=1,
    )

    final_docs = limit_context_docs(
        docs=final_docs,
        max_chars=final_max_chars,
        max_per_source=final_max_per_source,
    )

    for doc in final_docs or []:
        doc.metadata = dict(getattr(doc, "metadata", {}) or {})
        doc.metadata["dynamic_context_policy_mode"] = mode
        doc.metadata["dynamic_context_policy_top_n"] = final_top_n
        doc.metadata["dynamic_context_policy_max_chars"] = final_max_chars
        doc.metadata["dynamic_context_policy_max_per_source"] = final_max_per_source

    if debug:
        print_context_debug(
            question=question,
            mode=mode,
            docs=final_docs,
        )

    return final_docs


# ============================================================
# 7. DEBUG HELPER
# ============================================================

def print_context_debug(question, mode, docs):
    # Optional debug print para makita kung ano ang pinili.
    print(f"[CONTEXT FILTER] question={question}", flush=True)
    print(f"[CONTEXT FILTER] mode={mode}", flush=True)
    print(f"[CONTEXT FILTER] final_docs={len(docs or [])}", flush=True)

    for index, doc in enumerate(docs or [], start=1):
        metadata = get_metadata(doc)
        source_name = get_source_name(doc)
        source_key = get_source_key(doc)
        score = get_context_score(doc)
        original_rank = get_original_rank(doc)
        anchor_reason = metadata.get("context_anchor_reason", "")
        filter_scope = metadata.get("context_confident_filter_scope", "")
        primary_score = metadata.get("primary_evidence_score", "")
        retrieval_score = metadata.get("retrieval_agreement_score", "")
        direct_score = metadata.get("direct_window_score", "")
        answer_intent = metadata.get("answer_intent", metadata.get("dynamic_answer_intent", ""))
        answer_score = metadata.get("answer_pattern_score", "")
        dynamic_top_n = metadata.get("dynamic_single_fact_top_n", metadata.get("dynamic_context_policy_top_n", ""))
        dynamic_selection = metadata.get("dynamic_selection_strategy", "")
        dynamic_policy_max_chars = metadata.get("dynamic_context_policy_max_chars", "")
        semantic_rank = metadata.get("semantic_rank_for_primary", "")
        bm25_rank = metadata.get("bm25_rank_for_primary", "")
        final_cleanup = metadata.get("final_low_value_cleanup", "")
        removed_count = metadata.get("final_low_value_removed_count", "")
        evidence_snippet = metadata.get("evidence_snippet", "")

        page = metadata.get("page", "")
        chunk_id = metadata.get("chunk_id") or metadata.get("chunk_index") or ""
        neighbor_flag = metadata.get("neighbor_expanded", False)
        neighbor_reason = metadata.get("neighbor_expand_reason", "")
        neighbor_offset = metadata.get("neighbor_offset", "")

        print(
            f"[CONTEXT FILTER] {index}. "
            f"score={score:.4f} | "
            f"primary_score={primary_score} | "
            f"retrieval_score={retrieval_score} | "
            f"direct_score={direct_score} | "
            f"answer_intent={answer_intent} | "
            f"answer_score={answer_score} | "
            f"dynamic_top_n={dynamic_top_n} | "
            f"dynamic_selection={dynamic_selection} | "
            f"policy_max_chars={dynamic_policy_max_chars} | "
            f"semantic_rank={semantic_rank} | "
            f"bm25_rank={bm25_rank} | "
            f"final_cleanup={final_cleanup} | "
            f"removed_low_value={removed_count} | "
            f"evidence_snippet={evidence_snippet[:120]} | "
            f"original_rank={original_rank} | "
            f"page={page} | "
            f"chunk={chunk_id} | "
            f"neighbor={neighbor_flag} | "
            f"neighbor_offset={neighbor_offset} | "
            f"neighbor_reason={neighbor_reason} | "
            f"source_key={source_key} | "
            f"anchor_reason={anchor_reason} | "
            f"filter_scope={filter_scope} | "
            f"source={source_name}",
            flush=True,
        )
