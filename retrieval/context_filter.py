import re

from config.settings import (
    MAX_CONTEXT_CHARS,
    MIN_CONTEXT_LENGTH,
    MIN_QUALITY_SCORE,
)


# Characters na normal makita sa clean documents.
SAFE_PUNCTUATION = set(".,;:!?()[]{}'\"-/–—%#_+=<>@&*")

# Minimum number ng matched terms bago pumasa sa evidence guard.
MIN_EVIDENCE_TERM_MATCHES = 1

# Percent ng meaningful question terms na dapat makita sa retrieved context.
MIN_EVIDENCE_TERM_COVERAGE = 0.65

# Ilang missing meaningful terms lang ang papayagan para iwas partial-topic answer.
MAX_UNMATCHED_EVIDENCE_TERMS = 1

# Limit ng text na iche-check ng evidence guard para mabilis pa rin.
MAX_EVIDENCE_CONTEXT_CHARS = 5000

# Common words na hindi useful pang-check ng document evidence.
EVIDENCE_WEAK_TERMS = {
    # English common/question words.
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "could",
    "did", "do", "does", "for", "from", "give", "had", "has", "have",
    "he", "her", "his", "how", "i", "if", "in", "is", "it", "its",
    "me", "must", "my", "of", "on", "or", "our", "please", "shall",
    "she", "should", "show", "that", "the", "their", "them", "these",
    "they", "this", "those", "to", "was", "were", "what", "when",
    "where", "which", "who", "why", "will", "with", "would", "you",
    "your", "tell", "about", "answer", "example", "result", "correct",
    "always", "same", "different", "another",

    # Weak standalone number/time words.
    "zero", "one", "two", "three", "four", "five", "six", "seven",
    "eight", "nine", "ten", "day", "days", "week", "weeks", "month",
    "months", "year", "years",

    # Weak arithmetic/action words para hindi math-specific ang guard.
    "plus", "minus", "add", "adding", "sum", "total", "subtract",
    "difference", "times", "multiply", "multiplied", "divide", "divided",
    "equals", "equal", "calculate", "compute",

    # Filipino/Tagalog common/question words.
    "ako", "alin", "ang", "ano", "ba", "bakit", "dapat", "dito",
    "doon", "eh", "eto", "gano", "ganito", "ganyan", "ito", "iyon",
    "ka", "kailan", "kelan", "kapag", "kay", "kaya", "ko", "kung",
    "lang", "may", "meron", "mga", "mo", "na", "nang", "ng", "nga",
    "naman", "ni", "nila", "nito", "noong", "nung", "pa", "paano",
    "para", "po", "sa", "saan", "si", "sino", "sya", "siya", "to",
    "yan", "yung",
}

# Starters na kailangan mas strict dahil madalas may false premise.
ASSUMPTION_CHECK_STARTERS = (
    "why ",
    "how ",
    "when ",
    "why did ",
    "how did ",
    "why was ",
    "why is ",
    "bakit ",
    "paano ",
    "kailan ",
    "kelan ",
)


def is_readable_char(char):
    # Unicode-friendly para hindi ma-penalize ang Japanese text.
    return char.isalnum() or char.isspace() or char in SAFE_PUNCTUATION


def text_quality_score(text):
    # Basic readable score para ma-filter ang sirang OCR/noisy chunks.
    text = str(text or "")
    if not text.strip():
        return 0.0

    total_chars = max(len(text), 1)
    readable_chars = sum(1 for char in text if is_readable_char(char))
    readable_ratio = readable_chars / total_chars

    words = re.findall(r"\b\w+\b", text, flags=re.UNICODE)
    short_words = [word for word in words if len(word) <= 2]
    short_word_ratio = len(short_words) / max(len(words), 1)

    weird_chars = sum(1 for char in text if not is_readable_char(char))
    weird_ratio = weird_chars / total_chars

    return readable_ratio - (short_word_ratio * 0.25) - weird_ratio


def filter_low_quality_docs(docs, min_score=MIN_QUALITY_SCORE, min_length=MIN_CONTEXT_LENGTH):
    # Tanggalin ang sobrang ikli o maingay na chunks.
    clean_docs = []

    for doc in docs or []:
        text = (doc.page_content or "").strip()

        if len(text) < min_length:
            continue

        score = text_quality_score(text)

        if score >= min_score:
            doc.metadata = dict(doc.metadata or {})
            doc.metadata["quality_score"] = float(score)
            clean_docs.append(doc)

    return clean_docs


def limit_context_docs(docs, max_chars=MAX_CONTEXT_CHARS):
    # Limitahan ang final context size habang pinapanatili ang ranking order.
    selected_docs = []
    total_chars = 0

    for doc in docs or []:
        text_length = len(doc.page_content or "")

        if total_chars + text_length > max_chars:
            if not selected_docs:
                selected_docs.append(doc)
            break

        selected_docs.append(doc)
        total_chars += text_length

    return selected_docs


def count_keyword_matches(docs, keywords):
    # Bilangin kung ilang keywords ang nasa final context.
    if not docs or not keywords:
        return 0

    context_text = " ".join(doc.page_content or "" for doc in docs).lower()
    return sum(1 for keyword in keywords if str(keyword).lower() in context_text)


def has_min_keyword_matches(docs, keywords, min_matches=1):
    # Old basic keyword check kung kailangan pa ng ibang module.
    return count_keyword_matches(docs, keywords) >= min_matches


def normalize_evidence_term(term):
    # Linisin ang isang term bago gamitin pang-check sa final context.
    term = str(term or "").strip().lower().strip("_")

    if not term:
        return ""

    if term in EVIDENCE_WEAK_TERMS:
        return ""

    # Ignore short standalone numbers like 1 or 2, pero keep years or IDs.
    if term.isdigit() and len(term) < 3:
        return ""

    # Ignore one-character Latin tokens, pero keep Japanese/CJK tokens.
    if len(term) == 1 and not re.search(r"[\u3040-\u30ff\u3400-\u9fff]", term):
        return ""

    return term


def extract_evidence_terms(text):
    # Kunin lang ang meaningful terms mula English, Filipino, numbers, at Japanese/CJK.
    raw_terms = re.findall(
        r"[A-Za-z0-9_]+|[\u3040-\u30ff\u3400-\u9fff]+",
        str(text or ""),
    )
    terms = []
    seen = set()

    for raw_term in raw_terms:
        term = normalize_evidence_term(raw_term)

        if not term or term in seen:
            continue

        seen.add(term)
        terms.append(term)

    return terms


def has_searchable_question_terms(question):
    # Fast pre-retrieval guard.
    # Kapag walang useful search term ang fresh question, huwag na mag-RAG.
    return bool(extract_evidence_terms(question))


def build_evidence_context_text(docs, max_chars=MAX_EVIDENCE_CONTEXT_CHARS):
    # Gamitin lang ang final retrieved docs para mabilis ang checking.
    parts = []
    total_chars = 0

    for doc in docs or []:
        metadata = doc.metadata or {}
        metadata_text = " ".join([
            str(metadata.get("source", "")),
            str(metadata.get("file_name", "")),
            str(metadata.get("title", "")),
        ])
        content = f"{metadata_text}\n{doc.page_content or ''}".strip()

        if not content:
            continue

        remaining = max_chars - total_chars

        if remaining <= 0:
            break

        text_part = content[:remaining]
        parts.append(text_part)
        total_chars += len(text_part)

    return "\n".join(parts).lower()


def evidence_term_exists(term, context_text):
    # Normal words use word-boundary match para iwas false substring hits.
    # Japanese/CJK text uses substring match dahil walang spaces ang ibang text.
    term = str(term or "").lower()

    if not term:
        return False

    if re.search(r"[\u3040-\u30ff\u3400-\u9fff]", term):
        return term in context_text

    pattern = rf"(?<![a-z0-9_]){re.escape(term)}(?![a-z0-9_])"
    return re.search(pattern, context_text) is not None


def get_evidence_term_matches(terms, context_text):
    # Ihiwalay ang matched at missing terms para madaling i-debug.
    matched_terms = []
    missing_terms = []

    for term in terms or []:
        if evidence_term_exists(term, context_text):
            matched_terms.append(term)
        else:
            missing_terms.append(term)

    return matched_terms, missing_terms


def has_enough_term_coverage(terms, context_text):
    # Pigilan ang partial-topic answer.
    # Example: salary + Independence Day should not pass dahil may Independence Day lang.
    if not terms:
        return False

    matched_terms, missing_terms = get_evidence_term_matches(
        terms=terms,
        context_text=context_text,
    )

    # Kapag maikli ang question, dapat lahat ng meaningful terms ay present.
    if len(terms) <= 2:
        return len(missing_terms) == 0

    coverage = len(matched_terms) / len(terms)

    if coverage < MIN_EVIDENCE_TERM_COVERAGE:
        return False

    if len(missing_terms) > MAX_UNMATCHED_EVIDENCE_TERMS:
        return False

    return True


def has_document_evidence(question, retrieval_query, docs, debug=False):
    # Generic post-retrieval guard bago tawagin ang LLM.
    # Hindi ito tumatawag sa LLM, text check lang sa final retrieved docs.
    question_terms = extract_evidence_terms(question)
    query_terms = extract_evidence_terms(f"{question or ''}\n{retrieval_query or ''}")

    if not query_terms:
        if debug:
            print("Evidence guard: blocked because query has no searchable terms.")
        return False

    context_text = build_evidence_context_text(docs)

    if not context_text:
        if debug:
            print("Evidence guard: blocked because final docs have no text.")
        return False

    question_matched, question_missing = get_evidence_term_matches(
        terms=question_terms,
        context_text=context_text,
    )
    query_matched, query_missing = get_evidence_term_matches(
        terms=query_terms,
        context_text=context_text,
    )

    # Original question coverage ang pinaka-important check.
    # Dito mabablock ang tanong na may halong unsupported term gaya ng sahod.
    if not has_enough_term_coverage(question_terms, context_text):
        if debug:
            print(f"Evidence guard question terms: {question_terms}")
            print(f"Evidence guard question matched: {question_matched}")
            print(f"Evidence guard question missing: {question_missing}")
            print("Evidence guard: blocked because question coverage is too low.")
        return False

    # Second safety check para siguradong may minimum matches ang query terms.
    required_matches = MIN_EVIDENCE_TERM_MATCHES

    if len(query_terms) >= 3:
        required_matches = 2

    if debug:
        print(f"Evidence guard query terms: {query_terms}")
        print(f"Evidence guard query matched: {query_matched}")
        print(f"Evidence guard query missing: {query_missing}")
        print(f"Evidence guard matches: {len(query_matched)}/{required_matches}")

    return len(query_matched) >= required_matches


def needs_strict_assumption_check(question):
    # Mas strict sa why/how/when/bakit/paano/kelan dahil madalas may hidden assumption.
    normalized = " ".join(str(question or "").lower().split())
    return any(normalized.startswith(starter) for starter in ASSUMPTION_CHECK_STARTERS)


def get_document_key(doc):
    # Stable key para sa duplicate removal.
    metadata = doc.metadata or {}
    source = metadata.get("source", "")
    page = metadata.get("page", "")
    chunk_id = metadata.get("chunk_id") or metadata.get("chunk_index", "")

    if chunk_id != "":
        return (source, page, chunk_id)

    preview = (doc.page_content or "")[:250]
    return (source, page, preview)


def remove_duplicate_docs(docs):
    # Tanggalin duplicate chunks pero i-keep ang unang lumabas.
    unique_docs = []
    seen = set()

    for doc in docs or []:
        key = get_document_key(doc)

        if key in seen:
            continue

        seen.add(key)
        unique_docs.append(doc)

    return unique_docs


def get_chunk_index(doc):
    # Kunin ang chunk index mula metadata.
    chunk_index = (doc.metadata or {}).get("chunk_index")

    if chunk_index is None:
        return None

    try:
        return int(chunk_index)
    except (TypeError, ValueError):
        return None


def expand_neighbor_chunks(selected_docs, all_chunks, window=1):
    # Optional: isama ang katabing chunks para hindi bitin ang context.
    if not selected_docs or not all_chunks:
        return selected_docs

    chunks_by_source = {}

    for chunk in all_chunks:
        metadata = chunk.metadata or {}
        source = metadata.get("source", "")
        chunk_index = get_chunk_index(chunk)

        if chunk_index is None:
            continue

        chunks_by_source.setdefault(source, {})[chunk_index] = chunk

    expanded_docs = []

    for doc in selected_docs:
        metadata = doc.metadata or {}
        source = metadata.get("source", "")
        chunk_index = get_chunk_index(doc)

        if chunk_index is None:
            expanded_docs.append(doc)
            continue

        source_chunks = chunks_by_source.get(source, {})

        for index in range(chunk_index - window, chunk_index + window + 1):
            neighbor_doc = source_chunks.get(index)
            if neighbor_doc:
                expanded_docs.append(neighbor_doc)

    return remove_duplicate_docs(expanded_docs)
