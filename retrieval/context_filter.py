import re
import unicodedata
from collections import defaultdict
from pathlib import Path

from config.settings import (
    MAX_CONTEXT_CHARS,
    MAX_PER_SOURCE,
    MAX_EVIDENCE_CONTEXT_CHARS,
    MAX_UNMATCHED_EVIDENCE_TERMS,
    MIN_CONTEXT_LENGTH,
    MIN_EVIDENCE_TERM_COVERAGE,
    MIN_EVIDENCE_TERM_MATCHES,
    MIN_QUALITY_SCORE,
)


# Characters na normal makita sa clean documents.
SAFE_PUNCTUATION = set(".,;:!?()[]{}'\"-/–—%#_+=<>@&*")

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
    "yan", "yung", "tinatawag", "papel", "bilang", "kaugnay", "nauugnay",
    "pinagkaiba", "pagkakaiba", "compare", "comparison", "versus", "vs",
    "bawat", "isa", "each", "every", "respectively", "exact", "eksaktong",
}

# Small bilingual aliases para hindi mablock ang Tagalog question kapag English ang documents.
# Guard helper lang ito, hindi ito ginagamit bilang answer evidence.
EVIDENCE_TERM_ALIASES = {
    "apruba": ("approval", "approved", "approve"),
    "araw": ("day", "date"),
    "baguhin": ("change", "update", "modify"),
    "gabay": ("guide", "guideline"),
    "gumawa": ("create", "make", "prepare"),
    "hakbang": ("step", "steps", "procedure"),
    "himagsikan": ("revolution", "revolutionary"),
    "kailangan": ("required", "requirement", "must"),
    "layunin": ("purpose", "objective", "goal"),
    "lider": ("leader", "head"),
    "pamamahala": ("management", "governance"),
    "pamahalaan": ("government", "administration"),
    "pamahalaang": ("government", "administration"),
    "panuntunan": ("rule", "rules", "guideline"),
    "patakaran": ("policy", "rule"),
    "rebolusyonaryo": ("revolutionary", "revolution"),
    "responsable": ("responsible", "owner"),
    "sagot": ("answer", "response"),
    "tagapayo": ("adviser", "advisor", "consultant"),
    "tungkulin": ("role", "responsibility", "duty"),
    "utak": ("brain", "brains"),
    "unang": ("first",),
    "republika": ("republic", "republica"),
    "pilipinas": ("philippines", "philippine"),
    "kalayaan": ("independence", "liberty"),
    "deklarasyon": ("declaration", "proclamation"),
    "nagdeklara": ("declared", "proclaimed", "declaration", "proclamation"),
    "nagproklama": ("declared", "proclaimed", "declaration", "proclamation"),
    "tagapagtatag": ("founder", "founders", "founded"),
    "founder": ("founders", "founded"),
    "founders": ("founder", "founded"),
    "role": ("role", "responsibility", "duty"),
    "address": ("address", "location"),
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



def strip_accents(text):
    # Gawing accent-insensitive ang matching.
    # Example: "Andrés" sa document dapat mag-match sa "Andres" sa question.
    normalized = unicodedata.normalize("NFKD", str(text or ""))
    return "".join(char for char in normalized if not unicodedata.combining(char))


def normalize_match_text(text):
    # Shared normalization para sa evidence/source matching.
    return strip_accents(text).lower()

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


def limit_context_docs(docs, max_chars=MAX_CONTEXT_CHARS, max_per_source=None):
    # Limitahan ang final context size habang pinapanatili ang ranking order.
    # Kapag may max_per_source, iiwasan na puro isang source lang ang pumasok.
    selected_docs = []
    total_chars = 0
    source_counts = defaultdict(int)

    for doc in docs or []:
        source_key = get_context_source_key(doc)

        if max_per_source is not None and source_counts[source_key] >= max_per_source:
            continue

        text_length = len(doc.page_content or "")

        if total_chars + text_length > max_chars:
            if not selected_docs:
                selected_docs.append(doc)
            break

        selected_docs.append(doc)
        total_chars += text_length
        source_counts[source_key] += 1

    return selected_docs



RELATIONSHIP_QUERY_TERMS = {
    "connect",
    "connected",
    "connection",
    "relationship",
    "related",
    "compare",
    "comparison",
    "difference",
    "different",
    "versus",
    "vs",
    "cause",
    "result",
    "effect",
    "link",
    "lead",
    "leads",
    "led",
    "leading",
    "into",
    "spark",
    "sparked",
    "trigger",
    "triggered",
    "between",
    "nauugnay",
    "ugnay",
    "kaugnayan",
    "pinagkaiba",
    "pagkakaiba",
    "ikumpara",
    "kumpara",
}

SOURCE_WEAK_TOKENS = {
    "a",
    "an",
    "and",
    "ang",
    "article",
    "data",
    "file",
    "md",
    "of",
    "pdf",
    "the",
    "wiki",
    "wikipedia",
}


def normalize_source_text(text):
    # Normalize source/question text para mag-match ang dash, case, accent, at punctuation.
    normalized = normalize_match_text(text)
    normalized = normalized.replace("–", "-").replace("—", "-")
    normalized = re.sub(r"[^a-z0-9\u3040-\u30ff\u3400-\u9fff]+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def get_source_tokens(doc):
    # Kunin meaningful tokens mula sa source filename.
    metadata = doc.metadata or {}
    raw_source = metadata.get("source") or metadata.get("file_name") or metadata.get("title") or ""
    source_name = Path(str(raw_source)).stem
    tokens = normalize_source_text(source_name).split()
    return [token for token in tokens if token not in SOURCE_WEAK_TOKENS]


def get_context_source_key(doc):
    # Stable source key para sa source diversity checks.
    tokens = get_source_tokens(doc)

    if tokens:
        return " ".join(tokens)

    metadata = doc.metadata or {}
    raw_source = metadata.get("source") or metadata.get("file_name") or metadata.get("title") or ""
    return normalize_source_text(raw_source) or "unknown source"


def get_context_rerank_score(doc):
    # Kunin score kahit iba-iba ang metadata key na gamit ng retriever/reranker.
    metadata = doc.metadata or {}

    for key in ("rerank_score", "score", "hybrid_score", "semantic_score", "quality_score"):
        value = metadata.get(key)

        if value is None:
            continue

        try:
            return float(value)
        except (TypeError, ValueError):
            continue

    return 0.0


def is_relationship_query(question):
    # Detect kung compare/cross-doc/relationship type ang tanong.
    normalized_question = f" {normalize_source_text(question)} "

    for term in RELATIONSHIP_QUERY_TERMS:
        pattern = rf"(?<![a-z0-9_]){re.escape(term)}(?![a-z0-9_])"

        if re.search(pattern, normalized_question):
            return True

    return False


def get_source_question_match_score(doc, question):
    # Mas mataas kapag source filename mismo ay may words na nasa question.
    source_tokens = set(get_source_tokens(doc))
    question_tokens = set(normalize_source_text(question).split())

    if not source_tokens or not question_tokens:
        return 0

    return len(source_tokens.intersection(question_tokens))


def should_keep_source_head(doc, question):
    # Iwasan ang generic source na one-word lang ang match, pero payagan ang single-topic files.
    source_tokens = get_source_tokens(doc)
    match_score = get_source_question_match_score(doc, question)

    if match_score <= 0:
        return False

    if len(source_tokens) <= 2:
        return True

    return match_score >= 2


def select_final_context_docs(
    reranked_docs,
    question,
    top_n=4,
    max_chars=MAX_CONTEXT_CHARS,
    max_per_source=MAX_PER_SOURCE,
):
    # Normal questions: gamitin ang highest-ranked docs.
    # Huwag pilitin ang source diversity dito para hindi mapasukan ng distractor source.
    if not is_relationship_query(question):
        return limit_context_docs(
            docs=(reranked_docs or [])[:top_n],
            max_chars=max_chars,
            max_per_source=None,
        )

    # Cross-doc questions: unahin muna ang source coverage bago duplicate chunks.
    # Example target: Treaty of Paris + Spanish-American War + Philippine-American War.
    docs_by_source = defaultdict(list)

    for doc in reranked_docs or []:
        docs_by_source[get_context_source_key(doc)].append(doc)

    for source_key in docs_by_source:
        docs_by_source[source_key].sort(
            key=get_context_rerank_score,
            reverse=True,
        )

    source_heads = []

    for source_key, docs in docs_by_source.items():
        best_doc = docs[0]

        if not should_keep_source_head(best_doc, question):
            continue

        source_heads.append((
            get_source_question_match_score(best_doc, question),
            get_context_rerank_score(best_doc),
            source_key,
            best_doc,
        ))

    source_heads.sort(key=lambda item: (item[0], item[1]), reverse=True)

    selected_docs = []
    selected_doc_keys = set()
    selected_source_keys = set()

    def add_doc_if_new(doc):
        # Helper para hindi maulit ang parehong chunk.
        doc_key = get_document_key(doc)

        if doc_key in selected_doc_keys:
            return False

        selected_docs.append(doc)
        selected_doc_keys.add(doc_key)
        selected_source_keys.add(get_context_source_key(doc))
        return True

    # Pass 1: kunin ang best chunk ng sources na direktang tugma sa question.
    for _, _, source_key, doc in source_heads:
        if len(selected_docs) >= top_n:
            break

        if source_key in selected_source_keys:
            continue

        add_doc_if_new(doc)

    # Pass 2: kung kulang pa, kumuha muna ng ibang source bago duplicate source.
    for doc in reranked_docs or []:
        if len(selected_docs) >= top_n:
            break

        source_key = get_context_source_key(doc)

        if source_key in selected_source_keys:
            continue

        add_doc_if_new(doc)

    # Pass 3: kung kulang pa rin, saka lang payagan ang duplicate source.
    for doc in reranked_docs or []:
        if len(selected_docs) >= top_n:
            break

        add_doc_if_new(doc)

    return limit_context_docs(
        docs=selected_docs,
        max_chars=max_chars,
        max_per_source=max_per_source,
    )


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
    term = normalize_match_text(term).strip().strip("_")

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
        r"[A-Za-zÀ-ÖØ-öø-ÿ0-9_]+|[\u3040-\u30ff\u3400-\u9fff]+",
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

    return normalize_match_text("\n".join(parts))




ATTRIBUTE_HEAD_WORDS = {
    "address",
    "credential",
    "credentials",
    "id",
    "key",
    "number",
    "passcode",
    "password",
    "secret",
    "token",
    "url",
}

ATTRIBUTE_WEAK_MODIFIERS = {
    "a", "an", "ang", "ano", "exact", "eksaktong", "official", "private",
    "specific", "the", "what", "which",
}

MULTI_PART_MARKERS = (
    " and ",
    " at ",
    " saka ",
    " pati ",
    " also ",
    ",",
    ";",
)


def normalize_attribute_text(text):
    # Normalize para pareho ang "Wi-Fi password", "wifi password", at "wi fi password".
    normalized = normalize_match_text(text)
    normalized = normalized.replace("wi-fi", "wifi")
    normalized = normalized.replace("wi fi", "wifi")
    normalized = re.sub(r"[^a-z0-9_\u3040-\u30ff\u3400-\u9fff]+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def contains_phrase(text, phrase):
    # Exact phrase check na may word boundary para hindi mag-match sa maling substring.
    normalized_text = normalize_attribute_text(text)
    normalized_phrase = normalize_attribute_text(phrase)

    if not normalized_phrase:
        return False

    pattern = rf"(?<![a-z0-9_]){re.escape(normalized_phrase)}(?![a-z0-9_])"
    return re.search(pattern, normalized_text) is not None


def is_multi_part_question(question):
    # Kapag may supported part at unsupported part, prompt ang bahalang mag-partial answer.
    normalized_question = f" {normalize_attribute_text(question)} "
    return any(marker in normalized_question for marker in MULTI_PART_MARKERS)


def extract_exact_attribute_phrases(question):
    # Kunin ang compound attribute na hinihingi ng user.
    # Hindi ito naka-base sa topic/document; naka-base ito sa requested field shape.
    # Example: "wifi password", "passport number", "employee id", "api key".
    normalized_question = normalize_attribute_text(question)
    tokens = normalized_question.split()
    phrases = []
    seen = set()

    for index, token in enumerate(tokens):
        if token not in ATTRIBUTE_HEAD_WORDS:
            continue

        start = max(0, index - 3)
        phrase_tokens = tokens[start:index + 1]

        while phrase_tokens and phrase_tokens[0] in ATTRIBUTE_WEAK_MODIFIERS:
            phrase_tokens = phrase_tokens[1:]

        if len(phrase_tokens) < 2:
            continue

        phrase = " ".join(phrase_tokens)

        if phrase in seen:
            continue

        seen.add(phrase)
        phrases.append(phrase)

    return phrases


def get_missing_exact_attribute_phrases(question, context_text):
    # Ibalik ang requested compound attributes na wala mismo sa context.
    missing_phrases = []

    for phrase in extract_exact_attribute_phrases(question):
        if not contains_phrase(context_text, phrase):
            missing_phrases.append(phrase)

    return missing_phrases


def should_block_exact_attribute_gap(question, context_text):
    # Block lang kapag ang tanong ay mainly humihingi ng isang exact attribute.
    # Kapag multi-part ang tanong, hayaan ang prompt na sumagot ng supported part.
    missing_phrases = get_missing_exact_attribute_phrases(
        question=question,
        context_text=context_text,
    )

    if not missing_phrases:
        return False

    if is_multi_part_question(question):
        return False

    return True


def get_evidence_term_variants(term):
    # Isama ang konting English/Tagalog equivalent para sa term coverage check.
    term = normalize_match_text(term)

    variants = [term]
    variants.extend(EVIDENCE_TERM_ALIASES.get(term, ()))

    return [variant for variant in variants if variant]


def evidence_term_exists(term, context_text):
    # Normal words use word-boundary match para iwas false substring hits.
    # Japanese/CJK text uses substring match dahil walang spaces ang ibang text.
    for variant in get_evidence_term_variants(term):
        if re.search(r"[\u3040-\u30ff\u3400-\u9fff]", variant):
            if variant in context_text:
                return True

            continue

        pattern = rf"(?<![a-z0-9_]){re.escape(variant)}(?![a-z0-9_])"

        if re.search(pattern, context_text) is not None:
            return True

    return False


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
    # Pigilan ang totally unrelated context, pero huwag masyadong strict sa paraphrase/translation.
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

    if coverage >= MIN_EVIDENCE_TERM_COVERAGE and len(missing_terms) <= MAX_UNMATCHED_EVIDENCE_TERMS:
        return True

    # Backup rule para sa multilingual/paraphrased questions na may sapat na strong matches.
    if len(matched_terms) >= 3 and coverage >= 0.30:
        return True

    # Entity/source fallback para sa translated questions.
    # Example: Tagalog question + English document can still be valid when named entities match.
    if len(matched_terms) >= 2 and len(missing_terms) <= max(MAX_UNMATCHED_EVIDENCE_TERMS + 2, 4):
        return True

    return False


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

    if should_block_exact_attribute_gap(question, context_text):
        if debug:
            missing_phrases = get_missing_exact_attribute_phrases(question, context_text)
            print(f"Evidence guard exact attribute missing: {missing_phrases}")
            print("Evidence guard: blocked because requested exact attribute is not in context.")
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
    # Kapag multi-part ang question, huwag i-block agad dahil puwedeng partial answer.
    if is_multi_part_question(question):
        if len(question_matched) < MIN_EVIDENCE_TERM_MATCHES:
            if debug:
                print(f"Evidence guard question terms: {question_terms}")
                print(f"Evidence guard question matched: {question_matched}")
                print(f"Evidence guard question missing: {question_missing}")
                print("Evidence guard: blocked because multi-part question has no supported part.")
            return False
    elif not has_enough_term_coverage(question_terms, context_text):
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

    # Kapag multi-part ang question, sapat na may supported part.
    # Ang missing detail ay dapat sagutin ng prompt as "not stated", hindi i-block buong answer.
    if is_multi_part_question(question):
        required_matches = MIN_EVIDENCE_TERM_MATCHES

    # Kapag may strong original question match, huwag i-block dahil lang mahaba ang rewritten query.
    # Useful ito sa Tagalog/English mixed retrieval.
    if len(question_matched) >= 2:
        required_matches = min(required_matches, len(query_matched))

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
