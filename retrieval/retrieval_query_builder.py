import re


COMPARISON_MARKERS = (
    "compare",
    "vs",
    "versus",
    "difference",
    "different",
    "pinagkaiba",
    "pagkakaiba",
    "ihambing",
)

RELATION_MARKERS = (
    "connect",
    "connection",
    "relationship",
    "relate",
    "related",
    "link",
    "linked",
    "nauugnay",
    "kaugnay",
    "ugnayan",
)

TOPIC_JOINER_WORDS = {
    "and",
    "of",
    "the",
    "de",
    "del",
    "ni",
    "at",
}

TOPIC_WEAK_WORDS = {
    "how",
    "why",
    "what",
    "when",
    "where",
    "who",
    "which",
    "paano",
    "bakit",
    "ano",
    "kailan",
    "sino",
    "saan",
}


CONTENT_WEAK_WORDS = {
    "a",
    "an",
    "and",
    "ang",
    "ano",
    "anong",
    "as",
    "at",
    "because",
    "bakit",
    "by",
    "did",
    "does",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "its",
    "kailan",
    "na",
    "ng",
    "ni",
    "niya",
    "of",
    "on",
    "or",
    "paano",
    "sa",
    "saan",
    "si",
    "sino",
    "sinong",
    "the",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
}


NON_LATIN_QUERY_PATTERN = re.compile(
    r"[぀-ヿ㐀-鿿][぀-ヿ㐀-鿿A-Za-z0-9\s()（）\-–—年]+"
)

JAPANESE_REFERENCE_MARKERS = (
    "japanese article",
    "japanese doc",
    "japanese document",
    "japanese source",
)

LEADING_TOPIC_WORDS = (
    "compare",
    "difference between",
    "different between",
    "pinagkaiba ng",
    "pagkakaiba ng",
    "ihambing ang",
    "ano ang pinagkaiba ng",
    "ano ang pagkakaiba ng",
)


def normalize_query_text(text):
    # Gawing consistent ang spaces at dash variants.
    text = str(text or "").strip()
    text = text.replace("–", "-").replace("—", "-")
    text = re.sub(r"\s+", " ", text)
    return text


def normalize_key(text):
    # Gumawa ng lowercase key para sa duplicate checking.
    return normalize_query_text(text).lower()


def add_unique_query(queries, query):
    # Magdagdag ng query kung hindi empty at hindi duplicate.
    query = normalize_query_text(query)

    if not query:
        return

    key = normalize_key(query)

    if key in {normalize_key(item) for item in queries}:
        return

    queries.append(query)


def contains_any_marker(text, markers):
    # Generic detector para sa comparison o relationship query.
    normalized = normalize_key(text)
    return any(marker in normalized for marker in markers)


def looks_like_japanese_reference(question):
    # Detect kapag ang tanong ay explicitly tumutukoy sa Japanese article/doc/source.
    return contains_any_marker(question, JAPANESE_REFERENCE_MARKERS)


def extract_non_latin_phrases(question):
    # Kunin ang Japanese/non-Latin title phrases kapag nasa tanong mismo.
    # Generic ito: hindi naka-hardcode sa specific history file.
    text = normalize_query_text(question)
    phrases = []

    for match in NON_LATIN_QUERY_PATTERN.finditer(text):
        phrase = clean_topic_text(match.group(0))
        phrase = phrase.strip(" .,:;?!")

        if len(phrase) < 2:
            continue

        add_unique_query(phrases, phrase)

    return phrases


def build_japanese_reference_query(question):
    # Kapag may "Japanese doc/article" pero walang Japanese title sa question,
    # gumawa ng broader query gamit exact English terms sa question.
    if not looks_like_japanese_reference(question):
        return ""

    text = normalize_query_text(question)
    keywords = []

    for token in re.findall(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*", text):
        lower = token.lower()

        if lower in TOPIC_WEAK_WORDS or lower in {"the", "a", "an", "by", "in", "on", "sa", "ang", "ng", "doc", "article", "japanese"}:
            continue

        if token[:1].isupper() or token.isdigit() or len(token) >= 6:
            keywords.append(token)

    if not keywords:
        return ""

    return " ".join(["Japanese document", *keywords])


def looks_like_comparison(question):
    # True kapag mukhang comparison question.
    return contains_any_marker(question, COMPARISON_MARKERS)


def looks_like_relation_question(question):
    # True kapag mukhang cross-document / relationship question.
    return contains_any_marker(question, RELATION_MARKERS)


def remove_leading_topic_words(text):
    # Alisin ang generic instruction words para topic na lang ang matira.
    cleaned = normalize_query_text(text)

    for prefix in LEADING_TOPIC_WORDS:
        pattern = rf"^{re.escape(prefix)}\s+"
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)

    return cleaned.strip(" .,:;?!")


def clean_topic_text(text):
    # Linisin ang isang extracted topic bago gawing retrieval query.
    text = normalize_query_text(text)
    text = remove_leading_topic_words(text)
    text = re.sub(r"^(how did|how does|why did|why does|paano|bakit)\s+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(huwag|do not|dont|don't)\b.*$", "", text, flags=re.IGNORECASE)
    return text.strip(" .,:;?!")


def split_comparison_topics(question):
    # Kunin ang dalawang side ng A vs B o A versus B.
    text = normalize_query_text(question)

    patterns = (
        r"(.+?)\s+vs\.?\s+(.+?)(?:[.?!]|$)",
        r"(.+?)\s+versus\s+(.+?)(?:[.?!]|$)",
        r"difference\s+between\s+(.+?)\s+and\s+(.+?)(?:[.?!]|$)",
        r"pinagkaiba\s+ng\s+(.+?)\s+at\s+(.+?)(?:[.?!]|$)",
        r"pagkakaiba\s+ng\s+(.+?)\s+at\s+(.+?)(?:[.?!]|$)",
    )

    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)

        if not match:
            continue

        left = clean_topic_text(match.group(1))
        right = clean_topic_text(match.group(2))
        return [topic for topic in (left, right) if topic]

    return []


def is_topic_piece(token):
    # Check kung mukhang part ng title/proper noun ang token.
    if not token:
        return False

    if token.lower() in TOPIC_JOINER_WORDS:
        return True

    if token[:1].isupper():
        return True

    if "-" in token and any(part[:1].isupper() for part in token.split("-")):
        return True

    if token.isdigit() and len(token) >= 3:
        return True

    return False


def extract_named_topics(question):
    # Generic proper-noun extractor para sa relationship/cross-doc queries.
    # Wala itong domain-specific aliases; kinukuha lang nito ang existing words sa question.
    text = normalize_query_text(question)
    tokens = re.findall(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*", text)

    topics = []
    current = []

    for token in tokens:
        if is_topic_piece(token):
            # Joiner words like "of" and "the" should not start a topic by themselves.
            if token.lower() in TOPIC_JOINER_WORDS and not current:
                continue

            current.append(token)
            continue

        if current:
            topic = clean_topic_text(" ".join(current))
            add_unique_query(topics, topic)
            current = []

    if current:
        topic = clean_topic_text(" ".join(current))
        add_unique_query(topics, topic)

    cleaned_topics = []

    for topic in topics:
        words = topic.split()

        if normalize_key(topic) in TOPIC_WEAK_WORDS:
            continue

        if len(words) < 2 and not any(char.isdigit() for char in topic) and len(topic) < 4:
            continue

        add_unique_query(cleaned_topics, topic)

    return cleaned_topics


def extract_descriptive_phrases(question):
    # Kumuha ng important clue phrases mula sa "called/known as" at "bilang/as" patterns.
    # Generic ito at hindi naka-base sa specific tao, event, o document.
    text = normalize_query_text(question)
    phrase_patterns = (
        r"\b(?:tinatawag\s+na|kilala\s+bilang|known\s+as|called|referred\s+to\s+as)\s+(.+?)(?:\s+(?:dahil|because|for|as|and\s+later)\b|[.?!]|$)",
        r"\b(?:bilang|as|role\s+as|served\s+as|serve\s+as)\s+(.+?)(?:[.?!]|$)",
    )

    phrases = []

    for pattern in phrase_patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            phrase = clean_topic_text(match.group(1))
            phrase = re.sub(r"^(ang|the|a|an)\s+", "", phrase, flags=re.IGNORECASE)
            phrase = phrase.strip(" .,:;?!")

            if len(phrase) < 3:
                continue

            add_unique_query(phrases, phrase)

    return phrases


def build_content_keyword_query(question):
    # Gumawa ng compact keyword query mula sa content words.
    # Useful ito sa multilingual paraphrase dahil natatanggal ang filler words.
    text = normalize_query_text(question)
    tokens = re.findall(r"[A-Za-zÀ-ÿ0-9]+(?:-[A-Za-zÀ-ÿ0-9]+)*", text)
    keywords = []

    for token in tokens:
        lower = token.lower()

        if lower in CONTENT_WEAK_WORDS:
            continue

        if len(lower) <= 2 and not lower.isdigit():
            continue

        keywords.append(token)

    if len(keywords) < 3:
        return ""

    return " ".join(keywords)


def build_descriptive_lookup_queries(question):
    # Extra queries para sa descriptive/paraphrased lookup questions.
    # Hindi ito translation dictionary; nire-rewrite lang nito ang sariling words ng user.
    if looks_like_comparison(question) or looks_like_relation_question(question):
        return []

    queries = []
    phrases = extract_descriptive_phrases(question)

    if len(phrases) >= 2:
        add_unique_query(queries, " ".join(phrases))

    content_query = build_content_keyword_query(question)
    add_unique_query(queries, content_query)

    for phrase in phrases:
        add_unique_query(queries, phrase)

    return queries


def build_relation_bridge_query(question, topics):
    # Gumawa ng bridge query gamit ang topics mismo at generic relation words.
    if not topics:
        return ""

    topic_text = " ".join(topics)
    return f"{topic_text} relationship connection cause result"


def build_retrieval_queries(question, rewritten_question=None, enabled=True, max_queries=3):
    # Gumawa ng retrieval queries nang hindi ginagalaw ang original answer question.
    # Kapag normal direct question, usually isang query lang ang ibabalik.
    original = normalize_query_text(question)
    rewritten = normalize_query_text(rewritten_question or "")

    queries = []

    add_unique_query(queries, rewritten or original)

    if original and rewritten and normalize_key(original) != normalize_key(rewritten):
        add_unique_query(queries, original)

    for phrase in extract_non_latin_phrases(original):
        add_unique_query(queries, phrase)

    japanese_reference_query = build_japanese_reference_query(original)
    add_unique_query(queries, japanese_reference_query)

    if not enabled:
        return queries[:max_queries]

    for descriptive_query in build_descriptive_lookup_queries(original):
        add_unique_query(queries, descriptive_query)

    if looks_like_comparison(original):
        for topic in split_comparison_topics(original):
            add_unique_query(queries, topic)

    if looks_like_relation_question(original):
        topics = extract_named_topics(original)
        bridge_query = build_relation_bridge_query(original, topics)
        add_unique_query(queries, bridge_query)

        if topics:
            add_unique_query(queries, topics[-1])

        for topic in topics:
            add_unique_query(queries, topic)

    return queries[:max(1, max_queries)]


def combine_retrieval_queries(queries):
    # Pagsamahin ang queries para magamit ng evidence guard/debug.
    if isinstance(queries, str):
        return queries

    return "\n".join(normalize_query_text(query) for query in queries or [] if normalize_query_text(query))


def get_document_key(doc):
    # Stable key para matanggal duplicate docs mula sa multiple retrieval queries.
    metadata = doc.metadata or {}
    source = metadata.get("source", "")
    page = metadata.get("page", "")
    chunk_id = metadata.get("chunk_id") or metadata.get("chunk_index", "")

    if chunk_id != "":
        return (source, page, chunk_id)

    preview = (doc.page_content or "")[:250]
    return (source, page, preview)


def interleave_unique_documents(document_groups, max_docs=15):
    # Ihalo ang results per query para hindi matabunan ng first query ang ibang query.
    selected_docs = []
    seen = set()

    groups = [list(group or []) for group in document_groups or []]
    max_group_length = max((len(group) for group in groups), default=0)

    for index in range(max_group_length):
        for group in groups:
            if index >= len(group):
                continue

            doc = group[index]
            key = get_document_key(doc)

            if key in seen:
                continue

            seen.add(key)
            selected_docs.append(doc)

            if len(selected_docs) >= max_docs:
                return selected_docs

    return selected_docs
