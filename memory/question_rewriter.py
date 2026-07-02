"""Rewrite follow-up questions into standalone retrieval queries."""

import re

CLARIFY_PREFIX = "CLARIFY:"

REFERENCE_WORDS = {
    "it", "this", "that", "these", "those", "there",
    "he", "she", "they", "them", "him", "her",
    "its", "his", "their",
}

PERSON_REFERENCE_WORDS = {"he", "him", "his", "she", "her"}
GROUP_REFERENCE_WORDS = {"they", "them", "their"}
TOPIC_REFERENCE_WORDS = {"it", "its", "this", "that", "these", "those", "there"}

NON_PERSON_TOPIC_WORDS = {
    "policy", "procedure", "process", "rule", "rules", "manual",
    "document", "documents", "system", "event", "events", "war",
    "occupation", "revolution", "cruelty", "cruelties", "issue",
    "problem", "requirement", "requirements", "review", "coding",
    "software", "release", "attendance", "leave", "treaty", "agreement",
    "constitution", "law", "battle", "revolt", "independence", "article",
    "protocol", "ratification", "sovereignty", "territory", "country",
    "nation", "government", "organization", "source", "wikipedia",
}

GROUP_TOPIC_WORDS = {
    "people", "employees", "users", "developers", "reviewers", "team",
    "teams", "group", "groups", "members", "army", "workers",
    "customers", "clients", "students", "citizens", "ladies", "women",
    "men", "girls", "boys", "families", "nations", "countries",
}

PERSON_RELATED_GROUP_WORDS = {
    "ladies", "women", "men", "girls", "boys", "people", "army", "team",
    "group", "family", "families", "members", "relationship", "relationships",
}

QUESTION_WORDS = {
    "who", "what", "where", "when", "why", "how", "which", "did", "does",
    "do", "is", "are", "was", "were", "can", "could", "should", "would",
}

QUESTION_PREFIX_PATTERNS = [
    r"^(?:can you|could you|please)?\s*(?:explain|describe|discuss|tell me about|give details about|give me details about)\s+(.+)$",
    r"^(?:what do you know about|information about|details about)\s+(.+)$",
    r"^(?:who|what|where|when)\s+(?:is|are|was|were)\s+(.+)$",
    r"^(?:why|how|when|what)\s+(?:did|does|do|is|are|was|were)\s+(.+)$",
]

TRAILING_FILLER_PATTERNS = [
    r"\s+in detail$",
    r"\s+in details$",
    r"\s+with details$",
    r"\s+for me$",
    r"\s+please$",
]


def extract_text(response):
    # LangChain/Ollama response -> plain text.
    if hasattr(response, "content"):
        return response.content

    return str(response)


def normalize_space(text):
    # Collapse whitespace for easier matching.
    return re.sub(r"\s+", " ", str(text or "")).strip()


def normalize_memory_key(text):
    text = normalize_space(text).lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def looks_like_date_or_short_value(text):
    # Dates/numbers are answer values, not safe pronoun targets.
    text = normalize_space(text).strip(" ?.!,:;\"'")

    if not text:
        return True

    patterns = [
        r"^\d{4}$",
        r"^\d{4}-\d{2}-\d{2}$",
        r"^\d{1,2}[/-]\d{1,2}[/-]\d{2,4}$",
        r"^(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2},?\s*\d{0,4}$",
        r"^\d{1,2}\s+(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{4}$",
    ]

    lowered = text.lower()
    return any(re.search(pattern, lowered, flags=re.I) for pattern in patterns)


def ensure_question_mark(text):
    # Keep returned rewrite as a question.
    text = normalize_space(text).rstrip(". ")

    if not text.endswith("?"):
        text += "?"

    return text


def is_clarification(text):
    # Check if the rewriter says the follow-up is ambiguous.
    return str(text or "").strip().upper().startswith(CLARIFY_PREFIX)


def clean_rewritten_question(text, fallback_question):
    # Kunin lang ang unang useful line para iwas explanation ng small models.
    text = str(text or "").strip()

    if not text:
        return fallback_question

    for label in ["Standalone Question:", "Rewritten Question:", "Question:", "Answer:"]:
        if text.lower().startswith(label.lower()):
            text = text[len(label):].strip()

    for line in text.splitlines():
        line = line.strip().strip('"').strip("'")

        if not line:
            continue

        if is_clarification(line):
            message = line[len(CLARIFY_PREFIX):].strip()
            if message:
                return f"{CLARIFY_PREFIX} {message}"

            return f"{CLARIFY_PREFIX} Which person, item, or topic do you mean?"

        return ensure_question_mark(line)

    return fallback_question


def extract_user_lines(chat_history, current_question=None):
    # Return reference-bearing history lines in chronological order.
    # Raw assistant answers are intentionally ignored to avoid using dates/numbers as topics.
    current_key = normalize_space(current_question).lower()
    history_lines = []

    allowed_prefixes = (
        "user:",
        "source:",
        "sources:",
        "assistant source:",
        "assistant sources:",
        "topic:",
        "resolved question:",
    )

    for raw_line in str(chat_history or "").splitlines():
        line = raw_line.strip()
        lowered = line.lower()

        if not lowered.startswith(allowed_prefixes):
            continue

        value = normalize_space(line.split(":", 1)[1])

        if not value:
            continue

        if current_key and value.lower() == current_key:
            continue

        if looks_like_date_or_short_value(value):
            continue

        history_lines.append(value)

    return history_lines

def clean_topic_candidate(text):
    # Remove common question wrappers, source suffixes, and filler words from a topic candidate.
    topic = normalize_space(text).strip(" ?.!,:;\"'")

    if looks_like_date_or_short_value(topic):
        return ""

    # Source titles often come in as "José Rizal - Wikipedia".
    topic = re.sub(r"\s+[-–—]\s+(?:wikipedia|source|pdf|docx|pptx|xlsx|csv|markdown|md)\b.*$", "", topic, flags=re.I).strip()

    for pattern in TRAILING_FILLER_PATTERNS:
        topic = re.sub(pattern, "", topic, flags=re.I).strip()

    topic = re.sub(r"^(?:the topic|the subject|topic|subject|source|sources|assistant source|assistant sources)\s+(?:is|was|are|were)?\s*", "", topic, flags=re.I).strip()

    return topic.strip(" ?.!,:;\"'")


def extract_topic_from_user_question(question):
    # Extract a likely subject/topic from the user's previous clear question.
    question = clean_topic_candidate(question)

    if not question:
        return ""

    lowered = question.lower()

    for pattern in QUESTION_PREFIX_PATTERNS:
        match = re.match(pattern, lowered, flags=re.I)

        if not match:
            continue

        start, end = match.span(1)
        topic = question[start:end]
        topic = clean_topic_candidate(topic)

        if topic:
            return topic

    # Fallback: short direct topic-like user messages can be used as the topic.
    words = question.split()

    if 1 <= len(words) <= 10:
        return question

    return ""


def get_reference_words(question):
    # Return only contextual reference words.
    # "that" is ignored when it is a relative connector, e.g.
    # "the ladies that had relationship with Jose Rizal".
    text = normalize_space(question).lower()
    tokens = re.findall(r"[a-zA-Z']+", text)
    reference_words = set()

    for index, token in enumerate(tokens):
        if token not in REFERENCE_WORDS:
            continue

        if token == "that" and not is_contextual_that_reference(text, tokens, index):
            continue

        if token == "there" and not is_contextual_there_reference(text, tokens, index):
            continue

        reference_words.add(token)

    return reference_words


def is_contextual_followup_line(question):
    # Previous follow-up questions should not become future reference targets.
    text = normalize_space(question).lower().rstrip("?")

    if not text:
        return False

    reference = r"(?:it|this|that|these|those|he|she|they|them|him|her|its|his|their)"

    patterns = [
        rf"^(?:who|what|where|when|why|how|which)\s+{reference}\b",
        rf"^(?:who|what|where|when|why|how|which)\s+(?:is|are|was|were|did|does|do|can|could|should|would)\s+{reference}\b",
        rf"^(?:did|does|do|is|are|was|were|can|could|should|would)\s+{reference}\b",
        rf"^{reference}\b",
    ]

    return any(re.search(pattern, text, flags=re.I) for pattern in patterns)


def get_latest_topic(chat_history, current_question=None):
    # Use the most recent clear subject/source as the follow-up reference target.
    for history_line in reversed(extract_user_lines(chat_history, current_question=current_question)):
        if is_contextual_followup_line(history_line):
            continue

        topic = extract_topic_from_user_question(history_line)

        if topic:
            return topic

    return ""


def extract_person_name_candidate(text):
    # Extract a likely person name embedded in a larger topic.
    text = clean_topic_candidate(text)

    candidates = re.findall(
        r"\b[A-ZÀ-Þ][A-Za-zÀ-ÿ'’.-]*(?:\s+[A-ZÀ-Þ][A-Za-zÀ-ÿ'’.-]*){1,3}\b",
        text,
    )

    for candidate in reversed(candidates):
        candidate = clean_topic_candidate(candidate)
        tokens = {token.lower() for token in re.findall(r"[A-Za-zÀ-ÿ'’-]+", candidate)}

        if not candidate or tokens.intersection(NON_PERSON_TOPIC_WORDS):
            continue

        if looks_like_group_topic(candidate):
            continue

        return candidate

    # Support single hyphenated proper names like "Lapu-Lapu".
    match = re.search(r"\b[A-ZÀ-Þ][A-Za-zÀ-ÿ'’]+-[A-ZÀ-Þ][A-Za-zÀ-ÿ'’]+\b", text)
    if match:
        return match.group(0)

    return ""


def looks_like_person_topic(topic):
    # Generic heuristic: person names are usually proper-name phrases, not question fragments.
    topic = clean_topic_candidate(topic)
    words = re.findall(r"[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ'’-]*", topic)
    tokens = {word.lower() for word in words}

    if not words:
        return False

    if tokens.intersection(REFERENCE_WORDS) or tokens.intersection(QUESTION_WORDS):
        return False

    if tokens.intersection(NON_PERSON_TOPIC_WORDS) or tokens.intersection(PERSON_RELATED_GROUP_WORDS):
        return False

    if looks_like_group_topic(topic):
        return False

    if extract_person_name_candidate(topic):
        return True

    return False


def looks_like_group_topic(topic):
    # Generic heuristic for plural/group references.
    topic = clean_topic_candidate(topic).lower()
    tokens = set(re.findall(r"[a-zA-ZÀ-ÿ']+", topic))

    if tokens.intersection(GROUP_TOPIC_WORDS):
        return True

    return any(token.endswith("s") for token in tokens if len(token) > 3 and token not in {"paris"})


def looks_like_singular_non_person_topic(topic):
    # Compatible target for "it/its": one non-person thing, event, document, policy, treaty, etc.
    topic = clean_topic_candidate(topic)
    lowered = topic.lower()
    tokens = set(re.findall(r"[a-zA-ZÀ-ÿ']+", lowered))

    if not topic or is_contextual_followup_line(topic):
        return False

    if tokens.intersection(REFERENCE_WORDS) or tokens.intersection(QUESTION_WORDS):
        return False

    if tokens.intersection(PERSON_RELATED_GROUP_WORDS) or looks_like_group_topic(topic):
        return False

    if extract_person_name_candidate(topic) and not tokens.intersection(NON_PERSON_TOPIC_WORDS):
        return False

    words = re.findall(r"[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ'’-]*", topic)
    return 1 <= len(words) <= 12


def get_latest_compatible_topic(chat_history, current_question=None):
    # Pick the newest previous topic compatible with the current pronoun/reference.
    reference_words = get_reference_words(current_question)

    if not reference_words:
        return get_latest_topic(chat_history, current_question=current_question)

    wants_person = bool(reference_words.intersection(PERSON_REFERENCE_WORDS))
    wants_group = bool(reference_words.intersection(GROUP_REFERENCE_WORDS))
    wants_singular_topic = bool(reference_words.intersection({"it", "its"}))
    wants_general_topic = bool(reference_words.intersection({"this", "that", "there"}))
    wants_plural_topic = bool(reference_words.intersection({"these", "those"}))

    for history_line in reversed(extract_user_lines(chat_history, current_question=current_question)):
        if is_contextual_followup_line(history_line):
            continue

        topic = extract_topic_from_user_question(history_line)

        if not topic:
            continue

        if wants_person:
            person = extract_person_name_candidate(topic) or extract_person_name_candidate(history_line)
            if person:
                return person

            if looks_like_person_topic(topic):
                return topic

            continue

        if wants_group:
            if looks_like_group_topic(topic):
                return topic
            continue

        if wants_plural_topic:
            if looks_like_group_topic(topic):
                return topic
            continue

        if wants_singular_topic:
            if looks_like_singular_non_person_topic(topic):
                return topic
            continue

        if wants_general_topic:
            if looks_like_singular_non_person_topic(topic) or looks_like_group_topic(topic):
                return topic

    return ""


def is_contextual_that_reference(text, tokens, index):
    # "that" can be a real reference or only a relative connector.
    # Contextual examples: "what does that mean", "about that", "after that".
    # Non-contextual example: "the ladies that had relationship with Jose Rizal".
    if index == 0:
        return True

    previous_word = tokens[index - 1] if index > 0 else ""
    next_word = tokens[index + 1] if index + 1 < len(tokens) else ""

    if previous_word in {
        "about", "after", "before", "during", "from", "in", "on", "of",
        "to", "with", "without", "because", "beside", "near", "like",
    }:
        return True

    if next_word in {"mean", "means", "refer", "refers", "happen", "happened"}:
        return True

    if re.search(r"\b(?:what|why|how|when|where)\s+(?:is|was|are|were|did|does|do)\s+that\b", text):
        return True

    if re.search(r"\bthat\s*\?$", text):
        return True

    return False


def is_contextual_there_reference(text, tokens, index):
    # Avoid treating existential "were there / are there" as a previous-topic reference.
    if index == 0:
        return True

    previous_word = tokens[index - 1] if index > 0 else ""

    if previous_word in {"over", "from", "in", "there"}:
        return True

    return False


def contains_reference_word(question):
    # Detect unresolved references by their role in the question.
    # This is intentionally not a simple keyword check.
    # Connector examples that should stay standalone:
    # - "the policy that employees must follow"
    # - "the ladies that had relationship with Jose Rizal"
    text = normalize_space(question).lower().rstrip("?")

    if not text:
        return False

    reference = r"(?:it|this|that|these|those|he|she|they|them|him|her|its|his|their)"
    object_reference = r"(?:it|this|that|these|those|him|her|them)"
    preposition_reference = r"(?:it|this|that|these|those|there|he|she|they|them|him|her|its|his|their)"

    contextual_patterns = [
        # Direct pronoun subject: "it means", "he died", "they arrived".
        rf"^(?:there|{reference})\b",

        # WH + pronoun/reference subject:
        # "what it did", "what it caused", "who he was", "why she left",
        # "how they escaped", "when this happened".
        rf"^(?:who|what|where|when|why|how|which)\s+{reference}\b",

        # WH + auxiliary + pronoun/reference subject:
        # "when did he die", "what does it mean", "why was that important".
        rf"^(?:who|what|where|when|why|how|which)\s+(?:is|are|was|were|did|does|do|can|could|should|would)\s+{reference}\b",

        # Auxiliary + pronoun/reference subject:
        # "did he die", "does it apply", "can they use it".
        rf"^(?:did|does|do|is|are|was|were|can|could|should|would)\s+{reference}\b",

        # Imperative/question wrappers with only a reference as topic.
        rf"^(?:tell me about|explain|describe|discuss|give details about|give me details about)\s+{reference}\b",

        # Unresolved object reference at the end:
        # "who killed him", "did Lapu-Lapu kill him", "what caused it".
        rf"\b(?:about|after|before|during|from|in|on|of|to|with|without|because of|cause|caused|kill|killed|meet|met|see|saw|use|used|apply|applied)\s+{preposition_reference}\s*$",
        rf"\b{object_reference}\s*$",
    ]

    return any(re.search(pattern, text, flags=re.I) for pattern in contextual_patterns)


def has_clear_topic_text(topic):
    # A standalone question must contain an explicit subject, not only a pronoun/reference.
    topic = clean_topic_candidate(topic).lower()

    if not topic:
        return False

    tokens = re.findall(r"[a-zA-Z']+", topic)

    if not tokens:
        return False

    if set(tokens).issubset(REFERENCE_WORDS):
        return False

    leading_fillers = {
        "the", "a", "an", "to", "for", "of", "about", "with", "by", "from",
        "in", "on", "at", "into", "during", "after", "before", "over", "under",
    }
    meaningful_tokens = [token for token in tokens if token not in leading_fillers]

    if not meaningful_tokens:
        return False

    # If the extracted subject starts with a reference word, the question depends on history.
    # Examples: "he", "his works", "that event", "their policy", "to him".
    if meaningful_tokens[0] in REFERENCE_WORDS:
        return False

    # Object references still need history even when another explicit word exists.
    # Example: "Did Lapu-Lapu kill him?" has a subject but an unresolved object.
    unresolved_object_refs = {"it", "this", "that", "these", "those", "him", "her", "them"}
    if get_reference_words(topic).intersection(unresolved_object_refs):
        return False

    return True


def classify_question_dependency(question):
    # Generic classifier used by the chatbot to decide whether memory is needed.
    # Returns a small dict so debug logs/tests can show why a question was treated that way.
    question = normalize_space(question)

    if not question:
        return {
            "is_follow_up": False,
            "is_standalone": False,
            "reason": "empty question",
        }

    if contains_reference_word(question):
        return {
            "is_follow_up": True,
            "is_standalone": False,
            "reason": "question has an unresolved subject/object reference",
        }

    lowered = question.lower().rstrip("?")

    standalone_patterns = [
        r"^(?:who|what|where|when|which)\s+(?:is|are|was|were)\s+(.+)$",
        r"^(?:why|how|when|what|which)\s+(?:did|does|do|is|are|was|were)\s+(.+)$",
        r"^(?:who|what|where|when|why|how|which)\s+(?!is\b|are\b|was\b|were\b|did\b|does\b|do\b)(?:[a-zA-Z'’-]+)\s+(.+)$",
        r"^(?:is|are|was|were)\s+there\s+(.+)$",
        r"^(?:where|when|why|how)\s+(?:is|are|was|were)\s+there\s+(.+)$",
        r"^(?:did|does|do|is|are|was|were|can|could|should|would)\s+(.+)$",
        r"^(?:can you|could you|please)?\s*(?:explain|describe|discuss|tell me about|give details about|give me details about)\s+(.+)$",
        r"^(?:what do you know about|information about|details about)\s+(.+)$",
    ]

    for pattern in standalone_patterns:
        match = re.match(pattern, lowered, flags=re.I)

        if not match:
            continue

        topic = clean_topic_candidate(question[match.span(1)[0]:match.span(1)[1]])

        if has_clear_topic_text(topic):
            return {
                "is_follow_up": False,
                "is_standalone": True,
                "reason": "question has an explicit subject/topic",
            }

    words = question.split()
    if len(words) >= 5:
        return {
            "is_follow_up": False,
            "is_standalone": True,
            "reason": "long question without unresolved references",
        }

    if looks_like_short_followup(question):
        return {
            "is_follow_up": True,
            "is_standalone": False,
            "reason": "short contextual follow-up",
        }

    return {
        "is_follow_up": False,
        "is_standalone": False,
        "reason": "ambiguous short question without clear reference",
    }


def is_standalone_question(question):
    # Do not rewrite complete questions such as "Who is Jose Rizal?" using old chat topics.
    return classify_question_dependency(question).get("is_standalone", False)


def is_follow_up_question(question):
    # Public helper for tests and chatbot logic.
    return classify_question_dependency(question).get("is_follow_up", False)


def looks_like_short_followup(question):
    # Short questions are often contextual follow-ups.
    text = normalize_space(question).lower().rstrip("?")
    words = text.split()

    if not words:
        return False

    if contains_reference_word(text):
        return True

    if len(words) <= 4 and words[0] in {"why", "how", "when", "where", "what", "who"}:
        return True

    return False

def cleanup_reference_rewrite(text):
    # Small grammar cleanup for common follow-up rewrites.
    text = normalize_space(text)
    text = re.sub(r"\bdid\s+(.+?)\s+died\b", r"did \1 die", text, flags=re.I)
    text = re.sub(r"\bwhen\s+did\s+(.+?)\s+died\b", r"when did \1 die", text, flags=re.I)
    return text


def replace_reference_once(question, topic):
    # Replace the first clear reference with the compatible topic.
    def replacement(match):
        word = match.group(0).lower()

        if word in {"its", "his", "their"}:
            return f"{topic}'s"

        return topic

    rewritten = re.sub(
        r"\b(it|this|that|these|those|there|he|she|they|them|him|her|its|his|their)\b",
        replacement,
        question,
        count=1,
        flags=re.I,
    )

    return cleanup_reference_rewrite(rewritten)


def deterministic_followup_rewrite(question, topic):
    # Resolve common follow-up shapes without asking the LLM.
    question = normalize_space(question)
    topic = clean_topic_candidate(topic)

    if not question or not topic:
        return ""

    q = question.lower().rstrip("?")

    match = re.match(r"^what\s+(?:it|this|that|he|she|they)\s+did$", q)
    if match:
        return ensure_question_mark(f"What did {topic} do")

    match = re.match(r"^what\s+did\s+(?:it|this|that|he|she|they)\s+do$", q)
    if match:
        return ensure_question_mark(f"What did {topic} do")

    match = re.match(r"^what\s+(?:is|are|was|were)\s+(?:its|his|her|their)\s+(.+)$", q)
    if match:
        return ensure_question_mark(f"What {match.group(0).split()[1]} {topic}'s {match.group(1)}")

    if contains_reference_word(question):
        return ensure_question_mark(replace_reference_once(question, topic))

    if q in {"why", "why?"}:
        return ensure_question_mark(f"Why is {topic} important")

    if q in {"how", "how?"}:
        return ensure_question_mark(f"How does {topic} work")

    if q in {"when", "when?"}:
        return ensure_question_mark(f"When did {topic} happen")

    if q in {"where", "where?"}:
        return ensure_question_mark(f"Where did {topic} happen")

    if q in {"what happened", "what happened?"}:
        return ensure_question_mark(f"What happened to {topic}")

    return ""


def build_rewrite_prompt(question, chat_history):
    # Generic prompt used only when deterministic rewrite cannot resolve the follow-up.
    return f"""
Rewrite the follow-up question as one standalone question for document retrieval.

Conversation History:
{chat_history}

Follow-up Question:
{question}

Rules:
- Resolve he, him, or his only to the nearest previous person.
- Resolve she or her only to the nearest previous person.
- Resolve they, them, or their only to the nearest previous group or plural entity.
- Resolve it, this, that, these, or those to the nearest topic, item, event, document, policy, process, system, or requirement.
- Do not resolve a person pronoun to a country, group, event, document, policy, process, system, or topic.
- If the latest topic is not compatible with the pronoun, look further back in the conversation.
- If the follow-up already has a clear standalone subject, return it unchanged.
- If no compatible reference exists, return CLARIFY: Which person, item, or topic do you mean?
- Preserve the original meaning.
- Keep the same language as the follow-up question.
- Do not answer the question.
- Return only one standalone question or the CLARIFY line.

Standalone Question:
"""


def rewrite_question(question, chat_history, llm):
    # Rewrite only contextual follow-up questions.
    question = normalize_space(question)
    chat_history = str(chat_history or "").strip()

    if not question or not chat_history:
        return question

    if is_standalone_question(question):
        return ensure_question_mark(question)

    latest_topic = get_latest_compatible_topic(chat_history, current_question=question)
    deterministic_rewrite = deterministic_followup_rewrite(question, latest_topic)

    if deterministic_rewrite:
        return deterministic_rewrite

    prompt = build_rewrite_prompt(question=question, chat_history=chat_history)
    response = llm.invoke(prompt)

    rewritten = clean_rewritten_question(
        text=extract_text(response),
        fallback_question=question,
    )

    if is_clarification(rewritten) and latest_topic and looks_like_short_followup(question):
        fallback_rewrite = deterministic_followup_rewrite(question, latest_topic)
        if fallback_rewrite:
            return fallback_rewrite

    return rewritten
