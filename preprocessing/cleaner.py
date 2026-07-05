import re
from langchain_core.documents import Document
try:
    from config.settings import MIN_CLEAN_TEXT_LENGTH
except ImportError:
    MIN_CLEAN_TEXT_LENGTH = 100


WEB_NOISE_KEYWORDS = (
    "retrieved",
    "archived",
    "web.archive.org",
    "books.google.com",
    "internet archive",
    "librivox",
    "jstor",
)


NOISE_SECTION_TITLES = (
    "references",
    "reference",
    "further reading",
    "external links",
    "bibliography",
    "notes",
    "citations",
    "sources",
    "see also",
)


DOMAIN_EXTENSIONS = (
    "com",
    "org",
    "net",
    "gov",
    "edu",
    "jp",
    "ph",
    "io",
    "co",
)


def is_url_like_text(text):
    # Check whether text is URL/domain-like even after PDF extraction splits it.
    text = str(text or "").strip()

    if not text:
        return False

    domain_ext = "|".join(DOMAIN_EXTENSIONS)

    patterns = (
        r"https?\s*:\s*/\s*/\s*\S+",
        r"www\.\S+",
        rf"\b[a-zA-Z0-9.-]+\.({domain_ext})\b/\S*",
        r"web\.archive\.org",
        r"books\.google\.com",
        r"google\.com/books",
    )

    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def normalize_section_heading(line):
    # Make the possible section heading comparable.
    # Example: "== References ==" -> "references".
    line = str(line or "").strip().lower()

    # Remove common heading symbols from markdown/wiki/pdf extraction.
    line = re.sub(r"^[=#*_\-\s]+", "", line)
    line = re.sub(r"[=#*_\-\s]+$", "", line)

    # Remove numbering from the heading.
    # Example: "12 References" -> "references".
    line = re.sub(r"^\d+(\.\d+)*\s+", "", line)

    # Keep letters/numbers only for stable matching.
    line = re.sub(r"[^a-z0-9\s]+", " ", line)
    line = re.sub(r"\s+", " ", line).strip()
    return line


def is_noise_section_heading(line):
    # Detect common ending sections that usually create bad retrieval chunks.
    # This is generic for web/PDF docs and is not specific to history or company topics.
    heading = normalize_section_heading(line)

    if not heading:
        return False

    return heading in NOISE_SECTION_TITLES


def remove_noise_sections(text):
    # Remove the full ending/reference section before chunking.
    # Line-level cleaning alone is not enough when there are many citation/link lines.
    kept_lines = []

    for line in str(text or "").splitlines():
        if is_noise_section_heading(line):
            break

        kept_lines.append(line)

    return "\n".join(kept_lines)


def is_noise_line(line):
    # Remove only obvious noise lines to avoid deleting useful SOP/manual content.
    line = str(line or "").strip()
    lower = line.lower()

    if not line:
        return True

    # Example: 1 / 10
    if re.fullmatch(r"\d+\s*/\s*\d+", line):
        return True

    # Example: Page 1 or Page 1 of 10
    if re.fullmatch(r"page\s+\d+(\s+of\s+\d+)?", lower):
        return True

    # Example: 6/25/2026, 10:30 AM
    if re.fullmatch(r"\d{1,2}/\d{1,2}/\d{2,4},\s*\d{1,2}:\d{2}\s*(am|pm)?", lower):
        return True

    # Example: a line that only contains a URL/domain.
    if is_url_like_text(line) and len(line) < 220:
        return True

    # Example: [1] [2] [3]
    if re.fullmatch(r"(\[\d+\]\s*)+", line):
        return True

    # Scraped reference/footer lines. Limit this to short lines to avoid being too aggressive.
    if len(line) < 180 and any(keyword in lower for keyword in WEB_NOISE_KEYWORDS):
        return True

    # Simple Wikipedia title/footer noise.
    if lower.endswith("- wikipedia") and len(line) < 120:
        return True

    return False


def remove_markdown_and_links(text):
    # Convert common markdown syntax to plain text without deleting the main content.
    domain_ext = "|".join(DOMAIN_EXTENSIONS)

    # Markdown link: [label](url) -> label
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)

    # Normal URLs and PDF-broken URLs like "http s://example.com".
    text = re.sub(r"https?\s*:\s*/\s*/\s*\S+", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"h\s*t\s*t\s*p\s*s?\s*:\s*/\s*/\s*\S+", " ", text, flags=re.IGNORECASE)

    # Raw www links.
    text = re.sub(r"www\.\S+", " ", text, flags=re.IGNORECASE)

    # Domain with path/query like google.com/books?id=...
    text = re.sub(
        rf"\b[a-zA-Z0-9.-]+\.({domain_ext})\b/\S*",
        " ",
        text,
        flags=re.IGNORECASE,
    )

    # Common broken URL fragments from PDF extraction.
    # Examples: oogle.com/books, oks.google.com/books, le.com/books, chive.org/web.
    text = re.sub(
        r"\b(?:oogle|google|books\.google|oks\.google|s\.google|le|e|chive|rchive|hive|archive)\.(?:com|org)\S*",
        " ",
        text,
        flags=re.IGNORECASE,
    )

    # Domain-only leftovers. Keep this after path cleanup.
    text = re.sub(
        rf"\b[a-zA-Z0-9.-]+\.({domain_ext})\b",
        " ",
        text,
        flags=re.IGNORECASE,
    )

    # Percent-encoded URL/query leftovers.
    text = re.sub(r"%[0-9A-Fa-f]{2}", " ", text)
    text = re.sub(r"\b(id|qid|pg|dq|q|page|epage|artifactID)=\S+", " ", text, flags=re.IGNORECASE)

    # Citation markers.
    text = re.sub(r"\[\d+\]", " ", text)

    # Markdown / separator symbols.
    text = re.sub(r"^\s{0,3}#{1,6}\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"[-_=*#~—]{4,}", " ", text)
    text = re.sub(r"[_`~]+", " ", text)

    return text


def normalize_symbols(text):
    # Remove control characters and common extraction artifacts.
    text = text.replace("�", " ")
    text = re.sub(r"[\x00-\x09\x0B-\x1F\x7F-\x9F]", " ", text)
    text = re.sub(r"[«»¤©®™†‡•]", " ", text)
    text = re.sub(r"\|{2,}", " ", text)
    text = text.replace("<<", " ")
    return text


def normalize_spacing(text):
    # Fix excessive spaces and blank lines.
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


def clean_text(text):
    # General cleaning flow before chunking and embedding.
    # This is safe for company docs, SOPs, manuals, policies, and mixed-language text.
    if not text:
        return ""

    text = str(text)
    text = remove_markdown_and_links(text)
    text = normalize_symbols(text)
    text = remove_noise_sections(text)

    clean_lines = []
    for line in text.splitlines():
        line = line.strip()
        if not is_noise_line(line):
            clean_lines.append(line)

    return normalize_spacing("\n".join(clean_lines))


def clean_documents(docs, min_length=MIN_CLEAN_TEXT_LENGTH):
    # Clean a list of LangChain Document objects.
    cleaned_docs = []

    for doc in docs or []:
        cleaned_text = clean_text(getattr(doc, "page_content", ""))

        if len(cleaned_text) < min_length:
            continue

        cleaned_docs.append(
            Document(
                page_content=cleaned_text,
                metadata=dict(doc.metadata or {}),
            )
        )

    return cleaned_docs
