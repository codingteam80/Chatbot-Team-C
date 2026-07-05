from collections import defaultdict
from pathlib import Path
import re

from langchain_text_splitters import RecursiveCharacterTextSplitter

try:
    from config.settings import (
        ADD_RETRIEVAL_CONTEXT_PREFIX,
        CHUNK_OVERLAP,
        CHUNK_SIZE,
        MIN_DOCUMENT_LENGTH,
        USE_HISTORY_TEST_CATEGORY,
    )
except ImportError:
    # Fallback so the code remains runnable while settings.py is missing.
    CHUNK_SIZE = 900
    CHUNK_OVERLAP = 150
    MIN_DOCUMENT_LENGTH = 50
    ADD_RETRIEVAL_CONTEXT_PREFIX = True
    USE_HISTORY_TEST_CATEGORY = True


DEFAULT_SEPARATORS = [
    "\n# ",
    "\n## ",
    "\n### ",
    "\n\n",
    "\n",
    ". ",
    "。",
    "、",
    " ",
    "",
]

# ADD_RETRIEVAL_CONTEXT_PREFIX and USE_HISTORY_TEST_CATEGORY are settings-driven.
# Use USE_HISTORY_TEST_CATEGORY=True while using the history/Wikipedia sample data.
# Set this to False in config/settings.py when the data folder contains company documents.


JAPANESE_TEXT_PATTERN = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff]")
MARKDOWN_HEADING_PATTERN = re.compile(r"^\s{0,3}#{1,6}\s+(.+?)\s*$", re.MULTILINE)

CATEGORY_KEYWORDS = {
    "coding": [
        "coding",
        "code review",
        "source code",
        "secure coding",
        "misra",
        "software release",
        "design review",
        "programming",
        "developer",
        "コード",
        "レビュー",
        "設計",
        "リリース",
    ],
    "security": [
        "security",
        "secure",
        "password",
        "access control",
        "authentication",
        "authorization",
        "vulnerability",
        "セキュリティ",
        "パスワード",
        "認証",
    ],
    "incident": [
        "incident",
        "issue report",
        "escalation",
        "root cause",
        "corrective action",
        "障害",
        "インシデント",
        "報告",
    ],
    "it": [
        "it acceptable use",
        "acceptable use",
        "device",
        "network",
        "email use",
        "internet use",
        "情報システム",
        "ネットワーク",
        "メール",
    ],
    "hr": [
        "leave",
        "attendance",
        "absence",
        "holiday",
        "overtime",
        "employee",
        "hr",
        "勤怠",
        "休暇",
        "社員",
    ],
}

DOC_TYPE_KEYWORDS = {
    "sop": ["sop", "standard operating procedure", "procedure", "process", "手順", "標準手順"],
    "policy": ["policy", "policies", "規程", "ポリシー"],
    "manual": ["manual", "handbook", "マニュアル"],
    "guideline": ["guideline", "guide", "rules", "standard", "misra", "ガイドライン", "ルール", "規則"],
    "checklist": ["checklist", "check list", "チェックリスト"],
    "report": ["report", "summary", "報告", "レポート"],
    "article": ["article", "wikipedia"],
}

HISTORY_TEST_KEYWORDS = [
    "wikipedia",
    "bonifacio",
    "mabini",
    "aguinaldo",
    "gomburza",
    "rizal",
    "katipunan",
    "philippine revolution",
    "philippine american war",
    "spanish american war",
    "japanese occupation",
    "lapu",
    "magellan",
]

SEARCHABLE_METADATA_KEYS = [
    "title",
    "section",
    "category",
    "doc_type",
    "source",
    "file_name",
    "language",
]

REQUIRED_METADATA_KEYS = [
    "source",
    "file_name",
    "file_type",
    "title",
    "category",
    "doc_type",
    "section",
    "language",
    "chunk_index",
    "chunk_id",
]


def validate_chunk_settings(chunk_size, chunk_overlap):
    # Check whether chunk settings are valid.
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")

    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must not be negative")

    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")


def get_text(doc):
    # Safe getter for page_content.
    return str(getattr(doc, "page_content", "") or "")


def get_valid_documents(docs, min_length=MIN_DOCUMENT_LENGTH):
    # Remove empty or very short docs before chunking.
    valid_docs = []

    for doc in docs or []:
        text = get_text(doc).strip()
        if len(text) >= min_length:
            valid_docs.append(doc)

    return valid_docs


def clean_metadata_value(value):
    # Chroma metadata must be simple types only.
    if value is None:
        return ""

    if isinstance(value, (str, int, float, bool)):
        return value

    return str(value)


def normalize_metadata(metadata):
    # Clean metadata keys/values for stable vectorstore usage.
    normalized = {}

    for key, value in dict(metadata or {}).items():
        safe_key = str(key or "").strip()
        safe_value = clean_metadata_value(value)

        if safe_key and safe_value != "":
            normalized[safe_key] = safe_value

    return normalized


def get_first_existing(metadata, keys):
    # Get the first existing metadata field.
    for key in keys:
        value = str(metadata.get(key, "") or "").strip()
        if value:
            return value

    return ""


def get_file_name_from_path(path_value):
    # Get the file name from either a Windows path or Linux path.
    path_text = str(path_value or "").strip()

    if not path_text:
        return ""

    return path_text.replace("\\", "/").split("/")[-1]


def get_file_stem(file_name):
    # Use the file name as a readable title.
    if not file_name:
        return "Untitled document"

    return Path(file_name).stem or file_name


def get_file_type(file_name):
    # Get the extension, such as .pdf, .docx, or .md.
    if not file_name:
        return ""

    return Path(file_name).suffix.lower()


def normalize_for_detect(text):
    # Simple lowercase text for keyword detection.
    return " ".join(str(text or "").lower().replace("_", " ").replace("-", " ").split())


def detect_language(text, metadata):
    # Simple English/Japanese detection.
    existing_language = str(metadata.get("language", "") or "").strip().lower()

    if existing_language:
        return existing_language

    if JAPANESE_TEXT_PATTERN.search(str(text or "")):
        return "ja"

    return "en"


def get_metadata_haystack(metadata):
    # Use file/title metadata first to avoid confusion from random chunk words.
    return normalize_for_detect(
        " ".join([
            str(metadata.get("source", "") or ""),
            str(metadata.get("file_name", "") or ""),
            str(metadata.get("title", "") or ""),
        ])
    )




def is_history_test_document(metadata):
    # Optional test-data mode only.
    # When True, old Wikipedia/history sample files use category=history and doc_type=article.
    # When False, normal generic category/doc_type detection is used.
    if not USE_HISTORY_TEST_CATEGORY:
        return False

    haystack = get_metadata_haystack(metadata)

    for keyword in HISTORY_TEST_KEYWORDS:
        if normalize_for_detect(keyword) in haystack:
            return True

    return False


def detect_from_keywords(haystack, keyword_map, default_value):
    # Generic detector for category/doc_type.
    for label, keywords in keyword_map.items():
        for keyword in keywords:
            if normalize_for_detect(keyword) in haystack:
                return label

    return default_value


def detect_category(text, metadata):
    # Category = broad business area of the document.
    existing_category = str(metadata.get("category", "") or "").strip().lower()

    if existing_category:
        return existing_category

    if is_history_test_document(metadata):
        return "history"

    haystack = get_metadata_haystack(metadata)
    return detect_from_keywords(haystack, CATEGORY_KEYWORDS, "general")


def detect_doc_type(text, metadata):
    # Doc type = SOP, policy, manual, guideline, etc.
    existing_doc_type = str(metadata.get("doc_type", "") or "").strip().lower()

    if existing_doc_type:
        return existing_doc_type

    if is_history_test_document(metadata):
        return "article"

    haystack = get_metadata_haystack(metadata)
    return detect_from_keywords(haystack, DOC_TYPE_KEYWORDS, "document")


def extract_section(text, metadata):
    # Get the section from metadata, markdown heading, or first useful line.
    existing_section = get_first_existing(metadata, ["section", "heading", "header", "page_title"])

    if existing_section:
        return existing_section

    text = str(text or "")
    match = MARKDOWN_HEADING_PATTERN.search(text)

    if match:
        return match.group(1).strip()

    for line in text.splitlines():
        cleaned_line = line.strip().strip("-:：")

        if not cleaned_line:
            continue

        if cleaned_line.lower().startswith("retrieval context:"):
            continue

        if 3 <= len(cleaned_line) <= 100 and not cleaned_line.endswith("."):
            return cleaned_line

    return ""


def enrich_document_metadata(metadata, text):
    # Main document-level metadata enrichment.
    metadata = normalize_metadata(metadata)

    source = get_first_existing(metadata, ["source", "file_path", "path", "file_name", "filename", "name"])
    file_name = get_first_existing(metadata, ["file_name", "filename", "name"])

    if not file_name:
        file_name = get_file_name_from_path(source)

    if not source:
        source = file_name or "unknown"

    title = get_first_existing(metadata, ["title", "document_title", "doc_title"])
    if not title:
        title = get_file_stem(file_name or source)

    file_type = get_first_existing(metadata, ["file_type", "extension", "ext"])
    if not file_type:
        file_type = get_file_type(file_name or source)

    metadata["source"] = source
    metadata["file_name"] = file_name or get_file_name_from_path(source) or source
    metadata["file_type"] = file_type
    metadata["title"] = title
    metadata["language"] = detect_language(text, metadata)
    metadata["category"] = detect_category(text, metadata)
    metadata["doc_type"] = detect_doc_type(text, metadata)

    return metadata


def get_source_key(metadata):
    # Stable key for per-source chunk numbering.
    return (
        str(metadata.get("source", "") or "").strip()
        or str(metadata.get("file_name", "") or "").strip()
        or "unknown"
    )


def update_section_metadata(metadata, text, last_section_by_source):
    # Carry forward the previous section when the chunk is a continuation.
    source_key = get_source_key(metadata)
    section = extract_section(text, metadata)

    if section:
        metadata["section"] = section
        last_section_by_source[source_key] = section
        return metadata

    if source_key in last_section_by_source:
        metadata["section"] = last_section_by_source[source_key]
        return metadata

    metadata["section"] = str(metadata.get("title", "") or "General").strip() or "General"
    return metadata


def build_retrieval_context(metadata):
    # Compact metadata text version to embed together with the chunk.
    parts = []

    for key in SEARCHABLE_METADATA_KEYS:
        value = str(metadata.get(key, "") or "").strip()
        if value:
            parts.append(f"{key}: {value}")

    return " | ".join(parts)


def add_retrieval_context_prefix(chunk, metadata):
    # Prepend the metadata context to the chunk text.
    if not ADD_RETRIEVAL_CONTEXT_PREFIX:
        return chunk

    text = get_text(chunk).strip()

    if not text:
        return chunk

    if text.startswith("Retrieval context:"):
        return chunk

    retrieval_context = build_retrieval_context(metadata)

    if retrieval_context:
        chunk.page_content = f"Retrieval context: {retrieval_context}\n\n{text}"

    return chunk


def add_chunk_metadata(chunks, chunk_size, chunk_overlap):
    # Add retrieval-ready metadata to each chunk.
    source_counts = defaultdict(int)
    last_section_by_source = {}

    for global_index, chunk in enumerate(chunks or []):
        original_text = get_text(chunk)
        metadata = enrich_document_metadata(chunk.metadata or {}, original_text)
        metadata = update_section_metadata(metadata, original_text, last_section_by_source)

        source_key = get_source_key(metadata)
        source_chunk_index = source_counts[source_key]
        source_counts[source_key] += 1

        metadata["chunk_index"] = source_chunk_index
        metadata["global_chunk_index"] = global_index
        metadata["chunk_id"] = f"{source_key}::chunk_{source_chunk_index}"
        metadata["chunk_size"] = chunk_size
        metadata["chunk_overlap"] = chunk_overlap
        metadata["retrieval_context"] = build_retrieval_context(metadata)

        for required_key in REQUIRED_METADATA_KEYS:
            metadata.setdefault(required_key, "")

        chunk.metadata = normalize_metadata(metadata)
        add_retrieval_context_prefix(chunk, chunk.metadata)

    return chunks


def chunk_documents(docs, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    # Main function: cleaned docs -> metadata-ready chunks.
    validate_chunk_settings(chunk_size, chunk_overlap)
    valid_docs = get_valid_documents(docs)

    if not valid_docs:
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=DEFAULT_SEPARATORS,
    )

    chunks = splitter.split_documents(valid_docs)
    return add_chunk_metadata(chunks, chunk_size, chunk_overlap)
