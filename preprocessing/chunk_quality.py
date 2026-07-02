import re
from pathlib import Path

try:
    from config.settings import MIN_DOCUMENT_LENGTH
except ImportError:
    MIN_DOCUMENT_LENGTH = 50

try:
    from config.settings import MIN_CHUNK_WORDS
except ImportError:
    MIN_CHUNK_WORDS = 25

try:
    from config.settings import MIN_CHUNK_CHARS
except ImportError:
    MIN_CHUNK_CHARS = max(120, MIN_DOCUMENT_LENGTH)

try:
    from config.settings import ENABLE_CHUNK_QUALITY_FILTER
except ImportError:
    ENABLE_CHUNK_QUALITY_FILTER = True


CHUNK_QUALITY_REPORT_FILE = Path("reports") / "chunk_quality_report.txt"

REFERENCE_SECTION_NAMES = {
    "references",
    "reference",
    "bibliography",
    "external links",
    "further reading",
    "notes",
    "citations",
    "sources",
}

WEAK_TEXT_MARKERS = (
    "retrieved",
    "archived",
    "archive",
    "isbn",
    "doi",
    "external links",
    "further reading",
    "bibliography",
)


def normalize_text(text):
    # Lowercase + normalize spaces para stable ang debug/checking.
    text = str(text or "").lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def get_doc_text(doc):
    # Safe getter ng chunk text.
    return str(getattr(doc, "page_content", "") or "")


def get_doc_metadata(doc):
    # Safe getter ng chunk metadata.
    return dict(getattr(doc, "metadata", {}) or {})


def get_source_label(doc):
    # Human-readable source for reports.
    metadata = get_doc_metadata(doc)
    return (
        metadata.get("file_name")
        or metadata.get("source")
        or metadata.get("title")
        or "unknown"
    )


def get_chunk_label(doc):
    # Stable chunk id for reports.
    metadata = get_doc_metadata(doc)
    return metadata.get("chunk_id") or metadata.get("chunk_index") or "N/A"


def get_body_text(text):
    # Quality checking should focus on actual text, not the retrieval metadata prefix.
    text = str(text or "").strip()

    if not text.lower().startswith("retrieval context:"):
        return text

    if "\n\n" in text:
        return text.split("\n\n", 1)[1].strip()

    return re.sub(r"^retrieval context\s*:\s*", "", text, flags=re.IGNORECASE).strip()


def get_section_name(doc):
    # Get section from metadata. This is generic and not source-specific.
    metadata = get_doc_metadata(doc)
    section = (
        metadata.get("section")
        or metadata.get("heading")
        or metadata.get("section_title")
        or ""
    )
    return normalize_text(section)


def is_reference_section(doc):
    # Skip common reference/bibliography sections.
    section = get_section_name(doc)

    if not section:
        return False

    for section_name in REFERENCE_SECTION_NAMES:
        if section == section_name:
            return True

        if section.startswith(section_name + " "):
            return True

    return False


def count_words(text):
    # Count words for English/Japanese/mixed content.
    return len(re.findall(r"[A-Za-z0-9\u3040-\u30ff\u3400-\u9fff]+", str(text or "")))


def count_weak_markers(text):
    # Count generic reference/noise markers.
    normalized = normalize_text(text)
    count = 0

    for marker in WEAK_TEXT_MARKERS:
        if marker in normalized:
            count += 1

    return count


def looks_like_url_or_reference_list(text):
    # Detect chunks that are mostly references, links, archive/citation details.
    text = str(text or "")
    normalized = normalize_text(text)

    patterns = (
        r"https?\s*:\s*/\s*/",
        r"\bwww\.",
        r"\bdoi\b",
        r"\bisbn\b",
        r"\bretrieved\b",
        r"\barchived\b",
        r"\barchive\b",
        r"\b(accessed|publisher|press)\b",
    )

    hits = 0

    for pattern in patterns:
        if re.search(pattern, normalized):
            hits += 1

    return hits >= 3


def looks_like_metadata_only(raw_text, body_text):
    # Skip chunks where the prefix dominates and body is too small.
    raw_text = str(raw_text or "")
    body_text = str(body_text or "")

    if not raw_text.strip():
        return True

    if not raw_text.lower().startswith("retrieval context:"):
        return False

    if count_words(body_text) >= MIN_CHUNK_WORDS:
        return False

    return True


def is_too_short(body_text):
    # Very short chunks are usually weak final context.
    body_text = str(body_text or "").strip()

    if len(body_text) < MIN_CHUNK_CHARS:
        return True

    if count_words(body_text) < MIN_CHUNK_WORDS:
        return True

    return False


def get_chunk_quality_issue(doc):
    # Return empty string if chunk is usable.
    raw_text = get_doc_text(doc)
    body_text = get_body_text(raw_text)

    if not body_text.strip():
        return "empty_text"

    if is_reference_section(doc):
        return "reference_section"

    if looks_like_metadata_only(raw_text, body_text):
        return "metadata_only"

    if is_too_short(body_text):
        return "too_short"

    if looks_like_url_or_reference_list(body_text):
        return "reference_like_text"

    if count_weak_markers(body_text) >= 3:
        return "too_many_weak_markers"

    return ""


def add_quality_metadata(doc, status, issue=""):
    # Add debug metadata without changing the chunk text.
    metadata = get_doc_metadata(doc)
    body_text = get_body_text(get_doc_text(doc))

    metadata["chunk_quality_status"] = status
    metadata["chunk_quality_issue"] = issue
    metadata["chunk_body_chars"] = len(body_text)
    metadata["chunk_body_words"] = count_words(body_text)

    doc.metadata = metadata
    return doc


def filter_quality_chunks(chunks, report_path=CHUNK_QUALITY_REPORT_FILE):
    # Filter weak chunks before cache/embedding.
    # Generic ito: length, reference sections, URL/citation noise, metadata-only text.
    chunks = list(chunks or [])

    if not ENABLE_CHUNK_QUALITY_FILTER:
        kept_chunks = [add_quality_metadata(chunk, "not_filtered", "disabled") for chunk in chunks]
        write_chunk_quality_report(report_path, len(chunks), kept_chunks, [])
        return kept_chunks

    kept_chunks = []
    skipped_items = []

    for chunk in chunks:
        issue = get_chunk_quality_issue(chunk)

        if issue:
            chunk = add_quality_metadata(chunk, "skipped", issue)
            skipped_items.append(build_skip_item(chunk, issue))
            continue

        kept_chunks.append(add_quality_metadata(chunk, "kept", ""))

    # Safety fallback: never allow the filter to wipe out the whole dataset.
    if chunks and not kept_chunks:
        kept_chunks = [add_quality_metadata(chunk, "kept_by_fallback", "filter_removed_all") for chunk in chunks]
        skipped_items = []

    write_chunk_quality_report(report_path, len(chunks), kept_chunks, skipped_items)
    return kept_chunks


def build_skip_item(doc, issue):
    # Build one skipped record for the report.
    metadata = get_doc_metadata(doc)
    preview = get_body_text(get_doc_text(doc))[:260].replace("\n", " ")

    return {
        "reason": issue,
        "source": get_source_label(doc),
        "page": metadata.get("page", "N/A"),
        "chunk": get_chunk_label(doc),
        "words": metadata.get("chunk_body_words", "N/A"),
        "chars": metadata.get("chunk_body_chars", "N/A"),
        "preview": preview,
    }


def write_chunk_quality_report(report_path, total_chunks, kept_chunks, skipped_items):
    # Write simple debug report.
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    reason_counts = {}

    for item in skipped_items:
        reason = item.get("reason", "unknown")
        reason_counts[reason] = reason_counts.get(reason, 0) + 1

    lines = []
    lines.append("=" * 80)
    lines.append("CHUNK QUALITY REPORT")
    lines.append("=" * 80)
    lines.append(f"Total chunks   : {total_chunks}")
    lines.append(f"Kept chunks    : {len(kept_chunks)}")
    lines.append(f"Skipped chunks : {len(skipped_items)}")
    lines.append("")
    lines.append("CONFIG")
    lines.append("-" * 80)
    lines.append(f"Enabled         : {ENABLE_CHUNK_QUALITY_FILTER}")
    lines.append(f"Min chunk chars : {MIN_CHUNK_CHARS}")
    lines.append(f"Min chunk words : {MIN_CHUNK_WORDS}")
    lines.append("")
    lines.append("SKIPPED REASONS")
    lines.append("-" * 80)

    if reason_counts:
        for reason, count in sorted(reason_counts.items()):
            lines.append(f"{reason:<30} {count}")
    else:
        lines.append("None")

    lines.append("")
    lines.append("SKIPPED SAMPLES")
    lines.append("-" * 80)

    if skipped_items:
        for item in skipped_items[:100]:
            lines.append(f"Reason : {item['reason']}")
            lines.append(f"Source : {item['source']}")
            lines.append(f"Page   : {item['page']}")
            lines.append(f"Chunk  : {item['chunk']}")
            lines.append(f"Words  : {item['words']}")
            lines.append(f"Chars  : {item['chars']}")
            lines.append(f"Preview: {item['preview']}")
            lines.append("")
    else:
        lines.append("None")

    report_path.write_text("\n".join(lines), encoding="utf-8")
