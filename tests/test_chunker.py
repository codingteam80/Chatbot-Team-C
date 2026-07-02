import sys
from pathlib import Path
from collections import defaultdict


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_FOLDER = PROJECT_ROOT / "data"
REPORT_FOLDER = PROJECT_ROOT / "reports"
REPORT_FILE = REPORT_FOLDER / "chunker_test_report.txt"

PREVIEW_CHARS = 700
SAMPLE_CHUNKS_PER_FILE = 3
VERY_SHORT_CHUNK_CHARS = 80

# Required metadata na dapat meron sa bawat chunk.
REQUIRED_METADATA_KEYS = ["source"]

# Optional metadata na useful sa debugging, source display, at retrieval report.
OPTIONAL_METADATA_KEYS = [
    "file_name",
    "file_path",
    "file_type",
    "title",
    "section",
    "page",
    "chunk_index",
    "language",
]


if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from loaders.document_loader import load_documents
from preprocessing.cleaner import clean_documents
from preprocessing.chunker import chunk_documents


def get_text(doc):
    # Kunin ang text ng Document object.
    return str(getattr(doc, "page_content", "") or "")


def get_metadata(doc):
    # Kunin ang metadata ng Document object.
    return dict(getattr(doc, "metadata", {}) or {})


def get_source(doc):
    # Kunin ang source file mula sa metadata.
    metadata = get_metadata(doc)
    return (
        metadata.get("source")
        or metadata.get("file_path")
        or metadata.get("file_name")
        or "Unknown source"
    )


def group_by_source(docs):
    # I-group ang docs/chunks per source file.
    grouped = defaultdict(list)

    for doc in docs or []:
        grouped[get_source(doc)].append(doc)

    return dict(grouped)


def text_lengths(docs):
    # Kunin ang length ng bawat doc/chunk.
    return [len(get_text(doc).strip()) for doc in docs or []]


def average(values):
    # Compute average nang safe kahit empty list.
    if not values:
        return 0

    return round(sum(values) / len(values), 2)


def count_very_short_chunks(chunks):
    # Bilangin chunks na sobrang ikli.
    return sum(
        1
        for chunk in chunks or []
        if 0 < len(get_text(chunk).strip()) < VERY_SHORT_CHUNK_CHARS
    )


def count_empty_metadata(docs):
    # Bilangin chunks na walang kahit anong metadata.
    count = 0

    for doc in docs or []:
        metadata = get_metadata(doc)

        if not metadata:
            count += 1

    return count


def count_missing_source(docs):
    # Bilangin chunks na walang source metadata.
    count = 0

    for doc in docs or []:
        metadata = get_metadata(doc)

        if not metadata.get("source") and not metadata.get("file_name"):
            count += 1

    return count


def count_missing_required_metadata(docs, required_keys=None):
    # Bilangin chunks na kulang sa required metadata keys.
    required_keys = required_keys or REQUIRED_METADATA_KEYS
    count = 0

    for doc in docs or []:
        metadata = get_metadata(doc)

        if any(not metadata.get(key) for key in required_keys):
            count += 1

    return count


def get_metadata_key_counts(docs):
    # Bilangin kung ilang chunks ang meron ng bawat metadata key.
    key_counts = defaultdict(int)

    for doc in docs or []:
        metadata = get_metadata(doc)

        for key, value in metadata.items():
            if value not in [None, ""]:
                key_counts[key] += 1

    return dict(sorted(key_counts.items()))


def get_missing_metadata_examples(docs, required_keys=None, limit=5):
    # Kumuha ng sample chunks na kulang sa required metadata.
    required_keys = required_keys or REQUIRED_METADATA_KEYS
    examples = []

    for index, doc in enumerate(docs or []):
        metadata = get_metadata(doc)
        missing_keys = [key for key in required_keys if not metadata.get(key)]

        if missing_keys:
            examples.append({
                "chunk_number": index + 1,
                "missing_keys": missing_keys,
                "metadata": metadata,
                "preview": get_preview(get_text(doc)),
            })

        if len(examples) >= limit:
            break

    return examples


def format_metadata(metadata):
    # I-format ang metadata para mas madaling basahin sa report.
    if not metadata:
        return ["Metadata : {}"]

    lines = ["Metadata :"]

    for key in sorted(metadata.keys()):
        lines.append(f"  {key}: {metadata.get(key)}")

    return lines


def get_preview(text):
    # Gawing one-preview block ang text.
    text = " ".join(str(text or "").split())
    return text[:PREVIEW_CHARS]


def add_metadata_summary(lines, chunks):
    # Idagdag sa report ang metadata quality check.
    total_chunks = len(chunks or [])
    empty_metadata = count_empty_metadata(chunks)
    missing_source = count_missing_source(chunks)
    missing_required = count_missing_required_metadata(chunks)
    key_counts = get_metadata_key_counts(chunks)

    lines.extend([
        "METADATA SUMMARY",
        "-" * 80,
        f"Required keys            : {', '.join(REQUIRED_METADATA_KEYS)}",
        f"Optional useful keys     : {', '.join(OPTIONAL_METADATA_KEYS)}",
        f"Chunks with metadata     : {total_chunks - empty_metadata}/{total_chunks}",
        f"Empty metadata chunks    : {empty_metadata}",
        f"Missing source chunks    : {missing_source}",
        f"Missing required chunks  : {missing_required}",
        "Metadata key coverage    :",
    ])

    if not key_counts:
        lines.append("  No metadata keys found.")
    else:
        for key, count in key_counts.items():
            lines.append(f"  {key}: {count}/{total_chunks}")

    lines.append("")

    missing_examples = get_missing_metadata_examples(chunks)

    if missing_examples:
        lines.extend([
            "MISSING METADATA EXAMPLES",
            "-" * 80,
        ])

        for example in missing_examples:
            lines.extend([
                f"Chunk number : {example['chunk_number']}",
                f"Missing keys : {', '.join(example['missing_keys'])}",
                f"Metadata     : {example['metadata']}",
                "Preview      :",
                example["preview"] or "[Empty chunk]",
                "",
            ])


def build_report(raw_docs, cleaned_docs, chunks, load_report=None):
    # Gumawa ng readable chunk test report.
    raw_by_source = group_by_source(raw_docs)
    cleaned_by_source = group_by_source(cleaned_docs)
    chunks_by_source = group_by_source(chunks)

    chunk_lengths = text_lengths(chunks)
    all_sources = sorted(set(raw_by_source.keys()) | set(cleaned_by_source.keys()) | set(chunks_by_source.keys()))

    lines = [
        "=" * 80,
        "CHUNKER TEST REPORT",
        "=" * 80,
        f"Data folder       : {DATA_FOLDER}",
        f"Raw documents     : {len(raw_docs)}",
        f"Cleaned documents : {len(cleaned_docs)}",
        f"Chunks created    : {len(chunks)}",
        "",
        "CHUNK SIZE SUMMARY",
        "-" * 80,
        f"Min chunk chars   : {min(chunk_lengths) if chunk_lengths else 0}",
        f"Avg chunk chars   : {average(chunk_lengths)}",
        f"Max chunk chars   : {max(chunk_lengths) if chunk_lengths else 0}",
        f"Very short chunks : {count_very_short_chunks(chunks)}",
        f"Missing source    : {count_missing_source(chunks)}",
        "",
    ]

    add_metadata_summary(lines, chunks)

    if load_report:
        lines.extend([
            "LOADER SUMMARY",
            "-" * 80,
            f"Loaded files : {load_report.get('loaded_files', 0)}",
            f"Loaded docs  : {load_report.get('loaded_docs', 0)}",
            f"Skipped files: {len(load_report.get('skipped_files', []))}",
            f"Failed files : {len(load_report.get('failed_files', []))}",
            "",
        ])

    lines.extend([
        "=" * 80,
        "CHUNKS PER FILE",
        "=" * 80,
        "",
    ])

    if not all_sources:
        lines.append("No files/documents found.")
        return lines

    for file_index, source in enumerate(all_sources, start=1):
        file_raw_docs = raw_by_source.get(source, [])
        file_cleaned_docs = cleaned_by_source.get(source, [])
        file_chunks = chunks_by_source.get(source, [])
        file_chunk_lengths = text_lengths(file_chunks)

        lines.extend([
            "-" * 80,
            f"{file_index}. {source}",
            "-" * 80,
            f"Raw docs       : {len(file_raw_docs)}",
            f"Cleaned docs   : {len(file_cleaned_docs)}",
            f"Chunks         : {len(file_chunks)}",
            f"Min chars      : {min(file_chunk_lengths) if file_chunk_lengths else 0}",
            f"Avg chars      : {average(file_chunk_lengths)}",
            f"Max chars      : {max(file_chunk_lengths) if file_chunk_lengths else 0}",
            "",
        ])

        for chunk_index, chunk in enumerate(file_chunks[:SAMPLE_CHUNKS_PER_FILE], start=1):
            metadata = get_metadata(chunk)
            text = get_text(chunk)

            lines.extend([
                f"SAMPLE CHUNK {chunk_index}",
                f"Chars    : {len(text)}",
            ])
            lines.extend(format_metadata(metadata))
            lines.extend([
                "Preview  :",
                get_preview(text) or "[Empty chunk]",
                "",
            ])

        if len(file_chunks) > SAMPLE_CHUNKS_PER_FILE:
            lines.append(f"... {len(file_chunks) - SAMPLE_CHUNKS_PER_FILE} more chunks hidden for this file.")
            lines.append("")

    return lines


def main():
    # Main test para makita kung maayos ang chunks bago embedding.
    if not DATA_FOLDER.exists():
        print(f"Data folder not found: {DATA_FOLDER}")
        return

    raw_docs, load_report = load_documents(
        DATA_FOLDER,
        recursive=True,
        return_report=True,
    )

    cleaned_docs = clean_documents(raw_docs)
    chunks = chunk_documents(cleaned_docs)

    report_lines = build_report(
        raw_docs=raw_docs,
        cleaned_docs=cleaned_docs,
        chunks=chunks,
        load_report=load_report,
    )

    REPORT_FOLDER.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.write_text("\n".join(report_lines), encoding="utf-8")

    print("\n".join(report_lines))
    print(f"\nReport saved to: {REPORT_FILE}")


if __name__ == "__main__":
    main()
