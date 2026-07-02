import sys
from pathlib import Path
from collections import defaultdict


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_FOLDER = PROJECT_ROOT / "data"
REPORT_FOLDER = PROJECT_ROOT / "reports"
REPORT_FILE = REPORT_FOLDER / "test_cleaner_report.txt"
PREVIEW_CHARS = 800


if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from loaders.document_loader import load_documents
from preprocessing.cleaner import clean_documents


def get_source(doc):
    # Kunin ang file source sa metadata.
    metadata = dict(getattr(doc, "metadata", {}) or {})
    return metadata.get("source") or metadata.get("file_path") or metadata.get("file_name") or "Unknown source"


def group_docs_by_source(docs):
    # I-group ang documents per file para makita bawat file.
    grouped_docs = defaultdict(list)

    for doc in docs or []:
        grouped_docs[get_source(doc)].append(doc)

    return dict(grouped_docs)


def get_total_chars(docs):
    # Bilangin ang total characters ng documents.
    return sum(len(str(getattr(doc, "page_content", "") or "")) for doc in docs or [])


def get_preview(docs):
    # Kumuha ng maikling preview ng cleaned text.
    text_parts = []

    for doc in docs or []:
        text = str(getattr(doc, "page_content", "") or "").strip()

        if text:
            text_parts.append(text)

        if len("\n\n".join(text_parts)) >= PREVIEW_CHARS:
            break

    return "\n\n".join(text_parts)[:PREVIEW_CHARS].strip()


def build_report(raw_docs, cleaned_docs, load_report=None):
    # Gumawa ng simple report para makita kung gumana ang cleaner.
    raw_by_source = group_docs_by_source(raw_docs)
    cleaned_by_source = group_docs_by_source(cleaned_docs)

    all_sources = sorted(set(raw_by_source.keys()) | set(cleaned_by_source.keys()))

    lines = [
        "=" * 80,
        "CLEANER TEST REPORT",
        "=" * 80,
        f"Data folder       : {DATA_FOLDER}",
        f"Raw documents     : {len(raw_docs)}",
        f"Cleaned documents : {len(cleaned_docs)}",
        f"Raw characters    : {get_total_chars(raw_docs)}",
        f"Cleaned characters: {get_total_chars(cleaned_docs)}",
        "",
    ]

    if load_report:
        lines.extend([
            "=" * 80,
            "LOADER SUMMARY",
            "=" * 80,
            f"Loaded files : {load_report.get('loaded_files', 0)}",
            f"Loaded docs  : {load_report.get('loaded_docs', 0)}",
            f"Skipped files: {len(load_report.get('skipped_files', []))}",
            f"Failed files : {len(load_report.get('failed_files', []))}",
            "",
        ])

    lines.extend([
        "=" * 80,
        "CLEANED FILES",
        "=" * 80,
        "",
    ])

    if not all_sources:
        lines.append("No documents found.")
        return lines

    for index, source in enumerate(all_sources, start=1):
        raw_file_docs = raw_by_source.get(source, [])
        cleaned_file_docs = cleaned_by_source.get(source, [])

        raw_chars = get_total_chars(raw_file_docs)
        cleaned_chars = get_total_chars(cleaned_file_docs)
        preview = get_preview(cleaned_file_docs)

        lines.extend([
            "-" * 80,
            f"{index}. {source}",
            "-" * 80,
            f"Raw docs       : {len(raw_file_docs)}",
            f"Cleaned docs   : {len(cleaned_file_docs)}",
            f"Raw chars      : {raw_chars}",
            f"Cleaned chars  : {cleaned_chars}",
            "",
            "CLEANED PREVIEW:",
            preview if preview else "[No cleaned text after filtering]",
            "",
        ])

    return lines


def main():
    # Simple test para sa lahat ng documents sa data folder.
    if not DATA_FOLDER.exists():
        print(f"Data folder not found: {DATA_FOLDER}")
        return

    raw_docs, load_report = load_documents(
        DATA_FOLDER,
        recursive=True,
        return_report=True,
    )

    cleaned_docs = clean_documents(raw_docs)

    report_lines = build_report(
        raw_docs=raw_docs,
        cleaned_docs=cleaned_docs,
        load_report=load_report,
    )

    REPORT_FOLDER.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.write_text("\n".join(report_lines), encoding="utf-8")

    print("\n".join(report_lines))
    print(f"\nReport saved to: {REPORT_FILE}")


if __name__ == "__main__":
    main()
