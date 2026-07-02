import re
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_FOLDER = PROJECT_ROOT / "data"
REPORT_FOLDER = PROJECT_ROOT / "reports"
REPORT_FILE = REPORT_FOLDER / "cleaner_url_check_report.txt"
MAX_MATCHES_PER_DOC = 10


if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from loaders.document_loader import load_documents
from preprocessing.cleaner import clean_documents


URL_PATTERNS = [
    r"https?://\S+",
    r"www\.\S+",
    r"web\.archive\.org",
    r"books\.google\.com",
    r"google\.com/books",
    r"\b[a-zA-Z0-9.-]+\.(com|org|net|gov|edu|jp|ph|io|co)\b/\S*",
    r"\b[a-zA-Z0-9.-]+\.(com|org|net|gov|edu|jp|ph|io|co)\b",
]


def get_doc_source(doc):
    # Metadata source lang ito para malaman kung anong file may naiwan na URL-like text.
    metadata = dict(getattr(doc, "metadata", {}) or {})
    return metadata.get("source") or metadata.get("file_path") or metadata.get("file_name") or "Unknown source"


def find_url_like_text(text):
    # Hanapin ang URL-like text sa mismong content, hindi sa metadata.
    matches = []

    for pattern in URL_PATTERNS:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            start = max(match.start() - 100, 0)
            end = min(match.end() + 100, len(text))

            context = text[start:end]
            context = re.sub(r"\s+", " ", context).strip()

            matches.append({
                "pattern": pattern,
                "match": match.group(0),
                "context": context,
            })

    return matches


def build_report(cleaned_docs, load_report):
    # Gumawa ng report ng URL-like matches sa cleaned documents.
    lines = [
        "=" * 80,
        "CLEANER URL CHECK REPORT",
        "=" * 80,
        f"Data folder    : {DATA_FOLDER}",
        f"Loaded files   : {load_report.get('loaded_files', 0)}",
        f"Loaded docs    : {load_report.get('loaded_docs', 0)}",
        f"Cleaned docs   : {len(cleaned_docs)}",
        "",
        "Note:",
        "This test checks cleaned doc.page_content only.",
        "metadata['source'] is not checked because it is just the file reference for the Source UI.",
        "",
    ]

    total_matches = 0
    docs_with_matches = 0

    for doc_index, doc in enumerate(cleaned_docs, start=1):
        text = str(getattr(doc, "page_content", "") or "")
        source = get_doc_source(doc)
        matches = find_url_like_text(text)

        if not matches:
            continue

        docs_with_matches += 1
        total_matches += len(matches)

        lines.extend([
            "-" * 80,
            f"Doc index : {doc_index}",
            f"Source    : {source}",
            f"Matches   : {len(matches)}",
            "",
        ])

        for item_index, item in enumerate(matches[:MAX_MATCHES_PER_DOC], start=1):
            lines.extend([
                f"Match {item_index}:",
                f"Pattern : {item['pattern']}",
                f"Text    : {item['match']}",
                f"Context : {item['context']}",
                "",
            ])

        if len(matches) > MAX_MATCHES_PER_DOC:
            lines.append(f"... {len(matches) - MAX_MATCHES_PER_DOC} more matches hidden for this doc.")
            lines.append("")

    lines.extend([
        "=" * 80,
        "SUMMARY",
        "=" * 80,
        f"Docs with URL-like text : {docs_with_matches}",
        f"Total URL-like matches  : {total_matches}",
        "",
    ])

    if total_matches == 0:
        lines.append("OK: No URL-like text found in cleaned page_content.")
    else:
        lines.append("CHECK: URL-like text still exists in cleaned page_content.")
        lines.append("Review the matches above before making the cleaner more aggressive.")

    return lines


def main():
    # Main test: load documents from data folder, clean them, then scan cleaned content.
    if not DATA_FOLDER.exists():
        print(f"Data folder not found: {DATA_FOLDER}")
        return

    raw_docs, load_report = load_documents(
        DATA_FOLDER,
        recursive=True,
        return_report=True,
    )

    cleaned_docs = clean_documents(raw_docs)
    report_lines = build_report(cleaned_docs, load_report)

    REPORT_FOLDER.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.write_text("\n".join(report_lines), encoding="utf-8")

    print("\n".join(report_lines))
    print(f"\nReport saved to: {REPORT_FILE}")


if __name__ == "__main__":
    main()
