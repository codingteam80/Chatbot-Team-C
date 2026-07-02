import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1] if Path(__file__).resolve().parent.name == "tests" else Path.cwd()

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import DATA_PATH, LOAD_RECURSIVE
from loaders.document_loader import load_documents, validate_data_path
from preprocessing.cleaner import clean_documents
from preprocessing.chunker import chunk_documents
from preprocessing.chunk_quality import filter_quality_chunks


REPORT_DIR = Path("reports")
CHUNK_QUALITY_REPORT_FILE = REPORT_DIR / "chunk_quality_report.txt"
STANDALONE_REPORT_FILE = REPORT_DIR / "chunk_quality_standalone_report.txt"


def format_seconds(seconds):
    # Gawing readable ang seconds.
    if seconds < 60:
        return f"{seconds:.2f} sec"

    minutes = int(seconds // 60)
    remaining_seconds = seconds % 60
    return f"{minutes} min {remaining_seconds:.2f} sec"


def run_step(step_name, timings, function):
    # Patakbuhin ang step at sukatin ang oras.
    print(f"[START] {step_name}", flush=True)
    start_time = time.perf_counter()

    result = function()

    elapsed_time = time.perf_counter() - start_time
    timings[step_name] = elapsed_time
    print(f"[DONE]  {step_name} - {format_seconds(elapsed_time)}", flush=True)

    return result


def write_standalone_report(timings, docs, cleaned_docs, chunks_before, chunks_after):
    # Gumawa ng maliit na report para makita kung tumakbo ang standalone check.
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    total_time = sum(timings.values())
    skipped_chunks = chunks_before - chunks_after

    lines = []
    lines.append("=" * 80)
    lines.append("CHUNK QUALITY STANDALONE RUN")
    lines.append("=" * 80)
    lines.append(f"Data path              : {DATA_PATH}")
    lines.append(f"Load recursive         : {LOAD_RECURSIVE}")
    lines.append(f"Documents loaded       : {len(docs)}")
    lines.append(f"Cleaned documents      : {len(cleaned_docs)}")
    lines.append(f"Chunks before quality  : {chunks_before}")
    lines.append(f"Chunks after quality   : {chunks_after}")
    lines.append(f"Chunks skipped         : {skipped_chunks}")
    lines.append(f"Quality report file    : {CHUNK_QUALITY_REPORT_FILE}")
    lines.append("")
    lines.append("Timing:")

    for step_name, elapsed_time in timings.items():
        percent = (elapsed_time / total_time * 100) if total_time else 0
        lines.append(f"- {step_name:<24} {format_seconds(elapsed_time):>14} ({percent:>5.1f}%)")

    lines.append("")
    lines.append(f"Total time             : {format_seconds(total_time)}")

    if timings:
        slowest_step = max(timings, key=timings.get)
        lines.append(f"Main bottleneck        : {slowest_step}")

    STANDALONE_REPORT_FILE.write_text("\n".join(lines), encoding="utf-8")


def main():
    # Standalone chunk quality check only.
    # Hindi ito mag-eembed, hindi gagalaw sa Chroma, at hindi tatawag sa ingest.py.
    timings = {}

    print("Chunk quality standalone check started...")
    print(f"Data path      : {DATA_PATH}")
    print(f"Quality report : {CHUNK_QUALITY_REPORT_FILE}")
    print("")

    validate_data_path(DATA_PATH)

    docs, loader_report = run_step(
        "Load documents",
        timings,
        lambda: load_documents(DATA_PATH, recursive=LOAD_RECURSIVE, return_report=True),
    )

    if not docs:
        raise ValueError("No documents were loaded. Check your data folder.")

    cleaned_docs = run_step(
        "Clean documents",
        timings,
        lambda: clean_documents(docs),
    )

    if not cleaned_docs:
        raise ValueError("No cleaned documents were created. Check your cleaner.")

    chunks = run_step(
        "Chunk documents",
        timings,
        lambda: chunk_documents(cleaned_docs),
    )

    if not chunks:
        raise ValueError("No chunks were created. Check your chunker.")

    chunks_before_quality = len(chunks)

    kept_chunks = run_step(
        "Filter chunk quality",
        timings,
        lambda: filter_quality_chunks(
            chunks,
            report_path=CHUNK_QUALITY_REPORT_FILE,
        ),
    )

    if not kept_chunks:
        raise ValueError("No chunks passed quality filter. Check preprocessing/chunk_quality.py.")

    write_standalone_report(
        timings=timings,
        docs=docs,
        cleaned_docs=cleaned_docs,
        chunks_before=chunks_before_quality,
        chunks_after=len(kept_chunks),
    )

    print("")
    print("CHUNK QUALITY CHECK DONE")
    print(f"Loaded documents      : {len(docs)}")
    print(f"Cleaned documents     : {len(cleaned_docs)}")
    print(f"Chunks before quality : {chunks_before_quality}")
    print(f"Chunks after quality  : {len(kept_chunks)}")
    print(f"Chunks skipped        : {chunks_before_quality - len(kept_chunks)}")
    print(f"Quality report saved  : {CHUNK_QUALITY_REPORT_FILE.resolve()}")
    print(f"Run report saved      : {STANDALONE_REPORT_FILE.resolve()}")


if __name__ == "__main__":
    main()
