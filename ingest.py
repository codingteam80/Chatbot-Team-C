import json
import time
from pathlib import Path

from config.settings import (
    CHROMA_COLLECTION_NAME,
    CHROMA_PATH,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DATA_PATH,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_DEVICE,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_MODEL_REVISION,
    EMBEDDING_NORMALIZE,
    FORCE_REINGEST,
    LOAD_RECURSIVE,
    USE_E5_PREFIX,
)
from embeddings.embedding_model import get_embedding_model
from loaders.document_loader import load_documents, validate_data_path
from preprocessing.cleaner import clean_documents
from preprocessing.chunker import chunk_documents
from utils.chunk_cache import get_file_signature, save_chunks_cache
from vectorstore.chroma_store import (
    create_chroma_vectorstore,
    get_chroma_document_count,
    has_chroma_files,
    reset_chroma_folder,
)


REPORT_FILE = Path("reports") / "ingest_report.txt"
INGEST_META_FILE = Path(CHROMA_PATH) / "ingest_meta.json"


def format_seconds(seconds):
    # Format seconds into a readable value.
    if seconds < 60:
        return f"{seconds:.2f} sec"

    minutes = int(seconds // 60)
    remaining_seconds = seconds % 60
    return f"{minutes} min {remaining_seconds:.2f} sec"


def save_report(lines):
    # Save the ingest result in the reports folder.
    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.write_text("\n".join(lines), encoding="utf-8")


def add_section(lines, title):
    # Add a title/header to the report.
    lines.append("")
    lines.append("=" * 70)
    lines.append(title)
    lines.append("=" * 70)


def run_step(step_name, lines, timings, function):
    # Run one step, measure the time, and add it to the report.
    print(f"[START] {step_name}", flush=True)
    lines.append(f"[START] {step_name}")

    start_time = time.perf_counter()

    try:
        result = function()
    except Exception as error:
        elapsed_time = time.perf_counter() - start_time
        timings[step_name] = elapsed_time

        print(f"[FAILED] {step_name} - {format_seconds(elapsed_time)}", flush=True)
        lines.append(f"[FAILED] {step_name} - {format_seconds(elapsed_time)}")
        lines.append(f"Error type    : {type(error).__name__}")
        lines.append(f"Error message : {error}")
        raise

    elapsed_time = time.perf_counter() - start_time
    timings[step_name] = elapsed_time

    print(f"[DONE]  {step_name} - {format_seconds(elapsed_time)}", flush=True)
    lines.append(f"[DONE]  {step_name} - {format_seconds(elapsed_time)}")

    return result


def read_json(path):
    # Read the JSON file. Return None when it is missing or invalid.
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return None


def write_json(path, data):
    # Write the JSON file.
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def get_current_metadata():
    # Metadata used to determine whether re-ingest is needed.
    # Important: include USE_E5_PREFIX so ingest will not be skipped after changing E5 mode.
    return {
        "data_path": str(DATA_PATH),
        "data_signature": get_file_signature(DATA_PATH),
        "chroma_collection_name": CHROMA_COLLECTION_NAME,
        "embedding_model_name": EMBEDDING_MODEL_NAME,
        "embedding_model_revision": EMBEDDING_MODEL_REVISION,
        "embedding_device": EMBEDDING_DEVICE,
        "embedding_normalize": EMBEDDING_NORMALIZE,
        "embedding_batch_size": EMBEDDING_BATCH_SIZE,
        "use_e5_prefix": USE_E5_PREFIX,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "load_recursive": LOAD_RECURSIVE,
    }


def is_vector_db_current():
    # Check whether the Chroma vector database exists and is still updated.
    if not has_chroma_files(CHROMA_PATH):
        return False

    old_metadata = read_json(INGEST_META_FILE)
    current_metadata = get_current_metadata()

    return old_metadata == current_metadata


def add_config_summary(lines):
    # Add the ingest config to the report so the E5 prefix setting is easy to check.
    add_section(lines, "INGEST CONFIG")
    lines.append(f"Data path             : {DATA_PATH}")
    lines.append(f"Chroma path           : {CHROMA_PATH}")
    lines.append(f"Chroma collection     : {CHROMA_COLLECTION_NAME}")
    lines.append(f"Embedding model       : {EMBEDDING_MODEL_NAME}")
    lines.append(f"Embedding revision    : {EMBEDDING_MODEL_REVISION}")
    lines.append(f"Embedding device      : {EMBEDDING_DEVICE}")
    lines.append(f"Normalize embeddings  : {EMBEDDING_NORMALIZE}")
    lines.append(f"Embedding batch size  : {EMBEDDING_BATCH_SIZE}")
    lines.append(f"Use E5 prefix         : {USE_E5_PREFIX}")
    lines.append(f"Chunk size            : {CHUNK_SIZE}")
    lines.append(f"Chunk overlap         : {CHUNK_OVERLAP}")
    lines.append(f"Load recursive        : {LOAD_RECURSIVE}")
    lines.append(f"Force re-ingest       : {FORCE_REINGEST}")


def add_loader_summary(lines, loader_report):
    # Add the document loading result to the report.
    add_section(lines, "DOCUMENT LOADER SUMMARY")
    lines.append(f"Loaded files      : {loader_report.get('loaded_files', 0)}")
    lines.append(f"Loaded documents  : {loader_report.get('loaded_docs', 0)}")
    lines.append(f"Skipped files     : {len(loader_report.get('skipped_files', []))}")
    lines.append(f"Failed files      : {len(loader_report.get('failed_files', []))}")

    if loader_report.get("skipped_files"):
        lines.append("")
        lines.append("Skipped files:")
        for item in loader_report["skipped_files"]:
            lines.append(f"- {item.get('file_path')} | {item.get('reason')}")

    if loader_report.get("failed_files"):
        lines.append("")
        lines.append("Failed files:")
        for item in loader_report["failed_files"]:
            lines.append(
                f"- {item.get('file_path')} | "
                f"{item.get('error_type')}: {item.get('error_message')}"
            )


def add_final_summary(lines, timings, docs, cleaned_docs, chunks, vector_count):
    # Final summary of the full ingest process.
    total_time = sum(timings.values())

    add_section(lines, "INGESTION SUMMARY")
    lines.append(f"Documents loaded  : {len(docs)}")
    lines.append(f"Cleaned documents : {len(cleaned_docs)}")
    lines.append(f"Chunks created    : {len(chunks)}")
    lines.append(f"Vectors saved     : {vector_count}")
    lines.append(f"Vector DB folder  : {CHROMA_PATH}")
    lines.append(f"Metadata file     : {INGEST_META_FILE}")
    lines.append(f"Report file       : {REPORT_FILE}")

    lines.append("")
    lines.append("Timing:")

    for step_name, elapsed_time in timings.items():
        percent = (elapsed_time / total_time * 100) if total_time else 0
        lines.append(f"- {step_name:<24} {format_seconds(elapsed_time):>14} ({percent:>5.1f}%)")

    lines.append("")
    lines.append(f"Total time        : {format_seconds(total_time)}")

    if timings:
        slowest_step = max(timings, key=timings.get)
        lines.append(f"Main bottleneck   : {slowest_step}")


def add_skip_message(lines):
    # Message when the Chroma DB is still current.
    add_section(lines, "INGESTION SKIPPED")
    lines.append("Existing Chroma vector database is still updated.")
    lines.append("No embedding was performed.")
    lines.append("")
    lines.append("This skip check now includes embedding model, chunk settings, and USE_E5_PREFIX.")
    lines.append("")
    lines.append("To force re-ingestion in CMD:")
    lines.append("  set FORCE_REINGEST=1")
    lines.append("  python ingest.py")
    lines.append("")
    lines.append("To force re-ingestion in PowerShell:")
    lines.append('  $env:FORCE_REINGEST="1"')
    lines.append("  python ingest.py")


def main():
    # Main ingest flow:
    # 1. Check data folder
    # 2. Skip when Chroma is still updated.
    # 3. Load documents
    # 4. Clean documents
    # 5. Chunk documents
    # 6. Save chunk cache
    # 7. Embed chunks and save to Chroma
    # 8. Save report
    lines = []
    timings = {}

    add_section(lines, "RAG INGESTION STARTED")
    add_config_summary(lines)

    print("RAG ingestion started...")
    print(f"Data path     : {DATA_PATH}")
    print(f"Chroma path   : {CHROMA_PATH}")
    print(f"Report file   : {REPORT_FILE}")
    print(f"Use E5 prefix : {USE_E5_PREFIX}")
    print("")

    try:
        validate_data_path(DATA_PATH)

        if is_vector_db_current() and not FORCE_REINGEST:
            add_skip_message(lines)
            print("[SKIPPED] Existing ChromaDB is current.")
            return

        docs, loader_report = run_step(
            "Load documents",
            lines,
            timings,
            lambda: load_documents(DATA_PATH, recursive=LOAD_RECURSIVE, return_report=True),
        )
        add_loader_summary(lines, loader_report)

        if not docs:
            raise ValueError("No documents were loaded. Check your data folder.")

        cleaned_docs = run_step(
            "Clean documents",
            lines,
            timings,
            lambda: clean_documents(docs),
        )

        if not cleaned_docs:
            raise ValueError("No cleaned documents were created. Check your cleaner.")

        chunks = run_step(
            "Chunk documents",
            lines,
            timings,
            lambda: chunk_documents(cleaned_docs),
        )

        if not chunks:
            raise ValueError("No chunks were created. Check your chunker.")

        run_step(
            "Save chunk cache",
            lines,
            timings,
            lambda: save_chunks_cache(chunks, data_path=DATA_PATH),
        )

        embedding_model = run_step(
            "Load embedding model",
            lines,
            timings,
            get_embedding_model,
        )

        def embed_and_save():
            # Reset first to avoid duplicate vectors.
            if has_chroma_files(CHROMA_PATH):
                reset_chroma_folder(CHROMA_PATH)

            vectorstore = create_chroma_vectorstore(
                chunks=chunks,
                embedding_model=embedding_model,
                persist_directory=CHROMA_PATH,
            )

            vector_count = get_chroma_document_count(vectorstore)
            write_json(INGEST_META_FILE, get_current_metadata())

            return vector_count

        vector_count = run_step(
            "Embed and save vectors",
            lines,
            timings,
            embed_and_save,
        )

        add_final_summary(lines, timings, docs, cleaned_docs, chunks, vector_count)
        add_section(lines, "INGESTION SUCCESS")

        print("")
        print("INGESTION DONE")
        print(f"Documents loaded : {len(docs)}")
        print(f"Cleaned docs     : {len(cleaned_docs)}")
        print(f"Chunks created   : {len(chunks)}")
        print(f"Vectors saved    : {vector_count}")
        print(f"Report saved to  : {REPORT_FILE}")

    except Exception as error:
        add_section(lines, "INGESTION FAILED")
        lines.append(f"Error type    : {type(error).__name__}")
        lines.append(f"Error message : {error}")

        print("")
        print("[FAILED] RAG ingestion failed.")
        print(f"Error type    : {type(error).__name__}")
        print(f"Error message : {error}")
        print(f"Report saved to: {REPORT_FILE}")

    finally:
        save_report(lines)


if __name__ == "__main__":
    main()
