import argparse
import sys
import time
from pathlib import Path


# Hanapin ang project root kahit ilagay ang file sa project root or tests folder.
def prepare_project_path():
    current_path = Path(__file__).resolve()
    candidates = [current_path.parent, current_path.parent.parent]

    for candidate in candidates:
        if (candidate / "config").exists() and (candidate / "retrieval").exists():
            if str(candidate) not in sys.path:
                sys.path.insert(0, str(candidate))
            return candidate

    if str(current_path.parent) not in sys.path:
        sys.path.insert(0, str(current_path.parent))

    return current_path.parent


PROJECT_ROOT = prepare_project_path()

from config.settings import (  # noqa: E402
    CHROMA_COLLECTION_NAME,
    CHROMA_PATH,
    EMBEDDING_MODEL_NAME,
    SEMANTIC_K,
    USE_E5_PREFIX,
)
from embeddings.embedding_model import get_embedding_model  # noqa: E402
from retrieval.semantic_retriever import format_semantic_query, semantic_search  # noqa: E402


DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "reports" / "semantic_test_result.txt"

DEFAULT_TEST_QUERIES = [
    {
        "id": "R01",
        "query": "Who are the ladies that had relationship with Jose Rizal?",
    },
    {
        "id": "R02",
        "query": "Who killed Ferdinand Magellan?",
    },
    {
        "id": "R03",
        "query": "Did Lapu-Lapu kill Magellan?",
    },
    {
        "id": "R04",
        "query": "Who is the first Philippine president?",
    },
    {
        "id": "R05",
        "query": "What did the Treaty of Paris of 1898 say about the Philippines?",
    },
    {
        "id": "R06",
        "query": "What hardships did Filipinos experience during the Japanese occupation?",
    },
    {
        "id": "R07",
        "query": "Who founded the Katipunan or KKK on July 7, 1892, and what was its purpose against Spain?",
    },
    {
        "id": "R08",
        "query": "Which secret group tried to free Filipinos from Spanish rule through armed revolution before it was discovered in 1896?",
    },
    {
        "id": "R09",
        "query": "Kailan ipinagdiriwang ang Araw ng Kalayaan ng Pilipinas at anong pangyayari ang ginugunita nito?",
    },
    {
        "id": "R10",
        "query": "How did the Treaty of Paris connect the Spanish-American War to the Philippine-American War?",
    },
    {
        "id": "R11",
        "query": "Why did Jose Rizal become the Supremo of the Katipunan?",
    },
    {
        "id": "R12",
        "query": "What is the difference between the Philippine Revolution and the Katipunan? Is one an organization and the other a war/revolution?",
    },
]


# Gawing readable ang seconds sa terminal at report.
def format_seconds(seconds):
    if seconds < 60:
        return f"{seconds:.2f} sec"

    minutes = int(seconds // 60)
    remaining_seconds = seconds % 60
    return f"{minutes} min {remaining_seconds:.2f} sec"


# Print sa terminal at isulat din sa report lines.
def write_line(lines, text=""):
    print(text)
    lines.append(str(text))


# Linisin ang preview para madaling basahin sa terminal at report.
def clean_preview(text, max_chars=500):
    text = str(text or "").replace("\n", " ").replace("\r", " ")
    text = " ".join(text.split())

    if len(text) > max_chars:
        return text[:max_chars].rstrip() + "..."

    return text


# Safe getter para hindi mag-error kapag kulang metadata.
def get_metadata(doc):
    return dict(getattr(doc, "metadata", {}) or {})


# Kunin ang source label mula sa common metadata keys.
def get_source_label(metadata):
    return (
        metadata.get("file_name")
        or metadata.get("source_file")
        or metadata.get("source")
        or metadata.get("path")
        or "Unknown source"
    )


# Check kung aksidenteng pumasok ang E5 prefix sa actual chunk text.
def has_e5_prefix_leak(text):
    text = str(text or "").strip().lower()
    return text.startswith("query:") or text.startswith("passage:")


# Load Chroma gamit muna ang project helper kung meron.
def load_vectorstore(embedding_model):
    try:
        from vectorstore.chroma_store import load_chroma_vectorstore

        load_attempts = [
            lambda: load_chroma_vectorstore(embedding_model=embedding_model),
            lambda: load_chroma_vectorstore(embedding_function=embedding_model),
            lambda: load_chroma_vectorstore(embedding_model),
        ]

        last_error = None
        for attempt in load_attempts:
            try:
                return attempt()
            except TypeError as error:
                last_error = error

        if last_error:
            raise last_error

    except ImportError:
        pass

    try:
        from langchain_chroma import Chroma
    except ImportError:
        from langchain_community.vectorstores import Chroma

    return Chroma(
        collection_name=CHROMA_COLLECTION_NAME,
        persist_directory=CHROMA_PATH,
        embedding_function=embedding_model,
    )


# Print basic config para makita agad kung tama ang setup.
def add_config(lines, k, output_path):
    write_line(lines, "=" * 80)
    write_line(lines, "SEMANTIC RETRIEVAL TEST")
    write_line(lines, "=" * 80)
    write_line(lines, f"Project root        : {PROJECT_ROOT}")
    write_line(lines, f"Chroma path         : {CHROMA_PATH}")
    write_line(lines, f"Chroma collection   : {CHROMA_COLLECTION_NAME}")
    write_line(lines, f"Embedding model     : {EMBEDDING_MODEL_NAME}")
    write_line(lines, f"USE_E5_PREFIX       : {USE_E5_PREFIX}")
    write_line(lines, f"Semantic K          : {k}")
    write_line(lines, f"Report file         : {output_path}")
    write_line(lines, "")


# Add isang document result sa terminal at report.
def add_doc_result(lines, index, doc):
    metadata = get_metadata(doc)
    source = get_source_label(metadata)
    page = metadata.get("page", "N/A")
    chunk_id = metadata.get("chunk_id") or metadata.get("chunk_index") or "N/A"
    distance = metadata.get("semantic_distance", "N/A")
    similarity = metadata.get("semantic_similarity_score", "N/A")
    category = metadata.get("category", "N/A")
    doc_type = metadata.get("doc_type", "N/A")
    language = metadata.get("language", "N/A")
    preview = clean_preview(doc.page_content)

    write_line(lines, f"{index}. Source      : {source}")
    write_line(lines, f"   Page        : {page}")
    write_line(lines, f"   Chunk       : {chunk_id}")
    write_line(lines, f"   Category    : {category}")
    write_line(lines, f"   Doc type    : {doc_type}")
    write_line(lines, f"   Language    : {language}")
    write_line(lines, f"   Distance    : {distance}")
    write_line(lines, f"   Similarity  : {similarity}")

    if has_e5_prefix_leak(doc.page_content):
        write_line(lines, "   WARNING     : E5 prefix leaked into chunk text. Remove query:/passage: from page_content.")

    write_line(lines, f"   Preview     : {preview}")
    write_line(lines, "")


# Patakbuhin ang semantic search sa isang query.
def run_single_query(lines, vectorstore, query_item, k, debug=False):
    query_id = query_item.get("id", "CUSTOM")
    raw_query = str(query_item.get("query", "") or "").strip()
    vector_query = format_semantic_query(raw_query, use_e5_prefix=USE_E5_PREFIX)

    write_line(lines, "-" * 80)
    write_line(lines, f"Test ID            : {query_id}")
    write_line(lines, f"Raw query          : {raw_query}")
    write_line(lines, f"Semantic query     : {vector_query}")
    write_line(lines, "Note               : Semantic query lang dapat may query: kapag USE_E5_PREFIX=True.")
    write_line(lines, "-" * 80)

    start_time = time.perf_counter()
    docs = semantic_search(
        vectorstore=vectorstore,
        query=raw_query,
        k=k,
        use_e5_prefix=USE_E5_PREFIX,
        debug=debug,
    )
    elapsed_time = time.perf_counter() - start_time

    write_line(lines, f"Semantic time      : {format_seconds(elapsed_time)}")
    write_line(lines, f"Results found      : {len(docs)}")
    write_line(lines, "")

    if not docs:
        write_line(lines, "No semantic results found. Check if Chroma DB exists and ingest was completed.")
        write_line(lines, "")
        return {
            "id": query_id,
            "query": raw_query,
            "results": 0,
            "time": elapsed_time,
            "top_source": "N/A",
        }

    for index, doc in enumerate(docs, start=1):
        add_doc_result(lines, index, doc)

    top_metadata = get_metadata(docs[0])
    return {
        "id": query_id,
        "query": raw_query,
        "results": len(docs),
        "time": elapsed_time,
        "top_source": get_source_label(top_metadata),
    }


# Basahin ang queries mula sa text file, isang query bawat line.
def load_queries_from_file(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Query file not found: {path}")

    queries = []
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if line and not line.startswith("#"):
            queries.append(
                {
                    "id": f"Q{index:02d}",
                    "query": line,
                }
            )

    return queries


# Kunin ang queries depende sa command line arguments.
def resolve_queries(args):
    if args.query:
        return [
            {
                "id": "CUSTOM",
                "query": args.query,
            }
        ]

    if args.query_file:
        return load_queries_from_file(args.query_file)

    if args.interactive:
        user_query = input("Enter semantic test query: ").strip()
        if user_query:
            return [
                {
                    "id": "CUSTOM",
                    "query": user_query,
                }
            ]

    # Default behavior:
    # Kapag walang --query, --query-file, or --interactive, automatic batch test ang default queries.
    return DEFAULT_TEST_QUERIES


# I-save ang report sa txt file.
def save_report(lines, output_path):
    output_path = Path(output_path)

    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


# Summary para mabilis makita ang top source per query.
def add_summary(lines, summaries):
    write_line(lines, "")
    write_line(lines, "=" * 80)
    write_line(lines, "SUMMARY")
    write_line(lines, "=" * 80)

    for item in summaries:
        write_line(
            lines,
            f"{item['id']:<6} results={item['results']:<3} "
            f"time={format_seconds(item['time']):>10} "
            f"top_source={item['top_source']}"
        )


# Main entry point ng test script.
def main():
    parser = argparse.ArgumentParser(description="Test semantic retrieval only.")
    parser.add_argument("--query", type=str, default="", help="Single query to test.")
    parser.add_argument("--query-file", type=str, default="", help="Text file with one query per line.")
    parser.add_argument("--interactive", action="store_true", help="Enter one query manually.")
    parser.add_argument("--sample", action="store_true", help="Kept for compatibility. Default already runs sample queries.")
    parser.add_argument("--k", type=int, default=SEMANTIC_K, help="Number of semantic results.")
    parser.add_argument("--debug", action="store_true", help="Print semantic retriever debug logs.")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT_PATH), help="Report txt path.")
    args = parser.parse_args()

    lines = []
    summaries = []
    output_path = Path(args.output)

    add_config(lines, args.k, output_path)

    write_line(lines, "[START] Load embedding model")
    start_time = time.perf_counter()
    embedding_model = get_embedding_model()
    write_line(lines, f"[DONE]  Load embedding model - {format_seconds(time.perf_counter() - start_time)}")
    write_line(lines, "")

    write_line(lines, "[START] Load Chroma vectorstore")
    start_time = time.perf_counter()
    vectorstore = load_vectorstore(embedding_model)
    write_line(lines, f"[DONE]  Load Chroma vectorstore - {format_seconds(time.perf_counter() - start_time)}")
    write_line(lines, "")

    queries = resolve_queries(args)

    if args.sample:
        write_line(lines, "Mode               : sample/default queries")
    elif args.query:
        write_line(lines, "Mode               : single query from --query")
    elif args.query_file:
        write_line(lines, f"Mode               : query file {args.query_file}")
    elif args.interactive:
        write_line(lines, "Mode               : interactive query")
    else:
        write_line(lines, "Mode               : default batch queries")
    write_line(lines, "")

    for query_item in queries:
        summary = run_single_query(
            lines=lines,
            vectorstore=vectorstore,
            query_item=query_item,
            k=args.k,
            debug=args.debug,
        )
        summaries.append(summary)

    add_summary(lines, summaries)

    report_path = save_report(lines, output_path)

    print("")
    print("SEMANTIC TEST DONE")
    print(f"Report saved to: {report_path}")


if __name__ == "__main__":
    main()
