import argparse
import pickle
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

try:
    from config.settings import (  # noqa: E402
        BM25_K,
        BM25_WEIGHT,
        CHROMA_COLLECTION_NAME,
        CHROMA_PATH,
        DATA_PATH,
        EMBEDDING_MODEL_NAME,
        ENABLE_METADATA_BOOST,
        HYBRID_FINAL_K,
        RRF_K,
        SEMANTIC_K,
        SEMANTIC_WEIGHT,
        USE_E5_PREFIX,
    )
except ImportError:
    BM25_K = 9
    BM25_WEIGHT = 0.4
    CHROMA_COLLECTION_NAME = "rag_documents"
    CHROMA_PATH = "chroma_db"
    DATA_PATH = "data"
    EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-small"
    ENABLE_METADATA_BOOST = True
    HYBRID_FINAL_K = 11
    RRF_K = 60
    SEMANTIC_K = 9
    SEMANTIC_WEIGHT = 0.6
    USE_E5_PREFIX = True

try:
    from config.settings import CHUNK_CACHE_PATH  # noqa: E402
except ImportError:
    CHUNK_CACHE_PATH = "cache/chunks.pkl"

from embeddings.embedding_model import get_embedding_model  # noqa: E402
from retrieval.bm25_retriever import create_bm25_retriever  # noqa: E402
from retrieval.hybrid_retriever import hybrid_search  # noqa: E402
from retrieval.semantic_retriever import format_semantic_query  # noqa: E402


DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "reports" / "retrieval_semantic_to_hybrid_result.txt"

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


# Optional expected source hints para mas madali basahin ang summary.
EXPECTED_SOURCE_HINTS = {
    "R01": ["José Rizal", "Jose Rizal"],
    "R02": ["History of the Philippines", "Magellan", "Lapulapu", "Lapu"],
    "R03": ["History of the Philippines", "Magellan", "Lapulapu", "Lapu"],
    "R04": ["Emilio Aguinaldo"],
    "R05": ["Treaty of Paris"],
    "R06": ["Japanese occupation"],
    "R07": ["Katipunan"],
    "R08": ["Katipunan", "Philippine Revolution"],
    "R09": ["Independence Day"],
    "R10": ["Treaty of Paris", "Spanish–American War", "Spanish-American War", "Philippine–American War", "Philippine-American War"],
    "R11": ["Katipunan", "José Rizal", "Jose Rizal"],
    "R12": ["Philippine Revolution", "Katipunan"],
}


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


# Convert relative project paths to absolute paths.
def resolve_project_path(path_value):
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


# Linisin ang preview para madaling basahin sa terminal at report.
def clean_preview(text, max_chars=400):
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


# Basic expected-source check para sa report summary lang.
def is_expected_source(query_id, source):
    hints = EXPECTED_SOURCE_HINTS.get(query_id, [])
    source_text = str(source or "").lower()
    return any(str(hint).lower() in source_text for hint in hints)


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


# Extract list of chunks mula sa iba-ibang possible cache shapes.
def extract_chunks_from_cache(cache_data):
    if isinstance(cache_data, list):
        return cache_data

    if isinstance(cache_data, tuple):
        for item in cache_data:
            if isinstance(item, list):
                return item

    if isinstance(cache_data, dict):
        for key in ["chunks", "docs", "documents", "items"]:
            value = cache_data.get(key)
            if isinstance(value, list):
                return value

    return []


# Try load chunks from utils/chunk_cache kung may load function.
def load_chunks_with_project_helper():
    try:
        import utils.chunk_cache as chunk_cache
    except ImportError:
        return []

    function_names = [
        "load_chunks_cache",
        "load_chunk_cache",
        "load_cached_chunks",
        "get_chunks_cache",
    ]

    for function_name in function_names:
        function = getattr(chunk_cache, function_name, None)
        if not callable(function):
            continue

        attempts = [
            lambda: function(data_path=DATA_PATH),
            lambda: function(cache_path=CHUNK_CACHE_PATH),
            lambda: function(CHUNK_CACHE_PATH),
            lambda: function(),
        ]

        for attempt in attempts:
            try:
                chunks = extract_chunks_from_cache(attempt())
                if chunks:
                    return chunks
            except TypeError:
                continue
            except Exception:
                continue

    return []


# Try direct pickle load from cache/chunks.pkl.
def load_chunks_from_pickle_cache():
    candidate_paths = [
        resolve_project_path(CHUNK_CACHE_PATH),
        PROJECT_ROOT / "cache" / "chunks.pkl",
    ]

    seen = set()
    for path in candidate_paths:
        path = Path(path)
        if path in seen:
            continue
        seen.add(path)

        if not path.exists():
            continue

        with path.open("rb") as file:
            cache_data = pickle.load(file)

        chunks = extract_chunks_from_cache(cache_data)
        if chunks:
            return chunks

    return []


# Fallback kapag walang cache: rebuild chunks from data folder.
def rebuild_chunks_from_data(debug=False):
    from loaders.document_loader import load_documents
    from preprocessing.cleaner import clean_documents
    from preprocessing.chunker import chunk_documents

    if debug:
        print("[BM25] No chunk cache found. Rebuilding chunks from data folder...", flush=True)

    load_attempts = [
        lambda: load_documents(DATA_PATH, recursive=True, return_report=True),
        lambda: load_documents(DATA_PATH, return_report=True),
        lambda: load_documents(DATA_PATH, recursive=True),
        lambda: load_documents(DATA_PATH),
    ]

    docs = []
    last_error = None
    for attempt in load_attempts:
        try:
            loaded = attempt()
            if isinstance(loaded, tuple):
                docs = loaded[0]
            else:
                docs = loaded
            break
        except TypeError as error:
            last_error = error

    if not docs and last_error:
        raise last_error

    cleaned_docs = clean_documents(docs)
    chunks = chunk_documents(cleaned_docs)

    return enrich_chunks_if_possible(chunks)


# Optional metadata enrichment kung existing sa project.
def enrich_chunks_if_possible(chunks):
    try:
        import preprocessing.metadata_processor as metadata_processor
    except ImportError:
        return chunks

    function_names = [
        "add_or_enrich_metadata",
        "add_chunk_metadata",
        "enrich_chunk_metadata",
        "add_metadata_to_chunks",
        "process_metadata",
    ]

    for function_name in function_names:
        function = getattr(metadata_processor, function_name, None)
        if not callable(function):
            continue

        try:
            enriched = function(chunks)
            if enriched:
                return enriched
        except TypeError:
            continue
        except Exception:
            continue

    return chunks


# Load chunks para magawa ang BM25 index.
def load_chunks_for_bm25(debug=False):
    chunks = load_chunks_with_project_helper()
    if chunks:
        if debug:
            print(f"[BM25] Loaded chunks using project cache helper: {len(chunks)}", flush=True)
        return chunks

    chunks = load_chunks_from_pickle_cache()
    if chunks:
        if debug:
            print(f"[BM25] Loaded chunks from pickle cache: {len(chunks)}", flush=True)
        return chunks

    return rebuild_chunks_from_data(debug=debug)


# Add basic config sa report.
def add_config(lines, args, output_path):
    write_line(lines, "=" * 80)
    write_line(lines, "SEMANTIC TO HYBRID RETRIEVAL TEST")
    write_line(lines, "=" * 80)
    write_line(lines, f"Project root        : {PROJECT_ROOT}")
    write_line(lines, f"Chroma path         : {CHROMA_PATH}")
    write_line(lines, f"Chroma collection   : {CHROMA_COLLECTION_NAME}")
    write_line(lines, f"Data path           : {DATA_PATH}")
    write_line(lines, f"Chunk cache path    : {CHUNK_CACHE_PATH}")
    write_line(lines, f"Embedding model     : {EMBEDDING_MODEL_NAME}")
    write_line(lines, f"USE_E5_PREFIX       : {USE_E5_PREFIX}")
    write_line(lines, f"Semantic K          : {args.semantic_k}")
    write_line(lines, f"BM25 K              : {args.bm25_k}")
    write_line(lines, f"Hybrid final K      : {args.final_k}")
    print_k_label = "ALL" if int(args.print_k or 0) <= 0 else str(args.print_k)
    write_line(lines, f"Print per stage     : {print_k_label}")
    write_line(lines, f"RRF K               : {RRF_K}")
    write_line(lines, f"Semantic weight     : {SEMANTIC_WEIGHT}")
    write_line(lines, f"BM25 weight         : {BM25_WEIGHT}")
    write_line(lines, f"Metadata boost      : {not args.no_metadata_boost and ENABLE_METADATA_BOOST}")
    write_line(lines, f"Report file         : {output_path}")
    write_line(lines, "")


# Print isang document result sa report.
def add_doc_result(lines, index, doc, stage_name):
    metadata = get_metadata(doc)
    source = get_source_label(metadata)
    page = metadata.get("page", "N/A")
    chunk_id = metadata.get("chunk_id") or metadata.get("chunk_index") or "N/A"
    category = metadata.get("category", "N/A")
    doc_type = metadata.get("doc_type", "N/A")
    language = metadata.get("language", "N/A")
    semantic_rank = metadata.get("semantic_rank", "N/A")
    semantic_distance = metadata.get("semantic_distance", "N/A")
    semantic_similarity = metadata.get("semantic_similarity_score", "N/A")
    bm25_rank = metadata.get("bm25_rank", "N/A")
    bm25_score = metadata.get("bm25_rank_score", "N/A")
    hybrid_rank = metadata.get("hybrid_rank", "N/A")
    hybrid_score = metadata.get("hybrid_score", "N/A")
    metadata_boost_rank = metadata.get("metadata_boost_rank", "N/A")
    final_score = metadata.get("final_score", metadata.get("boosted_score", "N/A"))
    preview = clean_preview(doc.page_content)

    write_line(lines, f"{index}. Source          : {source}")
    write_line(lines, f"   Stage           : {stage_name}")
    write_line(lines, f"   Page            : {page}")
    write_line(lines, f"   Chunk           : {chunk_id}")
    write_line(lines, f"   Category        : {category}")
    write_line(lines, f"   Doc type        : {doc_type}")
    write_line(lines, f"   Language        : {language}")
    write_line(lines, f"   Semantic rank   : {semantic_rank}")
    write_line(lines, f"   Semantic dist   : {semantic_distance}")
    write_line(lines, f"   Semantic sim    : {semantic_similarity}")
    write_line(lines, f"   BM25 rank       : {bm25_rank}")
    write_line(lines, f"   BM25 rank score : {bm25_score}")
    write_line(lines, f"   Hybrid rank     : {hybrid_rank}")
    write_line(lines, f"   Hybrid score    : {hybrid_score}")
    write_line(lines, f"   Boost rank      : {metadata_boost_rank}")
    write_line(lines, f"   Final score     : {final_score}")

    if has_e5_prefix_leak(doc.page_content):
        write_line(lines, "   WARNING         : E5 prefix leaked into chunk text. Remove query:/passage: from page_content.")

    write_line(lines, f"   Preview         : {preview}")
    write_line(lines, "")


# Decide ilang results ang ipi-print sa bawat retrieval stage.
def get_stage_print_limit(docs, limit):
    # Kapag 0, negative, None, or invalid ang value, ipakita lahat.
    docs = docs or []

    try:
        limit = int(limit)
    except Exception:
        return len(docs)

    if limit <= 0:
        return len(docs)

    return min(limit, len(docs))


# Add stage results section.
def add_stage_results(lines, title, docs, limit):
    write_line(lines, "")
    write_line(lines, title)
    write_line(lines, "-" * len(title))

    if not docs:
        write_line(lines, "No results.")
        return

    print_limit = get_stage_print_limit(docs, limit)
    write_line(lines, f"Showing results : {print_limit} of {len(docs)}")
    write_line(lines, "")

    for index, doc in enumerate(docs[:print_limit], start=1):
        add_doc_result(lines, index, doc, title)


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
        user_query = input("Enter retrieval test query: ").strip()
        if user_query:
            return [
                {
                    "id": "CUSTOM",
                    "query": user_query,
                }
            ]

    return DEFAULT_TEST_QUERIES


# Patakbuhin Semantic -> BM25 -> Hybrid sa isang query.
def run_single_query(lines, vectorstore, bm25_retriever, query_item, args):
    query_id = query_item.get("id", "CUSTOM")
    raw_query = str(query_item.get("query", "") or "").strip()
    vector_query = format_semantic_query(raw_query, use_e5_prefix=USE_E5_PREFIX)

    write_line(lines, "")
    write_line(lines, "=" * 80)
    write_line(lines, f"Test ID            : {query_id}")
    write_line(lines, f"Raw query          : {raw_query}")
    write_line(lines, f"Semantic query     : {vector_query}")
    write_line(lines, "Note               : BM25 and Hybrid receive the raw query; only semantic vector search uses query:.")
    write_line(lines, "=" * 80)

    start_time = time.perf_counter()
    details = hybrid_search(
        query=raw_query,
        vectorstore=vectorstore,
        bm25_retriever=bm25_retriever,
        semantic_k=args.semantic_k,
        bm25_k=args.bm25_k,
        final_k=args.final_k,
        use_rrf=not args.no_rrf,
        use_metadata_boost=not args.no_metadata_boost,
        use_e5_prefix=USE_E5_PREFIX,
        debug=args.debug,
        return_details=True,
    )
    elapsed_time = time.perf_counter() - start_time

    semantic_docs = details.get("semantic_docs", [])
    bm25_docs = details.get("bm25_docs", [])
    hybrid_docs = details.get("hybrid_docs", [])
    query_info = details.get("query_info", {})

    write_line(lines, f"Retrieval time     : {format_seconds(elapsed_time)}")
    write_line(lines, f"Semantic results   : {len(semantic_docs)}")
    write_line(lines, f"BM25 results       : {len(bm25_docs)}")
    write_line(lines, f"Hybrid results     : {len(hybrid_docs)}")
    write_line(lines, f"Query info         : {query_info}")

    add_stage_results(lines, "SEMANTIC RESULTS", semantic_docs, args.print_k)
    add_stage_results(lines, "BM25 RESULTS", bm25_docs, args.print_k)
    add_stage_results(lines, "HYBRID RESULTS", hybrid_docs, args.print_k)

    semantic_top = get_source_label(get_metadata(semantic_docs[0])) if semantic_docs else "N/A"
    bm25_top = get_source_label(get_metadata(bm25_docs[0])) if bm25_docs else "N/A"
    hybrid_top = get_source_label(get_metadata(hybrid_docs[0])) if hybrid_docs else "N/A"

    return {
        "id": query_id,
        "query": raw_query,
        "time": elapsed_time,
        "semantic_results": len(semantic_docs),
        "bm25_results": len(bm25_docs),
        "hybrid_results": len(hybrid_docs),
        "semantic_top": semantic_top,
        "bm25_top": bm25_top,
        "hybrid_top": hybrid_top,
        "hybrid_top_expected": is_expected_source(query_id, hybrid_top),
    }


# Summary para mabilis makita kung gumaganda ba from semantic to hybrid.
def add_summary(lines, summaries):
    write_line(lines, "")
    write_line(lines, "=" * 80)
    write_line(lines, "SUMMARY")
    write_line(lines, "=" * 80)

    for item in summaries:
        expected_label = "OK" if item["hybrid_top_expected"] else "CHECK"
        write_line(
            lines,
            f"{item['id']:<6} "
            f"time={format_seconds(item['time']):>10} "
            f"S={item['semantic_results']:<2} "
            f"B={item['bm25_results']:<2} "
            f"H={item['hybrid_results']:<2} "
            f"hybrid_top={item['hybrid_top']} "
            f"[{expected_label}]",
        )

    write_line(lines, "")
    write_line(lines, "Legend:")
    write_line(lines, "S = semantic result count")
    write_line(lines, "B = BM25 result count")
    write_line(lines, "H = hybrid result count")
    write_line(lines, "OK/CHECK is only based on simple expected source hints, not final answer correctness.")


# I-save ang report sa txt file.
def save_report(lines, output_path):
    output_path = Path(output_path)

    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


# Main entry point ng test script.
def main():
    parser = argparse.ArgumentParser(description="Test semantic, BM25, and hybrid retrieval.")
    parser.add_argument("--query", type=str, default="", help="Single query to test.")
    parser.add_argument("--query-file", type=str, default="", help="Text file with one query per line.")
    parser.add_argument("--interactive", action="store_true", help="Enter one query manually.")
    parser.add_argument("--semantic-k", type=int, default=SEMANTIC_K, help="Number of semantic results.")
    parser.add_argument("--bm25-k", type=int, default=BM25_K, help="Number of BM25 results.")
    parser.add_argument("--final-k", type=int, default=HYBRID_FINAL_K, help="Number of hybrid results.")
    parser.add_argument("--print-k", type=int, default=0, help="Number of results to print per stage. Use 0 to print all results.")
    parser.add_argument("--debug", action="store_true", help="Print retriever debug logs.")
    parser.add_argument("--no-rrf", action="store_true", help="Disable RRF and just merge semantic + BM25.")
    parser.add_argument("--no-metadata-boost", action="store_true", help="Disable metadata boost.")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT_PATH), help="Report txt path.")
    args = parser.parse_args()

    lines = []
    summaries = []
    output_path = Path(args.output)

    add_config(lines, args, output_path)

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

    write_line(lines, "[START] Load chunks for BM25")
    start_time = time.perf_counter()
    chunks = load_chunks_for_bm25(debug=args.debug)
    write_line(lines, f"[DONE]  Load chunks for BM25 - {format_seconds(time.perf_counter() - start_time)}")
    write_line(lines, f"Chunks loaded      : {len(chunks)}")
    write_line(lines, "")

    if not chunks:
        raise ValueError("No chunks loaded for BM25. Run ingest.py first or check CHUNK_CACHE_PATH.")

    write_line(lines, "[START] Build BM25 retriever")
    start_time = time.perf_counter()
    bm25_retriever = create_bm25_retriever(chunks, k=args.bm25_k, debug=args.debug)
    write_line(lines, f"[DONE]  Build BM25 retriever - {format_seconds(time.perf_counter() - start_time)}")
    write_line(lines, "")

    queries = resolve_queries(args)

    if args.query:
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
            bm25_retriever=bm25_retriever,
            query_item=query_item,
            args=args,
        )
        summaries.append(summary)

    add_summary(lines, summaries)

    report_path = save_report(lines, output_path)

    print("")
    print("SEMANTIC TO HYBRID TEST DONE")
    print(f"Report saved to: {report_path}")


if __name__ == "__main__":
    main()
