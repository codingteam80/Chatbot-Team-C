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

from config.settings import (  # noqa: E402
    BM25_K,
    CHROMA_COLLECTION_NAME,
    CHROMA_PATH,
    EMBEDDING_MODEL_NAME,
    HYBRID_FINAL_K,
    RERANK_TOP_N,
    SEMANTIC_K,
    USE_E5_PREFIX,
)

try:  # noqa: E402
    from config.settings import CHUNK_CACHE_PATH
except ImportError:  # noqa: E402
    CHUNK_CACHE_PATH = "cache/chunks.pkl"

try:  # noqa: E402
    from config.settings import RERANK_POOL_TOP_N
except ImportError:  # noqa: E402
    RERANK_POOL_TOP_N = HYBRID_FINAL_K

from embeddings.embedding_model import get_embedding_model  # noqa: E402
from retrieval.bm25_retriever import create_bm25_retriever  # noqa: E402
from retrieval.hybrid_retriever import hybrid_search  # noqa: E402
from retrieval.reranker import load_reranker, rerank_documents  # noqa: E402
from retrieval.semantic_retriever import format_semantic_query  # noqa: E402


DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "reports" / "retrieval_hybrid_to_rerank_result.txt"

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


def format_seconds(seconds):
    if seconds < 60:
        return f"{seconds:.2f} sec"

    minutes = int(seconds // 60)
    remaining_seconds = seconds % 60
    return f"{minutes} min {remaining_seconds:.2f} sec"


def write_line(lines, text=""):
    print(text)
    lines.append(str(text))


def clean_preview(text, max_chars=380):
    text = str(text or "").replace("\n", " ").replace("\r", " ")
    text = " ".join(text.split())

    if len(text) > max_chars:
        return text[:max_chars].rstrip() + "..."

    return text


def get_metadata(doc):
    return dict(getattr(doc, "metadata", {}) or {})


def get_source_label(metadata):
    return (
        metadata.get("file_name")
        or metadata.get("source_file")
        or metadata.get("source")
        or metadata.get("path")
        or "Unknown source"
    )


def has_e5_prefix_leak(text):
    text = str(text or "").strip().lower()
    return text.startswith("query:") or text.startswith("passage:")


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


def extract_chunks_from_cache_object(cache_object):
    # Different projects save chunk cache differently.
    # This function accepts list directly or dict wrappers.
    if isinstance(cache_object, list):
        return cache_object

    if isinstance(cache_object, tuple):
        for item in cache_object:
            chunks = extract_chunks_from_cache_object(item)
            if chunks:
                return chunks

    if isinstance(cache_object, dict):
        for key in ("chunks", "docs", "documents", "data"):
            value = cache_object.get(key)
            chunks = extract_chunks_from_cache_object(value)
            if chunks:
                return chunks

    return []


def load_chunks_from_cache():
    cache_path = Path(CHUNK_CACHE_PATH)

    if not cache_path.is_absolute():
        cache_path = PROJECT_ROOT / cache_path

    if not cache_path.exists():
        raise FileNotFoundError(
            f"Chunk cache not found: {cache_path}. Run python ingest.py first."
        )

    try:
        from utils.chunk_cache import load_chunks_cache

        attempts = [
            lambda: load_chunks_cache(),
            lambda: load_chunks_cache(data_path=None),
            lambda: load_chunks_cache(cache_path),
            lambda: load_chunks_cache(str(cache_path)),
        ]

        for attempt in attempts:
            try:
                cache_object = attempt()
                chunks = extract_chunks_from_cache_object(cache_object)
                if chunks:
                    return chunks
            except TypeError:
                continue
            except Exception:
                break

    except ImportError:
        pass

    with cache_path.open("rb") as file:
        cache_object = pickle.load(file)

    chunks = extract_chunks_from_cache_object(cache_object)

    if not chunks:
        raise ValueError(
            f"Chunk cache was loaded but no chunks were found inside: {cache_path}"
        )

    return chunks


def add_config(lines, args, output_path):
    write_line(lines, "=" * 80)
    write_line(lines, "RETRIEVAL TEST: SEMANTIC -> BM25 -> HYBRID/RRF -> RERANK")
    write_line(lines, "=" * 80)
    write_line(lines, f"Project root        : {PROJECT_ROOT}")
    write_line(lines, f"Chroma path         : {CHROMA_PATH}")
    write_line(lines, f"Chroma collection   : {CHROMA_COLLECTION_NAME}")
    write_line(lines, f"Embedding model     : {EMBEDDING_MODEL_NAME}")
    write_line(lines, f"USE_E5_PREFIX       : {USE_E5_PREFIX}")
    write_line(lines, f"Semantic K          : {args.semantic_k}")
    write_line(lines, f"BM25 K              : {args.bm25_k}")
    write_line(lines, f"Hybrid final K      : {args.hybrid_k}")
    write_line(lines, f"Rerank pool K       : {args.pool_k}")
    write_line(lines, f"Rerank top N        : {args.top_n}")
    write_line(lines, f"Evidence check      : {args.use_evidence_check}")
    write_line(lines, f"Report file         : {output_path}")
    write_line(lines, "")


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
    bm25_rank = metadata.get("bm25_rank", "N/A")
    hybrid_rank = metadata.get("hybrid_rank", "N/A")
    hybrid_score = metadata.get("hybrid_score", "N/A")
    boost_rank = metadata.get("metadata_boost_rank", "N/A")
    rerank_rank = metadata.get("rerank_rank", "N/A")
    rerank_score = metadata.get("rerank_score", "N/A")
    original_rank = metadata.get("rerank_original_rank", "N/A")
    evidence_match_count = metadata.get("evidence_match_count", "N/A")
    evidence_proximity_ok = metadata.get("evidence_proximity_ok", "N/A")
    preview = clean_preview(doc.page_content)

    write_line(lines, f"{index}. Source      : {source}")
    write_line(lines, f"   Page        : {page}")
    write_line(lines, f"   Chunk       : {chunk_id}")
    write_line(lines, f"   Category    : {category}")
    write_line(lines, f"   Doc type    : {doc_type}")
    write_line(lines, f"   Language    : {language}")

    if stage_name in {"SEMANTIC", "HYBRID", "RERANK"}:
        write_line(lines, f"   Semantic rk : {semantic_rank}")
        write_line(lines, f"   Sem distance: {semantic_distance}")

    if stage_name in {"BM25", "HYBRID", "RERANK"}:
        write_line(lines, f"   BM25 rank   : {bm25_rank}")

    if stage_name in {"HYBRID", "RERANK"}:
        write_line(lines, f"   Hybrid rank : {hybrid_rank}")
        write_line(lines, f"   Hybrid score: {hybrid_score}")
        write_line(lines, f"   Boost rank  : {boost_rank}")

    if stage_name == "RERANK":
        write_line(lines, f"   Rerank rank : {rerank_rank}")
        write_line(lines, f"   Rerank score: {rerank_score}")
        write_line(lines, f"   Orig rank   : {original_rank}")
        write_line(lines, f"   Evidence    : matches={evidence_match_count}, proximity_ok={evidence_proximity_ok}")

    if has_e5_prefix_leak(doc.page_content):
        write_line(lines, "   WARNING     : E5 prefix leaked into chunk text. Remove query:/passage: from page_content.")

    write_line(lines, f"   Preview     : {preview}")
    write_line(lines, "")


def add_stage_results(lines, title, docs, limit, stage_name):
    write_line(lines, "")
    write_line(lines, title)
    write_line(lines, "-" * len(title))

    if not docs:
        write_line(lines, "No results.")
        return

    for index, doc in enumerate(docs[:limit], start=1):
        add_doc_result(lines, index, doc, stage_name=stage_name)


def run_single_query(lines, vectorstore, bm25_retriever, reranker, query_item, args):
    query_id = query_item.get("id", "CUSTOM")
    raw_query = str(query_item.get("query", "") or "").strip()
    vector_query = format_semantic_query(raw_query, use_e5_prefix=USE_E5_PREFIX)

    write_line(lines, "=" * 80)
    write_line(lines, f"Test ID            : {query_id}")
    write_line(lines, f"Raw query          : {raw_query}")
    write_line(lines, f"Semantic query     : {vector_query}")
    write_line(lines, "=" * 80)

    start_time = time.perf_counter()
    details = hybrid_search(
        query=raw_query,
        vectorstore=vectorstore,
        bm25_retriever=bm25_retriever,
        semantic_k=args.semantic_k,
        bm25_k=args.bm25_k,
        final_k=args.hybrid_k,
        debug=args.debug,
        return_details=True,
    )
    hybrid_elapsed = time.perf_counter() - start_time

    semantic_docs = details.get("semantic_docs", [])
    bm25_docs = details.get("bm25_docs", [])
    hybrid_docs = details.get("hybrid_docs", [])
    rerank_pool = hybrid_docs[: args.pool_k]

    start_time = time.perf_counter()
    reranked_items = rerank_documents(
        query=raw_query,
        documents=rerank_pool,
        reranker=reranker,
        top_n=args.top_n,
        return_scores=True,
        show_scores=False,
        debug=args.debug,
        use_evidence_check=args.use_evidence_check,
        require_proximity=args.require_proximity,
    )
    rerank_elapsed = time.perf_counter() - start_time

    reranked_docs = [doc for doc, _score in reranked_items]

    write_line(lines, f"Hybrid time        : {format_seconds(hybrid_elapsed)}")
    write_line(lines, f"Rerank time        : {format_seconds(rerank_elapsed)}")
    write_line(lines, f"Semantic results   : {len(semantic_docs)}")
    write_line(lines, f"BM25 results       : {len(bm25_docs)}")
    write_line(lines, f"Hybrid results     : {len(hybrid_docs)}")
    write_line(lines, f"Reranked results   : {len(reranked_docs)}")

    add_stage_results(lines, "SEMANTIC RESULTS", semantic_docs, args.stage_limit, "SEMANTIC")
    add_stage_results(lines, "BM25 RESULTS", bm25_docs, args.stage_limit, "BM25")
    add_stage_results(lines, "HYBRID/RRF RESULTS", hybrid_docs, args.stage_limit, "HYBRID")
    add_stage_results(lines, "RERANKED FINAL RESULTS", reranked_docs, args.top_n, "RERANK")

    top_source = "N/A"
    if reranked_docs:
        top_source = get_source_label(get_metadata(reranked_docs[0]))

    return {
        "id": query_id,
        "query": raw_query,
        "semantic_results": len(semantic_docs),
        "bm25_results": len(bm25_docs),
        "hybrid_results": len(hybrid_docs),
        "reranked_results": len(reranked_docs),
        "hybrid_time": hybrid_elapsed,
        "rerank_time": rerank_elapsed,
        "top_source": top_source,
    }


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


def save_report(lines, output_path):
    output_path = Path(output_path)

    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def add_summary(lines, summaries):
    write_line(lines, "")
    write_line(lines, "=" * 80)
    write_line(lines, "SUMMARY")
    write_line(lines, "=" * 80)

    for item in summaries:
        write_line(
            lines,
            f"{item['id']:<6} "
            f"sem={item['semantic_results']:<3} "
            f"bm25={item['bm25_results']:<3} "
            f"hybrid={item['hybrid_results']:<3} "
            f"rerank={item['reranked_results']:<3} "
            f"hybrid_time={format_seconds(item['hybrid_time']):>10} "
            f"rerank_time={format_seconds(item['rerank_time']):>10} "
            f"top_source={item['top_source']}",
        )


def main():
    parser = argparse.ArgumentParser(description="Test retrieval from semantic to hybrid to reranker.")
    parser.add_argument("--query", type=str, default="", help="Single query to test.")
    parser.add_argument("--query-file", type=str, default="", help="Text file with one query per line.")
    parser.add_argument("--interactive", action="store_true", help="Enter one query manually.")
    parser.add_argument("--semantic-k", type=int, default=SEMANTIC_K, help="Number of semantic results.")
    parser.add_argument("--bm25-k", type=int, default=BM25_K, help="Number of BM25 results.")
    parser.add_argument("--hybrid-k", type=int, default=HYBRID_FINAL_K, help="Number of hybrid/RRF results.")
    parser.add_argument("--pool-k", type=int, default=RERANK_POOL_TOP_N, help="Hybrid docs passed to reranker.")
    parser.add_argument("--top-n", type=int, default=RERANK_TOP_N, help="Final reranked docs.")
    parser.add_argument("--stage-limit", type=int, default=5, help="How many docs to print per non-final stage.")
    parser.add_argument("--use-evidence-check", action="store_true", help="Optional evidence gate inside reranker.")
    parser.add_argument("--require-proximity", action="store_true", help="Require proximity when evidence check is enabled.")
    parser.add_argument("--debug", action="store_true", help="Print debug logs from retrievers/reranker.")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT_PATH), help="Report txt path.")
    args = parser.parse_args()

    if args.pool_k > args.hybrid_k:
        args.pool_k = args.hybrid_k

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

    write_line(lines, "[START] Load chunks from cache")
    start_time = time.perf_counter()
    chunks = load_chunks_from_cache()
    write_line(lines, f"[DONE]  Load chunks from cache - {format_seconds(time.perf_counter() - start_time)}")
    write_line(lines, f"Chunks loaded       : {len(chunks)}")
    write_line(lines, "")

    write_line(lines, "[START] Create BM25 retriever")
    start_time = time.perf_counter()
    bm25_retriever = create_bm25_retriever(chunks=chunks, k=args.bm25_k, debug=args.debug)
    write_line(lines, f"[DONE]  Create BM25 retriever - {format_seconds(time.perf_counter() - start_time)}")
    write_line(lines, "")

    write_line(lines, "[START] Load reranker model")
    start_time = time.perf_counter()
    reranker = load_reranker(debug=args.debug)
    write_line(lines, f"[DONE]  Load reranker model - {format_seconds(time.perf_counter() - start_time)}")
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
            reranker=reranker,
            query_item=query_item,
            args=args,
        )
        summaries.append(summary)

    add_summary(lines, summaries)
    report_path = save_report(lines, output_path)

    print("")
    print("HYBRID TO RERANK TEST DONE")
    print(f"Report saved to: {report_path}")


if __name__ == "__main__":
    main()
