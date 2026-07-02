import inspect
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import config.settings as settings_module
except ImportError:
    settings_module = None


def get_setting(name, default_value="N/A"):
    # Read from config.settings when available.
    if settings_module is None:
        return default_value

    return getattr(settings_module, name, default_value)


BM25_K = get_setting("BM25_K", 9)
DATA_PATH = get_setting("DATA_PATH", "data")
HYBRID_FINAL_K = get_setting("HYBRID_FINAL_K", 11)
RERANK_TOP_N = get_setting("RERANK_TOP_N", 3)
SEMANTIC_K = get_setting("SEMANTIC_K", 9)

TRACKED_SETTING_NAMES = [
    "SEMANTIC_K",
    "BM25_K",
    "HYBRID_FINAL_K",
    "RERANK_TOP_N",
    "SINGLE_FACT_TOP_N",
    "CROSS_DOC_TOP_N",
    "COMPARISON_TOP_N",
    "NEGATIVE_TOP_N",
    "FALSE_PREMISE_TOP_N",
    "MIN_QUALITY_SCORE",
    "MAX_DOC_CHARS",
    "MAX_CONTEXT_CHARS",
    "MAX_PROMPT_CONTEXT_CHARS",
    "NEIGHBOR_WINDOW",
    "ENABLE_NEIGHBOR_EXPANSION",
    "DATA_PATH",
]

from embeddings.embedding_model import get_embedding_model
from loaders.document_loader import load_documents
from preprocessing.cleaner import clean_documents
from preprocessing.chunker import chunk_documents
from retrieval.bm25_retriever import create_bm25_retriever
from retrieval.context_filter import select_final_context_docs
from retrieval.hybrid_retriever import hybrid_search
from retrieval.query_analyzer import analyze_query
from retrieval.reranker import load_reranker, rerank_documents
from vectorstore.chroma_store import load_chroma_vectorstore

try:
    from utils import chunk_cache as chunk_cache_module
except ImportError:
    chunk_cache_module = None

REPORT_FILE = Path("reports") / "retrieval_report.txt"
PREVIEW_CHARS = 500


def format_seconds(seconds):
    # Human-readable seconds.
    if seconds < 60:
        return f"{seconds:.2f} sec"

    minutes = int(seconds // 60)
    remaining_seconds = seconds % 60
    return f"{minutes} min {remaining_seconds:.2f} sec"


def add_section(lines, title):
    # Add report section header.
    lines.append("")
    lines.append("=" * 80)
    lines.append(title)
    lines.append("=" * 80)


def save_report(lines):
    # Save report to reports/retrieval_report.txt.
    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.write_text("\n".join(lines), encoding="utf-8")


def normalize_for_compare(value):
    # Normalize setting values for simple env/config comparison.
    if isinstance(value, bool):
        return "true" if value else "false"

    return str(value).strip().lower()


def get_setting_sync_status(setting_name):
    # Compare imported config.settings value against current process env value.
    config_value = get_setting(setting_name, "<missing in config.settings>")
    env_value = os.environ.get(setting_name)

    if env_value is None:
        return config_value, "<not set>", "env not set; using config/default"

    if normalize_for_compare(config_value) == normalize_for_compare(env_value):
        return config_value, env_value, "SYNCED"

    return config_value, env_value, "MISMATCH - env exists but imported setting differs"


def add_settings_sync_report(lines):
    # Show whether this test script is reading config/settings/env values correctly.
    add_section(lines, "SETTINGS SYNC CHECK")
    lines.append(f"Python executable : {sys.executable}")
    lines.append(f"Current cwd       : {Path.cwd()}")
    lines.append(f"Script file       : {Path(__file__).resolve()}")
    lines.append(f"Project root      : {PROJECT_ROOT}")
    lines.append(f"config.settings   : {'LOADED' if settings_module is not None else 'NOT LOADED'}")
    lines.append(f"settings file     : {getattr(settings_module, '__file__', 'N/A')}")
    lines.append("")
    lines.append("Name                          | config.settings value | env value | status")
    lines.append("-" * 100)

    for setting_name in TRACKED_SETTING_NAMES:
        config_value, env_value, status = get_setting_sync_status(setting_name)
        lines.append(
            f"{setting_name:<29} | {str(config_value):<21} | {str(env_value):<9} | {status}"
        )

    lines.append("")
    lines.append("Note: If env value is set but status is MISMATCH, restart the app/test from the same .bat or check if config.settings reads that env variable.")


def get_function_source(function):
    # Show where an imported function actually came from.
    try:
        return inspect.getsourcefile(function) or "N/A"
    except Exception:
        return "N/A"


def add_function_signature_report(lines):
    # Show actual imported function signatures to confirm expected parameters.
    add_section(lines, "FUNCTION SIGNATURE CHECK")

    functions = [
        ("hybrid_search", hybrid_search),
        ("rerank_documents", rerank_documents),
        ("select_final_context_docs", select_final_context_docs),
        ("create_bm25_retriever", create_bm25_retriever),
    ]

    for function_name, function in functions:
        try:
            signature = inspect.signature(function)
        except Exception as error:
            signature = f"<signature unavailable: {type(error).__name__}>"

        lines.append(f"{function_name} source    : {get_function_source(function)}")
        lines.append(f"{function_name} signature : {signature}")
        lines.append("")


def add_call_parameter_report(lines, hybrid_count=None):
    # Show exact values this script will pass to retrieval functions.
    add_section(lines, "CALL PARAMETER CHECK")
    lines.append(f"create_bm25_retriever k              : {BM25_K}")
    lines.append(f"hybrid_search semantic_k             : {SEMANTIC_K}")
    lines.append(f"hybrid_search bm25_k                 : {BM25_K}")
    lines.append(f"hybrid_search final_k                : {HYBRID_FINAL_K}")

    if hybrid_count is None:
        lines.append("rerank_documents top_n               : len(hybrid_docs) after hybrid search")
    else:
        lines.append(f"rerank_documents top_n               : {hybrid_count} (len(hybrid_docs))")

    lines.append(f"select_final_context_docs top_n      : {RERANK_TOP_N}")
    lines.append(f"select_final_context_docs all_chunks : NOT PASSED in this retrieval test")


def get_chunk_id(doc):
    # Stable chunk identifier for flow debugging.
    metadata = get_metadata(doc)
    return str(metadata.get("chunk_id") or metadata.get("chunk_index") or id(doc))


def get_short_doc_label(doc):
    # Compact one-line doc label.
    metadata = get_metadata(doc)
    source = metadata.get("file_name") or metadata.get("source") or "Unknown source"
    page = metadata.get("page", "N/A")
    section = str(metadata.get("section", "") or "")[:70]
    return f"{source} | page={page} | section={section}"


def build_rank_map(docs):
    # Map chunk_id to 1-based rank.
    return {get_chunk_id(doc): index for index, doc in enumerate(docs or [], start=1)}


def add_chunk_flow_report(lines, semantic_docs, bm25_docs, hybrid_docs, reranked_docs, final_docs, max_rows=20):
    # Show which chunks survive or get dropped at each pipeline stage.
    add_section(lines, "CHUNK FLOW CHECK")

    semantic_rank = build_rank_map(semantic_docs)
    bm25_rank = build_rank_map(bm25_docs)
    hybrid_rank = build_rank_map(hybrid_docs)
    rerank_rank = build_rank_map(reranked_docs)
    final_rank = build_rank_map(final_docs)

    chunk_docs = {}
    ordered_chunk_ids = []

    for docs in [hybrid_docs, semantic_docs, bm25_docs, reranked_docs, final_docs]:
        for doc in docs or []:
            chunk_id = get_chunk_id(doc)
            if chunk_id not in chunk_docs:
                chunk_docs[chunk_id] = doc
                ordered_chunk_ids.append(chunk_id)

    lines.append("Chunk ID | semantic | bm25 | hybrid | rerank | final | label")
    lines.append("-" * 120)

    for chunk_id in ordered_chunk_ids[:max_rows]:
        doc = chunk_docs[chunk_id]
        lines.append(
            f"{chunk_id} | "
            f"S={semantic_rank.get(chunk_id, '-'):<3} | "
            f"B={bm25_rank.get(chunk_id, '-'):<3} | "
            f"H={hybrid_rank.get(chunk_id, '-'):<3} | "
            f"R={rerank_rank.get(chunk_id, '-'):<3} | "
            f"F={final_rank.get(chunk_id, '-'):<3} | "
            f"{get_short_doc_label(doc)}"
        )

    important_hybrid_limit = min(5, len(hybrid_docs or []))
    final_ids = set(final_rank)
    missing_top_hybrid = [
        doc for doc in (hybrid_docs or [])[:important_hybrid_limit]
        if get_chunk_id(doc) not in final_ids
    ]

    if missing_top_hybrid:
        lines.append("")
        lines.append(f"WARNING: These top {important_hybrid_limit} hybrid chunks were not included in FINAL RETRIEVED CHUNKS:")

        for doc in missing_top_hybrid:
            lines.append(f"- {get_chunk_id(doc)} | {get_short_doc_label(doc)}")
    else:
        lines.append("")
        lines.append(f"OK: Top {important_hybrid_limit} hybrid chunks are present in FINAL RETRIEVED CHUNKS.")


def run_step(step_name, timings, lines, function):
    # Run step with timer.
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


def call_with_optional_debug(function, *args, debug=False, **kwargs):
    # Call function with debug only kapag supported.
    parameters = inspect.signature(function).parameters

    if "debug" in parameters:
        return function(*args, debug=debug, **kwargs)

    return function(*args, **kwargs)


def get_metadata(doc):
    # Safe metadata getter.
    return dict(getattr(doc, "metadata", {}) or {})


def get_text(doc):
    # Safe text getter.
    return str(getattr(doc, "page_content", "") or "")


def get_source_label(doc):
    # Source label for report.
    metadata = get_metadata(doc)
    source = metadata.get("file_name") or metadata.get("source") or "Unknown source"
    page = metadata.get("page", "N/A")
    chunk_id = metadata.get("chunk_id") or metadata.get("chunk_index") or "Unknown chunk"
    return f"{source} | page={page} | {chunk_id}"


def get_preview(doc, max_chars=PREVIEW_CHARS):
    # Short preview for report.
    text = " ".join(get_text(doc).split())

    if len(text) <= max_chars:
        return text

    return text[:max_chars].rstrip() + "..."


def load_chunks_from_cache():
    # Try to load chunk cache if your project has utils/chunk_cache.py.
    if chunk_cache_module is None:
        return None, "utils.chunk_cache module not found"

    possible_loader_names = [
        "load_chunks_cache",
        "load_cached_chunks",
        "load_chunks_from_cache",
        "get_chunks_cache",
        "read_chunks_cache",
    ]

    for loader_name in possible_loader_names:
        loader = getattr(chunk_cache_module, loader_name, None)

        if loader is None:
            continue

        try:
            parameters = inspect.signature(loader).parameters

            if "data_path" in parameters:
                result = loader(data_path=DATA_PATH)
            elif "source_path" in parameters:
                result = loader(source_path=DATA_PATH)
            else:
                result = loader()

            if isinstance(result, tuple):
                result = result[0]

            if isinstance(result, dict):
                for key in ["chunks", "cached_chunks", "documents", "docs"]:
                    if result.get(key):
                        return result[key], f"Loaded from cache using {loader_name}()"

            if isinstance(result, list) and result:
                return result, f"Loaded from cache using {loader_name}()"

        except Exception as error:
            return None, f"{loader_name}() failed: {type(error).__name__}: {error}"

    return None, "No supported chunk cache loader found"


def load_chunks_for_bm25():
    # BM25 needs chunks.
    # Fast path: cache. Fallback: load -> clean -> chunk.
    chunks, cache_message = load_chunks_from_cache()

    if chunks:
        print(f"[CACHE] {cache_message}", flush=True)
        print(f"[CACHE] Chunks loaded: {len(chunks)}", flush=True)
        return chunks

    print(f"[CACHE] Not used: {cache_message}", flush=True)
    print("[CACHE] Fallback: load -> clean -> chunk raw documents.", flush=True)

    docs = load_documents(DATA_PATH)
    cleaned_docs = clean_documents(docs)
    return chunk_documents(cleaned_docs)


def add_query_info_report(lines, query_info):
    # Show detected query hints.
    add_section(lines, "QUERY HINTS")
    lines.append(f"Original query : {query_info.get('original_query')}")
    lines.append(f"Mode           : {query_info.get('mode')}")
    lines.append(f"Category hint  : {query_info.get('category') or 'none'}")
    lines.append(f"Doc type hint  : {query_info.get('doc_type') or 'none'}")
    lines.append(f"Language hint  : {query_info.get('language') or 'none'}")
    lines.append(f"Source hint    : {query_info.get('source_hint') or 'none'}")
    lines.append(f"Important terms: {query_info.get('important_terms')}")
    lines.append(f"Source keywords: {query_info.get('source_keywords')}")


def add_docs_to_report(lines, title, docs, score_keys=None, max_docs=9):
    # Print top docs with metadata.
    add_section(lines, title)

    if not docs:
        lines.append("No results.")
        return

    if score_keys is None:
        score_keys = []

    for index, doc in enumerate(docs[:max_docs], start=1):
        metadata = get_metadata(doc)
        lines.append(f"Rank    : {index}")
        lines.append(f"Source  : {get_source_label(doc)}")
        lines.append(f"Title   : {metadata.get('title', '')}")
        lines.append(f"Section : {metadata.get('section', '')}")
        lines.append(f"Category: {metadata.get('category', '')}")
        lines.append(f"Doc type: {metadata.get('doc_type', '')}")
        lines.append(f"Language: {metadata.get('language', '')}")

        for score_key in score_keys:
            if score_key in metadata:
                lines.append(f"{score_key}: {metadata.get(score_key)}")

        boost_reasons = metadata.get("metadata_boost_reasons")
        context_reasons = metadata.get("context_match_reasons")

        if boost_reasons:
            lines.append(f"Boost reasons  : {boost_reasons}")

        if context_reasons:
            lines.append(f"Context reasons: {context_reasons}")

        lines.append(f"Preview : {get_preview(doc)}")
        lines.append("-" * 80)


def add_timing_summary(lines, timings):
    # Show timing summary.
    add_section(lines, "TIMING SUMMARY")
    total_time = sum(timings.values())
    lines.append(f"Total measured time : {format_seconds(total_time)}")
    lines.append("")

    for step_name, elapsed_time in timings.items():
        percent = (elapsed_time / total_time * 100) if total_time else 0
        lines.append(f"- {step_name:<28} {format_seconds(elapsed_time):>14} ({percent:>5.1f}%)")

    if timings:
        bottleneck = max(timings, key=timings.get)
        lines.append("")
        lines.append(f"Main bottleneck     : {bottleneck}")
        lines.append(f"Bottleneck time     : {format_seconds(timings[bottleneck])}")


def main():
    # User input question.
    question = input("Enter your question: ").strip()

    if not question:
        print("No question entered.")
        return

    lines = []
    timings = {}
    query_info = analyze_query(question, debug=True)

    add_section(lines, "RETRIEVAL DEBUG TEST")
    lines.append(f"Question    : {question}")
    lines.append(f"Data path   : {DATA_PATH}")
    lines.append(f"Report file : {REPORT_FILE}")
    lines.append("")
    lines.append("Flow:")
    lines.append("1. Query analyzer")
    lines.append("2. Semantic search")
    lines.append("3. BM25 search")
    lines.append("4. RRF combine")
    lines.append("5. Metadata boost")
    lines.append("6. Rerank")
    lines.append("7. Final context filter")

    add_settings_sync_report(lines)
    add_function_signature_report(lines)
    add_call_parameter_report(lines)
    add_query_info_report(lines, query_info)

    try:
        embedding_model = run_step(
            "Load embedding model",
            timings,
            lines,
            get_embedding_model,
        )

        vectorstore = run_step(
            "Load Chroma vectorstore",
            timings,
            lines,
            lambda: load_chroma_vectorstore(embedding_model),
        )

        chunks = run_step(
            "Load chunks for BM25",
            timings,
            lines,
            load_chunks_for_bm25,
        )

        bm25_retriever = run_step(
            "Create BM25 retriever",
            timings,
            lines,
            lambda: call_with_optional_debug(
                create_bm25_retriever,
                chunks,
                k=BM25_K,
                debug=False,
            ),
        )

        reranker = run_step(
            "Load reranker",
            timings,
            lines,
            load_reranker,
        )

        hybrid_details = run_step(
            "Hybrid search",
            timings,
            lines,
            lambda: hybrid_search(
                query=question,
                vectorstore=vectorstore,
                bm25_retriever=bm25_retriever,
                semantic_k=SEMANTIC_K,
                bm25_k=BM25_K,
                final_k=HYBRID_FINAL_K,
                use_rrf=True,
                use_metadata_boost=True,
                debug=False,
                return_details=True,
            ),
        )

        semantic_docs = hybrid_details["semantic_docs"]
        bm25_docs = hybrid_details["bm25_docs"]
        hybrid_docs = hybrid_details["hybrid_docs"]

        add_call_parameter_report(lines, hybrid_count=len(hybrid_docs))

        reranked_docs_with_scores = run_step(
            "Rerank hybrid results",
            timings,
            lines,
            lambda: rerank_documents(
                query=question,
                documents=hybrid_docs,
                reranker=reranker,
                top_n=len(hybrid_docs),
                return_scores=True,
                debug=False,
            ),
        )

        reranked_docs = [doc for doc, score in reranked_docs_with_scores]

        final_docs = run_step(
            "Final context filter",
            timings,
            lines,
            lambda: select_final_context_docs(
                reranked_docs=reranked_docs,
                question=question,
                semantic_docs=semantic_docs,
                bm25_docs=bm25_docs,
                top_n=RERANK_TOP_N,
                debug=True,
            ),
        )

        add_docs_to_report(
            lines,
            "SEMANTIC RESULTS",
            semantic_docs,
            score_keys=["semantic_distance", "semantic_rank"],
            max_docs=SEMANTIC_K,
        )

        add_docs_to_report(
            lines,
            "BM25 RESULTS",
            bm25_docs,
            score_keys=["bm25_rank", "bm25_rank_score"],
            max_docs=BM25_K,
        )

        add_docs_to_report(
            lines,
            "RRF + METADATA BOOST RESULTS",
            hybrid_docs,
            score_keys=["hybrid_score", "metadata_boost_score", "metadata_boosted_score"],
            max_docs=HYBRID_FINAL_K,
        )

        add_docs_to_report(
            lines,
            "RERANKED RESULTS",
            reranked_docs,
            score_keys=["rerank_score", "rerank_rank", "rerank_original_rank"],
            max_docs=len(reranked_docs),
        )

        add_docs_to_report(
            lines,
            "FINAL RETRIEVED CHUNKS",
            final_docs,
            score_keys=["rerank_score", "metadata_boosted_score", "hybrid_score"],
            max_docs=len(final_docs),
        )

        add_chunk_flow_report(
            lines=lines,
            semantic_docs=semantic_docs,
            bm25_docs=bm25_docs,
            hybrid_docs=hybrid_docs,
            reranked_docs=reranked_docs,
            final_docs=final_docs,
        )

        add_section(lines, "FINAL CHECK")
        lines.append(f"Semantic results : {len(semantic_docs)}")
        lines.append(f"BM25 results     : {len(bm25_docs)}")
        lines.append(f"Hybrid candidates: {len(hybrid_docs)}")
        lines.append(f"Reranked results : {len(reranked_docs)}")
        lines.append(f"Final chunks     : {len(final_docs)}")
        lines.append(f"RERANK_TOP_N setting used as final top_n : {RERANK_TOP_N}")
        lines.append(f"HYBRID_FINAL_K setting used              : {HYBRID_FINAL_K}")
        lines.append(f"SEMANTIC_K/BM25_K settings used          : {SEMANTIC_K}/{BM25_K}")
        lines.append("")

        if len(reranked_docs) < len(hybrid_docs):
            lines.append("WARNING: rerank_documents returned fewer docs than hybrid candidates even though top_n=len(hybrid_docs) was requested.")
            lines.append("Check rerank_documents implementation, reranker score filtering, or internal top_n/default limits.")

        if len(final_docs) < int(RERANK_TOP_N):
            lines.append("WARNING: final_docs is lower than RERANK_TOP_N. Check quality filters, context mode limits, or final context filter logic.")

        lines.append("")
        lines.append("Goal: final chunks should come from the correct source/category/section before using the LLM.")

        add_timing_summary(lines, timings)
        save_report(lines)

        print("")
        print("RETRIEVAL DEBUG DONE")
        print(f"Semantic results : {len(semantic_docs)}")
        print(f"BM25 results     : {len(bm25_docs)}")
        print(f"Hybrid candidates: {len(hybrid_docs)}")
        print(f"Reranked results : {len(reranked_docs)}")
        print(f"Final chunks     : {len(final_docs)}")
        print(f"RERANK_TOP_N     : {RERANK_TOP_N}")
        print(f"HYBRID_FINAL_K   : {HYBRID_FINAL_K}")
        print(f"SEMANTIC/BM25_K  : {SEMANTIC_K}/{BM25_K}")
        print(f"Report saved to  : {REPORT_FILE}")

    except Exception as error:
        add_section(lines, "RETRIEVAL DEBUG FAILED")
        lines.append(f"Error type    : {type(error).__name__}")
        lines.append(f"Error message : {error}")
        add_timing_summary(lines, timings)
        save_report(lines)

        print("")
        print("[FAILED] Retrieval debug test failed.")
        print(f"Error type    : {type(error).__name__}")
        print(f"Error message : {error}")
        print(f"Report saved to: {REPORT_FILE}")


if __name__ == "__main__":
    main()
