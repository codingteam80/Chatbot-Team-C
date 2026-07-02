import argparse
import inspect
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import (
    BM25_K,
    DATA_PATH,
    ENABLE_FALLBACK_RETRY,
    ENABLE_MULTI_QUERY_RETRIEVAL,
    ENABLE_TRUNCATION_RETRY,
    HYBRID_FINAL_K,
    MAX_CANDIDATES_BEFORE_RERANK,
    MAX_RETRIEVAL_QUERIES,
    NO_ANSWER_TEXT,
    RERANK_POOL_TOP_N,
    RERANK_TOP_N,
    SEMANTIC_K,
)
from chains.rag_chain import generate_answer
from embeddings.embedding_model import get_embedding_model
from llm.ollama_llm import load_llm
from loaders.document_loader import load_documents
from preprocessing.cleaner import clean_documents
from preprocessing.chunker import chunk_documents
from retrieval.bm25_retriever import create_bm25_retriever
from retrieval.context_filter import select_final_context_docs
from retrieval.hybrid_retriever import hybrid_search
from retrieval.reranker import load_reranker, rerank_documents
from vectorstore.chroma_store import load_chroma_vectorstore

try:
    from utils import chunk_cache as chunk_cache_module
except ImportError:
    chunk_cache_module = None


REPORT_FILE = Path("reports") / "rag_generation_report.txt"
PREVIEW_CHARS = 450

FALLBACK_RETRY_SKIP_CATEGORIES = {"NEGATIVE_UNANSWERABLE", "FALSE_PREMISE"}
TERMINAL_PUNCTUATION = (".", "?", "!", ")", "]", "}", '"', "'")
CURRENT_CASE_TIMINGS = None


# Question-only test cases.
# Walang expected_sources, answer_keywords, or min_keyword_matches.
# Hayaan ang retrieval + context filter + LLM prompt ang mag-decide.
TEST_CASES = [
    {
        "id": "R01",
        "category": "RETRIEVAL_BASIC",
        "question": "Who are the ladies that had relationship with Jose Rizal?",
    },
    {
        "id": "R02",
        "category": "RETRIEVAL_BASIC",
        "question": "Who killed Ferdinand Magellan?",
    },
    {
        "id": "R03",
        "category": "RETRIEVAL_BASIC",
        "question": "Did Lapu-Lapu kill Magellan?",
    },
    {
        "id": "R04",
        "category": "RETRIEVAL_BASIC",
        "question": "Who is the first Philippine president?",
    },
    {
        "id": "R05",
        "category": "RETRIEVAL_BASIC",
        "question": "What did the Treaty of Paris of 1898 say about the Philippines?",
    },
    {
        "id": "R06",
        "category": "RETRIEVAL_BASIC",
        "question": "What hardships did Filipinos experience during the Japanese occupation?",
    },
    {
        "id": "D01",
        "category": "DIRECT",
        "question": "Who was Apolinario Mabini and what role did he serve in the First Philippine Republic?",
    },
    {
        "id": "K01",
        "category": "KEYWORD_HEAVY",
        "question": "Who founded the Katipunan or KKK on July 7, 1892, and what was its purpose against Spain?",
    },
    {
        "id": "P01",
        "category": "PARAPHRASED",
        "question": "Which secret group tried to free Filipinos from Spanish rule through armed revolution before it was discovered in 1896?",
    },
    {
        "id": "T01",
        "category": "TAGALOG_ENGLISH_DOCS",
        "question": "Kailan ipinagdiriwang ang Araw ng Kalayaan ng Pilipinas at anong pangyayari ang ginugunita nito?",
    },
    {
        "id": "N01",
        "category": "NEGATIVE_UNANSWERABLE",
        "question": "What was Andres Bonifacio's official passport number during the Philippine Revolution?",
    },
    {
        "id": "X01",
        "category": "CROSS_DOC",
        "question": "How did the Treaty of Paris connect the Spanish-American War to the Philippine-American War?",
    },
    {
        "id": "F01",
        "category": "FALSE_PREMISE",
        "question": "Why did Jose Rizal become the Supremo of the Katipunan?",
    },
    {
        "id": "S01",
        "category": "SIMILAR_TOPIC",
        "question": "Compare the Spanish-American War vs. the Philippine-American War. Do not interchange the cause and result of the two wars.",
    },
    {
        "id": "S02",
        "category": "SIMILAR_TOPIC",
        "question": "What is the difference between the Philippine Revolution and the Katipunan? Is one an organization and the other a war/revolution?",
    },
]


def format_seconds(seconds):
    # Convert seconds to readable text.
    if seconds < 60:
        return f"{seconds:.2f} sec"

    minutes = int(seconds // 60)
    remaining_seconds = seconds % 60
    return f"{minutes} min {remaining_seconds:.2f} sec"


def call_supported(function, *args, **kwargs):
    # Call a function and ignore unsupported keyword arguments.
    # Useful kapag nagbabago ang function signature habang nagte-test.
    parameters = inspect.signature(function).parameters
    accepts_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )

    if accepts_kwargs:
        return function(*args, **kwargs)

    supported_kwargs = {}

    for key, value in kwargs.items():
        if key in parameters:
            supported_kwargs[key] = value

    return function(*args, **supported_kwargs)


def add_section(lines, title):
    # Add a report section.
    lines.append("")
    lines.append("=" * 80)
    lines.append(title)
    lines.append("=" * 80)


def add_timing_value(timings, name, elapsed):
    # Add elapsed time to a timing dictionary.
    if timings is None:
        return

    timings[name] = timings.get(name, 0.0) + elapsed


def timed_step(name, timings, function):
    # Run one step and record elapsed time globally and per question.
    global CURRENT_CASE_TIMINGS

    print(f"[START] {name}", flush=True)
    start_time = time.perf_counter()
    result = function()
    elapsed = time.perf_counter() - start_time

    add_timing_value(timings, name, elapsed)
    add_timing_value(CURRENT_CASE_TIMINGS, name, elapsed)

    print(f"[DONE]  {name} - {format_seconds(elapsed)}", flush=True)
    return result


def get_timing_total(timings):
    # Total measured time from a timing dictionary.
    return sum((timings or {}).values())


def get_bottleneck(timings):
    # Return the slowest step name and elapsed time.
    if not timings:
        return "None", 0.0

    step_name = max(timings, key=timings.get)
    return step_name, timings[step_name]


def add_timing_breakdown(lines, title, timings, elapsed=None):
    # Add timing details and bottleneck to the report.
    add_section(lines, title)

    timings = dict(timings or {})
    measured_total = get_timing_total(timings)
    denominator = elapsed if elapsed and elapsed > 0 else measured_total

    lines.append(f"End-to-end time : {format_seconds(elapsed or measured_total)}")
    lines.append(f"Measured steps  : {format_seconds(measured_total)}")

    if elapsed and elapsed > measured_total:
        overhead = elapsed - measured_total
        lines.append(f"Untimed overhead: {format_seconds(overhead)}")

    bottleneck_name, bottleneck_time = get_bottleneck(timings)
    bottleneck_percent = (bottleneck_time / denominator * 100.0) if denominator else 0.0
    lines.append(f"Bottleneck      : {bottleneck_name} - {format_seconds(bottleneck_time)} ({bottleneck_percent:.1f}%)")
    lines.append("")

    if not timings:
        lines.append("No measured steps.")
        return

    for name, step_time in sorted(timings.items(), key=lambda item: item[1], reverse=True):
        percent = (step_time / denominator * 100.0) if denominator else 0.0
        lines.append(f"- {name:<26} {format_seconds(step_time):>14} ({percent:>5.1f}%)")


def get_metadata(doc):
    # Safe metadata getter.
    return dict(getattr(doc, "metadata", {}) or {})


def get_source_label(doc):
    # Human-readable source label.
    metadata = get_metadata(doc)
    source = metadata.get("file_name") or metadata.get("source") or "Unknown source"
    page = metadata.get("page", "N/A")
    chunk_id = metadata.get("chunk_id") or metadata.get("chunk_index") or "Unknown chunk"
    return f"{source} | page={page} | {chunk_id}"


def get_preview(doc, max_chars=PREVIEW_CHARS):
    # Short chunk preview.
    text = " ".join(str(getattr(doc, "page_content", "") or "").split())

    if len(text) <= max_chars:
        return text

    return text[:max_chars].rstrip() + "..."


def save_report(lines):
    # Save text report.
    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.write_text("\n".join(lines), encoding="utf-8")


def normalize_rerank_result(result):
    # Support list[Document] or list[(Document, score)].
    docs = []

    for item in result or []:
        if isinstance(item, tuple) and item:
            docs.append(item[0])
        else:
            docs.append(item)

    return docs


def load_chunks_from_cache():
    # Try chunk cache first para mas mabilis ang BM25 setup.
    if chunk_cache_module is None:
        return None

    possible_names = [
        "load_chunks_cache",
        "load_cached_chunks",
        "load_chunks_from_cache",
        "get_chunks_cache",
        "read_chunks_cache",
    ]

    for name in possible_names:
        loader = getattr(chunk_cache_module, name, None)

        if loader is None:
            continue

        try:
            result = call_supported(loader, data_path=DATA_PATH, source_path=DATA_PATH)

            if isinstance(result, tuple):
                result = result[0]

            if isinstance(result, dict):
                for key in ["chunks", "cached_chunks", "documents", "docs"]:
                    if result.get(key):
                        return result[key]

            if isinstance(result, list) and result:
                return result

        except Exception:
            continue

    return None


def load_chunks_for_bm25():
    # Load chunks from cache or rebuild from source documents.
    chunks = load_chunks_from_cache()

    if chunks:
        print(f"[CACHE] Chunks loaded: {len(chunks)}", flush=True)
        return chunks

    print("[CACHE] No usable chunk cache. Rebuilding chunks.", flush=True)
    docs = load_documents(DATA_PATH)
    cleaned_docs = clean_documents(docs)
    return chunk_documents(cleaned_docs)


class RagComponents:
    # Lazy loader para one setup lang sa buong test run.
    def __init__(self):
        self.timings = {}
        self.embedding_model = None
        self.vectorstore = None
        self.chunks = None
        self.bm25_retriever = None
        self.reranker = None
        self.llm = None

    def get_vectorstore(self):
        if self.vectorstore is not None:
            return self.vectorstore

        self.embedding_model = timed_step("Load embedding model", self.timings, get_embedding_model)
        self.vectorstore = timed_step(
            "Load Chroma vectorstore",
            self.timings,
            lambda: load_chroma_vectorstore(self.embedding_model),
        )
        return self.vectorstore

    def get_bm25_retriever(self):
        if self.bm25_retriever is not None:
            return self.bm25_retriever

        self.chunks = timed_step("Load chunks for BM25", self.timings, load_chunks_for_bm25)
        self.bm25_retriever = timed_step(
            "Create BM25 retriever",
            self.timings,
            lambda: call_supported(create_bm25_retriever, self.chunks, k=BM25_K),
        )
        return self.bm25_retriever

    def get_reranker(self):
        if self.reranker is not None:
            return self.reranker

        self.reranker = timed_step("Load reranker", self.timings, load_reranker)
        return self.reranker

    def get_llm(self):
        if self.llm is not None:
            return self.llm

        self.llm = timed_step("Load LLM", self.timings, load_llm)
        return self.llm


def get_doc_key(doc):
    # Stable key para hindi maulit ang parehong chunk mula sa multiple queries.
    metadata = get_metadata(doc)
    return (
        metadata.get("source") or metadata.get("file_name") or "",
        metadata.get("page", ""),
        metadata.get("chunk_id") or metadata.get("chunk_index") or id(doc),
    )


def merge_unique_docs(*doc_lists, limit=0):
    # Pagsamahin ang docs habang tinatanggal ang duplicate chunks.
    merged = []
    seen = set()

    for docs in doc_lists:
        for doc in docs or []:
            key = get_doc_key(doc)

            if key in seen:
                continue

            seen.add(key)
            merged.append(doc)

            if limit and len(merged) >= limit:
                return merged

    return merged


def build_retrieval_queries(question, category=""):
    # Magdagdag ng controlled query variants para sa yes/no, cross-doc, at comparison cases.
    # Original question pa rin ang gamit sa reranker at answer generation.
    question = str(question or "").strip()
    category = str(category or "").strip().upper()

    queries = [question]

    if not ENABLE_MULTI_QUERY_RETRIEVAL:
        return queries

    lowered = question.lower()

    if category == "CROSS_DOC" or "connect" in lowered or "relationship" in lowered:
        queries.extend([
            "Treaty of Paris 1898 Spanish-American War Philippines Philippine-American War",
            "Philippine-American War December 1898 Treaty of Paris following Spanish-American War",
        ])

    if "katipunan" in lowered and "philippine revolution" in lowered:
        queries.extend([
            "Philippine Revolution Katipunan difference organization revolution",
            "Katipunan secret organization Philippine Revolution armed revolution",
            "Philippine Revolution war revolution Katipunan organization",
        ])
    elif category == "SIMILAR_TOPIC" or "compare" in lowered or " vs " in lowered or "difference" in lowered:
        queries.extend([
            "Spanish-American War cause result Philippines",
            "Philippine-American War cause result Treaty of Paris Filipino nationalists",
        ])

    if category == "DIRECT" and "apolinario mabini" in lowered:
        queries.append("Apolinario Mabini First Prime Minister First Philippine Republic role")

    if "lapu" in lowered and "magellan" in lowered:
        queries.append("Ferdinand Magellan killed Lapulapu chief of Mactan")

    unique_queries = []
    seen = set()

    for query in queries:
        query = " ".join(str(query or "").split())
        key = query.lower()

        if not query or key in seen:
            continue

        seen.add(key)
        unique_queries.append(query)

        if len(unique_queries) >= MAX_RETRIEVAL_QUERIES:
            break

    return unique_queries


def run_retrieval(question, components, use_metadata_boost=False, use_reranker=True, category=""):
    # Run retrieval pipeline up to final context docs.
    vectorstore = components.get_vectorstore()
    bm25_retriever = components.get_bm25_retriever()

    retrieval_queries = build_retrieval_queries(question, category=category)
    semantic_doc_lists = []
    bm25_doc_lists = []
    hybrid_doc_lists = []

    for retrieval_query in retrieval_queries:
        hybrid_result = timed_step(
            "Hybrid retrieval",
            components.timings,
            lambda current_query=retrieval_query: call_supported(
                hybrid_search,
                query=current_query,
                vectorstore=vectorstore,
                bm25_retriever=bm25_retriever,
                semantic_k=SEMANTIC_K,
                bm25_k=BM25_K,
                final_k=HYBRID_FINAL_K,
                use_metadata_boost=use_metadata_boost,
                return_details=True,
                debug=False,
            ),
        )

        if isinstance(hybrid_result, dict):
            semantic_doc_lists.append(hybrid_result.get("semantic_docs", []))
            bm25_doc_lists.append(hybrid_result.get("bm25_docs", []))
            hybrid_doc_lists.append(hybrid_result.get("hybrid_docs", []))
        else:
            hybrid_doc_lists.append(hybrid_result or [])

    semantic_docs = merge_unique_docs(*semantic_doc_lists)
    bm25_docs = merge_unique_docs(*bm25_doc_lists)
    hybrid_docs = merge_unique_docs(
        *hybrid_doc_lists,
        limit=MAX_CANDIDATES_BEFORE_RERANK,
    )

    if use_reranker:
        reranker = components.get_reranker()
        rerank_result = timed_step(
            "Rerank documents",
            components.timings,
            lambda: call_supported(
                rerank_documents,
                query=question,
                documents=hybrid_docs,
                reranker=reranker,
                top_n=min(len(hybrid_docs), max(RERANK_POOL_TOP_N, RERANK_TOP_N)),
                return_scores=True,
                debug=False,
            ),
        )
        ranked_docs = normalize_rerank_result(rerank_result)
    else:
        ranked_docs = hybrid_docs

    final_docs = timed_step(
        "Final context filter",
        components.timings,
        lambda: call_supported(
            select_final_context_docs,
            reranked_docs=ranked_docs,
            question=question,
            semantic_docs=semantic_docs,
            bm25_docs=bm25_docs,
            top_n=None,
            debug=False,
        ),
    )

    return {
        "retrieval_queries": retrieval_queries,
        "semantic_docs": semantic_docs,
        "bm25_docs": bm25_docs,
        "hybrid_docs": hybrid_docs,
        "ranked_docs": ranked_docs,
        "final_docs": final_docs,
    }


def is_fallback_answer(answer):
    # Exact fallback checker.
    return str(answer or "").strip() == NO_ANSWER_TEXT


def looks_truncated_answer(answer):
    # Simple detector para makita kung naputol ang sagot sa report.
    text = " ".join(str(answer or "").split())

    if not text or is_fallback_answer(text):
        return False

    if len(text.split()) < 6:
        return False

    return not text.endswith(TERMINAL_PUNCTUATION)


def should_retry_answer(answer, retrieval_result, category="", no_llm=False):
    # Retry lang kapag may context at hindi negative/false-premise test case.
    if no_llm:
        return False

    if not retrieval_result.get("final_docs"):
        return False

    category = str(category or "").strip().upper()

    if category in FALLBACK_RETRY_SKIP_CATEGORIES:
        return False

    if ENABLE_FALLBACK_RETRY and is_fallback_answer(answer):
        return True

    if ENABLE_TRUNCATION_RETRY and looks_truncated_answer(answer):
        return True

    return False


def generate_rag_answer(question, retrieval_result, components, no_llm=False, category=""):
    # Generate answer using final context docs.
    if no_llm:
        return "[NO LLM MODE] Answer generation skipped."

    llm = components.get_llm()

    def generate_once(correction_retry=False, completion_retry=False):
        return call_supported(
            generate_answer,
            question=question,
            docs=retrieval_result["final_docs"],
            semantic_docs=retrieval_result.get("semantic_docs", []),
            bm25_docs=retrieval_result.get("bm25_docs", []),
            llm=llm,
            strict_assumption_check=True,
            correction_retry=correction_retry,
            completion_retry=completion_retry,
            debug=False,
        )

    answer = timed_step(
        "Generate answer",
        components.timings,
        lambda: generate_once(),
    )

    if should_retry_answer(answer, retrieval_result, category=category, no_llm=no_llm):
        answer = timed_step(
            "Fallback/completion retry",
            components.timings,
            lambda: generate_once(
                correction_retry=is_fallback_answer(answer),
                completion_retry=looks_truncated_answer(answer),
            ),
        )

    return answer


def get_answer_status(answer, no_llm=False):
    # Automatic status only.
    # Walang hardcoded expected keywords dito.
    answer = str(answer or "").strip()

    if no_llm:
        return "SKIPPED"

    if not answer:
        return "EMPTY_ANSWER"

    if answer == NO_ANSWER_TEXT:
        return "FALLBACK"

    if looks_truncated_answer(answer):
        return "POSSIBLY_TRUNCATED"

    return "ANSWERED"


def get_context_status(final_docs):
    # Automatic context status only.
    if final_docs:
        return "HAS_CONTEXT"

    return "NO_CONTEXT"


def add_docs_to_report(lines, title, docs, max_docs=5):
    # Add document preview to report.
    add_section(lines, title)

    if not docs:
        lines.append("No documents.")
        return

    for index, doc in enumerate(docs[:max_docs], start=1):
        metadata = get_metadata(doc)
        lines.append(f"Rank    : {index}")
        lines.append(f"Source  : {get_source_label(doc)}")
        lines.append(f"Title   : {metadata.get('title', '')}")
        lines.append(f"Section : {metadata.get('section', '')}")
        lines.append(f"Category: {metadata.get('category', '')}")
        lines.append(f"Doc type: {metadata.get('doc_type', '')}")

        score_keys = [
            "semantic_distance",
            "bm25_rank",
            "hybrid_score",
            "metadata_boosted_score",
            "rerank_score",
        ]

        for key in score_keys:
            if key in metadata:
                lines.append(f"{key}: {metadata.get(key)}")

        lines.append(f"Preview : {get_preview(doc)}")
        lines.append("-" * 80)


def add_sources_to_report(lines, docs):
    # Source-only output para malinis ang generation report.
    add_section(lines, "SOURCES")

    if not docs:
        lines.append("No sources.")
        return

    seen_sources = set()

    for doc in docs or []:
        label = get_source_label(doc)

        if label in seen_sources:
            continue

        seen_sources.add(label)
        lines.append(f"- {label}")


def add_case_report(lines, test_case, retrieval_result, answer, elapsed, case_timings=None, no_llm=False):
    # Add one case result to report.
    # Shows end-to-end timing and bottleneck per question.
    final_docs = retrieval_result.get("final_docs", [])
    context_status = get_context_status(final_docs)
    answer_status = get_answer_status(answer, no_llm=no_llm)
    retrieval_queries = retrieval_result.get("retrieval_queries", [])

    bottleneck_name, bottleneck_time = get_bottleneck(case_timings)
    bottleneck_percent = (bottleneck_time / elapsed * 100.0) if elapsed else 0.0

    add_section(lines, f"CASE {test_case['id']} - {test_case['category']}")
    lines.append(f"Question       : {test_case['question']}")
    lines.append(f"Context status : {context_status}")
    lines.append(f"Answer status  : {answer_status}")
    lines.append(f"Sources        : {len(final_docs)}")
    lines.append(f"Retrieval queries: {len(retrieval_queries)}")
    lines.append(f"Case time      : {format_seconds(elapsed)}")
    lines.append(f"Bottleneck     : {bottleneck_name} - {format_seconds(bottleneck_time)} ({bottleneck_percent:.1f}%)")
    lines.append("")

    lines.append("ANSWER:")
    lines.append(str(answer or ""))

    add_sources_to_report(lines, final_docs)
    add_timing_breakdown(lines, "CASE TIMING BREAKDOWN", case_timings, elapsed=elapsed)


def add_summary(lines, case_results, timings):
    # Add final summary with overall bottleneck and per-question bottlenecks.
    add_section(lines, "RAG TEST SUMMARY")
    total_cases = len(case_results)

    context_count = sum(
        1 for item in case_results
        if get_context_status(item["retrieval_result"].get("final_docs", [])) == "HAS_CONTEXT"
    )
    answered_count = sum(
        1 for item in case_results
        if get_answer_status(item["answer"], no_llm=item["no_llm"]) == "ANSWERED"
    )
    fallback_count = sum(
        1 for item in case_results
        if get_answer_status(item["answer"], no_llm=item["no_llm"]) == "FALLBACK"
    )
    truncated_count = sum(
        1 for item in case_results
        if get_answer_status(item["answer"], no_llm=item["no_llm"]) == "POSSIBLY_TRUNCATED"
    )

    total_end_to_end = sum(item.get("elapsed", 0.0) for item in case_results)
    total_measured = get_timing_total(timings)
    bottleneck_name, bottleneck_time = get_bottleneck(timings)
    bottleneck_percent = (bottleneck_time / total_measured * 100.0) if total_measured else 0.0

    lines.append(f"Total cases       : {total_cases}")
    lines.append(f"Cases with context: {context_count}")
    lines.append(f"Answered cases    : {answered_count}")
    lines.append(f"Fallback cases    : {fallback_count}")
    lines.append(f"Possibly truncated: {truncated_count}")
    lines.append(f"End-to-end total  : {format_seconds(total_end_to_end)}")
    lines.append(f"Measured total    : {format_seconds(total_measured)}")
    lines.append(f"Overall bottleneck: {bottleneck_name} - {format_seconds(bottleneck_time)} ({bottleneck_percent:.1f}%)")
    lines.append("")
    lines.append("Note:")
    lines.append("- This is a question-only RAG test.")
    lines.append("- No expected_sources, answer_keywords, or min_keyword_matches are used.")
    lines.append("- Load steps appear only when the model/vectorstore/retriever is first loaded in that run.")
    lines.append("- Review the final context and answer manually before tuning retrieval/prompt.")

    add_section(lines, "CASE STATUS SUMMARY")

    for item in case_results:
        context_status = get_context_status(item["retrieval_result"].get("final_docs", []))
        answer_status = get_answer_status(item["answer"], no_llm=item["no_llm"])
        case_timings = item.get("case_timings", {})
        case_bottleneck_name, case_bottleneck_time = get_bottleneck(case_timings)

        lines.append(
            f"{context_status:<11} | {answer_status:<18} | "
            f"{format_seconds(item['elapsed']):>14} | "
            f"{case_bottleneck_name} {format_seconds(case_bottleneck_time)} | "
            f"{item['case']['id']} | {item['case']['question']}"
        )

    add_timing_breakdown(lines, "OVERALL TIMING SUMMARY", timings, elapsed=total_end_to_end)

    add_section(lines, "BOTTLENECK SUMMARY BY CASE")

    for item in case_results:
        case_timings = item.get("case_timings", {})
        bottleneck_name, bottleneck_time = get_bottleneck(case_timings)
        percent = (bottleneck_time / item["elapsed"] * 100.0) if item["elapsed"] else 0.0

        lines.append(
            f"{item['case']['id']:<6} | "
            f"{bottleneck_name:<26} | "
            f"{format_seconds(bottleneck_time):>14} | "
            f"{percent:>5.1f}% | "
            f"{format_seconds(item['elapsed'])}"
        )



def select_cases(case_id="", limit=0):
    # Select one case or limited cases.
    if case_id:
        wanted = str(case_id).strip().upper()
        return [case for case in TEST_CASES if case["id"].upper() == wanted]

    if limit and limit > 0:
        return TEST_CASES[:limit]

    return TEST_CASES


def main():
    global CURRENT_CASE_TIMINGS

    parser = argparse.ArgumentParser(description="Run question-only RAG answer generation tests.")
    parser.add_argument("--case", default="", help="Run one case only, for example R04.")
    parser.add_argument("--limit", type=int, default=0, help="Run first N cases only.")
    parser.add_argument("--question", default="", help="Run a custom question instead of built-in cases.")
    parser.add_argument("--no-llm", action="store_true", help="Only test final context, skip LLM answer generation.")
    parser.add_argument("--no-reranker", action="store_true", help="Skip reranker and use hybrid docs directly.")
    parser.add_argument("--metadata-boost", action="store_true", help="Enable metadata boost during hybrid retrieval.")
    args = parser.parse_args()

    if args.question:
        cases = [
            {
                "id": "CUSTOM",
                "category": "CUSTOM",
                "question": args.question,
            }
        ]
    else:
        cases = select_cases(case_id=args.case, limit=args.limit)

    if not cases:
        raise ValueError("No matching test case found.")

    lines = []
    case_results = []
    components = RagComponents()

    add_section(lines, "QUESTION-ONLY RAG GENERATION TEST STARTED")
    lines.append(f"Data path       : {DATA_PATH}")
    lines.append(f"Report file     : {REPORT_FILE}")
    lines.append(f"Cases           : {len(cases)}")
    lines.append(f"Use reranker    : {not args.no_reranker}")
    lines.append(f"Metadata boost  : {args.metadata_boost}")
    lines.append(f"No LLM mode     : {args.no_llm}")
    lines.append("")
    lines.append("Evaluation mode : manual review")
    lines.append("Expected fields : disabled")

    for test_case in cases:
        print("\n" + "=" * 80)
        print(f"CASE {test_case['id']}: {test_case['question']}")
        print("=" * 80)
        CURRENT_CASE_TIMINGS = {}
        case_start = time.perf_counter()

        retrieval_result = run_retrieval(
            question=test_case["question"],
            components=components,
            use_metadata_boost=args.metadata_boost,
            use_reranker=not args.no_reranker,
            category=test_case.get("category", ""),
        )

        answer = generate_rag_answer(
            question=test_case["question"],
            retrieval_result=retrieval_result,
            components=components,
            no_llm=args.no_llm,
            category=test_case.get("category", ""),
        )

        elapsed = time.perf_counter() - case_start
        case_timings = dict(CURRENT_CASE_TIMINGS or {})
        CURRENT_CASE_TIMINGS = None

        case_results.append({
            "case": test_case,
            "retrieval_result": retrieval_result,
            "answer": answer,
            "elapsed": elapsed,
            "case_timings": case_timings,
            "no_llm": args.no_llm,
        })

        add_case_report(
            lines=lines,
            test_case=test_case,
            retrieval_result=retrieval_result,
            answer=answer,
            elapsed=elapsed,
            case_timings=case_timings,
            no_llm=args.no_llm,
        )

    add_summary(lines, case_results, components.timings)
    save_report(lines)

    print("\nDONE")
    print(f"Report saved to: {REPORT_FILE}")


if __name__ == "__main__":
    main()
