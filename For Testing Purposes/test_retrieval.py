import argparse
import time
from pathlib import Path

from test_utils import (
    clean_preview,
    filter_test_cases,
    format_seconds,
    format_timing,
    get_bottleneck,
    get_source_values,
    get_total_time,
    match_keywords,
    match_sources,
    prepare_project_path,
    print_section,
    timed_step,
    write_report,
)

PROJECT_ROOT = prepare_project_path(__file__)

from config.settings import (
    BM25_K,
    DATA_PATH,
    HYBRID_FINAL_K,
    MAX_CONTEXT_CHARS,
    MIN_QUALITY_SCORE,
    PREVIEW_CHARS,
    RERANK_TOP_N,
    SEMANTIC_K,
)
# Optional retrieval settings para compatible kahit hindi pa updated ang config/settings.py.
try:
    from config.settings import (
        BALANCED_RERANK_TOP_N,
        CANDIDATE_MAX_PER_SOURCE,
        ENABLE_CANDIDATE_BALANCING,
        ENABLE_MULTI_QUERY_RETRIEVAL,
        MAX_CANDIDATES_BEFORE_RERANK,
        MAX_RETRIEVAL_QUERIES,
    )
except ImportError:
    ENABLE_MULTI_QUERY_RETRIEVAL = True
    MAX_RETRIEVAL_QUERIES = 3
    MAX_CANDIDATES_BEFORE_RERANK = 15
    ENABLE_CANDIDATE_BALANCING = True
    CANDIDATE_MAX_PER_SOURCE = 2
    BALANCED_RERANK_TOP_N = 8

from embeddings.embedding_model import get_embedding_model
from retrieval.candidate_balancer import (
    balance_candidates,
    balance_reranked_documents,
)
from retrieval.context_filter import filter_low_quality_docs, limit_context_docs
from retrieval.hybrid_retriever import hybrid_search
from retrieval.reranker import load_reranker, rerank_documents
from retrieval.retrieval_query_builder import build_retrieval_queries
from utils.bm25_cache import load_or_create_bm25
from utils.chunk_cache import load_or_create_chunks
from vectorstore.chroma_store import load_chroma_vectorstore


REPORT_FILENAME = "result_retrieve.txt"
MIN_KEYWORD_MATCHES = 1


TEST_CASES = [
    {
        "id": "D01",
        "type": "direct",
        "query": "Who was Apolinario Mabini and what role did he serve in the First Philippine Republic?",
        "expected_sources": ["Apolinario Mabini - Wikipedia.md"],
        "must_have": [
            "Apolinario Mabini",
            "legal and constitutional adviser",
            "first Prime Minister",
            "First Philippine Republic",
        ],
    },
    {
        "id": "D02",
        "type": "direct",
        "query": "Sino si Andres Bonifacio?",
        "expected_sources": [
            "Andrés Bonifacio - Wikipedia.md",
            "Andres Bonifacio - Wikipedia.md",
        ],
        "must_have": [
            "Andrés Bonifacio",
            "Andres Bonifacio",
            "Filipino revolutionary leader",
            "Father of the Philippine Revolution",
        ],
    },
    {
        "id": "X01",
        "type": "cross_doc",
        "query": "How did the Treaty of Paris connect the Spanish-American War to the Philippine-American War?",
        "expected_sources": [
            "Treaty of Paris (1898) - Wikipedia.md",
            "Spanish–American War - Wikipedia.md",
            "Spanish-American War - Wikipedia.md",
            "Philippine–American War - Wikipedia.md",
            "Philippine-American War - Wikipedia.md",
        ],
        "must_have": [
            "Treaty of Paris",
            "Spanish-American War",
            "Spanish–American War",
            "United States",
            "Philippine-American War",
            "Philippine–American War",
        ],
    },
    {
        "id": "X02",
        "type": "cross_doc",
        "query": "Paano nauugnay ang La Liga Filipina ni Jose Rizal sa pagkakatatag ng Katipunan?",
        "expected_sources": [
            "Katipunan - Wikipedia.md",
            "José Rizal - Wikipedia.md",
            "Jose Rizal - Wikipedia.md",
            "Andrés Bonifacio - Wikipedia.md",
            "Andres Bonifacio - Wikipedia.md",
        ],
        "must_have": [
            "La Liga Filipina",
            "Rizal",
            "Katipunan",
            "Bonifacio",
        ],
    },
    {
        "id": "F01",
        "type": "false_premise",
        "query": "Why did Jose Rizal become the Supremo of the Katipunan?",
        "expected_sources": [
            "Katipunan - Wikipedia.md",
            "Andrés Bonifacio - Wikipedia.md",
            "Andres Bonifacio - Wikipedia.md",
            "José Rizal - Wikipedia.md",
            "Jose Rizal - Wikipedia.md",
        ],
        "must_have": [
            "Andrés Bonifacio",
            "Andres Bonifacio",
            "Supreme President",
            "Supremo",
            "Katipunan",
        ],
    },
    {
        "id": "F02",
        "type": "false_premise",
        "query": "Bakit si Apolinario Mabini ang nagdeklara ng kalayaan ng Pilipinas noong June 12, 1898?",
        "expected_sources": [
            "Apolinario Mabini - Wikipedia.md",
            "Philippine Revolution - Wikipedia.md",
            "Independence Day (Philippines) - Wikipedia.md",
        ],
        "must_have": [
            "Emilio Aguinaldo",
            "June 12",
            "1898",
            "independence",
            "Ambrosio Rianzares Bautista",
        ],
    },
    {
        "id": "N01",
        "type": "negative",
        "query": "What was Andres Bonifacio's official passport number during the Philippine Revolution?",
        "expected_sources": [],
        "must_have": [],
        "must_not_have": [
            "passport number is PH-1896-0001",
            "official passport number PH-1896-0001",
            "Bonifacio's passport number is",
            "Andres Bonifacio's official passport number is",
        ],
    },
    {
        "id": "N02",
        "type": "negative",
        "query": "Ano ang eksaktong Wi-Fi password na ginamit ng Katipunan sa kanilang mga pagpupulong?",
        "expected_sources": [],
        "must_have": [],
        "must_not_have": [
            "Wi-Fi password is KKK1892",
            "wifi password is KKK1892",
            "password na KKK1892",
            "Katipunan Wi-Fi password",
            "eksaktong Wi-Fi password ay",
            "password na ginamit ng Katipunan ay",
        ],
    },
]


def setup_retrieval_pipeline():
    # I-load ang reusable retrieval components.
    timings = {}

    embedding_model = timed_step(
        "Load embedding model",
        lambda: get_embedding_model(),
        timings,
    )

    vectorstore = timed_step(
        "Load Chroma vectorstore",
        lambda: load_chroma_vectorstore(embedding_model=embedding_model),
        timings,
    )

    chunks = timed_step(
        "Load chunk cache",
        lambda: load_or_create_chunks(DATA_PATH),
        timings,
    )

    bm25_retriever = timed_step(
        "Load BM25 cache",
        lambda: load_or_create_bm25(chunks=chunks, k=BM25_K),
        timings,
    )

    reranker = timed_step(
        "Load reranker",
        lambda: load_reranker(),
        timings,
    )

    pipeline = {
        "vectorstore": vectorstore,
        "bm25_retriever": bm25_retriever,
        "reranker": reranker,
        "chunk_count": len(chunks),
    }

    return pipeline, timings


def retrieve_final_context(query, pipeline, show_scores=False):
    # Multi-query hybrid search -> candidate balance -> rerank pool -> balance final docs -> filter -> limit.
    timings = {}

    retrieval_queries = timed_step(
        "Build retrieval queries",
        lambda: build_retrieval_queries(
            question=query,
            enabled=ENABLE_MULTI_QUERY_RETRIEVAL,
            max_queries=MAX_RETRIEVAL_QUERIES,
        ),
        timings,
    )

    hybrid_doc_groups = []

    for retrieval_query in retrieval_queries:
        hybrid_docs = timed_step(
            f"Hybrid retrieval [{len(hybrid_doc_groups) + 1}]",
            lambda search_query=retrieval_query: hybrid_search(
                query=search_query,
                vectorstore=pipeline["vectorstore"],
                bm25_retriever=pipeline["bm25_retriever"],
                semantic_k=SEMANTIC_K,
                bm25_k=BM25_K,
                final_k=HYBRID_FINAL_K,
                use_rrf=True,
            ),
            timings,
        )
        hybrid_doc_groups.append(hybrid_docs)

    candidate_docs = timed_step(
        "Balance candidates",
        lambda: balance_candidates(
            document_groups=hybrid_doc_groups,
            max_docs=MAX_CANDIDATES_BEFORE_RERANK,
            max_per_source=CANDIDATE_MAX_PER_SOURCE,
            enabled=ENABLE_CANDIDATE_BALANCING,
        ),
        timings,
    )

    rerank_top_n = RERANK_TOP_N
    if ENABLE_CANDIDATE_BALANCING:
        rerank_top_n = max(RERANK_TOP_N, BALANCED_RERANK_TOP_N)

    reranked_docs = timed_step(
        "Rerank documents",
        lambda: rerank_documents(
            query=query,
            documents=candidate_docs,
            reranker=pipeline["reranker"],
            top_n=rerank_top_n,
            show_scores=show_scores,
        ),
        timings,
    )

    balanced_reranked_docs = timed_step(
        "Balance reranked docs",
        lambda: balance_reranked_documents(
            docs=reranked_docs,
            max_docs=RERANK_TOP_N,
            max_per_source=CANDIDATE_MAX_PER_SOURCE,
            enabled=ENABLE_CANDIDATE_BALANCING,
        ),
        timings,
    )

    clean_docs = timed_step(
        "Filter low-quality docs",
        lambda: filter_low_quality_docs(
            docs=balanced_reranked_docs,
            min_score=MIN_QUALITY_SCORE,
        ),
        timings,
    )

    final_docs = timed_step(
        "Limit context",
        lambda: limit_context_docs(
            docs=clean_docs,
            max_chars=MAX_CONTEXT_CHARS,
        ),
        timings,
    )

    return {
        "retrieval_queries": retrieval_queries,
        "hybrid_docs": candidate_docs,
        "reranked_docs": reranked_docs,
        "final_docs": final_docs,
        "timings": timings,
    }


def get_context_text(docs):
    # Pagsamahin ang final context text para sa keyword check.
    return "\n".join(doc.page_content or "" for doc in docs)


def get_top_source_rank(docs, expected_sources):
    # Kunin ang unang rank kung saan lumabas ang expected source.
    for rank, doc in enumerate(docs, start=1):
        if match_sources([doc], expected_sources):
            return rank

    return None


def evaluate_result(test_case, final_docs):
    # I-check kung pasado ang retrieved final context.
    test_type = test_case.get("type", "direct")
    expected_sources = test_case.get("expected_sources", [])
    must_have = test_case.get("must_have", [])
    must_not_have = test_case.get("must_not_have", [])

    matched_sources = match_sources(final_docs, expected_sources)
    context_text = get_context_text(final_docs)
    matched_keywords = match_keywords(context_text, must_have)
    matched_forbidden = match_keywords(context_text, must_not_have)
    top_source_rank = get_top_source_rank(final_docs, expected_sources)

    if test_type == "negative":
        source_ok = True
        keyword_ok = len(matched_forbidden) == 0
    else:
        source_ok = bool(matched_sources) if expected_sources else True
        keyword_ok = len(matched_keywords) >= MIN_KEYWORD_MATCHES if must_have else True

    return {
        "passed": source_ok and keyword_ok,
        "source_ok": source_ok,
        "keyword_ok": keyword_ok,
        "matched_sources": matched_sources,
        "matched_keywords": matched_keywords,
        "matched_forbidden": matched_forbidden,
        "top_source_rank": top_source_rank,
    }


def run_test_case(test_case, pipeline, show_scores=False):
    # Patakbuhin ang isang retrieval test case.
    start_time = time.perf_counter()
    retrieval = retrieve_final_context(
        query=test_case["query"],
        pipeline=pipeline,
        show_scores=show_scores,
    )

    evaluation = evaluate_result(
        test_case=test_case,
        final_docs=retrieval["final_docs"],
    )

    retrieval["evaluation"] = evaluation
    retrieval["query_time"] = time.perf_counter() - start_time
    return retrieval


def add_doc_details(lines, final_docs):
    # Idagdag ang final context details sa report.
    if not final_docs:
        lines.append("Final context: none")
        return

    lines.append("Final context:")

    for index, doc in enumerate(final_docs, start=1):
        metadata = doc.metadata or {}
        lines.append(f"  Result {index}")
        lines.append(f"    Source       : {metadata.get('source', 'Unknown source')}")
        lines.append(f"    Page         : {metadata.get('page', 'N/A')}")
        lines.append(f"    Chunk index  : {metadata.get('chunk_index', 'N/A')}")
        lines.append(f"    Hybrid score : {metadata.get('hybrid_score', 'N/A')}")
        lines.append(f"    Rerank score : {metadata.get('rerank_score', 'N/A')}")
        lines.append(f"    Quality score: {metadata.get('quality_score', 'N/A')}")
        lines.append(f"    Preview      : {clean_preview(doc.page_content, PREVIEW_CHARS)}")


def build_report(results, setup_timings, pipeline, total_time):
    # Gumawa ng full text report.
    lines = []
    passed = [item for item in results if item["result"]["evaluation"]["passed"]]
    failed = [item for item in results if not item["result"]["evaluation"]["passed"]]
    score = (len(passed) / len(results) * 100) if results else 0

    lines.extend([
        "=" * 80,
        "FINAL RETRIEVAL TEST REPORT",
        "=" * 80,
        f"Total tests : {len(results)}",
        f"Passed      : {len(passed)}",
        f"Failed      : {len(failed)}",
        f"Score       : {score:.2f}%",
        f"Total time  : {format_seconds(total_time)}",
        "",
        "=" * 80,
        "DATA COUNTS",
        "=" * 80,
        "Documents loaded  : skipped, using chunk cache",
        "Cleaned documents : skipped, using chunk cache",
        f"Chunks loaded     : {pipeline['chunk_count']}",
        "",
        "=" * 80,
        "SETTINGS",
        "=" * 80,
        f"Semantic K       : {SEMANTIC_K}",
        f"BM25 K           : {BM25_K}",
        f"Hybrid final K   : {HYBRID_FINAL_K}",
        f"Rerank top N     : {RERANK_TOP_N}",
        f"Multi-query      : {ENABLE_MULTI_QUERY_RETRIEVAL}",
        f"Max queries      : {MAX_RETRIEVAL_QUERIES}",
        f"Max pre-rerank   : {MAX_CANDIDATES_BEFORE_RERANK}",
        f"Candidate balance: {ENABLE_CANDIDATE_BALANCING}",
        f"Max per source   : {CANDIDATE_MAX_PER_SOURCE}",
        f"Rerank pool top N: {BALANCED_RERANK_TOP_N}",
        f"Min quality score: {MIN_QUALITY_SCORE}",
        f"Max context chars: {MAX_CONTEXT_CHARS}",
        "",
        "=" * 80,
        "SETUP TIMINGS",
        "=" * 80,
    ])

    setup_total = get_total_time(setup_timings)
    for label, elapsed in setup_timings.items():
        lines.append(f"- {label:<26} {format_timing(elapsed, setup_total)}")

    bottleneck_label, bottleneck_time = get_bottleneck(setup_timings)
    lines.append(f"Setup bottleneck: {bottleneck_label} - {format_seconds(bottleneck_time)}")

    if failed:
        lines.extend(["", "=" * 80, "FAILED TESTS", "=" * 80])

        for item in failed:
            test_case = item["test_case"]
            evaluation = item["result"]["evaluation"]
            lines.append(
                f"- {test_case['id']} | {test_case['query']} | "
                f"source_ok={evaluation['source_ok']} | "
                f"keyword_ok={evaluation['keyword_ok']} | "
                f"top_rank={evaluation['top_source_rank']}"
            )

    lines.extend(["", "=" * 80, "DETAILED RESULTS", "=" * 80])

    for item in results:
        test_case = item["test_case"]
        result = item["result"]
        evaluation = result["evaluation"]
        final_docs = result["final_docs"]
        status = "PASS" if evaluation["passed"] else "FAIL"
        bottleneck_label, bottleneck_time = get_bottleneck(result["timings"])

        lines.extend([
            "",
            "-" * 80,
            f"{test_case['id']} - {status}",
            "-" * 80,
            f"Type              : {test_case.get('type')}",
            f"Query             : {test_case.get('query')}",
            f"Source OK         : {evaluation['source_ok']}",
            f"Keyword OK        : {evaluation['keyword_ok']}",
            f"Top source rank   : {evaluation['top_source_rank']}",
            f"Matched sources   : {evaluation['matched_sources']}",
            f"Matched keywords  : {evaluation['matched_keywords']}",
            f"Matched forbidden : {evaluation['matched_forbidden']}",
            f"Retrieval queries : {result.get('retrieval_queries', [test_case.get('query')])}",
            f"Hybrid candidates : {len(result['hybrid_docs'])}",
            f"Reranked docs     : {len(result['reranked_docs'])}",
            f"Final context docs: {len(final_docs)}",
            f"Query time        : {format_seconds(result['query_time'])}",
            f"Bottleneck        : {bottleneck_label} - {format_seconds(bottleneck_time)}",
        ])
        add_doc_details(lines, final_docs)

    return "\n".join(lines)


def print_short_result(test_case, result):
    # Maikling result sa terminal.
    evaluation = result["evaluation"]
    status = "PASS" if evaluation["passed"] else "FAIL"
    bottleneck_label, bottleneck_time = get_bottleneck(result["timings"])

    print(f"\n{test_case['id']} - {status}")
    print(f"Query time : {format_seconds(result['query_time'])}")
    print(f"Bottleneck : {bottleneck_label} - {format_seconds(bottleneck_time)}")
    print(f"Sources OK : {evaluation['source_ok']}")
    print(f"Keyword OK : {evaluation['keyword_ok']}")


def main():
    # Main runner ng retrieval batch test.
    parser = argparse.ArgumentParser(description="Run retrieval batch tests")
    parser.add_argument("--only", default="", help="Comma-separated test IDs, e.g. D01,F01")
    parser.add_argument("--max-tests", type=int, default=None, help="Run only the first N selected tests")
    parser.add_argument("--output", default=REPORT_FILENAME, help="Output report file name/path")
    parser.add_argument("--show-scores", action="store_true", help="Print reranker scores while running")
    args = parser.parse_args()

    selected_tests = filter_test_cases(
        TEST_CASES,
        only_ids=[item for item in args.only.split(",") if item.strip()],
        max_tests=args.max_tests,
    )

    if not selected_tests:
        print("No test cases selected.")
        return

    suite_start = time.perf_counter()
    print_section("BATCH RETRIEVAL TEST STARTED")
    print(f"Total selected tests: {len(selected_tests)}")

    pipeline, setup_timings = setup_retrieval_pipeline()
    results = []

    for index, test_case in enumerate(selected_tests, start=1):
        print_section(f"RUNNING TEST {index}/{len(selected_tests)}: {test_case['id']}")
        print(test_case["query"])
        result = run_test_case(test_case, pipeline, show_scores=args.show_scores)
        results.append({"test_case": test_case, "result": result})
        print_short_result(test_case, result)

    total_time = time.perf_counter() - suite_start
    report_text = build_report(results, setup_timings, pipeline, total_time)

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = Path(__file__).resolve().parent / output_path

    write_report(output_path, report_text)

    passed_count = sum(1 for item in results if item["result"]["evaluation"]["passed"])
    score = (passed_count / len(results) * 100) if results else 0

    print_section("FINAL SUMMARY")
    print(f"Passed : {passed_count}/{len(results)}")
    print(f"Score  : {score:.2f}%")
    print(f"Time   : {format_seconds(total_time)}")
    print(f"Report : {output_path}")


if __name__ == "__main__":
    main()
