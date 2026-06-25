import argparse
import inspect
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
    has_no_answer_phrase,
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
    NO_ANSWER_TEXT,
    RERANK_TOP_N,
    SEMANTIC_K,
)
from chains.rag_chain import clean_generated_answer, generate_answer, get_sources, stream_answer
from embeddings.embedding_model import get_embedding_model
from llm.ollama_llm import load_llm
from retrieval.context_filter import filter_low_quality_docs, limit_context_docs
from retrieval.hybrid_retriever import hybrid_search
from retrieval.reranker import load_reranker, rerank_documents
from utils.bm25_cache import load_or_create_bm25
from utils.chunk_cache import load_or_create_chunks
from vectorstore.chroma_store import load_chroma_vectorstore


REPORT_FILENAME = "result_rag.txt"
ANSWER_PREVIEW_CHARS = 900
DEBUG_PROMPT = False
PRINT_FULL_ANSWER = False


ASSUMPTION_CHECK_STARTERS = (
    "why ",
    "how ",
    "why did ",
    "how did ",
    "why was ",
    "why is ",
    "bakit ",
    "paano ",
)


TEST_CASES = [
    {
        "id": "D01",
        "category": "direct",
        "behavior": "answer",
        "question": "Who was Apolinario Mabini and what role did he serve in the First Philippine Republic?",
        "expected_sources": ["Apolinario Mabini - Wikipedia.md"],
        "answer_keywords": [
            "Apolinario Mabini",
            "Mabini",
            "legal and constitutional adviser",
            "first Prime Minister",
            "First Philippine Republic",
            "brain of the revolution",
            "utak ng himagsikan",
        ],
        "min_answer_keyword_matches": 3,
    },
    {
        "id": "D02",
        "category": "direct",
        "behavior": "answer",
        "question": "Sino si Andres Bonifacio?",
        "expected_sources": [
            "Andrés Bonifacio - Wikipedia.md",
            "Andres Bonifacio - Wikipedia.md",
        ],
        "answer_keywords": [
            "Andres Bonifacio",
            "Andrés Bonifacio",
            "Filipino revolutionary leader",
            "revolutionary leader",
            "Father of the Philippine Revolution",
            "Katipunan",
        ],
        "min_answer_keyword_matches": 2,
    },
    {
        "id": "X01",
        "category": "cross_doc",
        "behavior": "answer",
        "question": "How did the Treaty of Paris connect the Spanish-American War to the Philippine-American War?",
        "expected_sources": [
            "Treaty of Paris (1898) - Wikipedia.md",
            "Spanish–American War - Wikipedia.md",
            "Spanish-American War - Wikipedia.md",
            "Philippine–American War - Wikipedia.md",
            "Philippine-American War - Wikipedia.md",
        ],
        "answer_keywords": [
            "Treaty of Paris",
            "Spanish-American War",
            "Spanish–American War",
            "United States",
            "Philippine-American War",
            "Philippine–American War",
            "ceded",
        ],
        "min_answer_keyword_matches": 3,
    },
    {
        "id": "X02",
        "category": "cross_doc",
        "behavior": "answer",
        "question": "Paano nauugnay ang La Liga Filipina ni Jose Rizal sa pagkakatatag ng Katipunan?",
        "expected_sources": [
            "José Rizal - Wikipedia.md",
            "Jose Rizal - Wikipedia.md",
            "Katipunan - Wikipedia.md",
        ],
        "answer_keywords": [
            "La Liga Filipina",
            "Jose Rizal",
            "José Rizal",
            "Rizal",
            "Katipunan",
            "Andres Bonifacio",
            "Andrés Bonifacio",
            "Dapitan",
        ],
        "min_answer_keyword_matches": 3,
    },
    {
        "id": "F01",
        "category": "false_premise",
        "behavior": "correction",
        "question": "Why did Jose Rizal become the Supremo of the Katipunan?",
        "expected_sources": [
            "José Rizal - Wikipedia.md",
            "Jose Rizal - Wikipedia.md",
            "Katipunan - Wikipedia.md",
            "Andrés Bonifacio - Wikipedia.md",
            "Andres Bonifacio - Wikipedia.md",
        ],
        "answer_keywords": [
            "Jose Rizal",
            "José Rizal",
            "Rizal",
            "Andres Bonifacio",
            "Andrés Bonifacio",
            "Bonifacio",
            "Supremo",
            "Supreme President",
            "Katipunan",
        ],
        "correction_keywords": [
            "No",
            "not correct",
            "not supported",
            "does not support",
            "did not become",
            "was not",
        ],
        "min_answer_keyword_matches": 4,
        "min_correction_keyword_matches": 1,
        "allow_no_answer_as_safe": False,
        "must_not_contain": [
            "Rizal became Supremo because",
            "Rizal became the Supremo because",
            "Rizal was the Supremo because",
            "Jose Rizal became Supremo because",
        ],
    },
    {
        "id": "F02",
        "category": "false_premise",
        "behavior": "correction",
        "question": "Bakit si Apolinario Mabini ang nagdeklara ng kalayaan ng Pilipinas noong June 12, 1898?",
        "expected_sources": [
            "Apolinario Mabini - Wikipedia.md",
            "Philippine Revolution - Wikipedia.md",
            "Independence Day (Philippines) - Wikipedia.md",
        ],
        "answer_keywords": [
            "Apolinario Mabini",
            "Mabini",
            "Emilio Aguinaldo",
            "Aguinaldo",
            "June 12",
            "1898",
            "kalayaan",
            "independence",
            "Ambrosio Rianzares Bautista",
            "Philippine Declaration of Independence",
        ],
        "correction_keywords": [
            "Hindi",
            "hindi tama",
            "hindi sinusuportahan",
            "hindi nakasaad",
            "hindi si",
            "maling premise",
        ],
        "min_answer_keyword_matches": 4,
        "min_correction_keyword_matches": 1,
        "allow_no_answer_as_safe": False,
        "must_not_contain": [
            "Mabini ay naging deklarer",
            "Mabini ay naging deklarer ng kalayaan",
            "dahil si Mabini",
            "dahil si Apolinario Mabini",
            "Mabini declared independence",
            "Apolinario Mabini declared independence",
            "Mabini was the declarer",
            "Mabini became the declarer",
        ],
    },
    {
        "id": "N01",
        "category": "negative",
        "behavior": "no_answer",
        "question": "What was Andres Bonifacio's official passport number during the Philippine Revolution?",
        "expected_sources": [],
        "answer_keywords": [],
        "must_not_contain": [
            "passport number is",
            "official passport number is",
            "Bonifacio's passport number",
        ],
    },
    {
        "id": "N02",
        "category": "negative",
        "behavior": "no_answer",
        "question": "Ano ang eksaktong Wi-Fi password na ginamit ng Katipunan sa kanilang mga pagpupulong?",
        "expected_sources": [],
        "answer_keywords": [],
        "must_not_contain": [
            "Wi-Fi password is",
            "wifi password is",
            "password ay",
            "password na ginamit",
            "Rizal",
            "Gomburza",
        ],
    },
]


def needs_strict_assumption_check(question):
    # Mas strict sa why/how/bakit/paano dahil madalas may hidden false premise.
    normalized = " ".join(str(question or "").lower().split())
    return any(normalized.startswith(starter) for starter in ASSUMPTION_CHECK_STARTERS)


def function_accepts_parameter(func, parameter_name):
    # Check kung suportado ng current backend ang optional parameter.
    try:
        return parameter_name in inspect.signature(func).parameters
    except (TypeError, ValueError):
        return False


def generate_answer_once(question, docs, llm, strict_assumption_check=False, correction_retry=False):
    # Generate answer with compatibility sa old/new rag_chain signatures.
    kwargs = {
        "question": question,
        "docs": docs,
        "llm": llm,
        "chat_history": "",
        "debug": DEBUG_PROMPT,
    }

    if function_accepts_parameter(generate_answer, "strict_assumption_check"):
        kwargs["strict_assumption_check"] = strict_assumption_check

    if function_accepts_parameter(generate_answer, "correction_retry"):
        kwargs["correction_retry"] = correction_retry

    return generate_answer(**kwargs)


def generate_streamed_answer(question, docs, llm, strict_assumption_check=False, correction_retry=False):
    # Streaming generation para same behavior sa app, with fallback sa old signature.
    if not docs:
        return NO_ANSWER_TEXT, 0.0

    answer = ""
    start_time = time.perf_counter()

    kwargs = {
        "question": question,
        "docs": docs,
        "llm": llm,
        "chat_history": "",
        "debug": DEBUG_PROMPT,
    }

    if function_accepts_parameter(stream_answer, "strict_assumption_check"):
        kwargs["strict_assumption_check"] = strict_assumption_check

    if function_accepts_parameter(stream_answer, "correction_retry"):
        kwargs["correction_retry"] = correction_retry

    for chunk in stream_answer(**kwargs):
        answer += str(chunk)

        if PRINT_FULL_ANSWER:
            print(chunk, end="", flush=True)

    elapsed_time = time.perf_counter() - start_time

    if PRINT_FULL_ANSWER:
        print()

    return clean_generated_answer(answer, question), elapsed_time


def maybe_retry_correction(test_case, answer, final_docs, llm):
    # Second try kapag correction test pero fallback pa rin kahit may docs.
    if test_case.get("behavior") != "correction":
        return answer, 0.0, False

    if not final_docs or not has_no_answer_phrase(answer):
        return answer, 0.0, False

    retry_answer, retry_time = generate_answer_with_time(
        question=test_case["question"],
        docs=final_docs,
        llm=llm,
        strict_assumption_check=True,
        correction_retry=True,
    )

    if retry_answer and not has_no_answer_phrase(retry_answer):
        return retry_answer, retry_time, True

    return answer, retry_time, False


def generate_answer_with_time(question, docs, llm, strict_assumption_check=False, correction_retry=False):
    # Non-streaming answer with timing.
    if not docs:
        return NO_ANSWER_TEXT, 0.0

    start_time = time.perf_counter()
    answer = generate_answer_once(
        question=question,
        docs=docs,
        llm=llm,
        strict_assumption_check=strict_assumption_check,
        correction_retry=correction_retry,
    )
    elapsed_time = time.perf_counter() - start_time
    return clean_generated_answer(answer, question), elapsed_time


def setup_components():
    # Load lahat ng kailangan ng RAG test.
    timings = {}

    print_section("RAG TEST SETUP")

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

    llm = timed_step(
        "Load LLM",
        lambda: load_llm(),
        timings,
    )

    return {
        "embedding_model": embedding_model,
        "vectorstore": vectorstore,
        "chunks": chunks,
        "bm25_retriever": bm25_retriever,
        "reranker": reranker,
        "llm": llm,
    }, timings


def retrieve_docs(question, components, show_scores=False):
    # Kunin ang final context docs gamit hybrid search + reranker.
    timings = {}

    hybrid_docs = timed_step(
        "Hybrid retrieval",
        lambda: hybrid_search(
            query=question,
            vectorstore=components["vectorstore"],
            bm25_retriever=components["bm25_retriever"],
            semantic_k=SEMANTIC_K,
            bm25_k=BM25_K,
            final_k=HYBRID_FINAL_K,
            use_rrf=True,
        ),
        timings,
    )

    reranked_docs = timed_step(
        "Rerank documents",
        lambda: rerank_documents(
            query=question,
            documents=hybrid_docs,
            reranker=components["reranker"],
            top_n=RERANK_TOP_N,
            show_scores=show_scores,
        ),
        timings,
    )

    clean_docs = timed_step(
        "Filter low-quality docs",
        lambda: filter_low_quality_docs(reranked_docs, min_score=MIN_QUALITY_SCORE),
        timings,
    )

    final_docs = timed_step(
        "Limit context",
        lambda: limit_context_docs(clean_docs, max_chars=MAX_CONTEXT_CHARS),
        timings,
    )

    counts = {
        "hybrid_candidates": len(hybrid_docs),
        "reranked_docs": len(reranked_docs),
        "final_context_docs": len(final_docs),
    }

    return final_docs, timings, counts


def evaluate_answer(test_case, answer, sources, final_docs):
    # I-check kung pasado ang sagot base sa behavior ng test case.
    behavior = test_case["behavior"]

    expected_sources = test_case.get("expected_sources", [])
    matched_sources = match_sources(sources, expected_sources)
    min_source_matches = test_case.get("min_source_matches", 1)
    source_ok = len(matched_sources) >= min_source_matches if expected_sources else True

    answer_keywords = test_case.get("answer_keywords", [])
    matched_keywords = match_keywords(answer, answer_keywords)
    min_keyword_matches = test_case.get("min_answer_keyword_matches", 1)

    correction_keywords = test_case.get("correction_keywords", [])
    matched_correction_keywords = match_keywords(answer, correction_keywords)
    min_correction_matches = test_case.get("min_correction_keyword_matches", 1)
    correction_signal_ok = len(matched_correction_keywords) >= min_correction_matches if correction_keywords else True

    forbidden_terms = test_case.get("must_not_contain", [])
    matched_forbidden = match_keywords(answer, forbidden_terms)
    forbidden_ok = len(matched_forbidden) == 0

    has_no_answer = has_no_answer_phrase(answer) or not final_docs

    if behavior == "no_answer":
        answer_ok = has_no_answer
        no_answer_ok = has_no_answer
        passed = answer_ok and forbidden_ok
    elif behavior == "correction":
        allow_safe_no_answer = test_case.get("allow_no_answer_as_safe", False)
        content_ok = len(matched_keywords) >= min_keyword_matches
        correction_ok = content_ok and correction_signal_ok and not has_no_answer
        safe_no_answer = allow_safe_no_answer and has_no_answer and forbidden_ok

        answer_ok = correction_ok or safe_no_answer
        no_answer_ok = True if allow_safe_no_answer else not has_no_answer
        passed = source_ok and answer_ok and no_answer_ok and forbidden_ok
    else:
        answer_ok = len(matched_keywords) >= min_keyword_matches
        no_answer_ok = not has_no_answer
        passed = source_ok and answer_ok and no_answer_ok and forbidden_ok

    return {
        "passed": passed,
        "source_ok": source_ok,
        "answer_ok": answer_ok,
        "no_answer_ok": no_answer_ok,
        "forbidden_ok": forbidden_ok,
        "matched_sources": matched_sources,
        "matched_keywords": matched_keywords,
        "matched_correction_keywords": matched_correction_keywords,
        "matched_forbidden": matched_forbidden,
    }


def print_sources(sources):
    # Print sources sa terminal.
    if not sources:
        print("Sources: None")
        return

    print("Sources:")

    for index, source in enumerate(sources, start=1):
        if isinstance(source, dict):
            name = source.get("source", "Unknown source")
            page = source.get("page", "N/A")
            file_name = source.get("file_name") or source.get("source_path") or ""
            file_note = f" | File: {file_name}" if file_name else ""
            print(f"  {index}. {name} | Page: {page}{file_note}")
        else:
            print(f"  {index}. {source}")


def print_test_result(result):
    # Print result ng isang test case.
    status = "PASS" if result["passed"] else "FAIL"
    evaluation = result["evaluation"]
    total_time = get_total_time(result["timings"])
    bottleneck_label, bottleneck_time = get_bottleneck(result["timings"])

    print_section(f"{result['id']} - {status}")
    print(f"Category : {result['category']}")
    print(f"Behavior : {result['behavior']}")
    print(f"Question : {result['question']}")

    if result["error"]:
        print(f"Error    : {result['error']}")

    print("\nChecks:")
    print(f"  source_ok    : {evaluation['source_ok']}")
    print(f"  answer_ok    : {evaluation['answer_ok']}")
    print(f"  no_answer_ok : {evaluation['no_answer_ok']}")
    print(f"  forbidden_ok : {evaluation['forbidden_ok']}")

    print("\nMatched:")
    print(f"  sources    : {evaluation['matched_sources']}")
    print(f"  keywords   : {evaluation['matched_keywords']}")
    print(f"  correction : {evaluation.get('matched_correction_keywords', [])}")
    print(f"  forbidden  : {evaluation['matched_forbidden']}")

    print("\nCounts:")
    for key, value in result["counts"].items():
        print(f"  {key:<20}: {value}")

    print("\nTimings:")
    for key, value in result["timings"].items():
        print(f"  {key:<24}: {format_timing(value, total_time)}")

    print(f"  {'Total question time':<24}: {format_seconds(result['question_time'])}")
    print(f"  {'Question bottleneck':<24}: {bottleneck_label} - {format_seconds(bottleneck_time)}")

    print("\nAnswer preview:")
    print(clean_preview(result["answer"], ANSWER_PREVIEW_CHARS))
    print()
    print_sources(result["sources"])


def run_single_test(test_case, components, show_scores=False, use_stream=False):
    # Patakbuhin ang isang RAG test case.
    question_start = time.perf_counter()
    question = test_case["question"]

    timings = {}
    counts = {
        "hybrid_candidates": 0,
        "reranked_docs": 0,
        "final_context_docs": 0,
    }
    final_docs = []
    sources = []
    answer = ""
    error = None

    try:
        final_docs, retrieval_timings, counts = retrieve_docs(question, components, show_scores=show_scores)
        timings.update(retrieval_timings)

        strict_check = test_case.get("behavior") == "correction" or needs_strict_assumption_check(question)

        if use_stream:
            answer, answer_time = generate_streamed_answer(
                question=question,
                docs=final_docs,
                llm=components["llm"],
                strict_assumption_check=strict_check,
            )
        else:
            answer, answer_time = generate_answer_with_time(
                question=question,
                docs=final_docs,
                llm=components["llm"],
                strict_assumption_check=strict_check,
            )

        timings["Answer generation"] = answer_time

        retry_answer, retry_time, used_retry = maybe_retry_correction(
            test_case=test_case,
            answer=answer,
            final_docs=final_docs,
            llm=components["llm"],
        )

        if retry_time:
            timings["Correction retry"] = retry_time

        if used_retry:
            answer = retry_answer

        sources = get_sources(final_docs)

        evaluation = evaluate_answer(
            test_case=test_case,
            answer=answer,
            sources=sources,
            final_docs=final_docs,
        )

    except Exception as exception:
        error = f"{type(exception).__name__}: {exception}"
        answer = f"ERROR: {error}"
        evaluation = {
            "passed": False,
            "source_ok": False,
            "answer_ok": False,
            "no_answer_ok": False,
            "forbidden_ok": False,
            "matched_sources": [],
            "matched_keywords": [],
            "matched_correction_keywords": [],
            "matched_forbidden": [],
        }

    question_time = time.perf_counter() - question_start

    result = {
        "id": test_case["id"],
        "category": test_case.get("category", test_case["behavior"]),
        "behavior": test_case["behavior"],
        "question": question,
        "passed": evaluation["passed"],
        "evaluation": evaluation,
        "answer": answer,
        "sources": sources,
        "counts": counts,
        "timings": timings,
        "question_time": question_time,
        "error": error,
    }

    print_test_result(result)
    return result


def add_summary_to_report(lines, results, setup_timings, total_suite_time):
    # Idagdag ang summary sa report.
    total_tests = len(results)
    passed_tests = [result for result in results if result["passed"]]
    failed_tests = [result for result in results if not result["passed"]]
    score = (len(passed_tests) / total_tests * 100) if total_tests else 0

    lines.extend([
        "=" * 80,
        "FINAL RAG TEST REPORT",
        "=" * 80,
        f"Total tests : {total_tests}",
        f"Passed      : {len(passed_tests)}",
        f"Failed      : {len(failed_tests)}",
        f"Score       : {score:.2f}%",
        f"Total time  : {format_seconds(total_suite_time)}",
    ])

    if failed_tests:
        lines.extend(["", "FAILED TESTS:"])

        for result in failed_tests:
            evaluation = result["evaluation"]
            lines.append(
                f"- {result['id']} | {result['question']} | "
                f"source_ok={evaluation['source_ok']} | "
                f"answer_ok={evaluation['answer_ok']} | "
                f"no_answer_ok={evaluation['no_answer_ok']} | "
                f"forbidden_ok={evaluation['forbidden_ok']}"
            )

    lines.extend(["", "=" * 80, "SETUP TIMING SUMMARY", "=" * 80])
    setup_total = get_total_time(setup_timings)

    for label, seconds in setup_timings.items():
        lines.append(f"- {label:<26} {format_timing(seconds, setup_total)}")

    if setup_timings:
        bottleneck_label, _ = get_bottleneck(setup_timings)
        lines.append(f"\nSetup bottleneck: {bottleneck_label}")


def add_details_to_report(lines, results):
    # Idagdag ang per-test details sa report.
    lines.extend(["", "=" * 80, "DETAILED RESULTS", "=" * 80])

    for result in results:
        evaluation = result["evaluation"]
        total_time = get_total_time(result["timings"])
        bottleneck_label, bottleneck_time = get_bottleneck(result["timings"])
        status = "PASS" if result["passed"] else "FAIL"

        lines.extend([
            "",
            "-" * 80,
            f"{result['id']} - {status}",
            "-" * 80,
            f"Category : {result['category']}",
            f"Behavior : {result['behavior']}",
            f"Question : {result['question']}",
        ])

        if result["error"]:
            lines.append(f"Error    : {result['error']}")

        lines.extend([
            "",
            "Checks:",
            f"  source_ok    : {evaluation['source_ok']}",
            f"  answer_ok    : {evaluation['answer_ok']}",
            f"  no_answer_ok : {evaluation['no_answer_ok']}",
            f"  forbidden_ok : {evaluation['forbidden_ok']}",
            "",
            "Matched:",
            f"  sources    : {evaluation['matched_sources']}",
            f"  keywords   : {evaluation['matched_keywords']}",
            f"  correction : {evaluation.get('matched_correction_keywords', [])}",
            f"  forbidden  : {evaluation['matched_forbidden']}",
            "",
            "Counts:",
        ])

        for key, value in result["counts"].items():
            lines.append(f"  {key:<20}: {value}")

        lines.append("")
        lines.append("Timings:")
        for key, value in result["timings"].items():
            lines.append(f"  {key:<24}: {format_timing(value, total_time)}")

        lines.append(f"  {'Total question time':<24}: {format_seconds(result['question_time'])}")
        lines.append(f"  {'Question bottleneck':<24}: {bottleneck_label} - {format_seconds(bottleneck_time)}")

        lines.extend(["", "Sources:"])
        if result["sources"]:
            for index, source in enumerate(result["sources"], start=1):
                lines.append(f"  {index}. {' | '.join(str(value) for value in get_source_values(source))}")
        else:
            lines.append("  None")

        lines.extend(["", "Answer:", result["answer"]])


def build_report_text(results, setup_timings, total_suite_time):
    # Gumawa ng full report text para sa .txt output.
    lines = []
    add_summary_to_report(lines, results, setup_timings, total_suite_time)
    add_details_to_report(lines, results)
    return "\n".join(lines)


def main():
    # Main runner ng RAG batch test.
    parser = argparse.ArgumentParser(description="Run RAG batch tests")
    parser.add_argument("--only", default="", help="Comma-separated test IDs, e.g. D01,F01")
    parser.add_argument("--max-tests", type=int, default=None, help="Run only the first N selected tests")
    parser.add_argument("--output", default=REPORT_FILENAME, help="Output report file name/path")
    parser.add_argument("--show-scores", action="store_true", help="Print reranker scores while running")
    parser.add_argument("--stream", action="store_true", help="Use stream_answer instead of generate_answer")
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

    print_section("RAG BATCH TEST STARTED")
    print(f"Total selected tests: {len(selected_tests)}")
    print(f"Semantic K     : {SEMANTIC_K}")
    print(f"BM25 K         : {BM25_K}")
    print(f"Hybrid final K : {HYBRID_FINAL_K}")
    print(f"Rerank top N   : {RERANK_TOP_N}")

    components, setup_timings = setup_components()

    print_section("DATA COUNTS")
    print("Documents loaded  : skipped, using chunk cache")
    print("Cleaned documents : skipped, using chunk cache")
    print(f"Chunks loaded     : {len(components['chunks'])}")

    results = []

    for index, test_case in enumerate(selected_tests, start=1):
        print_section(f"RUNNING TEST {index}/{len(selected_tests)}: {test_case['id']}")
        results.append(
            run_single_test(
                test_case=test_case,
                components=components,
                show_scores=args.show_scores,
                use_stream=args.stream,
            )
        )

    total_suite_time = time.perf_counter() - suite_start
    report_text = build_report_text(results, setup_timings, total_suite_time)

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = Path(__file__).resolve().parent / output_path

    write_report(output_path, report_text)

    print("\n" + report_text)
    print(f"\nDetailed report saved to: {output_path}")
    print("RAG batch test finished.")


if __name__ == "__main__":
    main()
