import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import (
    DATA_PATH,
    ENABLE_FALLBACK_RETRY,
    ENABLE_MULTI_QUERY_RETRIEVAL,
    ENABLE_TRUNCATION_RETRY,
    MAX_RETRIEVAL_QUERIES,
    NO_ANSWER_TEXT,
)
from chains.chatbot import (
    build_retrieval_queries,
    ensure_final_context_docs,
    generate_chatbot_answer,
    load_chatbot_components,
    run_retrieval_core,
)


REPORT_FILE = Path("reports") / "rag_generation_report.txt"
PREVIEW_CHARS = 900

FALLBACK_RETRY_SKIP_CATEGORIES = {"NEGATIVE_UNANSWERABLE", "FALSE_PREMISE", "FOLLOW_UP_AMBIGUOUS"}
TERMINAL_PUNCTUATION = (".", "?", "!", ")", "]", "}", '"', "'")
CURRENT_CASE_TIMINGS = None


# Question-only test cases.
# Walang expected_sources, answer_keywords, or min_keyword_matches.
# This test now uses the same shared retrieval and answer functions as chat_ui.py.
TEST_CASES = [
    # User-provided Philippine history RAG questions.
    # These cases use the same shared retrieval and answer functions as chat_ui.py.
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
        "category": "ATTRIBUTION_CHECK",
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
        "id": "R05B",
        "category": "FOLLOW_UP_AMBIGUOUS",
        "question": "What it did?",
    },
    {
        "id": "R06",
        "category": "RETRIEVAL_BASIC",
        "question": "What hardships did Filipinos experience during the Japanese occupation?",
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
        "id": "S02",
        "category": "SIMILAR_TOPIC",
        "question": "What is the difference between the Philippine Revolution and the Katipunan? Is one an organization and the other a war/revolution?",
    },
    {
        "id": "D02",
        "category": "DATE_FACT",
        "question": "When is rizal's birthday?",
    },

    # Extra regression cases kept from the original RAG generation test.
    {
        "id": "D01",
        "category": "DIRECT",
        "question": "Who was Apolinario Mabini and what role did he serve in the First Philippine Republic?",
    },
    {
        "id": "N01",
        "category": "NEGATIVE_UNANSWERABLE",
        "question": "What was Andres Bonifacio's official passport number during the Philippine Revolution?",
    },
    {
        "id": "S01",
        "category": "SIMILAR_TOPIC",
        "question": "Compare the Spanish-American War vs. the Philippine-American War. Do not interchange the cause and result of the two wars.",
    },
]


def format_seconds(seconds):
    # Convert seconds to readable text.
    if seconds < 60:
        return f"{seconds:.2f} sec"

    minutes = int(seconds // 60)
    remaining_seconds = seconds % 60
    return f"{minutes} min {remaining_seconds:.2f} sec"


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


class RagComponents:
    # Lazy loader para one setup lang sa buong test run.
    def __init__(self):
        self.timings = {}
        self.components = None

    def get_all(self):
        if self.components is not None:
            return self.components

        self.components = timed_step(
            "Load chatbot components",
            self.timings,
            load_chatbot_components,
        )
        return self.components


def build_test_retrieval_queries(question):
    # Use the same generic query builder as the UI.
    return build_retrieval_queries(
        question=question,
        rewritten_question=question,
        enabled=ENABLE_MULTI_QUERY_RETRIEVAL,
        max_queries=MAX_RETRIEVAL_QUERIES,
    )


def run_retrieval(question, components, use_metadata_boost=False, use_reranker=True, category=""):
    # Run the same shared retrieval path used by chains.chatbot.ask_rag and ask_rag_stream.
    loaded = components.get_all()
    retrieval_queries = build_test_retrieval_queries(question)

    return timed_step(
        "Shared retrieval core",
        components.timings,
        lambda: run_retrieval_core(
            question=question,
            retrieval_queries=retrieval_queries,
            vectorstore=loaded["vectorstore"],
            bm25_retriever=loaded["bm25_retriever"],
            reranker=loaded["reranker"],
            use_metadata_boost=use_metadata_boost,
            use_reranker=use_reranker,
            debug=False,
        ),
    )


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
    # Generate answer using the same shared answer function used by chains.chatbot.
    loaded = components.get_all()

    if no_llm:
        ensure_final_context_docs(
            question=question,
            retrieval_result=retrieval_result,
            debug=False,
        )
        return "[NO LLM MODE] Answer generation skipped."


    def generate_once(correction_retry=False, completion_retry=False):
        return generate_chatbot_answer(
            question=question,
            retrieval_result=retrieval_result,
            llm=loaded["llm"],
            chat_history="",
            debug=False,
            correction_retry=correction_retry,
            completion_retry=completion_retry,
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
    lines.append("- The test now calls the same shared retrieval and answer functions used by the UI.")
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
    lines.append("Pipeline source : chains.chatbot shared core")

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
