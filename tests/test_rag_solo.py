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


REPORT_FILE = Path("reports") / "test_rag_solo.txt"
PREVIEW_CHARS = 450
CONTEXT_PREVIEW_CHARS = 1200
FALLBACK_RETRY_SKIP_CATEGORIES = {"NEGATIVE_UNANSWERABLE", "FALSE_PREMISE"}
TERMINAL_PUNCTUATION = (".", "?", "!", ")", "]", "}", '"', "'")
CURRENT_CASE_TIMINGS = None


SCORE_KEYS = [
    "semantic_score",
    "semantic_distance",
    "semantic_rank",
    "bm25_score",
    "bm25_rank",
    "rrf_score",
    "hybrid_score",
    "metadata_boosted_score",
    "rerank_score",
    "quality_score",
]


DOC_LIST_KEYS_PRIORITY = [
    "semantic_docs",
    "bm25_docs",
    "hybrid_docs",
    "retrieved_docs",
    "reranked_docs",
    "filtered_docs",
    "final_docs",
    "final_context_docs",
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


def first_existing(metadata, keys, default=""):
    # Return first metadata value that exists, including 0.
    for key in keys:
        value = metadata.get(key)
        if value is not None and value != "":
            return value
    return default


def get_source_label(doc):
    # Human-readable source label.
    metadata = get_metadata(doc)
    source = first_existing(metadata, ["file_name", "source"], "Unknown source")
    page = first_existing(metadata, ["page", "page_number"], "N/A")
    chunk_id = first_existing(metadata, ["chunk_id", "chunk_index", "chunk"], "Unknown chunk")
    return f"{source} | page={page} | chunk_id={chunk_id}"


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
    # Lazy loader para one setup lang sa buong solo test run.
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


def run_retrieval(question, components, use_metadata_boost=False, use_reranker=True, debug=False):
    # Run the same shared retrieval path used by chains.chatbot.ask_rag and ask_rag_stream.
    loaded = components.get_all()
    retrieval_queries = build_test_retrieval_queries(question)

    retrieval_result = timed_step(
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
            debug=debug,
        ),
    )

    # Keep retrieval queries visible in the solo report even if run_retrieval_core does not return them.
    if isinstance(retrieval_result, dict) and "retrieval_queries" not in retrieval_result:
        retrieval_result["retrieval_queries"] = retrieval_queries

    return retrieval_result


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


def should_retry_answer(answer, retrieval_result, no_llm=False):
    # Retry lang kapag may context.
    if no_llm:
        return False

    if not retrieval_result.get("final_docs"):
        return False

    if ENABLE_FALLBACK_RETRY and is_fallback_answer(answer):
        return True

    if ENABLE_TRUNCATION_RETRY and looks_truncated_answer(answer):
        return True

    return False


def generate_rag_answer(question, retrieval_result, components, no_llm=False, debug=False):
    # Generate answer using the same shared answer function used by chains.chatbot.
    loaded = components.get_all()

    if no_llm:
        ensure_final_context_docs(
            question=question,
            retrieval_result=retrieval_result,
            debug=debug,
        )
        return "[NO LLM MODE] Answer generation skipped."

    def generate_once(correction_retry=False, completion_retry=False):
        return generate_chatbot_answer(
            question=question,
            retrieval_result=retrieval_result,
            llm=loaded["llm"],
            chat_history="",
            debug=debug,
            correction_retry=correction_retry,
            completion_retry=completion_retry,
        )

    answer = timed_step(
        "Generate answer",
        components.timings,
        lambda: generate_once(),
    )

    if should_retry_answer(answer, retrieval_result, no_llm=no_llm):
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


def looks_like_doc_list(value):
    # Detect lists of LangChain Document-like objects.
    if not isinstance(value, list) or not value:
        return False

    return hasattr(value[0], "page_content") and hasattr(value[0], "metadata")


def add_retrieval_queries_to_report(lines, retrieval_queries):
    # Add retrieval query variants used by the pipeline.
    add_section(lines, "RETRIEVAL QUERIES")

    if not retrieval_queries:
        lines.append("No retrieval queries found.")
        return

    for index, query in enumerate(retrieval_queries, start=1):
        lines.append(f"[{index}] {query}")


def add_docs_to_report(lines, title, docs, max_docs=20):
    # Add detailed retrieved chunk logs to report.
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

        for key in SCORE_KEYS:
            if key in metadata:
                lines.append(f"{key}: {metadata.get(key)}")

        lines.append(f"Preview : {get_preview(doc, PREVIEW_CHARS)}")
        lines.append("-" * 80)


def add_retrieval_doc_lists_to_report(lines, retrieval_result):
    # Add every document list available from retrieval_result, including final_docs.
    add_section(lines, "RETRIEVAL RESULT KEYS")

    if not isinstance(retrieval_result, dict):
        lines.append(f"Retrieval result is not a dictionary: {type(retrieval_result)}")
        return

    keys = list(retrieval_result.keys())
    lines.append(", ".join(keys) if keys else "No keys found.")

    logged_keys = set()

    for key in DOC_LIST_KEYS_PRIORITY:
        value = retrieval_result.get(key)
        if looks_like_doc_list(value):
            add_docs_to_report(lines, f"RETRIEVED CHUNKS LOG - {key}", value)
            logged_keys.add(key)

    for key, value in retrieval_result.items():
        if key in logged_keys:
            continue

        if looks_like_doc_list(value):
            add_docs_to_report(lines, f"RETRIEVED CHUNKS LOG - {key}", value)
            logged_keys.add(key)

    if not logged_keys:
        lines.append("No document lists were found inside retrieval_result.")


def add_final_context_to_report(lines, final_docs):
    # Add the exact final context chunk order sent to the answer generator.
    add_section(lines, "FINAL CONTEXT SENT TO LLM")

    if not final_docs:
        lines.append("No final context documents.")
        return

    for index, doc in enumerate(final_docs, start=1):
        metadata = get_metadata(doc)
        lines.append(f"--- Context {index} ---")
        lines.append(f"Source : {get_source_label(doc)}")
        lines.append(f"Title  : {metadata.get('title', '')}")
        lines.append(f"Section: {metadata.get('section', '')}")

        for key in SCORE_KEYS:
            if key in metadata:
                lines.append(f"{key}: {metadata.get(key)}")

        lines.append("Text:")
        lines.append(get_preview(doc, CONTEXT_PREVIEW_CHARS))
        lines.append("-" * 80)


def add_sources_to_report(lines, docs):
    # Source-only output para mabilis makita kung anong files/chunks ang ginamit.
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


def add_run_report(lines, run_number, question, retrieval_result, answer, elapsed, case_timings=None, no_llm=False):
    # Add one solo run result to report.
    final_docs = retrieval_result.get("final_docs", []) if isinstance(retrieval_result, dict) else []
    context_status = get_context_status(final_docs)
    answer_status = get_answer_status(answer, no_llm=no_llm)
    retrieval_queries = retrieval_result.get("retrieval_queries", []) if isinstance(retrieval_result, dict) else []
    bottleneck_name, bottleneck_time = get_bottleneck(case_timings)
    bottleneck_percent = (bottleneck_time / elapsed * 100.0) if elapsed else 0.0

    add_section(lines, f"SOLO RAG RUN {run_number}")
    lines.append(f"Question       : {question}")
    lines.append(f"Context status : {context_status}")
    lines.append(f"Answer status  : {answer_status}")
    lines.append(f"Sources        : {len(final_docs)}")
    lines.append(f"Retrieval queries: {len(retrieval_queries)}")
    lines.append(f"Run time       : {format_seconds(elapsed)}")
    lines.append(f"Bottleneck     : {bottleneck_name} - {format_seconds(bottleneck_time)} ({bottleneck_percent:.1f}%)")

    add_retrieval_queries_to_report(lines, retrieval_queries)
    add_retrieval_doc_lists_to_report(lines, retrieval_result)
    add_final_context_to_report(lines, final_docs)

    add_section(lines, "ANSWER")
    lines.append(str(answer or ""))

    add_sources_to_report(lines, final_docs)
    add_timing_breakdown(lines, "RUN TIMING BREAKDOWN", case_timings, elapsed=elapsed)


def add_summary(lines, run_results, timings):
    # Add final summary with overall bottleneck and per-run bottlenecks.
    add_section(lines, "SOLO RAG TEST SUMMARY")

    total_runs = len(run_results)
    context_count = sum(
        1 for item in run_results
        if get_context_status(item["retrieval_result"].get("final_docs", [])) == "HAS_CONTEXT"
    )
    answered_count = sum(
        1 for item in run_results
        if get_answer_status(item["answer"], no_llm=item["no_llm"]) == "ANSWERED"
    )
    fallback_count = sum(
        1 for item in run_results
        if get_answer_status(item["answer"], no_llm=item["no_llm"]) == "FALLBACK"
    )
    truncated_count = sum(
        1 for item in run_results
        if get_answer_status(item["answer"], no_llm=item["no_llm"]) == "POSSIBLY_TRUNCATED"
    )

    total_end_to_end = sum(item.get("elapsed", 0.0) for item in run_results)
    total_measured = get_timing_total(timings)
    bottleneck_name, bottleneck_time = get_bottleneck(timings)
    bottleneck_percent = (bottleneck_time / total_measured * 100.0) if total_measured else 0.0

    lines.append(f"Total runs        : {total_runs}")
    lines.append(f"Runs with context : {context_count}")
    lines.append(f"Answered runs     : {answered_count}")
    lines.append(f"Fallback runs     : {fallback_count}")
    lines.append(f"Possibly truncated: {truncated_count}")
    lines.append(f"End-to-end total  : {format_seconds(total_end_to_end)}")
    lines.append(f"Measured total    : {format_seconds(total_measured)}")
    lines.append(f"Overall bottleneck: {bottleneck_name} - {format_seconds(bottleneck_time)} ({bottleneck_percent:.1f}%)")
    lines.append("")
    lines.append("How to read this:")
    lines.append("- If retrieved chunks/order change between repeated runs, retrieval/rerank is changing.")
    lines.append("- If retrieved chunks/order stay the same but answers differ, LLM generation/prompt is the likely cause.")
    lines.append("- The FINAL CONTEXT SENT TO LLM section is the most important part to compare.")

    add_section(lines, "RUN STATUS SUMMARY")

    for item in run_results:
        context_status = get_context_status(item["retrieval_result"].get("final_docs", []))
        answer_status = get_answer_status(item["answer"], no_llm=item["no_llm"])
        case_timings = item.get("case_timings", {})
        case_bottleneck_name, case_bottleneck_time = get_bottleneck(case_timings)

        lines.append(
            f"{context_status:<11} | {answer_status:<18} | "
            f"{format_seconds(item['elapsed']):>14} | "
            f"{case_bottleneck_name} {format_seconds(case_bottleneck_time)} | "
            f"RUN {item['run_number']}"
        )

    add_timing_breakdown(lines, "OVERALL TIMING SUMMARY", timings, elapsed=total_end_to_end)


def get_question_from_args(args):
    # Use CLI question first; otherwise ask interactively in terminal.
    question = str(args.question or "").strip()

    if question:
        return question

    return input("Enter your RAG question: ").strip()


def main():
    global CURRENT_CASE_TIMINGS

    parser = argparse.ArgumentParser(description="Run one custom RAG question and save debug logs to reports/test_rag_solo.txt.")
    parser.add_argument("--question", default="", help="Custom question. If empty, the script asks for input.")
    parser.add_argument("--repeat", type=int, default=1, help="Repeat the same question N times to compare stability.")
    parser.add_argument("--no-llm", action="store_true", help="Only test final context, skip LLM answer generation.")
    parser.add_argument("--no-reranker", action="store_true", help="Skip reranker and use hybrid docs directly.")
    parser.add_argument("--metadata-boost", action="store_true", help="Enable metadata boost during hybrid retrieval.")
    parser.add_argument("--debug", action="store_true", help="Enable debug output inside shared chatbot functions.")
    args = parser.parse_args()

    question = get_question_from_args(args)

    if not question:
        raise ValueError("Question is required.")

    repeat_count = max(1, int(args.repeat or 1))
    lines = []
    run_results = []
    components = RagComponents()

    add_section(lines, "SOLO RAG TEST STARTED")
    lines.append(f"Data path       : {DATA_PATH}")
    lines.append(f"Report file     : {REPORT_FILE}")
    lines.append(f"Question        : {question}")
    lines.append(f"Repeat count    : {repeat_count}")
    lines.append(f"Use reranker    : {not args.no_reranker}")
    lines.append(f"Metadata boost  : {args.metadata_boost}")
    lines.append(f"No LLM mode     : {args.no_llm}")
    lines.append(f"Debug mode      : {args.debug}")
    lines.append("")
    lines.append("Evaluation mode : manual review")
    lines.append("Output purpose  : compare retrieved chunks, final context, and answer stability")
    lines.append("Pipeline source : chains.chatbot shared core")

    for run_number in range(1, repeat_count + 1):
        print("\n" + "=" * 80)
        print(f"SOLO RAG RUN {run_number}: {question}")
        print("=" * 80)

        CURRENT_CASE_TIMINGS = {}
        run_start = time.perf_counter()

        retrieval_result = run_retrieval(
            question=question,
            components=components,
            use_metadata_boost=args.metadata_boost,
            use_reranker=not args.no_reranker,
            debug=args.debug,
        )

        answer = generate_rag_answer(
            question=question,
            retrieval_result=retrieval_result,
            components=components,
            no_llm=args.no_llm,
            debug=args.debug,
        )

        elapsed = time.perf_counter() - run_start
        case_timings = dict(CURRENT_CASE_TIMINGS or {})
        CURRENT_CASE_TIMINGS = None

        run_results.append({
            "run_number": run_number,
            "retrieval_result": retrieval_result,
            "answer": answer,
            "elapsed": elapsed,
            "case_timings": case_timings,
            "no_llm": args.no_llm,
        })

        add_run_report(
            lines=lines,
            run_number=run_number,
            question=question,
            retrieval_result=retrieval_result,
            answer=answer,
            elapsed=elapsed,
            case_timings=case_timings,
            no_llm=args.no_llm,
        )

    add_summary(lines, run_results, components.timings)
    save_report(lines)

    print("\nDONE")
    print(f"Report saved to: {REPORT_FILE}")


if __name__ == "__main__":
    main()
