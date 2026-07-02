"""Question-only standalone RAG answer generation test for InknowVa.

Purpose:
- Test only raw user questions, with no categories, no expected keywords, and no source hints.
- Let the retrieval, context filter, prompt, and LLM detect intent by themselves.
- Use the same UI backend entry point: chains.chatbot.ask_rag_stream().
- Use chat_history="" so this is for standalone questions only, not follow-ups.

Run from the project root:
    python tests/test_rag_generation_question_only.py --limit 3
    python tests/test_rag_generation_question_only.py
    python tests/test_rag_generation_question_only.py --question "When is rizal's birthday?"

Outputs:
    reports/rag_generation_question_only_report.txt
    reports/rag_generation_question_only_report.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

REPORT_DIR = PROJECT_ROOT / "reports"
REPORT_TXT = REPORT_DIR / "rag_generation_question_only_report.txt"
REPORT_JSONL = REPORT_DIR / "rag_generation_question_only_report.jsonl"
PREVIEW_CHARS = 900
TERMINAL_PUNCTUATION = (".", "?", "!", ")", "]", "}", '"', "'")

# Raw questions only. No category, no expected answer, no source hints.
QUESTIONS: List[str] = [
    "Who are the ladies that had relationship with Jose Rizal?",
    "Who killed Ferdinand Magellan?",
    "Did Lapu-Lapu kill Magellan?",
    "Who is the first Philippine president?",
    "What did the Treaty of Paris of 1898 say about the Philippines?",
    "What hardships did Filipinos experience during the Japanese occupation?",
    "Who founded the Katipunan or KKK on July 7, 1892, and what was its purpose against Spain?",
    "Which secret group tried to free Filipinos from Spanish rule through armed revolution before it was discovered in 1896?",
    "Kailan ipinagdiriwang ang Araw ng Kalayaan ng Pilipinas at anong pangyayari ang ginugunita nito?",
    "How did the Treaty of Paris connect the Spanish-American War to the Philippine-American War?",
    "Why did Jose Rizal become the Supremo of the Katipunan?",
    "What is the difference between the Philippine Revolution and the Katipunan? Is one an organization and the other a war/revolution?",
    "When is rizal's birthday?",
]


def format_seconds(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.2f} sec"

    minutes = int(seconds // 60)
    remaining = seconds % 60
    return f"{minutes} min {remaining:.2f} sec"


def looks_truncated_answer(answer: str) -> bool:
    text = " ".join(str(answer or "").split()).strip()

    if not text:
        return False

    if len(text.split()) < 6:
        return False

    return not text.endswith(TERMINAL_PUNCTUATION)


def get_answer_status(answer: str, error: str = "") -> str:
    if error:
        return "ERROR"

    text = str(answer or "").strip()

    if not text:
        return "EMPTY_ANSWER"

    if looks_truncated_answer(text):
        return "POSSIBLY_TRUNCATED"

    return "ANSWERED"


def source_label(doc: Any) -> str:
    metadata = dict(getattr(doc, "metadata", {}) or {})
    source = metadata.get("file_name") or metadata.get("source") or "Unknown source"
    page = metadata.get("page", "N/A")
    chunk_id = metadata.get("chunk_id") or metadata.get("chunk_index") or "N/A"
    section = metadata.get("section", "")
    return f"{source} | page={page} | chunk={chunk_id} | section={section}"


def doc_preview(doc: Any, limit: int = PREVIEW_CHARS) -> str:
    metadata = dict(getattr(doc, "metadata", {}) or {})
    evidence = str(metadata.get("evidence_snippet") or "").strip()
    text = evidence or str(getattr(doc, "page_content", "") or "")
    text = " ".join(text.split())

    if len(text) <= limit:
        return text

    return text[:limit].rstrip() + "..."


def select_questions(args: argparse.Namespace) -> List[Dict[str, Any]]:
    if args.question:
        return [{"id": "CUSTOM", "question": args.question.strip()}]

    selected = [
        {"id": f"Q{index:02d}", "question": question}
        for index, question in enumerate(QUESTIONS, start=1)
    ]

    if args.only:
        wanted = {value.strip().upper() for value in args.only}
        selected = [item for item in selected if item["id"].upper() in wanted]

    if args.limit and args.limit > 0:
        selected = selected[: args.limit]

    return selected


def run_one_question(chatbot: Any, components: Dict[str, Any], item: Dict[str, Any], debug: bool) -> Dict[str, Any]:
    question = item["question"]
    started = time.perf_counter()
    answer_parts: List[str] = []
    done_payload: Dict[str, Any] = {}
    events_seen: List[str] = []
    error = ""

    try:
        for event in chatbot.ask_rag_stream(
            question=question,
            vectorstore=components["vectorstore"],
            bm25_retriever=components["bm25_retriever"],
            reranker=components["reranker"],
            llm=components["llm"],
            chat_history="",
            debug=debug,
            all_chunks=components.get("chunks"),
        ):
            event_type = str(event.get("type"))
            events_seen.append(event_type)

            if event_type == "chunk":
                answer_parts.append(str(event.get("content", "")))
            elif event_type == "done":
                done_payload = dict(event)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"

    elapsed = time.perf_counter() - started
    answer = str(done_payload.get("answer") or "".join(answer_parts)).strip()
    final_docs = list(done_payload.get("documents") or [])
    sources = list(done_payload.get("sources") or [])
    suggestions = list(done_payload.get("suggestions") or [])

    return {
        "id": item["id"],
        "question": question,
        "answer_status": get_answer_status(answer=answer, error=error),
        "answer": answer,
        "elapsed_seconds": round(elapsed, 2),
        "events_seen": events_seen,
        "final_doc_count": len(final_docs),
        "source_count": len(sources),
        "suggestion_count": len(suggestions),
        "sources": sources,
        "final_docs": [
            {
                "label": source_label(doc),
                "preview": doc_preview(doc),
                "metadata": dict(getattr(doc, "metadata", {}) or {}),
            }
            for doc in final_docs
        ],
        "error": error,
    }


def write_reports(results: List[Dict[str, Any]], args: argparse.Namespace) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    status_counts: Dict[str, int] = {}
    for result in results:
        status = result.get("answer_status", "UNKNOWN")
        status_counts[status] = status_counts.get(status, 0) + 1

    lines: List[str] = []
    lines.append("=" * 100)
    lines.append("QUESTION-ONLY RAG ANSWER GENERATION TEST")
    lines.append("=" * 100)
    lines.append(f"Time             : {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"Project root     : {PROJECT_ROOT}")
    lines.append(f"Questions tested : {len(results)}")
    lines.append(f"Debug            : {args.debug}")
    lines.append("Chat history     : empty string / standalone questions")
    lines.append("Hints            : none; only raw question text is passed")
    lines.append("UI path          : chains.chatbot.ask_rag_stream()")
    lines.append("")
    lines.append("SUMMARY")
    lines.append("-" * 100)
    lines.append(" | ".join(f"{key}={value}" for key, value in sorted(status_counts.items())))
    lines.append("")
    lines.append("Question | status | seconds | final_docs | answer preview")
    lines.append("-" * 100)

    for result in results:
        preview = " ".join(result["answer"].split())[:180]
        lines.append(
            f"{result['id']} | {result['answer_status']} | {result['elapsed_seconds']} | "
            f"{result['final_doc_count']} | {preview}"
        )

    for result in results:
        lines.append("")
        lines.append("=" * 100)
        lines.append(f"{result['id']} - {result['question']}")
        lines.append("=" * 100)
        lines.append(f"Answer status  : {result['answer_status']}")
        lines.append(f"Elapsed seconds: {result['elapsed_seconds']}")
        lines.append(f"Final docs     : {result['final_doc_count']}")
        lines.append(f"Sources        : {result['source_count']}")
        lines.append(f"Suggestions    : {result['suggestion_count']}")
        if result.get("error"):
            lines.append(f"Error          : {result['error']}")
        lines.append("")
        lines.append("ANSWER")
        lines.append("-" * 100)
        lines.append(result["answer"] or "<empty answer>")
        lines.append("")
        lines.append("FINAL DOCS SENT TO LLM")
        lines.append("-" * 100)

        if not result["final_docs"]:
            lines.append("No final docs.")
        else:
            for index, doc in enumerate(result["final_docs"], start=1):
                metadata = doc.get("metadata", {}) or {}
                lines.append(f"[{index}] {doc['label']}")
                for key in [
                    "context_mode",
                    "context_reason",
                    "keep_reason",
                    "filter_scope",
                    "evidence_snippet",
                    "rerank_score",
                    "hybrid_score",
                    "metadata_boosted_score",
                ]:
                    if key in metadata:
                        lines.append(f"{key}: {metadata.get(key)}")
                lines.append(f"Preview: {doc['preview']}")
                lines.append("")

        lines.append("SOURCES")
        lines.append("-" * 100)
        if not result["sources"]:
            lines.append("No sources.")
        else:
            for index, source in enumerate(result["sources"], start=1):
                lines.append(f"[{index}] {source}")

    REPORT_TXT.write_text("\n".join(lines), encoding="utf-8")

    with REPORT_JSONL.open("w", encoding="utf-8") as file:
        for result in results:
            file.write(json.dumps(result, ensure_ascii=False, default=str))
            file.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run raw-question-only RAG answer-generation tests using the UI backend path."
    )
    parser.add_argument("--limit", type=int, default=0, help="Run only the first N questions.")
    parser.add_argument("--only", nargs="*", default=[], help="Run only specific generated IDs, for example: --only Q01 Q11")
    parser.add_argument("--question", default="", help="Run a single custom raw question.")
    parser.add_argument("--debug", action="store_true", help="Pass debug=True into ask_rag_stream().")
    parser.add_argument("--with-suggestions", action="store_true", help="Allow after-answer suggestion generation. Slower; does not affect the answer.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    selected = select_questions(args)

    if not selected:
        print("No questions selected.")
        return 1

    import chains.chatbot as chatbot

    if not args.with_suggestions:
        # Suggestions are generated after the answer and do not affect retrieval/prompt/answer.
        # Disable by default so answer-generation tests are faster.
        chatbot.safe_generate_suggestions = lambda *unused_args, **unused_kwargs: []

    print("Loading chatbot components...")
    components = chatbot.load_chatbot_components()
    print("Components loaded.")

    results: List[Dict[str, Any]] = []

    for item in selected:
        print("\n" + "=" * 100)
        print(f"{item['id']}: {item['question']}")
        print("=" * 100)
        result = run_one_question(
            chatbot=chatbot,
            components=components,
            item=item,
            debug=args.debug,
        )
        results.append(result)
        print(
            f"Status: {result['answer_status']} | "
            f"seconds={result['elapsed_seconds']} | final_docs={result['final_doc_count']}"
        )
        print("Answer:", " ".join(result["answer"].split())[:400])

    write_reports(results, args)

    print("\nDONE")
    print(f"Report written to: {REPORT_TXT}")
    print(f"JSONL written to : {REPORT_JSONL}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
