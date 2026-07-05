import argparse
import inspect
import json
import os
import re
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

from config.settings import NO_ANSWER_TEXT

from chains import chatbot as chatbot_module
from chains import rag_chain as rag_chain_module
from chains.chatbot import (
    build_retrieval_query_list,
    clean_generated_answer,
    generate_chatbot_answer_with_retry,
    get_sources,
    load_chatbot_components,
    run_retrieval_core,
)
from retrieval import context_filter as context_filter_module
from retrieval.context_filter import get_dynamic_retrieval_settings, select_final_context_docs
from retrieval.hybrid_retriever import hybrid_search
from retrieval.reranker import rerank_documents

try:
    from retrieval.query_analyzer import analyze_query
except ImportError:
    analyze_query = None

try:
    from retrieval.query_type_detector import get_query_type_label
except ImportError:
    get_query_type_label = None

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "query_expansion_config.json"
REPORT_FILE = Path("reports") / "rag_all_questions_report.txt"
PREVIEW_CHARS = 900

TRACKED_SETTING_NAMES = [
    "SEMANTIC_K",
    "BM25_K",
    "HYBRID_FINAL_K",
    "RERANK_TOP_N",
    "RERANK_POOL_TOP_N",
    "MAX_CANDIDATES_BEFORE_RERANK",
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
    "ENABLE_MULTI_QUERY_RETRIEVAL",
    "MAX_RETRIEVAL_QUERIES",
    "ENABLE_QUESTION_REWRITE",
    "ENABLE_FALLBACK_RETRY",
    "ENABLE_TRUNCATION_RETRY",
    "DATA_PATH",
]

GENERIC_STOPWORDS = {
    "the", "and", "or", "but", "for", "with", "from", "that", "this", "these", "those",
    "who", "what", "when", "where", "why", "how", "did", "does", "do", "is", "are",
    "was", "were", "be", "been", "being", "a", "an", "of", "to", "in", "on", "by",
    "as", "at", "it", "its", "he", "she", "they", "them", "his", "her", "their",
    "ang", "ng", "sa", "si", "ni", "mga", "ano", "sino", "kailan", "bakit", "paano",
    "ito", "iyan", "iyon", "na", "ba", "ko", "mo", "nya", "niya", "nila", "nito",
}


def read_json_config(config_path=DEFAULT_CONFIG_PATH):
    try:
        config_path = Path(config_path)
        return json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return {}


def normalize_text(text):
    text = str(text or "").lower()
    text = re.sub(r"[^a-z0-9\u3040-\u30ff\u3400-\u9fff]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_list(values):
    if isinstance(values, str):
        values = [values]
    if not isinstance(values, list):
        return []
    cleaned = []
    for value in values:
        value = str(value or "").strip()
        if value and value not in cleaned:
            cleaned.append(value)
    return cleaned


def load_rag_test_cases(config_path=DEFAULT_CONFIG_PATH):
    raw_config = read_json_config(config_path)
    cases = []

    # Preferred config for answer-level tests.
    # Supported forms:
    # "rag_test_questions": ["question", {...}]
    # or "rag_test_questions": {"questions": ["question", {...}]}
    rag_config = raw_config.get("rag_test_questions", []) if isinstance(raw_config, dict) else []
    if isinstance(rag_config, dict):
        raw_cases = rag_config.get("questions", [])
    else:
        raw_cases = rag_config

    # Fallback: reuse retrieval/context-filter questions so the RAG test is not hardcoded in Python.
    if not raw_cases:
        context_config = raw_config.get("context_filter_test_questions", {}) if isinstance(raw_config, dict) else {}
        raw_cases = context_config.get("questions", []) if isinstance(context_config, dict) else []

    seen_questions = set()
    for item in raw_cases or []:
        if isinstance(item, str):
            case = {"question": item}
        elif isinstance(item, dict):
            case = dict(item)
        else:
            continue

        question = str(case.get("question") or "").strip()
        if not question or question in seen_questions:
            continue

        case["question"] = question
        case["expected_all"] = normalize_list(case.get("expected_all", []))
        case["expected_any"] = normalize_list(case.get("expected_any", []))
        case["must_not_include"] = normalize_list(case.get("must_not_include", []))
        case["expected_source_any"] = normalize_list(case.get("expected_source_any", []))
        case["category"] = str(case.get("category", "")).strip()
        cases.append(case)
        seen_questions.add(question)

    return cases


def get_setting(name, default_value="N/A"):
    if settings_module is None:
        return default_value
    return getattr(settings_module, name, default_value)


def normalize_for_compare(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value).strip().lower()


def get_setting_sync_status(setting_name):
    config_value = get_setting(setting_name, "<missing in config.settings>")
    env_value = os.environ.get(setting_name)

    if env_value is None:
        return config_value, "<not set>", "env not set; using config/default"

    if normalize_for_compare(config_value) == normalize_for_compare(env_value):
        return config_value, env_value, "SYNCED"

    return config_value, env_value, "MISMATCH - env exists but imported setting differs"


def get_function_source(function):
    try:
        return inspect.getsourcefile(function) or "N/A"
    except Exception:
        return "N/A"


def get_function_signature(function):
    try:
        return str(inspect.signature(function))
    except Exception as error:
        return f"<signature unavailable: {type(error).__name__}>"


def add_section(lines, title):
    lines.append("")
    lines.append("=" * 80)
    lines.append(title)
    lines.append("=" * 80)


def add_subsection(lines, title):
    lines.append("")
    lines.append("-" * 80)
    lines.append(title)
    lines.append("-" * 80)


def format_seconds(seconds):
    seconds = float(seconds or 0)
    if seconds < 60:
        return f"{seconds:.2f} sec"
    minutes = int(seconds // 60)
    remaining = seconds % 60
    return f"{minutes} min {remaining:.2f} sec"


def time_step(label, timer_rows, function, *args, **kwargs):
    start = time.perf_counter()
    result = function(*args, **kwargs)
    elapsed = time.perf_counter() - start
    timer_rows.append((label, elapsed))
    return result


def save_report(lines, report_file=REPORT_FILE):
    report_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.write_text("\n".join(lines), encoding="utf-8")


def get_metadata(doc):
    return dict(getattr(doc, "metadata", {}) or {})


def get_doc_text(doc):
    return str(getattr(doc, "page_content", "") or "")


def get_chunk_id(doc):
    metadata = get_metadata(doc)
    return str(metadata.get("chunk_id") or metadata.get("chunk_index") or id(doc))


def get_source_label(doc):
    metadata = get_metadata(doc)
    source = metadata.get("file_name") or metadata.get("source") or "Unknown source"
    page = metadata.get("page", "N/A")
    chunk_id = metadata.get("chunk_id") or metadata.get("chunk_index") or "Unknown chunk"
    return f"{source} | page={page} | chunk={chunk_id}"


def get_preview(doc, limit=PREVIEW_CHARS):
    text = " ".join(get_doc_text(doc).split())
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "..."


def add_docs(lines, title, docs, max_docs=5):
    add_subsection(lines, title)
    docs = list(docs or [])
    lines.append(f"Count: {len(docs)}")

    if not docs:
        lines.append("No documents.")
        return

    score_keys = [
        "semantic_distance",
        "semantic_rank",
        "bm25_rank",
        "bm25_rank_score",
        "hybrid_score",
        "metadata_boosted_score",
        "rerank_score",
        "rerank_rank",
        "rerank_original_rank",
        "context_mode",
        "context_anchor_reason",
        "context_keep_reason",
        "context_confident_filter_scope",
    ]

    for index, doc in enumerate(docs[:max_docs], start=1):
        metadata = get_metadata(doc)
        lines.append("")
        lines.append(f"Rank    : {index}")
        lines.append(f"Source  : {get_source_label(doc)}")
        lines.append(f"Title   : {metadata.get('title', '')}")
        lines.append(f"Section : {metadata.get('section', '')}")

        for key in score_keys:
            if key in metadata:
                lines.append(f"{key}: {metadata.get(key)}")

        lines.append(f"Preview : {get_preview(doc)}")
        lines.append("-" * 80)

    if len(docs) > max_docs:
        lines.append(f"... {len(docs) - max_docs} more documents not shown.")


def add_settings_sync_report(lines):
    add_section(lines, "SETTINGS SYNC CHECK")
    lines.append(f"Python executable : {sys.executable}")
    lines.append(f"Current cwd       : {Path.cwd()}")
    lines.append(f"Script file       : {Path(__file__).resolve()}")
    lines.append(f"Project root      : {PROJECT_ROOT}")
    lines.append(f"config.settings   : {'LOADED' if settings_module is not None else 'NOT LOADED'}")
    lines.append(f"settings file     : {getattr(settings_module, '__file__', 'N/A')}")
    lines.append("")
    lines.append("Name                          | config.settings value | env value | status")
    lines.append("-" * 105)

    for setting_name in TRACKED_SETTING_NAMES:
        config_value, env_value, status = get_setting_sync_status(setting_name)
        lines.append(
            f"{setting_name:<29} | {str(config_value):<21} | {str(env_value):<9} | {status}"
        )


def add_function_sync_report(lines):
    add_section(lines, "RAG / RETRIEVAL FILE SYNC CHECK")

    functions = [
        ("chatbot.load_chatbot_components", load_chatbot_components),
        ("chatbot.build_retrieval_query_list", build_retrieval_query_list),
        ("chatbot.run_retrieval_core", run_retrieval_core),
        ("chatbot.generate_chatbot_answer_with_retry", generate_chatbot_answer_with_retry),
        ("rag_chain.prepare_context_docs", rag_chain_module.prepare_context_docs),
        ("rag_chain.build_prompt_from_context", rag_chain_module.build_prompt_from_context),
        ("rag_chain.generate_answer_with_context", rag_chain_module.generate_answer_with_context),
        ("rag_chain.select_final_context_docs", rag_chain_module.select_final_context_docs),
        ("retrieval.context_filter.select_final_context_docs", context_filter_module.select_final_context_docs),
        ("retrieval.context_filter.get_dynamic_retrieval_settings", get_dynamic_retrieval_settings),
        ("retrieval.hybrid_retriever.hybrid_search", hybrid_search),
        ("retrieval.reranker.rerank_documents", rerank_documents),
    ]

    if get_query_type_label is not None:
        functions.append(("retrieval.query_type_detector.get_query_type_label", get_query_type_label))

    for function_name, function in functions:
        lines.append(f"{function_name} source    : {get_function_source(function)}")
        lines.append(f"{function_name} signature : {get_function_signature(function)}")
        lines.append("")

    lines.append("Identity checks:")
    lines.append(
        "rag_chain.select_final_context_docs is retrieval.context_filter.select_final_context_docs : "
        f"{rag_chain_module.select_final_context_docs is context_filter_module.select_final_context_docs}"
    )
    lines.append(
        "chatbot.generate_answer_with_context is rag_chain.generate_answer_with_context      : "
        f"{chatbot_module.generate_answer_with_context is rag_chain_module.generate_answer_with_context}"
    )
    lines.append(
        "chatbot.clean_generated_answer is rag_chain.clean_generated_answer                  : "
        f"{chatbot_module.clean_generated_answer is rag_chain_module.clean_generated_answer}"
    )

    lines.append("")
    lines.append("Patch marker checks:")

    try:
        rag_chain_path = Path(get_function_source(rag_chain_module.generate_answer_with_context))
        rag_chain_text = rag_chain_path.read_text(encoding="utf-8", errors="ignore")

        markers = [
            "Hard lock for short date queries",
            "pre_llm_date_answer",
            "build_short_date_context_answer",
            "is_effective_empty_answer",
        ]

        for marker in markers:
            status = "FOUND" if marker in rag_chain_text else "MISSING"
            lines.append(f"{marker:<40}: {status}")

    except Exception as error:
        lines.append(f"Patch marker check error: {type(error).__name__}: {error}")


def add_loaded_components_report(lines, components):
    add_section(lines, "LOADED COMPONENTS CHECK")
    for key, value in components.items():
        lines.append(f"{key:<16}: {type(value).__module__}.{type(value).__name__}")

    chunks = components.get("chunks") or []
    lines.append(f"chunks count     : {len(chunks)}")


def get_question_info(question):
    info = {}
    if analyze_query is not None:
        try:
            info["analyzer"] = analyze_query(question)
        except Exception as error:
            info["analyzer_error"] = f"{type(error).__name__}: {error}"

    if get_query_type_label is not None:
        try:
            info["query_type"] = get_query_type_label(question)
        except Exception as error:
            info["query_type_error"] = f"{type(error).__name__}: {error}"

    try:
        info["dynamic_settings"] = dict(get_dynamic_retrieval_settings(question) or {})
    except Exception as error:
        info["dynamic_settings_error"] = f"{type(error).__name__}: {error}"

    return info


def answer_contains(answer, expected):
    return normalize_text(expected) in normalize_text(answer)


def source_contains(final_docs, expected):
    expected_key = normalize_text(expected)
    for doc in final_docs or []:
        label = get_source_label(doc)
        metadata = get_metadata(doc)
        searchable = " ".join([
            label,
            str(metadata.get("title", "")),
            str(metadata.get("source", "")),
            str(metadata.get("file_name", "")),
            str(metadata.get("section", "")),
        ])
        if expected_key and expected_key in normalize_text(searchable):
            return True
    return False


def get_answer_support_warnings(answer, context_docs, question):
    answer_key = normalize_text(answer)
    question_key = normalize_text(question)
    context_key = normalize_text(" ".join(get_doc_text(doc) for doc in context_docs or []))

    if not answer_key:
        return ["empty_answer"]

    if str(NO_ANSWER_TEXT).strip().lower() in str(answer).strip().lower():
        if context_docs:
            return ["no_answer_text_even_with_final_context"]
        return []

    tokens = []
    for token in answer_key.split():
        if len(token) < 4:
            continue
        if token in GENERIC_STOPWORDS:
            continue
        if token in question_key.split():
            continue
        if token not in tokens:
            tokens.append(token)

    unsupported = [token for token in tokens if token not in context_key]

    warnings = []
    if context_docs and len(unsupported) >= 8:
        warnings.append("many_answer_terms_not_found_in_context")
        warnings.append("unsupported_terms_sample=" + ", ".join(unsupported[:12]))

    return warnings


def evaluate_answer(case, answer, final_docs):
    expected_all = case.get("expected_all", [])
    expected_any = case.get("expected_any", [])
    must_not_include = case.get("must_not_include", [])
    expected_source_any = case.get("expected_source_any", [])

    checks_enabled = bool(expected_all or expected_any or must_not_include or expected_source_any)
    failures = []

    for expected in expected_all:
        if not answer_contains(answer, expected):
            failures.append(f"missing expected_all in answer: {expected}")

    if expected_any:
        if not any(answer_contains(answer, expected) for expected in expected_any):
            failures.append("missing any expected_any in answer: " + " | ".join(expected_any))

    for forbidden in must_not_include:
        if answer_contains(answer, forbidden):
            failures.append(f"found forbidden phrase in answer: {forbidden}")

    if expected_source_any:
        if not any(source_contains(final_docs, expected) for expected in expected_source_any):
            failures.append("missing expected source: " + " | ".join(expected_source_any))

    if not checks_enabled:
        return "REVIEW", ["no_expected_checks_configured"]

    if failures:
        return "FAIL", failures

    return "PASS", []


def run_one_question(case, components, no_llm=False, debug=False, max_docs_report=5):
    question = case["question"]
    timer_rows = []
    lines = []

    add_section(lines, f"QUESTION: {question}")

    info = get_question_info(question)
    analyzer = info.get("analyzer", {})
    if isinstance(analyzer, dict):
        lines.append(f"Question mode from analyzer : {analyzer.get('mode', '')}")
        lines.append(f"Important terms             : {analyzer.get('important_terms', [])}")
        lines.append(f"Source keywords             : {analyzer.get('source_keywords', [])}")
    elif info.get("analyzer_error"):
        lines.append(f"Question analyzer error     : {info.get('analyzer_error')}")

    if "query_type" in info:
        lines.append(f"Query type detector label   : {info.get('query_type')}")
    elif info.get("query_type_error"):
        lines.append(f"Query type detector error   : {info.get('query_type_error')}")

    dynamic_settings = info.get("dynamic_settings", {})
    if dynamic_settings:
        lines.append(f"Dynamic context mode        : {dynamic_settings.get('mode', '')}")
        lines.append(f"Dynamic retrieval settings  : {dynamic_settings}")
    elif info.get("dynamic_settings_error"):
        lines.append(f"Dynamic settings error      : {info.get('dynamic_settings_error')}")

    retrieval_queries = build_retrieval_query_list(
        question=question,
        rewritten_question=question,
    )

    lines.append("")
    lines.append("Retrieval queries:")
    for index, retrieval_query in enumerate(retrieval_queries, start=1):
        lines.append(f"[{index}] {retrieval_query}")

    retrieval_result = time_step(
        "run_retrieval_core",
        timer_rows,
        run_retrieval_core,
        question=question,
        retrieval_queries=retrieval_queries,
        vectorstore=components["vectorstore"],
        bm25_retriever=components["bm25_retriever"],
        reranker=components["reranker"],
        debug=debug,
    )

    if no_llm:
        answer = "[NO LLM MODE] Answer generation skipped."
        # Force final docs to be generated even without the LLM.
        from chains.chatbot import ensure_final_context_docs
        time_step(
            "ensure_final_context_docs",
            timer_rows,
            ensure_final_context_docs,
            question,
            retrieval_result,
            components.get("chunks", []),
            debug,
        )
    else:
        answer = time_step(
            "generate_chatbot_answer_with_retry",
            timer_rows,
            generate_chatbot_answer_with_retry,
            question=question,
            retrieval_result=retrieval_result,
            llm=components["llm"],
            chat_history="",
            debug=debug,
            category=case.get("category", ""),
            no_llm=False,
            all_chunks=components.get("chunks", []),
        )
        answer = clean_generated_answer(answer=answer, question=question)

    final_docs = retrieval_result.get("final_docs", []) or []
    sources = get_sources(final_docs)
    status, check_notes = evaluate_answer(case, answer, final_docs)
    support_warnings = get_answer_support_warnings(answer, final_docs, question)

    add_docs(lines, "SEMANTIC DOCS", retrieval_result.get("semantic_docs", []), max_docs=max_docs_report)
    add_docs(lines, "BM25 DOCS", retrieval_result.get("bm25_docs", []), max_docs=max_docs_report)
    add_docs(lines, "HYBRID DOCS", retrieval_result.get("hybrid_docs", []), max_docs=max_docs_report)
    add_docs(lines, "RANKED DOCS AFTER RERANK", retrieval_result.get("ranked_docs", []), max_docs=max_docs_report)
    add_docs(lines, "FINAL CONTEXT DOCS SENT TO LLM", final_docs, max_docs=max_docs_report)

    add_subsection(lines, "LLM ANSWER")
    lines.append(answer)

    add_subsection(lines, "SOURCES RETURNED TO UI")
    if sources:
        for index, source in enumerate(sources, start=1):
            lines.append(f"[{index}] {source}")
    else:
        lines.append("No sources.")

    add_subsection(lines, "ANSWER CHECK")
    lines.append(f"Status: {status}")
    lines.append("Check notes:")
    for note in check_notes:
        lines.append(f"- {note}")
    if not check_notes:
        lines.append("- none")

    lines.append("Support warnings:")
    for warning in support_warnings:
        lines.append(f"- {warning}")
    if not support_warnings:
        lines.append("- none")

    add_subsection(lines, "TIMING SUMMARY")
    total_time = sum(seconds for _, seconds in timer_rows)
    lines.append(f"Total measured time : {format_seconds(total_time)}")
    for label, seconds in timer_rows:
        pct = (seconds / total_time * 100) if total_time else 0
        lines.append(f"- {label:<38} {format_seconds(seconds):>14} ({pct:5.1f}%)")

    row = {
        "question": question,
        "status": status,
        "analyzer_mode": analyzer.get("mode", "") if isinstance(analyzer, dict) else "",
        "query_type": info.get("query_type", ""),
        "dynamic_mode": dynamic_settings.get("mode", "") if isinstance(dynamic_settings, dict) else "",
        "semantic_count": len(retrieval_result.get("semantic_docs", []) or []),
        "bm25_count": len(retrieval_result.get("bm25_docs", []) or []),
        "hybrid_count": len(retrieval_result.get("hybrid_docs", []) or []),
        "ranked_count": len(retrieval_result.get("ranked_docs", []) or []),
        "final_count": len(final_docs),
        "answer_preview": " ".join(str(answer).split())[:220],
        "final_sources": [get_source_label(doc) for doc in final_docs[:5]],
        "warnings": support_warnings,
        "check_notes": check_notes,
        "time_sec": total_time,
    }

    return lines, row


def add_batch_summary(lines, rows):
    add_section(lines, "BATCH SUMMARY")
    lines.append("Question | status | analyzer_mode | query_type | dynamic_mode | semantic | bm25 | hybrid | ranked | final | time")
    lines.append("-" * 150)

    for row in rows:
        lines.append(
            f"{row['question']} | {row['status']} | {row['analyzer_mode']} | {row['query_type']} | "
            f"{row['dynamic_mode']} | {row['semantic_count']} | {row['bm25_count']} | {row['hybrid_count']} | "
            f"{row['ranked_count']} | {row['final_count']} | {format_seconds(row['time_sec'])}"
        )
        lines.append(f"  Answer preview: {row['answer_preview']}")
        if row["warnings"]:
            lines.append("  Warnings: " + " | ".join(row["warnings"]))
        if row["check_notes"]:
            lines.append("  Check notes: " + " | ".join(row["check_notes"]))
        for index, source in enumerate(row["final_sources"], start=1):
            lines.append(f"  Final {index}: {source}")


def parse_args():
    parser = argparse.ArgumentParser(description="Run RAG answer tests using the same chatbot/RAG retrieval path.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to query_expansion_config.json")
    parser.add_argument("--report", default=str(REPORT_FILE), help="Output report path")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of questions. 0 means all.")
    parser.add_argument("--start", type=int, default=1, help="1-based start question index.")
    parser.add_argument("--no-llm", action="store_true", help="Skip answer generation; still checks final context sync.")
    parser.add_argument("--debug", action="store_true", help="Enable debug output in retrieval/RAG functions.")
    parser.add_argument("--max-docs-report", type=int, default=5, help="Max docs per stage to print per question.")
    return parser.parse_args()


def main():
    args = parse_args()
    config_path = Path(args.config)
    report_file = Path(args.report)

    cases = load_rag_test_cases(config_path)
    if args.start > 1:
        cases = cases[args.start - 1:]
    if args.limit and args.limit > 0:
        cases = cases[:args.limit]

    lines = []
    rows = []

    add_section(lines, "RAG ALL QUESTIONS DEBUG TEST")
    lines.append(f"Config path : {config_path}")
    lines.append(f"Report file : {report_file}")
    lines.append(f"No LLM mode : {args.no_llm}")
    lines.append("")
    lines.append("Test questions:")
    for index, case in enumerate(cases, start=1):
        lines.append(f"{index}. {case['question']}")

    add_settings_sync_report(lines)
    add_function_sync_report(lines)

    setup_timers = []
    components = time_step("load_chatbot_components", setup_timers, load_chatbot_components)
    add_loaded_components_report(lines, components)

    add_section(lines, "SETUP TIMING SUMMARY")
    total_setup = sum(seconds for _, seconds in setup_timers)
    lines.append(f"Total setup time : {format_seconds(total_setup)}")
    for label, seconds in setup_timers:
        pct = (seconds / total_setup * 100) if total_setup else 0
        lines.append(f"- {label:<38} {format_seconds(seconds):>14} ({pct:5.1f}%)")

    for case in cases:
        question_lines, row = run_one_question(
            case=case,
            components=components,
            no_llm=args.no_llm,
            debug=args.debug,
            max_docs_report=args.max_docs_report,
        )
        lines.extend(question_lines)
        rows.append(row)
        save_report(lines, report_file=report_file)

    add_batch_summary(lines, rows)
    save_report(lines, report_file=report_file)

    print(f"RAG report saved: {report_file.resolve()}")
    print(f"Questions tested: {len(rows)}")
    status_counts = {}
    for row in rows:
        status_counts[row["status"]] = status_counts.get(row["status"], 0) + 1
    print("Status counts:", status_counts)


if __name__ == "__main__":
    main()
