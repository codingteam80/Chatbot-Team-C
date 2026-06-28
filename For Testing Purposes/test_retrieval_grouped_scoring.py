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

# Optional settings para hindi mag-error kung luma pa ang settings.py.
try:
    from config.settings import (
        ENABLE_MULTI_QUERY_RETRIEVAL,
        MAX_CANDIDATES_BEFORE_RERANK,
        MAX_RETRIEVAL_QUERIES,
    )
except ImportError:
    ENABLE_MULTI_QUERY_RETRIEVAL = True
    MAX_RETRIEVAL_QUERIES = 3
    MAX_CANDIDATES_BEFORE_RERANK = 15

# Optional candidate balancer settings para compatible pa rin kahit luma ang settings.py.
try:
    from config.settings import (
        BALANCED_RERANK_TOP_N,
        CANDIDATE_MAX_PER_SOURCE,
        ENABLE_CANDIDATE_BALANCING,
    )
except ImportError:
    ENABLE_CANDIDATE_BALANCING = True
    CANDIDATE_MAX_PER_SOURCE = 2
    BALANCED_RERANK_TOP_N = 8

from embeddings.embedding_model import get_embedding_model
from retrieval.candidate_balancer import (
    balance_candidates,
    balance_reranked_documents,
)
from retrieval.context_filter import (
    filter_low_quality_docs,
    limit_context_docs,
    select_final_context_docs,
)
from retrieval.hybrid_retriever import hybrid_search
from retrieval.reranker import load_reranker, rerank_documents
from retrieval.retrieval_query_builder import build_retrieval_queries
from utils.bm25_cache import load_or_create_bm25
from utils.chunk_cache import load_or_create_chunks
from vectorstore.chroma_store import load_chroma_vectorstore


REPORT_FILENAME = "result_retrieve.txt"
MIN_KEYWORD_MATCHES = 1


BASELINE_TYPES = {
    "direct",
    "keyword_heavy",
    "paraphrased",
    "tagalog_english_docs",
    "negative",
}

STRESS_TYPES = {
    "cross_doc",
    "false_premise",
    "similar_topic",
}

GROUP_ORDER = ["baseline", "stress", "uncategorized"]


def get_test_group(test_case):
    # Baseline = normal retrieval questions. Stress = cross-doc/false-premise/confusion tests.
    explicit_group = test_case.get("test_group") or test_case.get("suite")
    if explicit_group:
        return explicit_group

    test_type = test_case.get("type", "uncategorized")
    if test_type in BASELINE_TYPES:
        return "baseline"

    if test_type in STRESS_TYPES:
        return "stress"

    return "uncategorized"


def score_percent(passed_count, total_count):
    # Safe percentage helper para walang division by zero.
    return (passed_count / total_count * 100) if total_count else 0


def get_item_passed(item):
    # Kunin ang pass/fail value from retrieval result item.
    return item["result"]["evaluation"]["passed"]


def add_score_line(lines, label, items):
    # Isang formatted score line para reusable sa baseline/stress/type scoring.
    total_count = len(items)
    passed_count = sum(1 for item in items if get_item_passed(item))
    failed_count = total_count - passed_count
    lines.append(
        f"{label:<28}: {passed_count}/{total_count} passed | "
        f"{failed_count} failed | {score_percent(passed_count, total_count):.2f}%"
    )


def group_result_items(results):
    # I-group ang retrieval results by baseline/stress.
    grouped = {group: [] for group in GROUP_ORDER}

    for item in results:
        group = get_test_group(item["test_case"])
        if group not in grouped:
            grouped[group] = []
        grouped[group].append(item)

    return grouped


def add_score_breakdown(lines, results):
    # Ipakita ang score by baseline/stress at by type.
    lines.extend(["", "=" * 80, "SCORE BREAKDOWN", "=" * 80])

    grouped = group_result_items(results)
    for group in GROUP_ORDER:
        if grouped.get(group):
            add_score_line(lines, group.upper(), grouped[group])

    lines.extend(["", "TYPE SCORES:"])
    test_types = sorted({item["test_case"].get("type", "uncategorized") for item in results})
    for test_type in test_types:
        type_results = [item for item in results if item["test_case"].get("type", "uncategorized") == test_type]
        add_score_line(lines, test_type, type_results)


def source_variants(*base_names):
    # Gumawa ng .md at .pdf source variants para gumana sa lumang data at uploaded PDF data.
    sources = []

    for base_name in base_names:
        for extension in (".md", ".pdf"):
            sources.append(f"{base_name}{extension}")

    return sources


TEST_CASES = [
    {
        "id": "D01",
        "type": "direct",
        "query": "Who was Apolinario Mabini and what role did he serve in the First Philippine Republic?",
        "expected_sources": source_variants("Apolinario Mabini - Wikipedia"),
        "must_have": [
            "Apolinario Mabini",
            "legal and constitutional adviser",
            "first Prime Minister",
            "First Philippine Republic",
        ],
        "min_keyword_matches": 2,
    },
    {
        "id": "D02",
        "type": "direct",
        "query": "Sino si Andres Bonifacio?",
        "expected_sources": source_variants(
            "Andrés Bonifacio - Wikipedia",
            "Andres Bonifacio - Wikipedia",
        ),
        "must_have": [
            "Andrés Bonifacio",
            "Andres Bonifacio",
            "Filipino revolutionary leader",
            "Father of the Philippine Revolution",
            "Katipunan",
        ],
        "min_keyword_matches": 2,
    },
    {
        "id": "K01",
        "type": "keyword_heavy",
        "query": "Katipunan KKK July 7 1892 founders Deodato Arellano Andres Bonifacio Valentin Diaz Ladislao Diwa Jose Dizon Teodoro Plata purpose independence Spain",
        "expected_sources": source_variants("Katipunan - Wikipedia"),
        "must_have": [
            "Katipunan",
            "July 7, 1892",
            "Deodato Arellano",
            "Andrés Bonifacio",
            "Valentin Diaz",
            "Ladislao Diwa",
            "Jose Dizon",
            "Teodoro Plata",
            "independence from the Spanish Empire",
        ],
        "min_keyword_matches": 3,
    },
    {
        "id": "K02",
        "type": "keyword_heavy",
        "query": "Gomburza Cavite mutiny 1872 garrote Bagumbayan Mariano Gomes Jose Burgos Jacinto Zamora secularization friars Rizal El filibusterismo",
        "expected_sources": source_variants("Gomburza - Wikipedia"),
        "must_have": [
            "Gomburza",
            "Mariano Gómes",
            "Jose Burgos",
            "Jacinto Zamora",
            "February 17, 1872",
            "Cavite mutiny",
            "El filibusterismo",
            "secularization",
        ],
        "min_keyword_matches": 3,
    },
    {
        "id": "P01",
        "type": "paraphrased",
        "query": "Which secret group tried to free Filipinos from Spanish rule through armed revolution before it was discovered in 1896?",
        "expected_sources": source_variants("Katipunan - Wikipedia"),
        "must_have": [
            "Katipunan",
            "secret society",
            "armed revolution",
            "Spanish Empire",
            "discovery by Spanish authorities",
            "August 1896",
        ],
        "min_keyword_matches": 2,
    },
    {
        "id": "P02",
        "type": "paraphrased",
        "query": "Sinong lider ang tinatawag na utak ng himagsikan dahil sa papel niya bilang tagapayo ng pamahalaang rebolusyonaryo?",
        "expected_sources": source_variants("Apolinario Mabini - Wikipedia"),
        "must_have": [
            "Apolinario Mabini",
            "utak ng himagsikan",
            "brain of the revolution",
            "legal and constitutional adviser",
            "Revolutionary Government",
        ],
        "min_keyword_matches": 2,
    },
    {
        "id": "T01",
        "type": "tagalog_english_docs",
        "query": "Kailan ipinagdiriwang ang Araw ng Kalayaan ng Pilipinas at anong pangyayari ang ginugunita nito?",
        "expected_sources": source_variants("Independence Day (Philippines) - Wikipedia"),
        "must_have": [
            "June 12",
            "Araw ng Kalayaan",
            "Philippine independence from Spain",
            "declaration of Philippine independence from Spain in 1898",
            "national holiday",
        ],
        "min_keyword_matches": 2,
    },
    {
        "id": "T02",
        "type": "tagalog_english_docs",
        "query": "Ano ang nangyari sa Pilipinas noong World War II nang sakupin ito ng Japan?",
        "expected_sources": source_variants("Japanese occupation of the Philippines - Wikipedia"),
        "must_have": [
            "Japanese Empire",
            "occupied the Commonwealth of the Philippines",
            "1942 and 1945",
            "World War II",
            "Bataan Death March",
            "MacArthur",
        ],
        "min_keyword_matches": 2,
    },
    {
        "id": "X01",
        "type": "cross_doc",
        "query": "How did the Treaty of Paris connect the Spanish-American War to the Philippine-American War?",
        "expected_sources": source_variants(
            "Treaty of Paris (1898) - Wikipedia",
            "Spanish–American War - Wikipedia",
            "Spanish-American War - Wikipedia",
            "Philippine–American War - Wikipedia",
            "Philippine-American War - Wikipedia",
        ),
        "expected_source_groups": [
            source_variants("Treaty of Paris (1898) - Wikipedia"),
            source_variants("Spanish–American War - Wikipedia", "Spanish-American War - Wikipedia"),
            source_variants("Philippine–American War - Wikipedia", "Philippine-American War - Wikipedia"),
        ],
        "must_have": [
            "Treaty of Paris",
            "Spanish-American War",
            "Spanish–American War",
            "United States",
            "Philippine-American War",
            "Philippine–American War",
            "annexation",
        ],
        "min_keyword_matches": 4,
    },
    {
        "id": "X02",
        "type": "cross_doc",
        "query": "Paano nauugnay ang La Liga Filipina ni Jose Rizal sa pagkakatatag ng Katipunan?",
        "expected_sources": source_variants(
            "Katipunan - Wikipedia",
            "José Rizal - Wikipedia",
            "Jose Rizal - Wikipedia",
            "Andrés Bonifacio - Wikipedia",
            "Andres Bonifacio - Wikipedia",
        ),
        "expected_source_groups": [
            source_variants("Katipunan - Wikipedia"),
            source_variants("José Rizal - Wikipedia", "Jose Rizal - Wikipedia"),
        ],
        "must_have": [
            "La Liga Filipina",
            "Rizal",
            "Katipunan",
            "Bonifacio",
            "representation to the Spanish Parliament",
            "arrest and deportation",
        ],
        "min_keyword_matches": 3,
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
    {
        "id": "F01",
        "type": "false_premise",
        "query": "Why did Jose Rizal become the Supremo of the Katipunan?",
        "expected_sources": source_variants(
            "Katipunan - Wikipedia",
            "Andrés Bonifacio - Wikipedia",
            "Andres Bonifacio - Wikipedia",
            "José Rizal - Wikipedia",
            "Jose Rizal - Wikipedia",
        ),
        "must_have": [
            "Andrés Bonifacio",
            "Andres Bonifacio",
            "Supreme President",
            "Supremo",
            "Katipunan",
            "Rizal",
        ],
        "must_not_have": [
            "Rizal became the Supremo",
            "Jose Rizal became the Supremo",
            "Rizal was the Supremo of the Katipunan",
        ],
        "min_keyword_matches": 2,
    },
    {
        "id": "F02",
        "type": "false_premise",
        "query": "Bakit si Apolinario Mabini ang nagdeklara ng kalayaan ng Pilipinas noong June 12, 1898?",
        "expected_sources": source_variants(
            "Apolinario Mabini - Wikipedia",
            "Emilio Aguinaldo - Wikipedia",
            "Philippine Revolution - Wikipedia",
            "Independence Day (Philippines) - Wikipedia",
        ),
        "must_have": [
            "Emilio Aguinaldo",
            "June 12",
            "1898",
            "independence",
            "Ambrosio Rianzares Bautista",
        ],
        "must_not_have": [
            "Mabini declared Philippine independence on June 12, 1898",
            "Apolinario Mabini declared independence on June 12, 1898",
            "si Apolinario Mabini ang nagdeklara",
        ],
        "min_keyword_matches": 2,
    },
    {
        "id": "S01",
        "type": "similar_topic",
        "query": "Compare Spanish-American War vs Philippine-American War. Huwag pagpalitin ang cause at result ng dalawang war.",
        "expected_sources": source_variants(
            "Spanish–American War - Wikipedia",
            "Spanish-American War - Wikipedia",
            "Philippine–American War - Wikipedia",
            "Philippine-American War - Wikipedia",
        ),
        "expected_source_groups": [
            source_variants("Spanish–American War - Wikipedia", "Spanish-American War - Wikipedia"),
            source_variants("Philippine–American War - Wikipedia", "Philippine-American War - Wikipedia"),
        ],
        "must_have": [
            "USS Maine",
            "Spanish-American War",
            "Spanish–American War",
            "Philippine-American War",
            "Philippine–American War",
            "annexation",
            "Treaty of Paris",
            "Battle of Manila",
        ],
        "min_keyword_matches": 4,
    },
    {
        "id": "S02",
        "type": "similar_topic",
        "query": "Ano ang pinagkaiba ng Philippine Revolution at Katipunan? Organization ba yung isa o war/revolution yung isa?",
        "expected_sources": source_variants(
            "Philippine Revolution - Wikipedia",
            "Katipunan - Wikipedia",
        ),
        "expected_source_groups": [
            source_variants("Philippine Revolution - Wikipedia"),
            source_variants("Katipunan - Wikipedia"),
        ],
        "must_have": [
            "Philippine Revolution",
            "war of independence",
            "Katipunan",
            "revolutionary organization",
            "1896 to 1898",
            "founded in 1892",
        ],
        "min_keyword_matches": 4,
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


def get_doc_merge_key(doc):
    # Stable key para hindi maulit ang parehong chunk mula sa multiple queries.
    metadata = doc.metadata or {}
    source = metadata.get("source", "")
    page = metadata.get("page", "")
    chunk_id = metadata.get("chunk_id") or metadata.get("chunk_index", "")

    if chunk_id != "":
        return (source, page, chunk_id)

    # Fallback kapag walang chunk_index metadata.
    preview = (doc.page_content or "")[:250]
    return (source, page, preview)


def merge_docs_round_robin(doc_groups, max_docs):
    # Pagsamahin ang candidates nang pantay kada retrieval query.
    # Mas okay ito kaysa simple extend dahil hindi natatabunan ng unang query ang ibang topics.
    if not doc_groups:
        return []

    merged_docs = []
    seen_keys = set()
    max_group_length = max(len(group) for group in doc_groups)

    for rank_index in range(max_group_length):
        for group in doc_groups:
            if rank_index >= len(group):
                continue

            doc = group[rank_index]
            doc_key = get_doc_merge_key(doc)

            if doc_key in seen_keys:
                continue

            seen_keys.add(doc_key)
            merged_docs.append(doc)

            if len(merged_docs) >= max_docs:
                return merged_docs

    return merged_docs


def run_multi_query_hybrid_search(query, pipeline):
    # Gumawa muna ng retrieval queries bago mag-hybrid search.
    # Direct questions usually 1 query lang; compare/cross-doc can become 2 to 3 queries.
    retrieval_queries = build_retrieval_queries(
        question=query,
        enabled=ENABLE_MULTI_QUERY_RETRIEVAL,
        max_queries=MAX_RETRIEVAL_QUERIES,
    )

    query_doc_groups = []

    for retrieval_query in retrieval_queries:
        docs = hybrid_search(
            query=retrieval_query,
            vectorstore=pipeline["vectorstore"],
            bm25_retriever=pipeline["bm25_retriever"],
            semantic_k=SEMANTIC_K,
            bm25_k=BM25_K,
            final_k=HYBRID_FINAL_K,
            use_rrf=True,
        )
        query_doc_groups.append(docs)

    balanced_docs = balance_candidates(
        document_groups=query_doc_groups,
        max_docs=MAX_CANDIDATES_BEFORE_RERANK,
        max_per_source=CANDIDATE_MAX_PER_SOURCE,
        enabled=ENABLE_CANDIDATE_BALANCING,
    )

    return {
        "retrieval_queries": retrieval_queries,
        "hybrid_docs": balanced_docs,
        "query_doc_groups": query_doc_groups,
    }


SOURCE_DIVERSITY_TEST_TYPES = {"cross_doc", "similar_topic"}


def should_use_source_diverse_context(test_case):
    # Gamitin lang ang source-diverse selection kapag kailangan talaga ng multiple sources.
    if test_case.get("type") in SOURCE_DIVERSITY_TEST_TYPES:
        return True

    return bool(test_case.get("expected_source_groups"))


def select_test_final_context(clean_docs, query, test_case):
    # Cross-doc/similar-topic: kailangan source coverage muna bago score.
    if should_use_source_diverse_context(test_case):
        return select_final_context_docs(
            reranked_docs=clean_docs,
            question=query,
            top_n=RERANK_TOP_N,
            max_chars=MAX_CONTEXT_CHARS,
            max_per_source=CANDIDATE_MAX_PER_SOURCE,
        )

    # Normal/direct/paraphrased questions: huwag pilitin ang source diversity.
    return limit_context_docs(
        docs=clean_docs[:RERANK_TOP_N],
        max_chars=MAX_CONTEXT_CHARS,
    )


def retrieve_final_context(test_case, pipeline, show_scores=False):
    # Multi-query hybrid search -> merge candidates -> rerank once -> quality filter -> context selection.
    query = test_case["query"]
    timings = {}

    hybrid_result = timed_step(
        "Hybrid retrieval",
        lambda: run_multi_query_hybrid_search(
            query=query,
            pipeline=pipeline,
        ),
        timings,
    )

    hybrid_docs = hybrid_result["hybrid_docs"]
    retrieval_queries = hybrid_result["retrieval_queries"]

    # Important: rerank once using the original test query, not every expanded retrieval query.
    # Kapag candidate balancing is enabled, rerank more than final top N then balance down.
    rerank_top_n = RERANK_TOP_N

    if ENABLE_CANDIDATE_BALANCING:
        rerank_top_n = max(RERANK_TOP_N, BALANCED_RERANK_TOP_N)

    reranked_docs = timed_step(
        "Rerank documents",
        lambda: rerank_documents(
            query=query,
            documents=hybrid_docs,
            reranker=pipeline["reranker"],
            top_n=rerank_top_n,
            show_scores=show_scores,
        ),
        timings,
    )

    clean_docs = timed_step(
        "Filter low-quality docs",
        lambda: filter_low_quality_docs(
            docs=reranked_docs,
            min_score=MIN_QUALITY_SCORE,
        ),
        timings,
    )

    final_docs = timed_step(
        "Select final context",
        lambda: select_test_final_context(
            clean_docs=clean_docs,
            query=query,
            test_case=test_case,
        ),
        timings,
    )

    return {
        "retrieval_queries": retrieval_queries,
        "hybrid_docs": hybrid_docs,
        "reranked_docs": reranked_docs,
        "final_docs": final_docs,
        "timings": timings,
    }


def get_context_text(docs):
    # Pagsamahin ang final context text para sa keyword check.
    return "\n".join(doc.page_content or "" for doc in docs)


def match_source_groups(docs, expected_source_groups):
    # I-check kung may match ang bawat required source group.
    matched_groups = []
    missing_groups = []

    for index, source_group in enumerate(expected_source_groups, start=1):
        matched_sources = match_sources(docs, source_group)

        if matched_sources:
            matched_groups.append({
                "group": index,
                "matched_sources": matched_sources,
            })
        else:
            missing_groups.append({
                "group": index,
                "expected_sources": source_group,
            })

    return matched_groups, missing_groups


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
    expected_source_groups = test_case.get("expected_source_groups", [])
    must_have = test_case.get("must_have", [])
    must_not_have = test_case.get("must_not_have", [])
    min_keyword_matches = test_case.get("min_keyword_matches", MIN_KEYWORD_MATCHES)

    matched_sources = match_sources(final_docs, expected_sources)
    matched_source_groups, missing_source_groups = match_source_groups(
        final_docs,
        expected_source_groups,
    )
    context_text = get_context_text(final_docs)
    matched_keywords = match_keywords(context_text, must_have)
    matched_forbidden = match_keywords(context_text, must_not_have)
    top_source_rank = get_top_source_rank(final_docs, expected_sources)

    if test_type == "negative":
        source_ok = True
        keyword_ok = len(matched_forbidden) == 0
    else:
        if expected_source_groups:
            source_ok = len(missing_source_groups) == 0
        else:
            source_ok = bool(matched_sources) if expected_sources else True

        keyword_ok = len(matched_keywords) >= min_keyword_matches if must_have else True
        keyword_ok = keyword_ok and len(matched_forbidden) == 0

    return {
        "passed": source_ok and keyword_ok,
        "source_ok": source_ok,
        "keyword_ok": keyword_ok,
        "matched_sources": matched_sources,
        "matched_source_groups": matched_source_groups,
        "missing_source_groups": missing_source_groups,
        "matched_keywords": matched_keywords,
        "matched_forbidden": matched_forbidden,
        "min_keyword_matches": min_keyword_matches,
        "top_source_rank": top_source_rank,
    }


def run_test_case(test_case, pipeline, show_scores=False):
    # Patakbuhin ang isang retrieval test case.
    start_time = time.perf_counter()
    retrieval = retrieve_final_context(
        test_case=test_case,
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


def add_failed_tests_by_group(lines, failed):
    # Ipakita ang failed tests na grouped by baseline/stress.
    if not failed:
        return

    lines.extend(["", "=" * 80, "FAILED TESTS BY GROUP", "=" * 80])
    grouped_failed = group_result_items(failed)

    for group in GROUP_ORDER:
        group_failed = grouped_failed.get(group, [])
        if not group_failed:
            continue

        lines.extend(["", f"{group.upper()} FAILED TESTS:"])
        for item in group_failed:
            test_case = item["test_case"]
            evaluation = item["result"]["evaluation"]
            lines.append(
                f"- {test_case['id']} | {test_case['query']} | "
                f"source_ok={evaluation['source_ok']} | "
                f"keyword_ok={evaluation['keyword_ok']} | "
                f"top_rank={evaluation['top_source_rank']}"
            )


def add_single_detail_to_report(lines, item):
    # Idagdag ang detailed result ng isang retrieval test case.
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
        f"Group             : {get_test_group(test_case)}",
        f"Type              : {test_case.get('type')}",
        f"Query             : {test_case.get('query')}",
        f"Source OK         : {evaluation['source_ok']}",
        f"Keyword OK        : {evaluation['keyword_ok']}",
        f"Top source rank   : {evaluation['top_source_rank']}",
        f"Matched sources   : {evaluation['matched_sources']}",
        f"Matched groups    : {evaluation['matched_source_groups']}",
        f"Missing groups    : {evaluation['missing_source_groups']}",
        f"Matched keywords  : {evaluation['matched_keywords']}",
        f"Required keywords : {evaluation['min_keyword_matches']}",
        f"Matched forbidden : {evaluation['matched_forbidden']}",
        f"Retrieval queries : {result.get('retrieval_queries', [test_case.get('query')])}",
        f"Hybrid candidates : {len(result['hybrid_docs'])}",
        f"Reranked docs     : {len(result['reranked_docs'])}",
        f"Final context docs: {len(final_docs)}",
        f"Query time        : {format_seconds(result['query_time'])}",
        f"Bottleneck        : {bottleneck_label} - {format_seconds(bottleneck_time)}",
    ])
    add_doc_details(lines, final_docs)


def add_grouped_details_to_report(lines, results):
    # Ihiwalay ang detailed results by baseline/stress.
    grouped = group_result_items(results)

    for group in GROUP_ORDER:
        group_items = grouped.get(group, [])
        if not group_items:
            continue

        lines.extend(["", "=" * 80, f"{group.upper()} DETAILED RESULTS", "=" * 80])
        for item in group_items:
            add_single_detail_to_report(lines, item)


def build_report(results, setup_timings, pipeline, total_time):
    # Gumawa ng full text report.
    lines = []
    passed = [item for item in results if get_item_passed(item)]
    failed = [item for item in results if not get_item_passed(item)]
    score = score_percent(len(passed), len(results))

    lines.extend([
        "=" * 80,
        "FINAL RETRIEVAL TEST REPORT",
        "=" * 80,
        f"Total tests : {len(results)}",
        f"Passed      : {len(passed)}",
        f"Failed      : {len(failed)}",
        f"Score       : {score:.2f}%",
        f"Total time  : {format_seconds(total_time)}",
    ])

    add_score_breakdown(lines, results)

    lines.extend([
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

    add_failed_tests_by_group(lines, failed)
    add_grouped_details_to_report(lines, results)

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
    for group in GROUP_ORDER:
        group_count = sum(1 for test_case in selected_tests if get_test_group(test_case) == group)
        if group_count:
            print(f"{group.title()} tests : {group_count}")

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

    grouped = group_result_items(results)
    for group in GROUP_ORDER:
        group_items = grouped.get(group, [])
        if not group_items:
            continue

        group_passed = sum(1 for item in group_items if get_item_passed(item))
        group_score = score_percent(group_passed, len(group_items))
        print(f"{group.title():<10}: {group_passed}/{len(group_items)} - {group_score:.2f}%")

    print(f"Time   : {format_seconds(total_time)}")
    print(f"Report : {output_path}")


if __name__ == "__main__":
    main()
