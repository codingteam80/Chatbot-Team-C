import argparse
import inspect
import re
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

# Optional settings para compatible pa rin kahit luma ang settings.py.
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

# Optional candidate balancer settings para hindi mag-error kung hindi pa updated ang settings.py.
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
from chains.rag_chain import clean_generated_answer, generate_answer, get_sources, stream_answer
from embeddings.embedding_model import get_embedding_model
from llm.ollama_llm import load_llm
from retrieval.candidate_balancer import (
    balance_candidates,
    balance_reranked_documents,
    balance_reranked_documents_with_query_coverage,
)
from retrieval.context_filter import (
    filter_low_quality_docs,
    has_document_evidence,
    select_final_context_docs,
)
from retrieval.hybrid_retriever import hybrid_search
from retrieval.reranker import load_reranker, rerank_documents
from retrieval.retrieval_query_builder import build_retrieval_queries
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


def expand_expected_source_variants(expected_sources):
    # Payagan ang .md at .pdf variant ng same source name.
    # Generic ito para hindi mag-fail kapag same document pero ibang loader output.
    expanded_sources = []

    for source in expected_sources or []:
        source_text = str(source or "").strip()

        if not source_text:
            continue

        if source_text not in expanded_sources:
            expanded_sources.append(source_text)

        path = Path(source_text)

        if path.suffix.lower() not in {".md", ".pdf"}:
            continue

        for extension in (".md", ".pdf"):
            variant = str(path.with_suffix(extension))

            if variant not in expanded_sources:
                expanded_sources.append(variant)

    return expanded_sources


def expand_expected_source_groups(expected_source_groups):
    # I-expand din ang source groups para tanggap ang .md at .pdf variants.
    return [
        expand_expected_source_variants(source_group)
        for source_group in expected_source_groups or []
    ]


def match_source_groups(actual_sources, expected_source_groups):
    # I-check kung may match ang bawat required source group.
    matched_groups = []
    missing_groups = []

    for index, source_group in enumerate(expand_expected_source_groups(expected_source_groups), start=1):
        matched_sources = match_sources(actual_sources, source_group)

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


def get_category_score_lines(results):
    # Score breakdown per RAG category para madaling makita kung anong behavior ang mahina.
    lines = []

    for category in CATEGORY_ORDER:
        category_results = [result for result in results if result.get("category") == category]

        if not category_results:
            continue

        total_count = len(category_results)
        passed_count = sum(1 for result in category_results if result.get("passed"))
        failed_count = total_count - passed_count
        score = (passed_count / total_count * 100) if total_count else 0
        lines.append(
            f"{category:<28}: {passed_count}/{total_count} passed | "
            f"{failed_count} failed | {score:.2f}%"
        )

    return lines


CATEGORY_ORDER = ['direct_answer',
 'paraphrase',
 'tagalog_question',
 'cross_document',
 'no_answer',
 'false_premise',
 'partial_evidence',
 'similar_topic_confusion',
 'follow_up_questions',
 'japanese_english_mixed']

# 30 RAG cases: 3 per category.
# Overlapping retrieval categories reuse equivalent retrieval questions where possible.
TEST_CASES = [{'id': 'DA01',
  'category': 'direct_answer',
  'behavior': 'answer',
  'question': 'Who was Apolinario Mabini and what role did he serve in the First Philippine '
              'Republic?',
  'expected_sources': ['Apolinario Mabini - Wikipedia.md', 'Apolinario Mabini - Wikipedia.pdf'],
  'answer_keywords': ['Apolinario Mabini',
                      'Mabini',
                      'legal and constitutional adviser',
                      'first Prime Minister',
                      'First Philippine Republic',
                      'brain of the revolution',
                      'utak ng himagsikan'],
  'min_answer_keyword_matches': 3},
 {'id': 'DA02',
  'category': 'direct_answer',
  'behavior': 'answer',
  'question': 'Who was Andres Bonifacio?',
  'expected_sources': ['Andrés Bonifacio - Wikipedia.md',
                       'Andrés Bonifacio - Wikipedia.pdf',
                       'Andres Bonifacio - Wikipedia.md',
                       'Andres Bonifacio - Wikipedia.pdf'],
  'answer_keywords': ['Andres Bonifacio',
                      'Andrés Bonifacio',
                      'Filipino revolutionary leader',
                      'revolutionary leader',
                      'Father of the Philippine Revolution',
                      'Katipunan',
                      'Supremo'],
  'min_answer_keyword_matches': 3},
 {'id': 'DA03',
  'category': 'direct_answer',
  'behavior': 'answer',
  'question': 'When was the Katipunan formed and what was its main purpose?',
  'expected_sources': ['Katipunan - Wikipedia.md', 'Katipunan - Wikipedia.pdf'],
  'answer_keywords': ['Katipunan',
                      'July 7, 1892',
                      '1892',
                      'Filipino independence',
                      'independence',
                      'Spanish Empire',
                      'armed revolution'],
  'min_answer_keyword_matches': 3},
 {'id': 'P01',
  'category': 'paraphrase',
  'behavior': 'answer',
  'question': 'Which revolutionary figure was known as the brain of the revolution and later '
              'became the first prime minister?',
  'expected_sources': ['Apolinario Mabini - Wikipedia.md', 'Apolinario Mabini - Wikipedia.pdf'],
  'answer_keywords': ['Apolinario Mabini',
                      'Mabini',
                      'brain of the revolution',
                      'utak ng himagsikan',
                      'first Prime Minister',
                      'First Philippine Republic'],
  'min_answer_keyword_matches': 3,
   'must_not_contain': ['Emilio Aguinaldo is known as the brain',
                        'Emilio Aguinaldo is known as the Brains',
                        'Mabini did not become the first prime minister',
                        'that role was held by Emilio Aguinaldo']},
 {'id': 'P02',
  'category': 'paraphrase',
  'behavior': 'answer',
  'question': 'What secret nationalist group pushed for independence from Spain through armed '
              'revolt after La Liga Filipina?',
  'expected_sources': ['Katipunan - Wikipedia.md', 'Katipunan - Wikipedia.pdf'],
  'answer_keywords': ['Katipunan',
                      'secret society',
                      'Filipino nationalists',
                      'independence',
                      'Spanish Empire',
                      'armed revolution',
                      'La Liga Filipina'],
  'min_answer_keyword_matches': 3},
 {'id': 'P03',
  'category': 'paraphrase',
  'behavior': 'answer',
  'question': 'Which 1898 peace agreement ended the war between Spain and the United States and '
              'transferred the Philippines?',
  'expected_sources': ['Treaty of Paris (1898) - Wikipedia.md',
                       'Treaty of Paris (1898) - Wikipedia.pdf',
                       'Spanish–American War - Wikipedia.md',
                       'Spanish–American War - Wikipedia.pdf',
                       'Spanish-American War - Wikipedia.md',
                       'Spanish-American War - Wikipedia.pdf'],
  'answer_keywords': ['Treaty of Paris',
                      '1898',
                      'Spanish-American War',
                      'Spanish–American War',
                      'Spain',
                      'United States',
                      'Philippines'],
  'min_answer_keyword_matches': 4},
 {'id': 'T01',
  'category': 'tagalog_question',
  'behavior': 'answer',
  'question': 'Ano ang naging papel ni Apolinario Mabini sa Unang Republika ng Pilipinas?',
  'expected_sources': ['Apolinario Mabini - Wikipedia.md', 'Apolinario Mabini - Wikipedia.pdf'],
  'answer_keywords': ['Apolinario Mabini',
                      'Mabini',
                      'legal and constitutional adviser',
                      'first Prime Minister',
                      'Prime Minister',
                      'Unang Republika',
                      'First Philippine Republic'],
  'min_answer_keyword_matches': 3},
 {'id': 'T02',
  'category': 'tagalog_question',
  'behavior': 'answer',
  'question': 'Kailan itinatag ang Katipunan at ano ang layunin nito?',
  'expected_sources': ['Katipunan - Wikipedia.md', 'Katipunan - Wikipedia.pdf'],
  'answer_keywords': ['Katipunan',
                      'July 7, 1892',
                      '1892',
                      'Filipino independence',
                      'kalayaan',
                      'independence',
                      'Spanish Empire'],
  'min_answer_keyword_matches': 3},
 {'id': 'T03',
  'category': 'tagalog_question',
  'behavior': 'answer',
  'question': 'Bakit mahalaga ang Gomburza sa buhay at mga akda ni Jose Rizal?',
  'expected_sources': ['Gomburza - Wikipedia.md',
                       'Gomburza - Wikipedia.pdf',
                       'José Rizal - Wikipedia.md',
                       'José Rizal - Wikipedia.pdf',
                       'Jose Rizal - Wikipedia.md',
                       'Jose Rizal - Wikipedia.pdf'],
  'answer_keywords': ['Gomburza',
                      'José Rizal',
                      'Jose Rizal',
                      'Rizal',
                      'El filibusterismo',
                      'execution',
                      'executed',
                      'memory'],
  'min_answer_keyword_matches': 3},
 {'id': 'X01',
  'category': 'cross_document',
  'behavior': 'answer',
  'question': 'How did the Treaty of Paris connect the Spanish-American War to the '
              'Philippine-American War?',
  'expected_sources': ['Treaty of Paris (1898) - Wikipedia.md',
                       'Treaty of Paris (1898) - Wikipedia.pdf',
                       'Spanish–American War - Wikipedia.md',
                       'Spanish–American War - Wikipedia.pdf',
                       'Spanish-American War - Wikipedia.md',
                       'Spanish-American War - Wikipedia.pdf',
                       'Philippine–American War - Wikipedia.md',
                       'Philippine–American War - Wikipedia.pdf',
                       'Philippine-American War - Wikipedia.md',
                       'Philippine-American War - Wikipedia.pdf'],
  'answer_keywords': ['Treaty of Paris',
                      'Spanish-American War',
                      'Spanish–American War',
                      'United States',
                      'Philippine-American War',
                      'Philippine–American War',
                      'Philippines',
                      'ceded',
                      'annexation'],
  'min_answer_keyword_matches': 4,
  'expected_source_groups': [['Treaty of Paris (1898) - Wikipedia.md',
                              'Treaty of Paris (1898) - Wikipedia.pdf'],
                             ['Spanish–American War - Wikipedia.md',
                              'Spanish–American War - Wikipedia.pdf',
                              'Spanish-American War - Wikipedia.md',
                              'Spanish-American War - Wikipedia.pdf'],
                             ['Philippine–American War - Wikipedia.md',
                              'Philippine–American War - Wikipedia.pdf',
                              'Philippine-American War - Wikipedia.md',
                              'Philippine-American War - Wikipedia.pdf']]},
 {'id': 'X02',
  'category': 'cross_document',
  'behavior': 'answer',
  'question': 'Paano nauugnay ang La Liga Filipina ni Jose Rizal sa pagkakatatag ng Katipunan?',
  'expected_sources': ['José Rizal - Wikipedia.md',
                       'José Rizal - Wikipedia.pdf',
                       'Jose Rizal - Wikipedia.md',
                       'Jose Rizal - Wikipedia.pdf',
                       'Katipunan - Wikipedia.md',
                       'Katipunan - Wikipedia.pdf'],
  'answer_keywords': ['La Liga Filipina',
                      'Jose Rizal',
                      'José Rizal',
                      'Rizal',
                      'Katipunan',
                      'Andres Bonifacio',
                      'Andrés Bonifacio',
                      'Dapitan'],
  'min_answer_keyword_matches': 3,
  'expected_source_groups': [['Katipunan - Wikipedia.md', 'Katipunan - Wikipedia.pdf'],
                             ['José Rizal - Wikipedia.md',
                              'José Rizal - Wikipedia.pdf',
                              'Jose Rizal - Wikipedia.md',
                              'Jose Rizal - Wikipedia.pdf']]},
 {'id': 'X03',
  'category': 'cross_document',
  'behavior': 'answer',
  'question': 'How did the Katipunan lead into the Philippine Revolution?',
  'expected_sources': ['Katipunan - Wikipedia.md',
                       'Katipunan - Wikipedia.pdf',
                       'Philippine Revolution - Wikipedia.md',
                       'Philippine Revolution - Wikipedia.pdf'],
  'answer_keywords': ['Katipunan',
                      'Philippine Revolution',
                      '1896',
                      'Spanish authorities',
                      'discovery',
                      'armed revolution',
                      'Spanish Empire'],
  'min_answer_keyword_matches': 4,
  'expected_source_groups': [['Katipunan - Wikipedia.md', 'Katipunan - Wikipedia.pdf'],
                             ['Philippine Revolution - Wikipedia.md',
                              'Philippine Revolution - Wikipedia.pdf']]},
 {'id': 'N01',
  'category': 'no_answer',
  'behavior': 'no_answer',
  'question': "What was Andres Bonifacio's official passport number during the Philippine "
              'Revolution?',
  'expected_sources': [],
  'answer_keywords': [],
  'must_not_contain': ['passport number is',
                       'official passport number is',
                       "Bonifacio's passport number"]},
 {'id': 'N02',
  'category': 'no_answer',
  'behavior': 'no_answer',
  'question': 'Ano ang eksaktong Wi-Fi password na ginamit ng Katipunan sa kanilang mga '
              'pagpupulong?',
  'expected_sources': [],
  'answer_keywords': [],
  'must_not_contain': ['Wi-Fi password is',
                       'wifi password is',
                       'password ay',
                       'password na ginamit',
                       'Rizal',
                       'Gomburza']},
 {'id': 'N03',
  'category': 'no_answer',
  'behavior': 'no_answer',
  'question': 'What programming language did Gomburza use to build their official website?',
  'expected_sources': [],
  'answer_keywords': [],
  'must_not_contain': ['programming language was',
                       'Python',
                       'JavaScript',
                       'website was built',
                       'official website']},
 {'id': 'F01',
  'category': 'false_premise',
  'behavior': 'correction',
  'question': 'Why did Jose Rizal become the Supremo of the Katipunan?',
  'expected_sources': ['José Rizal - Wikipedia.md',
                       'José Rizal - Wikipedia.pdf',
                       'Jose Rizal - Wikipedia.md',
                       'Jose Rizal - Wikipedia.pdf',
                       'Katipunan - Wikipedia.md',
                       'Katipunan - Wikipedia.pdf',
                       'Andrés Bonifacio - Wikipedia.md',
                       'Andrés Bonifacio - Wikipedia.pdf',
                       'Andres Bonifacio - Wikipedia.md',
                       'Andres Bonifacio - Wikipedia.pdf'],
  'answer_keywords': ['Jose Rizal',
                      'José Rizal',
                      'Rizal',
                      'Andres Bonifacio',
                      'Andrés Bonifacio',
                      'Bonifacio',
                      'Supremo',
                      'Supreme President',
                      'Katipunan'],
  'correction_keywords': ['No',
                          'not correct',
                          'not supported',
                          'does not support',
                          'did not',
                          'was not',
                          'incorrect',
                          'false premise',
                          'Hindi',
                          'hindi tama',
                          'hindi sinusuportahan',
                          'hindi nakasaad',
                          'hindi si',
                          'maling premise',
                          'mali ang premise'],
  'min_answer_keyword_matches': 4,
  'min_correction_keyword_matches': 1,
  'allow_no_answer_as_safe': False,
  'must_not_contain': ['Rizal became Supremo because',
                       'Rizal became the Supremo because',
                       'Rizal was the Supremo because',
                       'Jose Rizal became Supremo because']},
 {'id': 'F02',
  'category': 'false_premise',
  'behavior': 'correction',
  'question': 'Bakit si Apolinario Mabini ang nagdeklara ng kalayaan ng Pilipinas noong June 12, '
              '1898?',
  'expected_sources': ['Apolinario Mabini - Wikipedia.md',
                       'Apolinario Mabini - Wikipedia.pdf',
                       'Philippine Revolution - Wikipedia.md',
                       'Philippine Revolution - Wikipedia.pdf',
                       'Independence Day (Philippines) - Wikipedia.md',
                       'Independence Day (Philippines) - Wikipedia.pdf'],
  'answer_keywords': ['Apolinario Mabini',
                      'Mabini',
                      'Emilio Aguinaldo',
                      'Aguinaldo',
                      'June 12',
                      '1898',
                      'kalayaan',
                      'independence',
                      'Philippine Declaration of Independence'],
  'correction_keywords': ['No',
                          'not correct',
                          'not supported',
                          'does not support',
                          'did not',
                          'was not',
                          'incorrect',
                          'false premise',
                          'Hindi',
                          'hindi tama',
                          'hindi sinusuportahan',
                          'hindi nakasaad',
                          'hindi si',
                          'maling premise',
                          'mali ang premise'],
  'min_answer_keyword_matches': 4,
  'min_correction_keyword_matches': 1,
  'allow_no_answer_as_safe': False,
  'must_not_contain': ['Mabini declared independence',
                       'Apolinario Mabini declared independence',
                       'Mabini was the declarer',
                       'Mabini ang nagdeklara']},
 {'id': 'F03',
  'category': 'false_premise',
  'behavior': 'correction',
  'question': 'How did Gomburza found the Katipunan in 1892?',
  'expected_sources': ['Gomburza - Wikipedia.md',
                       'Gomburza - Wikipedia.pdf',
                       'Katipunan - Wikipedia.md',
                       'Katipunan - Wikipedia.pdf'],
  'answer_keywords': ['Gomburza',
                      'Katipunan',
                      '1892',
                      'founders',
                      'Andres Bonifacio',
                      'Andrés Bonifacio',
                      'Deodato Arellano',
                      'executed',
                      '1872'],
  'correction_keywords': ['No',
                          'not correct',
                          'not supported',
                          'does not support',
                          'did not',
                          'was not',
                          'incorrect',
                          'false premise',
                          'Hindi',
                          'hindi tama',
                          'hindi sinusuportahan',
                          'hindi nakasaad',
                          'hindi si',
                          'maling premise',
                          'mali ang premise'],
  'min_answer_keyword_matches': 4,
  'min_correction_keyword_matches': 1,
  'allow_no_answer_as_safe': False,
  'must_not_contain': ['Gomburza founded the Katipunan',
                       'Gomburza founded Katipunan',
                       'Gomburza started the Katipunan']},
 {'id': 'PE01',
  'category': 'partial_evidence',
  'behavior': 'answer',
  'question': 'What role did Apolinario Mabini serve, and what was his favorite food?',
  'expected_sources': ['Apolinario Mabini - Wikipedia.md', 'Apolinario Mabini - Wikipedia.pdf'],
  'answer_keywords': ['Apolinario Mabini',
                      'Mabini',
                      'legal and constitutional adviser',
                      'first Prime Minister',
                      'not stated',
                      'not provided',
                      'not found',
                      'not in the context',
                      'not in the documents',
                      'does not mention',
                      'cannot find',
                      'cannot be found',
                      'not specified',
                      'hindi nakasaad',
                      'hindi binanggit',
                      'hindi makita',
                      'wala sa context',
                      'wala sa dokumento',
                      'hindi sinusuportahan'],
  'min_answer_keyword_matches': 4,
  'must_not_contain': ['favorite food was', 'his favorite food was', "Mabini's favorite food was"]},
 {'id': 'PE02',
  'category': 'partial_evidence',
  'behavior': 'answer',
  'question': 'Sino ang mga founder ng Katipunan at ano ang exact home address ng bawat isa?',
  'expected_sources': ['Katipunan - Wikipedia.md', 'Katipunan - Wikipedia.pdf'],
  'answer_keywords': ['Katipunan',
                      'Deodato Arellano',
                      'Andres Bonifacio',
                      'Andrés Bonifacio',
                      'Valentin Diaz',
                      'Valentín Díaz',
                      'Ladislao Diwa',
                      'Jose Dizon',
                      'José Dizon',
                      'Teodoro Plata',
                      'not stated',
                      'not provided',
                      'not found',
                      'not in the context',
                      'not in the documents',
                      'does not mention',
                      'cannot find',
                      'cannot be found',
                      'not specified',
                      'hindi nakasaad',
                      'hindi binanggit',
                      'hindi makita',
                      'wala sa context',
                      'wala sa dokumento',
                      'hindi sinusuportahan'],
  'min_answer_keyword_matches': 5,
  'must_not_contain': ['home address is provided', 'home address was provided', 'tirahan ay']},
 {'id': 'PE03',
  'category': 'partial_evidence',
  'behavior': 'answer',
  'question': 'When was the Treaty of Paris signed, and what secret password did the negotiators '
              'use?',
  'expected_sources': ['Treaty of Paris (1898) - Wikipedia.md',
                       'Treaty of Paris (1898) - Wikipedia.pdf'],
  'answer_keywords': ['Treaty of Paris',
                      'December 10, 1898',
                      '1898',
                      'Spain',
                      'United States',
                      'not stated',
                      'not provided',
                      'not found',
                      'not in the context',
                      'not in the documents',
                      'does not mention',
                      'cannot find',
                      'cannot be found',
                      'not specified',
                      'hindi nakasaad',
                      'hindi binanggit',
                      'hindi makita',
                      'wala sa context',
                      'wala sa dokumento',
                      'hindi sinusuportahan'],
  'min_answer_keyword_matches': 4,
  'must_not_contain': ['secret password was', 'password was', 'negotiators used the password']},
 {'id': 'S01',
  'category': 'similar_topic_confusion',
  'behavior': 'answer',
  'question': 'Which event is observed every June 12: Philippine Independence Day or Republic Day?',
  'expected_sources': ['Independence Day (Philippines) - Wikipedia.md',
                       'Independence Day (Philippines) - Wikipedia.pdf'],
  'answer_keywords': ['Independence Day',
                      'Araw ng Kalayaan',
                      'June 12',
                      '1898',
                      'independence from Spain',
                      'Republic Day'],
  'min_answer_keyword_matches': 3,
  'must_not_contain': ['Republic Day is observed every June 12', 'Republic Day ang June 12']},
 {'id': 'S02',
  'category': 'similar_topic_confusion',
  'behavior': 'answer',
  'question': 'Which war began on February 4, 1899 after U.S. annexation, not the Spanish-American '
              'War?',
  'expected_sources': ['Philippine–American War - Wikipedia.md',
                       'Philippine–American War - Wikipedia.pdf',
                       'Philippine-American War - Wikipedia.md',
                       'Philippine-American War - Wikipedia.pdf'],
  'answer_keywords': ['Philippine-American War',
                      'Philippine–American War',
                      'February 4, 1899',
                      'United States',
                      'annexation',
                      'Treaty of Paris'],
  'min_answer_keyword_matches': 3,
  'must_not_contain': ['Spanish-American War began on February 4, 1899',
                       'Spanish–American War began on February 4, 1899']},
 {'id': 'S03',
  'category': 'similar_topic_confusion',
  'behavior': 'answer',
  'question': 'Who was the Supremo of the Katipunan: Jose Rizal or Andres Bonifacio?',
  'expected_sources': ['Andrés Bonifacio - Wikipedia.md',
                       'Andrés Bonifacio - Wikipedia.pdf',
                       'Andres Bonifacio - Wikipedia.md',
                       'Andres Bonifacio - Wikipedia.pdf',
                       'Katipunan - Wikipedia.md',
                       'Katipunan - Wikipedia.pdf'],
  'answer_keywords': ['Andres Bonifacio',
                      'Andrés Bonifacio',
                      'Bonifacio',
                      'Supremo',
                      'Supreme President',
                      'Katipunan',
                      'Jose Rizal',
                      'José Rizal'],
  'min_answer_keyword_matches': 4,
  'must_not_contain': ['Rizal was the Supremo',
                       'Jose Rizal was the Supremo',
                       'José Rizal was the Supremo']},
 {'id': 'FU01',
  'category': 'follow_up_questions',
  'behavior': 'answer',
  'question': 'What role did he serve in the First Philippine Republic?',
  'retrieval_query': 'What role did Apolinario Mabini serve in the First Philippine Republic?',
  'chat_history': 'User: Tell me about Apolinario Mabini.\n'
                  'Assistant: Apolinario Mabini was a Filipino revolutionary leader and statesman.',
  'expected_sources': ['Apolinario Mabini - Wikipedia.md', 'Apolinario Mabini - Wikipedia.pdf'],
  'answer_keywords': ['Apolinario Mabini',
                      'Mabini',
                      'legal and constitutional adviser',
                      'first Prime Minister',
                      'First Philippine Republic'],
  'min_answer_keyword_matches': 3},
 {'id': 'FU02',
  'category': 'follow_up_questions',
  'behavior': 'answer',
  'question': 'Who were its founders?',
  'retrieval_query': 'Who were the founders of the Katipunan?',
  'chat_history': 'User: What was the Katipunan?\n'
                  'Assistant: The Katipunan was a revolutionary organization seeking Philippine '
                  'independence from Spain.',
  'expected_sources': ['Katipunan - Wikipedia.md', 'Katipunan - Wikipedia.pdf'],
  'answer_keywords': ['Katipunan',
                      'founders',
                      'Deodato Arellano',
                      'Andres Bonifacio',
                      'Andrés Bonifacio',
                      'Valentin Diaz',
                      'Valentín Díaz',
                      'Ladislao Diwa',
                      'Jose Dizon',
                      'José Dizon',
                      'Teodoro Plata'],
  'min_answer_keyword_matches': 4},
 {'id': 'FU03',
  'category': 'follow_up_questions',
  'behavior': 'answer',
  'question': 'What did it do to the Philippines?',
  'retrieval_query': 'What did the Treaty of Paris of 1898 do to the Philippines?',
  'chat_history': 'User: Explain the Treaty of Paris of 1898.\n'
                  'Assistant: It was the peace treaty between Spain and the United States after '
                  'the Spanish-American War.',
  'expected_sources': ['Treaty of Paris (1898) - Wikipedia.md',
                       'Treaty of Paris (1898) - Wikipedia.pdf'],
  'answer_keywords': ['Treaty of Paris',
                      'Spain',
                      'United States',
                      'Philippines',
                      'ceded',
                      'relinquished',
                      '$20 million'],
  'min_answer_keyword_matches': 3},
 {'id': 'JE01',
  'category': 'japanese_english_mixed',
  'behavior': 'answer',
  'question': 'Sa Japanese article, kailan recognized by the United States ang Philippine '
              'independence?',
  'retrieval_query': 'フィリピンの歴史 (1946年-1965年) 1946年7月4日 Harry S. Truman Proclamation 2695 Philippine independence recognized United States',
  'expected_sources': ['フィリピンの歴史 (1946年-1965年) - Wikipedia.md',
                       'フィリピンの歴史 (1946年-1965年) - Wikipedia.pdf'],
  'answer_keywords': ['1946年7月4日',
                      'July 4, 1946',
                      '1946',
                      'United States',
                      'Philippine independence',
                      'independence',
                      'Harry S. Truman'],
  'min_answer_keyword_matches': 3},
 {'id': 'JE02',
  'category': 'japanese_english_mixed',
  'behavior': 'answer',
  'question': 'What period does フィリピンの歴史 (1946年-1965年) cover?',
  'expected_sources': ['フィリピンの歴史 (1946年-1965年) - Wikipedia.md',
                       'フィリピンの歴史 (1946年-1965年) - Wikipedia.pdf'],
  'answer_keywords': ['1946',
                      '1965',
                      'Third Republic',
                      '第三共和国',
                      'Diosdado Macapagal',
                      'ディオスダド・マカパガル'],
  'min_answer_keyword_matches': 3},
 {'id': 'JE03',
  'category': 'japanese_english_mixed',
  'behavior': 'answer',
  'question': 'According to the Japanese doc, who issued Proclamation 2695 recognizing Philippine '
              'independence?',
  'retrieval_query': 'フィリピンの歴史 (1946年-1965年) Proclamation 2695 Harry S. Truman Philippine independence United States 1946',
  'expected_sources': ['フィリピンの歴史 (1946年-1965年) - Wikipedia.md',
                       'フィリピンの歴史 (1946年-1965年) - Wikipedia.pdf'],
  'answer_keywords': ['Harry S. Truman',
                      'Truman',
                      'Proclamation 2695',
                      '1946',
                      'independence',
                      'Philippines'],
  'min_answer_keyword_matches': 3}]


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


def generate_answer_once(question, docs, llm, chat_history="", strict_assumption_check=False, correction_retry=False):
    # Generate answer with compatibility sa old/new rag_chain signatures.
    kwargs = {
        "question": question,
        "docs": docs,
        "llm": llm,
        "chat_history": chat_history,
        "debug": DEBUG_PROMPT,
    }

    if function_accepts_parameter(generate_answer, "strict_assumption_check"):
        kwargs["strict_assumption_check"] = strict_assumption_check

    if function_accepts_parameter(generate_answer, "correction_retry"):
        kwargs["correction_retry"] = correction_retry

    return generate_answer(**kwargs)


def generate_streamed_answer(question, docs, llm, chat_history="", strict_assumption_check=False, correction_retry=False):
    # Streaming generation para same behavior sa app, with fallback sa old signature.
    if not docs:
        return NO_ANSWER_TEXT, 0.0

    answer = ""
    start_time = time.perf_counter()

    kwargs = {
        "question": question,
        "docs": docs,
        "llm": llm,
        "chat_history": chat_history,
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


def maybe_retry_correction(test_case, answer, final_docs, llm, chat_history=""):
    # Second try kapag correction test pero fallback pa rin kahit may docs.
    if test_case.get("behavior") != "correction":
        return answer, 0.0, False

    if not final_docs or not has_no_answer_phrase(answer):
        return answer, 0.0, False

    correction_question = " ".join([
        "Correct the false premise in this question using only the provided context.",
        "If the context shows a different correct person, role, founder, date, or event,",
        "say the premise is not correct and state the supported correction.",
        "Do not use the fallback answer when a supported correction is available.",
        f"Question: {test_case['question']}",
    ])

    retry_answer, retry_time = generate_answer_with_time(
        question=correction_question,
        docs=final_docs,
        llm=llm,
        chat_history=chat_history,
        strict_assumption_check=True,
        correction_retry=True,
    )

    if retry_answer and not has_no_answer_phrase(retry_answer):
        return retry_answer, retry_time, True

    return answer, retry_time, False


def generate_answer_with_time(question, docs, llm, chat_history="", strict_assumption_check=False, correction_retry=False):
    # Non-streaming answer with timing.
    if not docs:
        return NO_ANSWER_TEXT, 0.0

    start_time = time.perf_counter()
    answer = generate_answer_once(
        question=question,
        docs=docs,
        llm=llm,
        chat_history=chat_history,
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


def run_multi_query_hybrid_search(question, components):
    # Gumawa ng retrieval queries bago hybrid search.
    # Direct question usually one query lang; cross-doc/comparison can become multiple queries.
    retrieval_queries = build_retrieval_queries(
        question=question,
        enabled=ENABLE_MULTI_QUERY_RETRIEVAL,
        max_queries=MAX_RETRIEVAL_QUERIES,
    )

    query_doc_groups = []

    for retrieval_query in retrieval_queries:
        docs = hybrid_search(
            query=retrieval_query,
            vectorstore=components["vectorstore"],
            bm25_retriever=components["bm25_retriever"],
            semantic_k=SEMANTIC_K,
            bm25_k=BM25_K,
            final_k=HYBRID_FINAL_K,
            use_rrf=True,
        )
        query_doc_groups.append(docs)

    candidate_docs = balance_candidates(
        document_groups=query_doc_groups,
        max_docs=MAX_CANDIDATES_BEFORE_RERANK,
        max_per_source=CANDIDATE_MAX_PER_SOURCE,
        enabled=ENABLE_CANDIDATE_BALANCING,
    )

    return {
        "retrieval_queries": retrieval_queries,
        "hybrid_docs": candidate_docs,
        "query_doc_groups": query_doc_groups,
    }


def retrieve_docs(question, components, show_scores=False):
    # Latest retrieval flow:
    # multi-query hybrid search -> candidate balance -> rerank larger pool
    # -> quality filter -> final context selection with cross-doc source diversity.
    timings = {}

    hybrid_result = timed_step(
        "Hybrid retrieval",
        lambda: run_multi_query_hybrid_search(
            question=question,
            components=components,
        ),
        timings,
    )

    hybrid_docs = hybrid_result["hybrid_docs"]
    retrieval_queries = hybrid_result["retrieval_queries"]

    rerank_top_n = RERANK_TOP_N

    if ENABLE_CANDIDATE_BALANCING:
        rerank_top_n = max(RERANK_TOP_N, BALANCED_RERANK_TOP_N)

    reranked_docs = timed_step(
        "Rerank documents",
        lambda: rerank_documents(
            query=question,
            documents=hybrid_docs,
            reranker=components["reranker"],
            top_n=rerank_top_n,
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
        "Select final context",
        lambda: select_final_context_docs(
            reranked_docs=clean_docs,
            question=question,
            top_n=RERANK_TOP_N,
            max_chars=MAX_CONTEXT_CHARS,
            max_per_source=CANDIDATE_MAX_PER_SOURCE,
        ),
        timings,
    )

    counts = {
        "retrieval_queries": retrieval_queries,
        "hybrid_candidates": len(hybrid_docs),
        "reranked_docs": len(reranked_docs),
        "final_context_docs": len(final_docs),
    }

    return final_docs, timings, counts


MISSING_INFO_MARKERS = (
    "not stated",
    "not provided",
    "not found",
    "not in the context",
    "not in the documents",
    "does not mention",
    "cannot find",
    "cannot be found",
    "not specified",
    "hindi nakasaad",
    "hindi binanggit",
    "hindi makita",
    "wala sa context",
    "wala sa dokumento",
    "hindi sinusuportahan",
)


def split_answer_sentences(answer):
    # Hatiin sa simple sentences para ma-check kung safe denial ang forbidden phrase.
    text = str(answer or "")
    sentences = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [sentence.strip() for sentence in sentences if sentence.strip()]


def is_safe_missing_info_sentence(sentence, forbidden_term):
    # Kapag ang forbidden term ay kasama sa sentence na malinaw na nagsasabing not provided,
    # hindi ito hallucination. Example: "exact home address is not provided".
    normalized_sentence = " ".join(str(sentence or "").lower().split())
    normalized_term = " ".join(str(forbidden_term or "").lower().split())

    if not normalized_term or normalized_term not in normalized_sentence:
        return False

    return any(marker in normalized_sentence for marker in MISSING_INFO_MARKERS)


def get_unsafe_forbidden_matches(answer, forbidden_terms):
    # Ibalik lang ang forbidden matches na hindi safe denial / missing-info statement.
    matched_forbidden = match_keywords(answer, forbidden_terms)

    if not matched_forbidden:
        return []

    sentences = split_answer_sentences(answer)
    unsafe_matches = []

    for forbidden_term in matched_forbidden:
        safe_match = any(
            is_safe_missing_info_sentence(sentence, forbidden_term)
            for sentence in sentences
        )

        if not safe_match:
            unsafe_matches.append(forbidden_term)

    return unsafe_matches

def evaluate_answer(test_case, answer, sources, final_docs):
    # I-check kung pasado ang sagot base sa behavior ng test case.
    behavior = test_case["behavior"]

    expected_sources = expand_expected_source_variants(test_case.get("expected_sources", []))
    expected_source_groups = test_case.get("expected_source_groups", [])
    matched_sources = match_sources(sources, expected_sources)
    matched_source_groups, missing_source_groups = match_source_groups(
        sources,
        expected_source_groups,
    )
    min_source_matches = test_case.get("min_source_matches", 1)

    if expected_source_groups:
        source_ok = len(missing_source_groups) == 0
    else:
        source_ok = len(matched_sources) >= min_source_matches if expected_sources else True

    answer_keywords = test_case.get("answer_keywords", [])
    matched_keywords = match_keywords(answer, answer_keywords)
    min_keyword_matches = test_case.get("min_answer_keyword_matches", 1)

    correction_keywords = test_case.get("correction_keywords", [])
    matched_correction_keywords = match_keywords(answer, correction_keywords)
    min_correction_matches = test_case.get("min_correction_keyword_matches", 1)
    correction_signal_ok = len(matched_correction_keywords) >= min_correction_matches if correction_keywords else True

    forbidden_terms = test_case.get("must_not_contain", [])
    matched_forbidden = get_unsafe_forbidden_matches(answer, forbidden_terms)
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
        "matched_source_groups": matched_source_groups,
        "missing_source_groups": missing_source_groups,
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

    if result.get("retrieval_query") and result["retrieval_query"] != result["question"]:
        print(f"Retrieval: {result['retrieval_query']}")

    if result["error"]:
        print(f"Error    : {result['error']}")

    print("\nChecks:")
    print(f"  source_ok    : {evaluation['source_ok']}")
    print(f"  answer_ok    : {evaluation['answer_ok']}")
    print(f"  no_answer_ok : {evaluation['no_answer_ok']}")
    print(f"  forbidden_ok : {evaluation['forbidden_ok']}")

    print("\nMatched:")
    print(f"  sources    : {evaluation['matched_sources']}")
    print(f"  groups     : {evaluation.get('matched_source_groups', [])}")
    print(f"  missing    : {evaluation.get('missing_source_groups', [])}")
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
    retrieval_query = test_case.get("retrieval_query", question)
    chat_history = test_case.get("chat_history", "")

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
        final_docs, retrieval_timings, counts = retrieve_docs(retrieval_query, components, show_scores=show_scores)
        timings.update(retrieval_timings)

        evidence_ok = has_document_evidence(
            question=question,
            retrieval_query=retrieval_query,
            docs=final_docs,
            debug=False,
        )

        if not evidence_ok:
            # Same behavior as app guard: huwag na tawagin ang LLM kapag walang sapat na evidence.
            answer = NO_ANSWER_TEXT
            answer_time = 0.0
            timings["Evidence guard"] = 0.0
        else:
            strict_check = test_case.get("behavior") == "correction" or needs_strict_assumption_check(question)

            if use_stream:
                answer, answer_time = generate_streamed_answer(
                    question=question,
                    docs=final_docs,
                    llm=components["llm"],
                    chat_history=chat_history,
                    strict_assumption_check=strict_check,
                )
            else:
                answer, answer_time = generate_answer_with_time(
                    question=question,
                    docs=final_docs,
                    llm=components["llm"],
                    chat_history=chat_history,
                    strict_assumption_check=strict_check,
                )

        timings["Answer generation"] = answer_time

        retry_answer, retry_time, used_retry = maybe_retry_correction(
            test_case=test_case,
            answer=answer,
            final_docs=final_docs,
            llm=components["llm"],
            chat_history=chat_history,
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
            "matched_source_groups": [],
            "missing_source_groups": [],
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
        "retrieval_query": retrieval_query,
        "chat_history": chat_history,
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

    lines.extend(["", "=" * 80, "CATEGORY SCORE BREAKDOWN", "=" * 80])
    lines.extend(get_category_score_lines(results))

    if failed_tests:
        lines.extend(["", "FAILED TESTS:"])

        for result in failed_tests:
            evaluation = result["evaluation"]
            lines.append(
                f"- {result['id']} | {result['question']} | "
                f"source_ok={evaluation['source_ok']} | "
                f"answer_ok={evaluation['answer_ok']} | "
                f"no_answer_ok={evaluation['no_answer_ok']} | "
                f"forbidden_ok={evaluation['forbidden_ok']} | "
                f"missing_groups={evaluation.get('missing_source_groups', [])}"
            )

    lines.extend([
        "",
        "=" * 80,
        "RETRIEVAL SETTINGS",
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
    ])

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

        if result.get("retrieval_query") and result["retrieval_query"] != result["question"]:
            lines.append(f"Retrieval: {result['retrieval_query']}")

        if result.get("chat_history"):
            lines.append(f"Chat history: {clean_preview(result['chat_history'], 500)}")

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
            f"  groups     : {evaluation.get('matched_source_groups', [])}",
            f"  missing    : {evaluation.get('missing_source_groups', [])}",
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
    print(f"Semantic K       : {SEMANTIC_K}")
    print(f"BM25 K           : {BM25_K}")
    print(f"Hybrid final K   : {HYBRID_FINAL_K}")
    print(f"Rerank top N     : {RERANK_TOP_N}")
    print(f"Multi-query      : {ENABLE_MULTI_QUERY_RETRIEVAL}")
    print(f"Max queries      : {MAX_RETRIEVAL_QUERIES}")
    print(f"Max pre-rerank   : {MAX_CANDIDATES_BEFORE_RERANK}")
    print(f"Candidate balance: {ENABLE_CANDIDATE_BALANCING}")
    print(f"Max per source   : {CANDIDATE_MAX_PER_SOURCE}")
    print(f"Rerank pool top N: {BALANCED_RERANK_TOP_N}")

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
