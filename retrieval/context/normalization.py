import json
import re
import unicodedata
from collections import defaultdict
from copy import copy
from pathlib import Path

try:
    from config.settings import (
        MAX_CONTEXT_CHARS,
        MAX_PER_SOURCE,
        SINGLE_FACT_TOP_N,
        CROSS_DOC_TOP_N,
        COMPARISON_TOP_N,
        NEGATIVE_TOP_N,
        FALSE_PREMISE_TOP_N,
        ENABLE_NEIGHBOR_EXPANSION,
        NEIGHBOR_WINDOW,
    )
except ImportError:
    MAX_CONTEXT_CHARS = 6000
    MAX_PER_SOURCE = 2
    SINGLE_FACT_TOP_N = 3
    CROSS_DOC_TOP_N = 5
    COMPARISON_TOP_N = 4
    NEGATIVE_TOP_N = 2
    FALSE_PREMISE_TOP_N = 2
    ENABLE_NEIGHBOR_EXPANSION = True
    NEIGHBOR_WINDOW = 1

try:
    from retrieval.query_analyzer import analyze_query
except ImportError:
    analyze_query = None


# ============================================================
# MODE-AWARE CONTEXT FILTER
# ============================================================
# Purpose:
# - Select which chunks will be passed to the LLM.
# - Beginner-friendly so it is easier to debug.
# - Word lists are loaded from config/query_expansion_config.json.
#
# Flow:
# 1. Detect mode:
#    - single_fact
#    - cross_doc
#
# 2. If cross_doc:
#    - detect anchor source from the question when possible
# - put anchor source first, then supporting sources
#    - do not force weak chunks just for diversity
#
# 3. If single_fact:
#    - check if question has entity/source match
# - if yes, anchor to matching source
#    - if no, anchor to top reranked source
#
# 4. Apply confident filter only inside the selected anchor source.
#
# Expected pipeline:
# semantic + BM25 -> RRF -> reranker -> context_filter -> LLM
# ============================================================


# ============================================================
# 1. BASIC HELPERS
# ============================================================

DEFAULT_CONTEXT_CONFIG_PATH = Path("config") / "query_expansion_config.json"


def normalize_text(text):
    # Lowercase and remove punctuation.
    # Normalize accents so "Andrés" and "Andres" match.
    text = str(text or "").lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = re.sub(r"[^a-z0-9\u3040-\u30ff\u3400-\u9fff]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_config_list(values):
    # Convert the JSON list into a set of normalized tokens.
    # All editable words come from JSON, not Python code.
    if isinstance(values, str):
        values = [values]

    if not isinstance(values, list):
        return set()

    normalized_values = set()

    for value in values:
        value = normalize_text(value)

        if value:
            normalized_values.add(value)

    return normalized_values


def read_context_json(config_path=DEFAULT_CONTEXT_CONFIG_PATH):
    # Read the shared query/context config.
    # Use an empty config when it is missing or invalid.
    try:
        config_path = Path(config_path)

        if not config_path.exists():
            return {}

        return json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return {}


def load_context_filter_terms(config_path=DEFAULT_CONTEXT_CONFIG_PATH):
    # All terms here are configurable in JSON.
    # This is Python logic only, with no domain-specific word list.
    raw_config = read_context_json(config_path=config_path)

    if not isinstance(raw_config, dict):
        raw_config = {}

    context_config = raw_config.get("context_filter", {})

    if not isinstance(context_config, dict):
        context_config = {}

    stopwords = normalize_config_list(raw_config.get("stopwords", []))
    question_weak_tokens = stopwords.union(
        normalize_config_list(context_config.get("question_weak_tokens", []))
    )

    source_weak_tokens = normalize_config_list(
        context_config.get("source_weak_tokens", [])
    )

    source_anchor_weak_tokens = question_weak_tokens.union(
        normalize_config_list(context_config.get("source_anchor_weak_tokens", []))
    )

    cross_doc_terms = normalize_config_list(
        context_config.get("cross_doc_terms", [])
    )

    return {
        "source_weak_tokens": source_weak_tokens,
        "question_weak_tokens": question_weak_tokens,
        "source_anchor_weak_tokens": source_anchor_weak_tokens,
        "cross_doc_terms": cross_doc_terms,
    }


CONTEXT_FILTER_TERMS = load_context_filter_terms()
SOURCE_WEAK_TOKENS = CONTEXT_FILTER_TERMS["source_weak_tokens"]
QUESTION_WEAK_TOKENS = CONTEXT_FILTER_TERMS["question_weak_tokens"]
SOURCE_ANCHOR_WEAK_TOKENS = CONTEXT_FILTER_TERMS["source_anchor_weak_tokens"]
CROSS_DOC_TERMS = CONTEXT_FILTER_TERMS["cross_doc_terms"]


def load_low_value_filter_config(config_path=DEFAULT_CONTEXT_CONFIG_PATH):
    # Low-value/reference/infobox-like chunk rules are configurable in JSON.
    raw_config = read_context_json(config_path=config_path)
    context_config = raw_config.get("context_filter", {}) if isinstance(raw_config, dict) else {}

    if not isinstance(context_config, dict):
        context_config = {}

    low_value_config = context_config.get("low_value", {})

    if not isinstance(low_value_config, dict):
        low_value_config = {}

    return {
        "reference_markers": normalize_config_list(low_value_config.get("reference_markers", [])),
        "low_value_section_phrases": normalize_config_list(low_value_config.get("low_value_section_phrases", [])),
        "low_value_body_phrases": normalize_config_list(low_value_config.get("low_value_body_phrases", [])),
    }


LOW_VALUE_FILTER_CONFIG = load_low_value_filter_config()
LOW_VALUE_REFERENCE_MARKERS = LOW_VALUE_FILTER_CONFIG["reference_markers"]
LOW_VALUE_SECTION_PHRASES = LOW_VALUE_FILTER_CONFIG["low_value_section_phrases"]
LOW_VALUE_BODY_PHRASES = LOW_VALUE_FILTER_CONFIG["low_value_body_phrases"]


def get_metadata(doc):
    # Safe metadata getter.
    return dict(getattr(doc, "metadata", {}) or {})


def get_source_name(doc):
    # Get the most readable source name.
    metadata = get_metadata(doc)

    return (
        metadata.get("file_name")
        or metadata.get("source")
        or metadata.get("title")
        or "unknown source"
    )


def get_source_key(doc):
    # Create a stable source key to detect the same document/source.
    # Example:
    # "data/Apolinario Mabini - Wikipedia.pdf"
    # -> "apolinario mabini"
    source_name = Path(str(get_source_name(doc))).stem
    tokens = normalize_text(source_name).split()

    useful_tokens = []

    for token in tokens:
        if token in SOURCE_WEAK_TOKENS:
            continue

        useful_tokens.append(token)

    if useful_tokens:
        return " ".join(useful_tokens)

    return normalize_text(source_name) or "unknown source"


def get_document_key(doc):
    # Create a stable key to avoid repeating the same chunk.
    metadata = get_metadata(doc)

    source = metadata.get("source") or metadata.get("file_name") or ""
    page = metadata.get("page", "")
    chunk_id = metadata.get("chunk_id") or metadata.get("chunk_index") or ""

    if chunk_id != "":
        return (str(source), str(page), str(chunk_id))

    preview = str(getattr(doc, "page_content", "") or "")[:250]
    return (str(source), str(page), preview)


def remove_duplicate_docs(docs):
    # Remove duplicate chunks while preserving the current order.
    unique_docs = []
    seen_keys = set()

    for doc in docs or []:
        doc_key = get_document_key(doc)

        if doc_key in seen_keys:
            continue

        seen_keys.add(doc_key)
        unique_docs.append(doc)

    return unique_docs


# Public names exported by this compatibility/refactor module.
__all__ = [
    'json',
    're',
    'unicodedata',
    'defaultdict',
    'copy',
    'Path',
    'MAX_CONTEXT_CHARS',
    'MAX_PER_SOURCE',
    'SINGLE_FACT_TOP_N',
    'CROSS_DOC_TOP_N',
    'COMPARISON_TOP_N',
    'NEGATIVE_TOP_N',
    'FALSE_PREMISE_TOP_N',
    'ENABLE_NEIGHBOR_EXPANSION',
    'NEIGHBOR_WINDOW',
    'analyze_query',
    'DEFAULT_CONTEXT_CONFIG_PATH',
    'normalize_text',
    'normalize_config_list',
    'read_context_json',
    'load_context_filter_terms',
    'CONTEXT_FILTER_TERMS',
    'SOURCE_WEAK_TOKENS',
    'QUESTION_WEAK_TOKENS',
    'SOURCE_ANCHOR_WEAK_TOKENS',
    'CROSS_DOC_TERMS',
    'load_low_value_filter_config',
    'LOW_VALUE_FILTER_CONFIG',
    'LOW_VALUE_REFERENCE_MARKERS',
    'LOW_VALUE_SECTION_PHRASES',
    'LOW_VALUE_BODY_PHRASES',
    'get_metadata',
    'get_source_name',
    'get_source_key',
    'get_document_key',
    'remove_duplicate_docs',
]
