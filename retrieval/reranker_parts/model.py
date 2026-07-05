import hashlib
import re
from functools import cmp_to_key

from FlagEmbedding import FlagReranker

from config.settings import (
    RERANK_BATCH_SIZE,
    RERANK_MAX_CHARS,
    RERANK_MAX_LENGTH,
    RERANK_TOP_N,
    RERANK_USE_FP16,
    RERANKER_MODEL_NAME,
)

try:
    from config.settings import RERANK_TIE_MARGIN
except ImportError:
    # Default value when it is not yet in settings.py.
    # 0.15 means a score gap of 0.15 or lower is considered a tie.
    RERANK_TIE_MARGIN = 0.15


try:
    from retrieval.query_expander import (
        DEFAULT_CONFIG_PATH as QUERY_EXPANSION_CONFIG_PATH,
        get_config_stopwords,
        normalize_text as normalize_query_text,
    )
except ImportError:
    QUERY_EXPANSION_CONFIG_PATH = None

    def get_config_stopwords(config_path=None):
        # Empty fallback when query_expander.py has not been added yet.
        return set()

    def normalize_query_text(text):
        # Minimal fallback normalize.
        return " ".join(re.findall(r"[a-z0-9]+", str(text or "").lower()))


def get_question_stopwords():
    # Stopwords are loaded from config/query_expansion_config.json.
    # There is no hardcoded stopword list in reranker.py.
    return set(get_config_stopwords(QUERY_EXPANSION_CONFIG_PATH))


def load_reranker(model_name=RERANKER_MODEL_NAME, use_fp16=RERANK_USE_FP16, debug=False):
    # Load the reranker model.
    # Reranker = second judge that selects the most relevant chunks.
    if debug:
        print(f"[RERANKER] Loading model: {model_name}", flush=True)
        print(f"[RERANKER] use_fp16={use_fp16}", flush=True)

    reranker = FlagReranker(model_name, use_fp16=use_fp16)

    if debug:
        print("[RERANKER] Model loaded.", flush=True)

    return reranker


def normalize_scores(scores):
    # Convert scores to a normal Python list for easier sorting.
    if isinstance(scores, (int, float)):
        return [float(scores)]

    if hasattr(scores, "tolist"):
        return scores.tolist()

    return list(scores)


def trim_document_text(text, max_chars=RERANK_MAX_CHARS):
    # Trim chunk text before passing it to the reranker.
    # Purpose: faster processing and avoiding overly long input.
    clean_text = " ".join(str(text or "").split())

    if len(clean_text) <= max_chars:
        return clean_text

    return clean_text[:max_chars].rstrip()




def strip_retrieval_context_prefix(text):
    # Remove the metadata prefix when evidence checks use the text.
    # Reason: metadata helps retrieval/reranking, but evidence should come from actual chunk content.
    text = str(text or "")

    if not text.startswith("Retrieval context:"):
        return text

    parts = text.split("\n\n", 1)

    if len(parts) == 2:
        return parts[1]

    return text

def get_document_label(doc):
    # Human-readable label for in debug output.
    metadata = dict(getattr(doc, "metadata", {}) or {})

    source = metadata.get("file_name") or metadata.get("source") or "Unknown source"
    page = metadata.get("page", "N/A")
    section = metadata.get("section") or "Unknown section"
    category = metadata.get("category") or "Unknown category"
    doc_type = metadata.get("doc_type") or "Unknown type"
    chunk_id = metadata.get("chunk_id") or metadata.get("chunk_index") or "Unknown chunk"

    return f"{source} | page={page} | section={section} | {category}/{doc_type} | {chunk_id}"


def get_document_dedup_key(doc):
    # Create a stable key to remove duplicate chunks before reranking.
    # Priority 1: source + chunk_id/chunk_index when available.
    # Priority 2: source + page + exact normalized text hash.
    # Priority 3: text hash only when metadata is incomplete.
    metadata = dict(getattr(doc, "metadata", {}) or {})

    source = (
        metadata.get("source")
        or metadata.get("file_name")
        or metadata.get("file")
        or ""
    )
    chunk_id = (
        metadata.get("chunk_id")
        or metadata.get("chunk_index")
        or metadata.get("id")
        or ""
    )
    page = metadata.get("page", "")

    source = str(source or "").strip().lower()
    chunk_id = str(chunk_id or "").strip().lower()
    page = str(page or "").strip().lower()

    if source and chunk_id:
        return ("chunk", source, chunk_id)

    text = strip_retrieval_context_prefix(getattr(doc, "page_content", ""))
    normalized_text = " ".join(str(text or "").lower().split())

    if normalized_text:
        text_hash = hashlib.md5(
            normalized_text.encode("utf-8", errors="ignore")
        ).hexdigest()

        if source or page:
            return ("text_with_location", source, page, text_hash)

        return ("text", text_hash)

    # Fallback to avoid accidentally merging empty docs with missing metadata.
    return ("object", id(doc))


def deduplicate_documents(documents, debug=False):
    # Remove duplicate docs before creating query-document pairs.
    # Reranking is faster because the same chunk is not scored repeatedly.
    # Keep the first occurrence because it usually follows the RRF/original retrieval order.
    unique_docs = []
    seen_keys = {}
    duplicate_count = 0

    for input_rank, doc in enumerate(documents, start=1):
        key = get_document_dedup_key(doc)

        if key in seen_keys:
            duplicate_count += 1
            kept_doc = seen_keys[key]
            kept_doc.metadata = dict(getattr(kept_doc, "metadata", {}) or {})
            kept_doc.metadata["rerank_duplicate_count"] = int(
                kept_doc.metadata.get("rerank_duplicate_count", 1)
            ) + 1
            continue

        doc.metadata = dict(getattr(doc, "metadata", {}) or {})
        doc.metadata["rerank_first_input_rank"] = input_rank
        doc.metadata["rerank_duplicate_count"] = 1

        seen_keys[key] = doc
        unique_docs.append(doc)

    if debug:
        print(
            f"[RERANKER] Dedup before rerank: "
            f"input={len(documents)}, unique={len(unique_docs)}, removed={duplicate_count}",
            flush=True,
        )

    return unique_docs


# Public names exported by this compatibility/refactor module.
__all__ = [
    'hashlib',
    're',
    'cmp_to_key',
    'FlagReranker',
    'RERANK_BATCH_SIZE',
    'RERANK_MAX_CHARS',
    'RERANK_MAX_LENGTH',
    'RERANK_TOP_N',
    'RERANK_USE_FP16',
    'RERANKER_MODEL_NAME',
    'RERANK_TIE_MARGIN',
    'QUERY_EXPANSION_CONFIG_PATH',
    'get_config_stopwords',
    'normalize_query_text',
    'get_question_stopwords',
    'load_reranker',
    'normalize_scores',
    'trim_document_text',
    'strip_retrieval_context_prefix',
    'get_document_label',
    'get_document_dedup_key',
    'deduplicate_documents',
]
