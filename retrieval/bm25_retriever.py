import re

from langchain_community.retrievers import BM25Retriever

try:
    from config.settings import BM25_K
except ImportError:
    BM25_K = 9

TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+|[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff]+")


def preprocess_text(text):
    # Simple tokenizer for English + Japanese text.
    # BM25 is keyword-based, kaya important ang clean tokens.
    if not text:
        return []

    return TOKEN_PATTERN.findall(str(text).lower())


def get_doc_label(doc):
    # Human-readable debug label.
    metadata = dict(getattr(doc, "metadata", {}) or {})
    source = metadata.get("file_name") or metadata.get("source") or "Unknown source"
    chunk_id = metadata.get("chunk_id") or metadata.get("chunk_index") or "Unknown chunk"
    return f"{source} | {chunk_id}"


def create_bm25_retriever(chunks, k=BM25_K, use_preprocessing=True, debug=False):
    # Gumawa ng BM25 index mula sa chunks.
    # Input chunks should already have metadata prefix from chunker.py.
    if not chunks:
        raise ValueError("No chunks received. Run load -> clean -> chunk first.")

    if debug:
        print(f"[BM25] Building BM25 index from {len(chunks)} chunks...", flush=True)

    if use_preprocessing:
        retriever = BM25Retriever.from_documents(
            chunks,
            preprocess_func=preprocess_text,
        )
    else:
        retriever = BM25Retriever.from_documents(chunks)

    retriever.k = k

    if debug:
        print(f"[BM25] Ready. Top K = {retriever.k}", flush=True)

    return retriever


def bm25_search(bm25_retriever, query, k=None, debug=False):
    # Maghanap gamit exact/keyword matching.
    query = str(query or "").strip()

    if not query:
        if debug:
            print("[BM25] Empty query. Returning no results.", flush=True)
        return []

    if bm25_retriever is None:
        raise ValueError("No BM25 retriever received. Create it first using create_bm25_retriever().")

    if k is not None:
        bm25_retriever.k = k

    docs = bm25_retriever.invoke(query)

    for rank, doc in enumerate(docs, start=1):
        doc.metadata = dict(getattr(doc, "metadata", {}) or {})
        doc.metadata["bm25_rank"] = rank
        doc.metadata["retrieval_rank_1"] = rank
        doc.metadata["bm25_rank_score"] = 1.0 / rank

    if debug:
        print(f"[BM25] Query: {query}", flush=True)
        print(f"[BM25] Results: {len(docs)}", flush=True)

        for index, doc in enumerate(docs[:5], start=1):
            print(f"[BM25] Top {index}: {get_doc_label(doc)}", flush=True)

    return docs
