import hashlib
import json
import pickle
from pathlib import Path

from config.settings import BM25_CACHE_META_PATH, BM25_CACHE_PATH, BM25_K, FORCE_CACHE_REBUILD
from retrieval.bm25_retriever import create_bm25_retriever


def get_chunks_signature(chunks):
    # Chunk signature used to rebuild BM25 when chunks change.
    items = []

    for doc in chunks or []:
        metadata = doc.metadata or {}
        source = metadata.get("source", "")
        chunk_id = metadata.get("chunk_id", "")
        text = doc.page_content or ""
        text_hash = hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()
        items.append(f"{source}|{chunk_id}|{len(text)}|{text_hash}")

    raw_signature = "\n".join(sorted(items))
    return hashlib.sha256(raw_signature.encode("utf-8")).hexdigest()


def get_cache_metadata(chunks, k):
    # Metadata used to determine whether the BM25 cache is stale.
    return {
        "chunks_signature": get_chunks_signature(chunks),
        "k": k,
    }


def read_json(path):
    # Safe JSON reader.
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return None


def write_json(path, data):
    # Safe JSON writer.
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def cache_is_valid(chunks, k, cache_path=BM25_CACHE_PATH, meta_path=BM25_CACHE_META_PATH):
    # Valid only when the cache exists and the chunk signature matches.
    if not Path(cache_path).exists():
        return False

    saved_metadata = read_json(meta_path)
    return saved_metadata == get_cache_metadata(chunks, k)


def save_bm25_cache(retriever, chunks, k, cache_path=BM25_CACHE_PATH, meta_path=BM25_CACHE_META_PATH):
    # Save the BM25 retriever and metadata.
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    with cache_path.open("wb") as file:
        pickle.dump(retriever, file)

    write_json(meta_path, get_cache_metadata(chunks, k))


def load_bm25_cache(cache_path=BM25_CACHE_PATH):
    # Load the BM25 cache.
    with Path(cache_path).open("rb") as file:
        return pickle.load(file)


def load_or_create_bm25(chunks, k=BM25_K, force_rebuild=False):
    # Use the BM25 cache when valid; rebuild when chunks change.
    force_rebuild = force_rebuild or FORCE_CACHE_REBUILD

    if cache_is_valid(chunks, k) and not force_rebuild:
        print("[CACHE] Loading BM25...")
        return load_bm25_cache()

    print("[CACHE] Building BM25...")
    retriever = create_bm25_retriever(chunks=chunks, k=k)
    save_bm25_cache(retriever, chunks=chunks, k=k)
    print("[CACHE] BM25 saved.")
    return retriever
