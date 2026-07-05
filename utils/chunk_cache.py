import hashlib
import json
import pickle
from pathlib import Path

from config.settings import (
    CHUNK_CACHE_META_PATH,
    CHUNK_CACHE_PATH,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    FORCE_CACHE_REBUILD,
)
from loaders.document_loader import load_documents
from preprocessing.cleaner import clean_documents
from preprocessing.chunker import chunk_documents


def get_file_signature(data_path):
    # Create a signature using each file path, size, and modified time.
    data_path = Path(data_path)
    items = []

    if not data_path.exists():
        return "missing"

    for file_path in sorted(data_path.rglob("*")):
        if not file_path.is_file():
            continue

        stat = file_path.stat()
        relative_path = str(file_path.relative_to(data_path)).replace("\\", "/")
        items.append(f"{relative_path}|{stat.st_size}|{stat.st_mtime_ns}")

    raw_signature = "\n".join(items)
    return hashlib.sha256(raw_signature.encode("utf-8")).hexdigest()


def get_cache_metadata(data_path):
    # Metadata used to determine whether the chunk cache is stale.
    return {
        "data_signature": get_file_signature(data_path),
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
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


def cache_is_valid(data_path, cache_path=CHUNK_CACHE_PATH, meta_path=CHUNK_CACHE_META_PATH):
    # Valid only when existing cache at tugma the metadata.
    if not Path(cache_path).exists():
        return False

    saved_metadata = read_json(meta_path)
    return saved_metadata == get_cache_metadata(data_path)


def save_chunks_cache(chunks, data_path="data", cache_path=CHUNK_CACHE_PATH, meta_path=CHUNK_CACHE_META_PATH):
    # Save chunks and metadata.
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    with cache_path.open("wb") as file:
        pickle.dump(chunks, file)

    write_json(meta_path, get_cache_metadata(data_path))


def load_chunks_cache(cache_path=CHUNK_CACHE_PATH):
    # Load the chunk cache.
    with Path(cache_path).open("rb") as file:
        return pickle.load(file)


def build_chunks(data_path="data"):
    # Full load -> clean -> chunk flow.
    docs = load_documents(data_path)
    cleaned_docs = clean_documents(docs)
    return chunk_documents(cleaned_docs)


def load_or_create_chunks(data_path="data", force_rebuild=False):
    # Use the chunk cache when valid; rebuild when files are new or changed.
    force_rebuild = force_rebuild or FORCE_CACHE_REBUILD

    if cache_is_valid(data_path) and not force_rebuild:
        print("[CACHE] Loading chunks...")
        return load_chunks_cache()

    print("[CACHE] Building chunks...")
    chunks = build_chunks(data_path)
    save_chunks_cache(chunks, data_path=data_path)
    print("[CACHE] Chunks saved.")
    return chunks
