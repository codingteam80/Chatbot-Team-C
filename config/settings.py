import os


def get_env_string(name, default_value):
    # Kunin string value mula sa environment variable.
    value = os.getenv(name)
    if value and value.strip():
        return value.strip()
    return default_value


def get_env_int(name, default_value):
    # Kunin integer value mula sa environment variable.
    try:
        return int(os.getenv(name, default_value))
    except (TypeError, ValueError):
        return default_value


def get_env_float(name, default_value):
    # Kunin float value mula sa environment variable.
    try:
        return float(os.getenv(name, default_value))
    except (TypeError, ValueError):
        return default_value


def get_env_bool(name, default_value):
    # Kunin boolean value mula sa environment variable.
    value = os.getenv(name)
    if value is None:
        return default_value

    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


# Paths
DATA_PATH = get_env_string("DATA_PATH", "data")
CHROMA_PATH = get_env_string("CHROMA_PATH", get_env_string("PERSIST_DIR", "chroma_db"))
CHROMA_COLLECTION_NAME = get_env_string("CHROMA_COLLECTION_NAME", "rag_documents")
CACHE_DIR = get_env_string("CACHE_DIR", "cache")
CHUNK_CACHE_PATH = get_env_string("CHUNK_CACHE_PATH", f"{CACHE_DIR}/chunks.pkl")
CHUNK_CACHE_META_PATH = get_env_string("CHUNK_CACHE_META_PATH", f"{CACHE_DIR}/chunks_meta.json")
BM25_CACHE_PATH = get_env_string("BM25_CACHE_PATH", f"{CACHE_DIR}/bm25.pkl")
BM25_CACHE_META_PATH = get_env_string("BM25_CACHE_META_PATH", f"{CACHE_DIR}/bm25_meta.json")
INGEST_RESULT_FILE = get_env_string("INGEST_RESULT_FILE", "ingest_result.txt")
FORCE_REINGEST = get_env_bool("FORCE_REINGEST", False)
FORCE_CACHE_REBUILD = get_env_bool("FORCE_CACHE_REBUILD", False)

# Fallback answer
NO_ANSWER_TEXT = "I cannot find the answer in the provided documents."

# Document loading and preprocessing
LOAD_RECURSIVE = get_env_bool("LOAD_RECURSIVE", True)
MIN_CLEAN_TEXT_LENGTH = get_env_int("MIN_CLEAN_TEXT_LENGTH", 100)
MIN_DOCUMENT_LENGTH = get_env_int("MIN_DOCUMENT_LENGTH", 50)
CHUNK_SIZE = get_env_int("CHUNK_SIZE", 1200)
CHUNK_OVERLAP = get_env_int("CHUNK_OVERLAP", 200)

# Embedding settings
EMBEDDING_MODEL_NAME = get_env_string("EMBEDDING_MODEL_NAME", "intfloat/multilingual-e5-base")
EMBEDDING_MODEL_REVISION = get_env_string("EMBEDDING_MODEL_REVISION", "main")
EMBEDDING_DEVICE = get_env_string("EMBEDDING_DEVICE", "cpu")
EMBEDDING_NORMALIZE = get_env_bool("EMBEDDING_NORMALIZE", True)
EMBEDDING_BATCH_SIZE = get_env_int("EMBEDDING_BATCH_SIZE", 32)

# Retrieval settings
SEMANTIC_K = get_env_int("SEMANTIC_K", 6)
BM25_K = get_env_int("BM25_K", 6)
HYBRID_FINAL_K = get_env_int("HYBRID_FINAL_K", 8)
RRF_K = get_env_int("RRF_K", 60)
SEMANTIC_WEIGHT = get_env_float("SEMANTIC_WEIGHT", 0.6)
BM25_WEIGHT = get_env_float("BM25_WEIGHT", 0.4)
USE_E5_PREFIX = get_env_bool("USE_E5_PREFIX", False)

# MMR settings
MMR_FETCH_K = get_env_int("MMR_FETCH_K", 18)
MMR_LAMBDA = get_env_float("MMR_LAMBDA", 0.5)

# Reranker settings
RERANKER_MODEL_NAME = get_env_string("RERANKER_MODEL_NAME", "BAAI/bge-reranker-base")
RERANK_TOP_N = get_env_int("RERANK_TOP_N", 3)
RERANK_MAX_CHARS = get_env_int("RERANK_MAX_CHARS", 700)
RERANK_MAX_LENGTH = get_env_int("RERANK_MAX_LENGTH", 320)
RERANK_BATCH_SIZE = get_env_int("RERANK_BATCH_SIZE", 4)
RERANK_USE_FP16 = get_env_bool("RERANK_USE_FP16", False)

# Context settings
MIN_QUALITY_SCORE = get_env_float("MIN_QUALITY_SCORE", 0.45)
MIN_CONTEXT_LENGTH = get_env_int("MIN_CONTEXT_LENGTH", 80)
MAX_CONTEXT_CHARS = get_env_int("MAX_CONTEXT_CHARS", 6000)
MAX_DOC_CHARS = get_env_int("MAX_DOC_CHARS", 2600)
MAX_PROMPT_CONTEXT_CHARS = get_env_int("MAX_PROMPT_CONTEXT_CHARS", 10000)
PREVIEW_CHARS = get_env_int("PREVIEW_CHARS", 500)
SOURCE_TOP_N = get_env_int("SOURCE_TOP_N", 3)

# Neighbor chunk expansion
ENABLE_NEIGHBOR_EXPANSION = get_env_bool("ENABLE_NEIGHBOR_EXPANSION", False)
NEIGHBOR_WINDOW = get_env_int("NEIGHBOR_WINDOW", 1)

# Conversation helpers
ENABLE_QUESTION_REWRITE = get_env_bool("ENABLE_QUESTION_REWRITE", True)
ENABLE_FALSE_PREMISE_RETRY = get_env_bool("ENABLE_FALSE_PREMISE_RETRY", True)

# Ollama settings
LLM_MODEL_NAME = get_env_string("OLLAMA_MODEL", "qwen2.5:3b")
LLM_TEMPERATURE = get_env_float("OLLAMA_TEMPERATURE", 0.2)
LLM_NUM_CTX = get_env_int("OLLAMA_NUM_CTX", 4096)
LLM_NUM_PREDICT = get_env_int("OLLAMA_NUM_PREDICT", 512)
LLM_TOP_P = get_env_float("OLLAMA_TOP_P", 0.85)
LLM_REPEAT_PENALTY = get_env_float("OLLAMA_REPEAT_PENALTY", 1.1)
