import os


def get_env_string(name, default_value):
    # Get a string value from an environment variable.
    value = os.getenv(name)
    if value and value.strip():
        return value.strip()
    return default_value


def get_env_int(name, default_value):
    # Get an integer value from an environment variable.
    try:
        return int(os.getenv(name, default_value))
    except (TypeError, ValueError):
        return default_value


def get_env_float(name, default_value):
    # Get a float value from an environment variable.
    try:
        return float(os.getenv(name, default_value))
    except (TypeError, ValueError):
        return default_value


def get_env_bool(name, default_value):
    # Get a boolean value from an environment variable.
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
USE_HISTORY_TEST_CATEGORY = get_env_bool("USE_HISTORY_TEST_CATEGORY", True)
ADD_RETRIEVAL_CONTEXT_PREFIX = get_env_bool("ADD_RETRIEVAL_CONTEXT_PREFIX", True)

# Embedding settings
# USE_E5_PREFIX=True means:
# - document embeddings should use "passage:" during ingest
# - semantic query embeddings should use "query:" during vector search
# - BM25, reranker, and final LLM context should still use raw text
EMBEDDING_MODEL_NAME = get_env_string("EMBEDDING_MODEL_NAME", "intfloat/multilingual-e5-small")
EMBEDDING_MODEL_REVISION = get_env_string("EMBEDDING_MODEL_REVISION", "main")
EMBEDDING_DEVICE = get_env_string("EMBEDDING_DEVICE", "cpu")
EMBEDDING_NORMALIZE = get_env_bool("EMBEDDING_NORMALIZE", True)
EMBEDDING_BATCH_SIZE = get_env_int("EMBEDDING_BATCH_SIZE", 32)
USE_E5_PREFIX = get_env_bool("USE_E5_PREFIX", True)

# Retrieval settings
SEMANTIC_K = get_env_int("SEMANTIC_K", 9)
BM25_K = get_env_int("BM25_K", 9)
HYBRID_FINAL_K = get_env_int("HYBRID_FINAL_K", 12)
RRF_K = get_env_int("RRF_K", 60)
SEMANTIC_WEIGHT = get_env_float("SEMANTIC_WEIGHT", 0.6)
BM25_WEIGHT = get_env_float("BM25_WEIGHT", 0.4)
#---------------------------------------------------------------------#
# Metadata boost settings
# Keep these small because RRF scores are usually around 0.006 to 0.02.
# For history test data, category/doc_type are disabled because most chunks share the same values.
ENABLE_METADATA_BOOST = get_env_bool("ENABLE_METADATA_BOOST", False)
METADATA_BOOST_CATEGORY = get_env_float("METADATA_BOOST_CATEGORY", 0.0)
METADATA_BOOST_DOC_TYPE = get_env_float("METADATA_BOOST_DOC_TYPE", 0.0)
METADATA_BOOST_LANGUAGE = get_env_float("METADATA_BOOST_LANGUAGE", 0.0)
METADATA_BOOST_SOURCE_HINT = get_env_float("METADATA_BOOST_SOURCE_HINT", 0.0020)
METADATA_BOOST_TITLE_TERM = get_env_float("METADATA_BOOST_TITLE_TERM", 0.0015)
METADATA_BOOST_SECTION_TERM = get_env_float("METADATA_BOOST_SECTION_TERM", 0.0015)
METADATA_BOOST_MAX = get_env_float("METADATA_BOOST_MAX", 0.0025)


# Multi-query retrieval settings
ENABLE_MULTI_QUERY_RETRIEVAL = get_env_bool("ENABLE_MULTI_QUERY_RETRIEVAL", False)
MAX_RETRIEVAL_QUERIES = get_env_int("MAX_RETRIEVAL_QUERIES", 3)
MAX_CANDIDATES_BEFORE_RERANK = get_env_int("MAX_CANDIDATES_BEFORE_RERANK", 14)

# Candidate balancing settings
ENABLE_CANDIDATE_BALANCING = get_env_bool("ENABLE_CANDIDATE_BALANCING", False)
CANDIDATE_MAX_PER_SOURCE = get_env_int("CANDIDATE_MAX_PER_SOURCE", 2)
BALANCED_RERANK_TOP_N = get_env_int("BALANCED_RERANK_TOP_N", 12)
#---------------------------------------------------------------------#

# MMR settings
MMR_FETCH_K = get_env_int("MMR_FETCH_K", 18)
MMR_LAMBDA = get_env_float("MMR_LAMBDA", 0.5)

# Reranker settings
RERANKER_MODEL_NAME = get_env_string("RERANKER_MODEL_NAME", "BAAI/bge-reranker-base")
RERANK_TOP_N = get_env_int("RERANK_TOP_N", 5)
RERANK_MAX_CHARS = get_env_int("RERANK_MAX_CHARS", 500)
RERANK_MAX_LENGTH = get_env_int("RERANK_MAX_LENGTH", 256)
RERANK_BATCH_SIZE = get_env_int("RERANK_BATCH_SIZE", 4)
RERANK_USE_FP16 = get_env_bool("RERANK_USE_FP16", False)
RERANK_POOL_TOP_N = get_env_int("RERANK_POOL_TOP_N", 10)
RERANK_USE_EVIDENCE_CHECK = get_env_bool("RERANK_USE_EVIDENCE_CHECK", True)
RERANK_REQUIRE_PROXIMITY_AUTO = get_env_bool("RERANK_REQUIRE_PROXIMITY_AUTO", True)
RERANK_REQUIRE_PROXIMITY_DEFAULT = get_env_bool("RERANK_REQUIRE_PROXIMITY_DEFAULT", False)

# Context settings
MIN_QUALITY_SCORE = get_env_float("MIN_QUALITY_SCORE", 0.45)
MIN_CONTEXT_LENGTH = get_env_int("MIN_CONTEXT_LENGTH", 80)
MAX_CONTEXT_CHARS = get_env_int("MAX_CONTEXT_CHARS", 8000)
MAX_PER_SOURCE = get_env_int("MAX_PER_SOURCE", 2)
MAX_DOC_CHARS = get_env_int("MAX_DOC_CHARS", 1500)
MAX_PROMPT_CONTEXT_CHARS = get_env_int("MAX_PROMPT_CONTEXT_CHARS", 8500)
PREVIEW_CHARS = get_env_int("PREVIEW_CHARS", 500)
SOURCE_TOP_N = get_env_int("SOURCE_TOP_N", 6)

# Mode-aware context filter settings
SINGLE_FACT_TOP_N = get_env_int("SINGLE_FACT_TOP_N", 5)
SINGLE_FACT_CLEAR_WINNER_GAP = get_env_float("SINGLE_FACT_CLEAR_WINNER_GAP", 1.20)
SINGLE_FACT_CLEAR_WINNER_KEEP = get_env_int("SINGLE_FACT_CLEAR_WINNER_KEEP", 1)
CROSS_DOC_TOP_N = get_env_int("CROSS_DOC_TOP_N", 6)
COMPARISON_TOP_N = get_env_int("COMPARISON_TOP_N", 6)
NEGATIVE_TOP_N = get_env_int("NEGATIVE_TOP_N", 2)
FALSE_PREMISE_TOP_N = get_env_int("FALSE_PREMISE_TOP_N", 2)

# Evidence guard settings
MIN_EVIDENCE_TERM_MATCHES = get_env_int("MIN_EVIDENCE_TERM_MATCHES", 1)
MIN_EVIDENCE_TERM_COVERAGE = get_env_float("MIN_EVIDENCE_TERM_COVERAGE", 0.45)
MAX_UNMATCHED_EVIDENCE_TERMS = get_env_int("MAX_UNMATCHED_EVIDENCE_TERMS", 3)
MAX_EVIDENCE_CONTEXT_CHARS = get_env_int("MAX_EVIDENCE_CONTEXT_CHARS", 5000)

# Neighbor chunk expansion
ENABLE_NEIGHBOR_EXPANSION = get_env_bool("ENABLE_NEIGHBOR_EXPANSION", True)
NEIGHBOR_WINDOW = get_env_int("NEIGHBOR_WINDOW", 2)

# Conversation helpers
ENABLE_QUESTION_REWRITE = get_env_bool("ENABLE_QUESTION_REWRITE", True)
ENABLE_FALSE_PREMISE_RETRY = get_env_bool("ENABLE_FALSE_PREMISE_RETRY", True)
CLARIFY_PREFIX = get_env_string("CLARIFY_PREFIX", "CLARIFY:")
MEMORY_KEY = get_env_string("MEMORY_KEY", "chat_memory")
MAX_HISTORY_CHARS = get_env_int("MAX_HISTORY_CHARS", 12000)

# Ollama settings
LLM_MODEL_NAME = get_env_string("OLLAMA_MODEL", "qwen3:1.7b")
LLM_TEMPERATURE = get_env_float("OLLAMA_TEMPERATURE", 0.1)
LLM_NUM_CTX = get_env_int("OLLAMA_NUM_CTX", 8192)
LLM_NUM_PREDICT = get_env_int("OLLAMA_NUM_PREDICT", 1536)    
LLM_TOP_P = get_env_float("OLLAMA_TOP_P", 0.5)
LLM_REPEAT_PENALTY = get_env_float("OLLAMA_REPEAT_PENALTY", 1.15)


# Generation retry settings
# When context exists but the answer still falls back, retry with a softer prompt.
ENABLE_FALLBACK_RETRY = get_env_bool("ENABLE_FALLBACK_RETRY", True)
ENABLE_TRUNCATION_RETRY = get_env_bool("ENABLE_TRUNCATION_RETRY", True)
