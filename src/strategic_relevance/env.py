import os

from .config import AgentConfig


def load_config() -> AgentConfig:
    return AgentConfig(
        alpha_sparse=_get_float("SR_ALPHA_SPARSE", 0.6),
        alpha_dense=_get_float("SR_ALPHA_DENSE", 0.4),
        sparse_use_bm25=_get_bool("SR_SPARSE_USE_BM25", True),
        sparse_use_keyword_hits=_get_bool("SR_SPARSE_USE_KEYWORD_HITS", True),
        matched_intent_threshold=_get_float("SR_MATCHED_INTENT_THRESHOLD", 0.45),
        matched_intent_top_k=int(os.getenv("SR_MATCHED_INTENT_TOP_K", "3")),
        why_max_keywords=int(os.getenv("SR_WHY_MAX_KEYWORDS", "5")),
        embeddings_enabled=_get_bool("SR_EMBEDDINGS_ENABLED", False),
        embeddings_base_url=os.getenv("SR_EMBEDDINGS_BASE_URL"),
        embeddings_api_key=os.getenv("SR_EMBEDDINGS_API_KEY"),
        embeddings_model=os.getenv("SR_EMBEDDINGS_MODEL"),
        cache_sqlite_path=os.getenv("SR_CACHE_SQLITE_PATH"),
    )


def _get_bool(name, default):
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "y", "on"}


def _get_float(name, default):
    v = os.getenv(name)
    if v is None:
        return default
    return float(v)
