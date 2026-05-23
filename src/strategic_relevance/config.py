class AgentConfig(object):
    def __init__(
        self,
        alpha_sparse=0.6,
        alpha_dense=0.4,
        sparse_use_bm25=True,
        sparse_use_keyword_hits=True,
        matched_intent_threshold=0.45,
        matched_intent_top_k=3,
        why_max_keywords=5,
        embeddings_enabled=False,
        embeddings_base_url=None,
        embeddings_api_key=None,
        embeddings_model=None,
        cache_sqlite_path=None,
    ):
        self.alpha_sparse = float(alpha_sparse)
        self.alpha_dense = float(alpha_dense)
        self.sparse_use_bm25 = bool(sparse_use_bm25)
        self.sparse_use_keyword_hits = bool(sparse_use_keyword_hits)
        self.matched_intent_threshold = float(matched_intent_threshold)
        self.matched_intent_top_k = int(matched_intent_top_k)
        self.why_max_keywords = int(why_max_keywords)
        self.embeddings_enabled = bool(embeddings_enabled)
        self.embeddings_base_url = embeddings_base_url
        self.embeddings_api_key = embeddings_api_key
        self.embeddings_model = embeddings_model
        self.cache_sqlite_path = cache_sqlite_path

    def to_dict(self):
        return {
            "alpha_sparse": self.alpha_sparse,
            "alpha_dense": self.alpha_dense,
            "sparse_use_bm25": self.sparse_use_bm25,
            "sparse_use_keyword_hits": self.sparse_use_keyword_hits,
            "matched_intent_threshold": self.matched_intent_threshold,
            "matched_intent_top_k": self.matched_intent_top_k,
            "why_max_keywords": self.why_max_keywords,
            "embeddings_enabled": self.embeddings_enabled,
            "embeddings_base_url": self.embeddings_base_url,
            "embeddings_api_key": self.embeddings_api_key,
            "embeddings_model": self.embeddings_model,
            "cache_sqlite_path": self.cache_sqlite_path,
        }
