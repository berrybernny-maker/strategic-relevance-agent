from .bm25 import BM25Okapi
from .config import AgentConfig
from .embeddings import OpenAICompatEmbeddings, cosine_similarity
from .models import EventCard, ScoredEvent, ScoredIntent, StrategicIntent
from .sparse import keyword_hit_score
from .storage import SqliteEmbeddingCache
from .text import tokenize


class StrategicRelevanceAgent(object):
    def __init__(self, config, intents, embeddings=None):
        self.config = config
        self.intents = intents
        self.embeddings = embeddings

    @classmethod
    def from_config(cls, config, intents):
        embeddings = None
        if config.embeddings_enabled:
            if not (config.embeddings_base_url and config.embeddings_api_key and config.embeddings_model):
                raise ValueError("embeddings enabled but base_url/api_key/model not configured")
            cache = SqliteEmbeddingCache(config.cache_sqlite_path) if config.cache_sqlite_path else None
            embeddings = OpenAICompatEmbeddings(
                base_url=config.embeddings_base_url,
                api_key=config.embeddings_api_key,
                model=config.embeddings_model,
                cache=cache,
            )
        return cls(config=config, intents=intents, embeddings=embeddings)

    def update_intents(self, intents):
        self.intents = intents

    def score_events(self, events, include_debug=False):
        if not self.intents:
            raise ValueError("no intents configured")
        events_text = [e.text_for_scoring() for e in events]
        tokenized_corpus = [tokenize(t) for t in events_text]
        bm25 = BM25Okapi(tokenized_corpus) if self.config.sparse_use_bm25 else None
        bm25_norm_per_intent = {}
        if bm25:
            for it in self.intents:
                q = tokenize(it.text_for_embedding())
                raw = bm25.get_scores(q)
                bm25_norm_per_intent[it.intent_id] = _minmax_list(raw)

        intent_dense_vecs = {}
        event_dense_vecs = None
        if self.embeddings:
            intent_texts = [it.text_for_embedding() for it in self.intents]
            intent_vecs = self.embeddings.embed(intent_texts)
            for it, v in zip(self.intents, intent_vecs):
                intent_dense_vecs[it.intent_id] = v
            event_dense_vecs = self.embeddings.embed(events_text)

        scored = []
        for idx, evt in enumerate(events):
            per_intent = []
            for it in self.intents:
                hit = keyword_hit_score(events_text[idx], it) if self.config.sparse_use_keyword_hits else None
                hit_score = hit.score if hit else 0.0
                bm25_norm = 0.0
                if bm25:
                    bm25_norm = bm25_norm_per_intent.get(it.intent_id, [0.0] * len(events))[idx]
                sparse_score = max(hit_score, bm25_norm) if self.config.sparse_use_bm25 else hit_score

                dense_score = None
                if self.embeddings and event_dense_vecs is not None:
                    v_evt = event_dense_vecs[idx]
                    v_it = intent_dense_vecs.get(it.intent_id)
                    if v_it is not None:
                        sim = cosine_similarity(v_evt, v_it)
                        dense_score = (sim + 1.0) / 2.0

                relevance = self._fuse_scores(sparse_score, dense_score)
                per_intent.append(
                    ScoredIntent(
                        intent_id=it.intent_id,
                        name=it.name,
                        relevance_score=relevance,
                        sparse_score=sparse_score,
                        dense_score=dense_score,
                        hit_keywords=(hit.hit_keywords if hit else []),
                    )
                )

            per_intent.sort(key=lambda x: x.relevance_score, reverse=True)

            threshold = self.config.matched_intent_threshold
            matched = [x.intent_id for x in per_intent if x.relevance_score >= threshold]
            if not matched:
                matched = [x.intent_id for x in per_intent[: self.config.matched_intent_top_k]]

            top = per_intent[0]
            why = _why_relevant(evt, top, self.config.why_max_keywords)
            scored.append(
                ScoredEvent(
                    event_id=evt.canonical_id(),
                    relevance_score=top.relevance_score,
                    matched_intents=matched,
                    why_relevant=why,
                    intents=(per_intent if include_debug else None),
                )
            )

        scored.sort(key=lambda x: x.relevance_score, reverse=True)
        return scored

    def _fuse_scores(self, sparse_score, dense_score):
        s = max(0.0, min(1.0, float(sparse_score)))
        if dense_score is None:
            return s
        d = max(0.0, min(1.0, float(dense_score)))
        a_s = float(self.config.alpha_sparse)
        a_d = float(self.config.alpha_dense)
        if a_s + a_d <= 0.0:
            return max(s, d)
        w = a_s + a_d
        return (a_s * s + a_d * d) / w


def _minmax_list(values):
    if not values:
        return []
    mn = min(values)
    mx = max(values)
    if mx <= mn:
        return [0.0 for _ in values]
    return [float((v - mn) / (mx - mn)) for v in values]


def _why_relevant(evt, top, max_keywords):
    kw = [k for k in (top.hit_keywords or []) if k][: int(max_keywords)]
    parts = []
    region = evt.get("region") or evt.get("market") or evt.get("country")
    if region:
        parts.append(str(region))
    topic = evt.get("topic")
    if topic:
        parts.append(str(topic))
    vertical = evt.get("strategic_vertical")
    if vertical:
        parts.append(str(vertical))
    head = " / ".join(parts)
    kw_part = "命中关键词: %s" % (", ".join(kw)) if kw else "关键词命中弱"
    score_part = "相关性: %.2f" % top.relevance_score
    if top.dense_score is None:
        detail = "sparse=%.2f" % top.sparse_score
    else:
        detail = "sparse=%.2f, dense=%.2f" % (top.sparse_score, top.dense_score)
    intent_part = "匹配意图: %s" % top.name
    if head:
        return "%s | %s | %s | %s (%s)" % (head, intent_part, kw_part, score_part, detail)
    return "%s | %s | %s (%s)" % (intent_part, kw_part, score_part, detail)
