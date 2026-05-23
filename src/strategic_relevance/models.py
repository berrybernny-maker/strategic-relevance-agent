import copy


class EventCard(object):
    def __init__(self, data):
        self.data = copy.deepcopy(data) if isinstance(data, dict) else {}

    def canonical_id(self):
        for k in ("event_id", "id", "canonical_event_key", "primary_source_id"):
            v = self.data.get(k)
            if v:
                return v
        raise ValueError("missing identifier: event_id/id/canonical_event_key/primary_source_id")

    def text_for_scoring(self):
        parts = []
        for k in ("title", "summary", "content", "text", "event_text", "clean_text"):
            v = self.data.get(k)
            if isinstance(v, str) and v.strip():
                parts.append(v)
        extra = self._extra_text_fields()
        if extra:
            parts.append(extra)
        text = "\n".join(parts).strip()
        if text:
            return text
        fallback = []
        for k in ("topic", "strategic_vertical", "region"):
            v = self.data.get(k)
            if isinstance(v, str) and v.strip():
                fallback.append(v)
        return " ".join(fallback).strip()

    def _extra_text_fields(self):
        skip = {
            "event_id",
            "id",
            "title",
            "summary",
            "content",
            "market",
            "region",
            "country",
            "topic",
            "event_type",
            "source_credibility",
            "novelty_score",
            "evidence",
        }
        text_values = []
        for k, v in self.data.items():
            if k in skip:
                continue
            if isinstance(v, str):
                text_values.append(v)
            elif isinstance(v, list):
                for item in v:
                    if isinstance(item, str):
                        text_values.append(item)
        return "\n".join(text_values).strip()

    def get(self, key, default=None):
        return self.data.get(key, default)

    def to_dict(self):
        return copy.deepcopy(self.data)


class StrategicIntent(object):
    def __init__(self, data):
        self.intent_id = str(data.get("intent_id") or "").strip()
        self.name = str(data.get("name") or "").strip()
        self.description = data.get("description")
        self.keywords = data.get("keywords") or []
        self.keyword_weights = data.get("keyword_weights") or None
        self.synonyms = data.get("synonyms") or None
        if not self.intent_id or not self.name:
            raise ValueError("intent_id and name are required")

    def text_for_embedding(self):
        parts = [self.name]
        if isinstance(self.description, str) and self.description.strip():
            parts.append(self.description)
        if self.keywords:
            parts.append(" ".join([str(x) for x in self.keywords if x]))
        return "\n".join(parts).strip()

    def to_dict(self):
        return {
            "intent_id": self.intent_id,
            "name": self.name,
            "description": self.description,
            "keywords": self.keywords,
            "keyword_weights": self.keyword_weights,
            "synonyms": self.synonyms,
        }


class ScoredIntent(object):
    def __init__(self, intent_id=None, name=None, relevance_score=0.0, sparse_score=0.0, dense_score=None, hit_keywords=None, **_):
        self.intent_id = intent_id
        self.name = name
        self.relevance_score = float(relevance_score)
        self.sparse_score = float(sparse_score)
        self.dense_score = dense_score if dense_score is None else float(dense_score)
        self.hit_keywords = hit_keywords or []

    def to_dict(self):
        return {
            "intent_id": self.intent_id,
            "name": self.name,
            "relevance_score": self.relevance_score,
            "sparse_score": self.sparse_score,
            "dense_score": self.dense_score,
            "hit_keywords": self.hit_keywords,
        }


class ScoredEvent(object):
    def __init__(self, event_id=None, relevance_score=0.0, matched_intents=None, why_relevant="", intents=None, **_):
        self.event_id = event_id
        self.relevance_score = float(relevance_score)
        self.matched_intents = matched_intents or []
        self.why_relevant = why_relevant
        self.intents = intents

    def to_dict(self):
        return {
            "event_id": self.event_id,
            "relevance_score": self.relevance_score,
            "matched_intents": self.matched_intents,
            "why_relevant": self.why_relevant,
            "intents": [x.to_dict() for x in self.intents] if self.intents else None,
        }
