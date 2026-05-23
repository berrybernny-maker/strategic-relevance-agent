from .text import normalize_text


class KeywordHit(object):
    def __init__(self, score, hit_keywords):
        self.score = float(score)
        self.hit_keywords = hit_keywords or []


def keyword_hit_score(event_text, intent):
    text = normalize_text(event_text)
    keywords = _expand_keywords(intent)
    weights = _weights_for(intent, keywords)
    total = sum(weights.values()) if weights else 0.0
    if total <= 0.0:
        return KeywordHit(score=0.0, hit_keywords=[])
    hit = []
    s = 0.0
    for kw, w in weights.items():
        if kw and kw in text:
            s += w
            hit.append(kw)
    return KeywordHit(score=max(0.0, min(1.0, s / total)), hit_keywords=hit)


def _expand_keywords(intent):
    keywords = [normalize_text(k) for k in (intent.keywords or []) if k and str(k).strip()]
    if not intent.synonyms:
        return list(dict.fromkeys(keywords))
    expanded = []
    for k in keywords:
        expanded.append(k)
        syns = intent.synonyms.get(k) or intent.synonyms.get(k.lower()) if intent.synonyms else None
        if syns:
            expanded.extend([normalize_text(s) for s in syns if s and s.strip()])
    return list(dict.fromkeys([k for k in expanded if k]))


def _weights_for(intent, keywords):
    if not keywords:
        return {}
    if not intent.keyword_weights:
        w = 1.0 / len(keywords)
        return {k: w for k in keywords}
    weights = {}
    for k in keywords:
        wk = intent.keyword_weights.get(k) if intent.keyword_weights else None
        if wk is None:
            wk = intent.keyword_weights.get(k.lower()) if intent.keyword_weights else None
        weights[k] = float(wk) if wk is not None else 1.0
    s = sum(weights.values())
    if s <= 0.0:
        return {k: 1.0 / len(keywords) for k in keywords}
    return {k: v / s for k, v in weights.items()}
