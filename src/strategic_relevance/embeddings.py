import json
import math
import time
import urllib.error
import urllib.request

from .storage import make_cache_key


class OpenAICompatEmbeddings(object):
    def __init__(self, base_url, api_key, model, cache=None, timeout_s=30.0):
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.cache = cache
        self.timeout_s = float(timeout_s)

    def embed(self, texts):
        texts = list(texts)
        out = [None for _ in texts]
        missing = []

        if self.cache:
            for i, t in enumerate(texts):
                k = make_cache_key(self.model, t)
                v = self.cache.get(k)
                if v is None:
                    missing.append((i, t))
                else:
                    out[i] = v
        else:
            missing = list(enumerate(texts))

        if missing:
            fetched = self._fetch_embeddings([t for _, t in missing])
            for (i, t), emb in zip(missing, fetched):
                out[i] = emb
                if self.cache:
                    self.cache.set(make_cache_key(self.model, t), emb)

        if any(v is None for v in out):
            raise RuntimeError("embedding fetch incomplete")
        return out

    def _fetch_embeddings(self, texts):
        url = self.base_url.rstrip("/") + "/embeddings"
        payload = {"model": self.model, "input": texts}
        body = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer %s" % self.api_key,
        }
        req = urllib.request.Request(url, data=body, headers=headers, method="POST")
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                raw = resp.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            raise RuntimeError("embedding http error: %s" % e.read().decode("utf-8"))
        if time.time() - t0 > self.timeout_s:
            raise RuntimeError("embedding timeout")
        data = json.loads(raw)
        items = data.get("data") or []
        items_sorted = sorted(items, key=lambda x: x.get("index", 0))
        return [it["embedding"] for it in items_sorted]


def cosine_similarity(a, b):
    if not a or not b:
        return 0.0
    if len(a) != len(b):
        n = min(len(a), len(b))
        a = a[:n]
        b = b[:n]
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        fx = float(x)
        fy = float(y)
        dot += fx * fy
        na += fx * fx
        nb += fy * fy
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return float(dot / (math.sqrt(na) * math.sqrt(nb)))
