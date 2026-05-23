import math


class BM25Okapi(object):
    def __init__(self, corpus, k1=1.5, b=0.75):
        self.corpus = corpus
        self.k1 = float(k1)
        self.b = float(b)
        self.doc_len = [len(doc) for doc in corpus]
        self.avgdl = (sum(self.doc_len) / float(len(self.doc_len))) if self.doc_len else 0.0
        self.df = {}
        self.tf = []
        for doc in corpus:
            freqs = {}
            for w in doc:
                freqs[w] = freqs.get(w, 0) + 1
            self.tf.append(freqs)
            for w in freqs.keys():
                self.df[w] = self.df.get(w, 0) + 1
        self.N = len(corpus)
        self.idf = {}
        for w, n in self.df.items():
            self.idf[w] = math.log(1.0 + (self.N - n + 0.5) / (n + 0.5))

    def get_scores(self, query_tokens):
        scores = [0.0 for _ in range(self.N)]
        for q in query_tokens:
            idf = self.idf.get(q, 0.0)
            for i in range(self.N):
                f = self.tf[i].get(q, 0)
                if f <= 0:
                    continue
                dl = self.doc_len[i] if self.doc_len else 0.0
                denom = f + self.k1 * (1.0 - self.b + self.b * (dl / self.avgdl if self.avgdl else 0.0))
                scores[i] += idf * (f * (self.k1 + 1.0)) / denom if denom else 0.0
        return scores
