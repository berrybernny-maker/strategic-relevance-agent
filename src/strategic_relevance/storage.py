import hashlib
import json
import sqlite3


def make_cache_key(model, text):
    h = hashlib.sha256()
    h.update(model.encode("utf-8"))
    h.update(b"\n")
    h.update(text.encode("utf-8"))
    return h.hexdigest()


class SqliteEmbeddingCache(object):
    def __init__(self, path):
        self.path = path
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                "create table if not exists embeddings (key text primary key, embedding text not null)"
            )
            conn.commit()

    def get(self, key):
        with sqlite3.connect(self.path) as conn:
            cur = conn.execute("select embedding from embeddings where key = ?", (key,))
            row = cur.fetchone()
            if not row:
                return None
            return json.loads(row[0])

    def set(self, key, embedding):
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                "insert or replace into embeddings(key, embedding) values (?, ?)",
                (key, json.dumps(embedding)),
            )
            conn.commit()
