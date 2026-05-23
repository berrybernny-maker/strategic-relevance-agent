import argparse
import json
import os
from http.server import BaseHTTPRequestHandler, HTTPServer

from .env import load_config
from .models import EventCard, StrategicIntent
from .scorer import StrategicRelevanceAgent

_config = load_config()
_agent = StrategicRelevanceAgent.from_config(_config, intents=[])


def _startup():
    p = os.getenv("SR_INTENTS_PATH")
    if not p:
        return
    try:
        with open(p, "r", encoding="utf-8") as f:
            raw = json.load(f)
        intents = _parse_intents_payload(raw)
        _agent.update_intents(intents)
    except Exception:
        return


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/health":
            self._json(
                200,
                {
                    "status": "ok",
                    "intents": len(_agent.intents),
                    "embeddings_enabled": bool(_config.embeddings_enabled),
                },
            )
            return
        if self.path == "/intents":
            self._json(200, [it.to_dict() for it in _agent.intents])
            return
        self._json(404, {"error": "not found"})

    def do_PUT(self):
        if self.path != "/intents":
            self._json(404, {"error": "not found"})
            return
        payload = self._read_json()
        try:
            intents = _parse_intents_payload(payload)
            _agent.update_intents(intents)
        except Exception as e:
            self._json(400, {"error": str(e)})
            return
        self._json(200, {"status": "ok", "count": len(intents)})

    def do_POST(self):
        if self.path != "/score":
            self._json(404, {"error": "not found"})
            return
        payload = self._read_json()
        try:
            events_raw = payload.get("events") if isinstance(payload, dict) else None
            include_debug = bool(payload.get("include_debug")) if isinstance(payload, dict) else False
            top_k = payload.get("top_k") if isinstance(payload, dict) else None
            threshold = payload.get("threshold") if isinstance(payload, dict) else None
            if not isinstance(events_raw, list):
                raise ValueError("events must be a list")
            events = [EventCard(x) for x in events_raw]
            scored = _agent.score_events(events, include_debug=include_debug)
            out = scored
            if threshold is not None:
                out = [x for x in out if x.relevance_score >= float(threshold)]
            if top_k is not None:
                out = out[: int(top_k)]
            self._json(200, [x.to_dict() for x in out])
        except Exception as e:
            self._json(400, {"error": str(e)})

    def _read_json(self):
        length = int(self.headers.get("Content-Length") or "0")
        raw = self.rfile.read(length).decode("utf-8") if length > 0 else ""
        if not raw.strip():
            return {}
        return json.loads(raw)

    def _json(self, code, obj):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def _parse_intents_payload(raw):
    if isinstance(raw, dict) and "intents" in raw:
        raw = raw["intents"]
    if not isinstance(raw, list):
        raise ValueError("intents must be a list")
    return [StrategicIntent(x) for x in raw]


def serve(host="0.0.0.0", port=8000):
    _startup()
    httpd = HTTPServer((host, int(port)), _Handler)
    httpd.serve_forever()


def main():
    parser = argparse.ArgumentParser(prog="sr-agent-server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    serve(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
