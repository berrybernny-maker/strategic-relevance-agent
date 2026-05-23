import argparse
import json

from .env import load_config
from .models import EventCard, StrategicIntent
from .scorer import StrategicRelevanceAgent


def main():
    parser = argparse.ArgumentParser(prog="sr-agent")
    sub = parser.add_subparsers(dest="cmd")

    p_score = sub.add_parser("score")
    p_score.add_argument("--intents", required=True)
    p_score.add_argument("--events", required=True)
    p_score.add_argument("--out", required=True)
    p_score.add_argument("--include-debug", action="store_true")

    args = parser.parse_args()
    if not getattr(args, "cmd", None):
        parser.print_help()
        raise SystemExit(2)

    if args.cmd == "score":
        _cmd_score(args)


def _cmd_score(args):
    intents_raw = _load_json(args.intents, "intents")
    events_raw = _load_json(args.events, "events")
    intents = [StrategicIntent(x) for x in intents_raw]
    events = [EventCard(x) for x in events_raw]
    cfg = load_config()
    agent = StrategicRelevanceAgent.from_config(cfg, intents=intents)
    scored = agent.score_events(events, include_debug=bool(args.include_debug))
    _dump_json(args.out, [x.to_dict() for x in scored])


def _load_json(path, key):
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw, dict) and key in raw:
        raw = raw[key]
    if not isinstance(raw, list):
        raise ValueError("invalid %s json" % key)
    return raw


def _dump_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
