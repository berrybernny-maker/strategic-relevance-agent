import json
import os
import unittest

from strategic_relevance.config import AgentConfig
from strategic_relevance.models import EventCard, StrategicIntent
from strategic_relevance.scorer import StrategicRelevanceAgent


class TestBasic(unittest.TestCase):
    def test_scores_run(self):
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        with open(os.path.join(root, "examples", "intents.sample.json"), "r", encoding="utf-8") as f:
            intents_raw = json.load(f)
        with open(os.path.join(root, "examples", "events.sample.json"), "r", encoding="utf-8") as f:
            events_raw = json.load(f)
        intents = [StrategicIntent(x) for x in intents_raw["intents"]]
        events = [EventCard(x) for x in events_raw["events"]]
        cfg = AgentConfig(embeddings_enabled=False)
        agent = StrategicRelevanceAgent.from_config(cfg, intents=intents)
        scored = agent.score_events(events, include_debug=True)
        self.assertEqual(len(scored), len(events))
        self.assertTrue(scored[0].event_id)
        self.assertTrue(scored[0].matched_intents)
        self.assertIsNotNone(scored[0].intents)


if __name__ == "__main__":
    unittest.main()
