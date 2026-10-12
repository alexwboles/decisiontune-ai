import json
import os
import sys
import threading
import time
import unittest
import urllib.error
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from decisiontune.backends import MockBackend
from decisiontune.dataset import parse_example
from decisiontune.serve import decide_remote, run_server

EX = parse_example({
    "state": "Ticket: double charge.",
    "questions": [
        {"id": "team", "type": "choice", "question": "Which team?",
         "options": ["billing", "technical"]},
        {"id": "urgent", "type": "noul", "question": "Urgent?"},
    ],
    "answers": {"team": "billing", "urgent": "yes"},
}, "t:1")


def post(url, payload):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as r:
        return r.status, json.loads(r.read().decode())


class TestServer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = run_server(MockBackend(mode="first", first_prob=0.9, latency_ms=1),
                                "mock-test", port=18091)
        cls.t = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.t.start()
        time.sleep(0.3)
        cls.base = "http://127.0.0.1:18091"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()

    def test_systemone_shape(self):
        status, body = post(self.base + "/v1/systemone", {
            "state": EX.state,
            "questions": [{"id": q.id, "type": q.type, "question": q.question,
                           "options": q.options or None} for q in EX.questions],
        })
        self.assertEqual(status, 200)
        ans = body["answers"]
        for qid, probs in ans.items():
            self.assertAlmostEqual(sum(probs.values()), 1.0, places=6)
            for p in probs.values():
                self.assertGreaterEqual(p, 0.0)
                self.assertLessEqual(p, 1.0)
        # mock "first" mode puts 0.9 on the first option
        self.assertAlmostEqual(ans["team"]["billing"], 0.9)
        self.assertIn("latency_ms", body)
        self.assertEqual(body["model"], "mock-test")

    def test_decide_remote_client(self):
        resp = decide_remote(self.base, EX.state, EX.questions)
        self.assertIn("answers", resp)
        self.assertAlmostEqual(resp["answers"]["urgent"]["yes"], 0.9)

    def test_models_endpoint(self):
        with urllib.request.urlopen(self.base + "/v1/models", timeout=10) as r:
            body = json.loads(r.read().decode())
        self.assertEqual(body["models"][0]["backend"], "mock")

    def test_dashboard_html(self):
        with urllib.request.urlopen(self.base + "/", timeout=10) as r:
            html = r.read().decode()
        self.assertIn("decisiontune", html)
        self.assertIn("/v1/systemone", html)
        self.assertNotIn("border-radius", html)  # sharp edges only

    def test_bad_requests(self):
        for payload, want in [
            ({"questions": []}, 400),
            ({"state": "x"}, 400),
            ({"state": "x", "questions": [{"id": "q", "type": "bogus", "question": "?"}]}, 400),
        ]:
            req = urllib.request.Request(self.base + "/v1/systemone",
                                         data=json.dumps(payload).encode(),
                                         headers={"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=10) as r:
                    self.fail(f"expected HTTP error, got {r.status}")
            except urllib.error.HTTPError as e:
                self.assertEqual(e.code, want, payload)

    def test_404(self):
        try:
            urllib.request.urlopen(self.base + "/nope", timeout=10)
            self.fail("expected 404")
        except urllib.error.HTTPError as e:
            self.assertEqual(e.code, 404)


if __name__ == "__main__":
    unittest.main()
