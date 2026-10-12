import os
import sys
import threading
import time
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from decisiontune.backends import MockBackend
from decisiontune.dataset import parse_example
from decisiontune.evaluate import EvalConfig, _brier, _ece, run_eval
from decisiontune.report import to_html, to_markdown, to_text
from decisiontune.serve import run_server

# Labels alternate so a "first-option" mock gets exactly 50% right.
EXAMPLES = [
    parse_example({
        "state": f"case {i}",
        "questions": [{"id": "q", "type": "choice", "question": "Pick?",
                       "options": ["alpha", "beta"]}],
        "answers": {"q": "alpha" if i % 2 == 0 else "beta"},
    }, f"t:{i}")
    for i in range(10)
]


class TestEvaluate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # tuned: puts 0.9 on first option -> 50% accuracy on alternating labels
        cls.tuned = run_server(MockBackend(mode="first", first_prob=0.9, latency_ms=2),
                               "tuned-mock", port=18092)
        # baseline: uniform -> 50% accuracy, worst calibration confidence
        cls.base = run_server(MockBackend(mode="uniform", latency_ms=1),
                              "base-mock", port=18093)
        for s in (cls.tuned, cls.base):
            threading.Thread(target=s.serve_forever, daemon=True).start()
        time.sleep(0.5)

    @classmethod
    def tearDownClass(cls):
        cls.tuned.shutdown()
        cls.base.shutdown()

    def test_metrics(self):
        cfg = EvalConfig(tuned_url="http://127.0.0.1:18092",
                         baseline_url="http://127.0.0.1:18093")
        rep = run_eval(EXAMPLES, cfg)
        self.assertEqual(rep.n_examples, 10)
        self.assertEqual(rep.n_questions, 10)
        # tuned: 0.9 confidence, right half the time -> acc 0.5, ECE ~ |0.9-0.5| = 0.4
        self.assertAlmostEqual(rep.tuned["accuracy"], 0.5)
        self.assertAlmostEqual(rep.tuned["ece"], 0.4, places=2)
        # baseline uniform: acc 0.5, confidence 0.5 -> perfectly calibrated
        self.assertAlmostEqual(rep.baseline["accuracy"], 0.5)
        self.assertAlmostEqual(rep.baseline["ece"], 0.0, places=2)
        # agreement: tuned always picks "alpha"; uniform baseline ties 0.5/0.5 and
        # _argmax breaks ties toward the larger key ("beta") -> agree 0.0
        self.assertAlmostEqual(rep.agreement, 0.0)
        # latency metrics present and sane
        self.assertGreater(rep.tuned["latency_ms"]["p50"], 0)
        self.assertGreaterEqual(rep.tuned["latency_ms"]["p95"],
                                rep.tuned["latency_ms"]["p50"])
        self.assertIn("choice", rep.per_type)

    def test_ece_brier_units(self):
        # perfect confidence on correct answers -> ECE 0, Brier 0
        self.assertAlmostEqual(_ece([1.0] * 5, [True] * 5), 0.0)
        # 0.5 confidence, always right -> ECE 0.5
        self.assertAlmostEqual(_ece([0.5] * 4, [True] * 4), 0.5, places=6)
        self.assertAlmostEqual(_brier({"a": 1.0, "b": 0.0}, ["a", "b"], "a"), 0.0)
        self.assertAlmostEqual(_brier({"a": 0.5, "b": 0.5}, ["a", "b"], "a"), 0.25)

    def test_report_rendering(self):
        cfg = EvalConfig(tuned_url="http://127.0.0.1:18092",
                         baseline_url="http://127.0.0.1:18093")
        rep = run_eval(EXAMPLES[:4], cfg)
        text = to_text(rep, "tuned-mock", "base-mock")
        self.assertIn("accuracy", text)
        self.assertIn("agreement", text)
        md = to_markdown(rep)
        self.assertIn("| metric |", md)
        html = to_html(rep)
        self.assertIn("<table>", html)
        self.assertIn("decisiontune", html)
        self.assertNotIn("border-radius", html)


class TestMinAccuracyGate(unittest.TestCase):
    """--min-accuracy: exit 2 when tuned accuracy falls below the bar (CI gate)."""

    @classmethod
    def setUpClass(cls):
        from decisiontune.dataset import write_jsonl
        import tempfile
        fd, cls.data = tempfile.mkstemp(suffix=".jsonl")
        os.close(fd)
        write_jsonl(cls.data, EXAMPLES)
        # tuned mock: 50% accuracy on the alternating labels
        cls.tuned = run_server(MockBackend(mode="first", first_prob=0.9, latency_ms=1),
                               "tuned-mock", port=18094)
        cls.base = run_server(MockBackend(mode="uniform", latency_ms=1),
                              "base-mock", port=18095)
        for s in (cls.tuned, cls.base):
            threading.Thread(target=s.serve_forever, daemon=True).start()
        time.sleep(0.5)

    @classmethod
    def tearDownClass(cls):
        cls.tuned.shutdown()
        cls.base.shutdown()
        os.unlink(cls.data)

    def _run_gate(self, min_accuracy):
        import io
        from contextlib import redirect_stdout
        from decisiontune.cli import build_parser
        args = build_parser().parse_args([
            "eval", "--data", self.data,
            "--tuned", "http://127.0.0.1:18094",
            "--baseline", "http://127.0.0.1:18095",
            "--min-accuracy", str(min_accuracy),
        ])
        with redirect_stdout(io.StringIO()):
            return args.fn(args)

    def test_gate_fails_below_bar(self):
        self.assertEqual(self._run_gate(0.9), 2)

    def test_gate_passes_above_bar(self):
        self.assertEqual(self._run_gate(0.4), 0)

    def test_preset_fills_train_recipe(self):
        import io, json
        from contextlib import redirect_stdout
        from decisiontune.cli import build_parser, PRESETS
        # preset applied to flags left at defaults…
        args = build_parser().parse_args(
            ["train", "--data", self.data, "--preset", "clef", "--dry-run"])
        buf = io.StringIO()
        with redirect_stdout(buf):
            self.assertEqual(args.fn(args), 0)
        # …but explicit flags win
        args2 = build_parser().parse_args(
            ["train", "--data", self.data, "--preset", "clef",
             "--batch-size", "8", "--dry-run"])
        buf2 = io.StringIO()
        with redirect_stdout(buf2):
            self.assertEqual(args2.fn(args2), 0)
        out1 = buf.getvalue()
        out2 = buf2.getvalue()
        plan1 = json.loads(out1[out1.index("{"):])
        plan2 = json.loads(out2[out2.index("{"):])
        self.assertEqual(plan1["optimizer"]["batch_size"],
                         PRESETS["clef"]["batch_size"])
        self.assertEqual(plan2["optimizer"]["batch_size"], 8)


if __name__ == "__main__":
    unittest.main()
