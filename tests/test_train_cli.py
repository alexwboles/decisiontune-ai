import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from decisiontune.cli import main as cli_main
from decisiontune.dataset import load_jsonl
from decisiontune.train import TrainConfig, build_rows, plan

REPO = os.path.join(os.path.dirname(__file__), "..")


class TestTrainPlan(unittest.TestCase):
    def test_build_rows(self):
        exs = load_jsonl(os.path.join(REPO, "examples", "guardrail.jsonl"))
        rows = build_rows(exs)
        # 16 examples x 2 questions each
        self.assertEqual(len(rows), 32)
        self.assertIn("prompt", rows[0])
        self.assertIn("completion", rows[0])
        self.assertIn(rows[0]["completion"], ("yes", "no"))

    def test_plan_structure(self):
        exs = load_jsonl(os.path.join(REPO, "examples", "ticket-routing.jsonl"))
        p = plan(exs, TrainConfig(model_id="SomeOrg/some-9b", out_dir="/tmp/x"))
        self.assertEqual(p["model_id"], "SomeOrg/some-9b")
        self.assertEqual(p["train_rows"], 24)  # 12 examples x 2 questions
        self.assertIn("noul", p["by_question_type"])
        self.assertGreater(p["est_steps"], 0)
        self.assertIn("Answer:", p["sample_prompt"])

    def test_dry_run_needs_no_torch(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = cli_main(["train", "--data",
                           os.path.join(REPO, "examples", "guardrail.jsonl"),
                           "--model", "FakeOrg/fake-9b", "--dry-run"])
        self.assertEqual(rc, 0)
        p = json.loads(buf.getvalue())
        self.assertEqual(p["model_id"], "FakeOrg/fake-9b")
        self.assertEqual(p["train_rows"], 32)

    def test_real_train_fails_cleanly_without_torch(self):
        from decisiontune.train import train
        exs = load_jsonl(os.path.join(REPO, "examples", "guardrail.jsonl"))[:1]
        with self.assertRaises(RuntimeError) as ctx:
            train(exs, TrainConfig())
        self.assertIn("pip install", str(ctx.exception))


class TestCli(unittest.TestCase):
    def test_validate_examples(self):
        for name in ("guardrail.jsonl", "ticket-routing.jsonl", "lead-qualification.jsonl"):
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = cli_main(["validate", os.path.join(REPO, "examples", name)])
            self.assertEqual(rc, 0, name)
            self.assertIn("dataset OK", buf.getvalue())

    def test_split_cli(self):
        with tempfile.TemporaryDirectory() as d:
            tr, ev = os.path.join(d, "tr.jsonl"), os.path.join(d, "ev.jsonl")
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = cli_main(["split", os.path.join(REPO, "examples", "guardrail.jsonl"),
                               "--train-out", tr, "--eval-out", ev,
                               "--eval-fraction", "0.25", "--seed", "1"])
            self.assertEqual(rc, 0)
            self.assertEqual(len(load_jsonl(tr)) + len(load_jsonl(ev)), 16)
            self.assertEqual(len(load_jsonl(ev)), 4)

    def test_validate_bad_file(self):
        fd, path = tempfile.mkstemp(suffix=".jsonl")
        with os.fdopen(fd, "w") as f:
            f.write('{"state": "x", "questions": [], "answers": {}}\n')
        try:
            rc = cli_main(["validate", path])
            self.assertEqual(rc, 1)
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()
