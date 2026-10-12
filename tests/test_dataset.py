import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from decisiontune.dataset import (
    DatasetError, check_label_coverage, load_jsonl, parse_example, split, stats,
)

GOOD = {
    "state": "Ticket: double charge on invoice.",
    "questions": [
        {"id": "team", "type": "choice", "question": "Which team?",
         "options": ["billing", "technical"], "criteria": {"billing": "money"}},
        {"id": "urgent", "type": "noul", "question": "Urgent?"},
        {"id": "sev", "type": "score", "question": "Severity?", "levels": 3},
    ],
    "answers": {"team": "billing", "urgent": "yes", "sev": 2},
}


def write_tmp(lines):
    fd, path = tempfile.mkstemp(suffix=".jsonl")
    with os.fdopen(fd, "w") as f:
        for line in lines:
            f.write(json.dumps(line) + "\n")
    return path


class TestDataset(unittest.TestCase):
    def test_parse_good(self):
        ex = parse_example(GOOD, "t:1")
        self.assertEqual(len(ex.questions), 3)
        self.assertEqual(ex.answers["team"], "billing")

    def test_rejects(self):
        bad = dict(GOOD)
        bad["answers"] = {"team": "nope", "urgent": "yes", "sev": 2}
        with self.assertRaises(DatasetError):
            parse_example(bad, "t:1")
        bad2 = dict(GOOD)
        bad2["answers"] = {"team": "billing", "urgent": "maybe", "sev": 2}
        with self.assertRaises(DatasetError):
            parse_example(bad2, "t:1")
        bad3 = dict(GOOD)
        bad3["answers"] = {"team": "billing", "urgent": "yes", "sev": 9}
        with self.assertRaises(DatasetError):
            parse_example(bad3, "t:1")
        with self.assertRaises(DatasetError):
            parse_example({"state": "x"}, "t:1")
        dup = dict(GOOD)
        dup["questions"] = [GOOD["questions"][0], GOOD["questions"][0]]
        with self.assertRaises(DatasetError):
            parse_example(dup, "t:1")

    def test_load_and_stats(self):
        path = write_tmp([GOOD, GOOD])
        try:
            exs = load_jsonl(path)
            self.assertEqual(len(exs), 2)
            s = stats(exs)
            self.assertEqual(s["examples"], 2)
            self.assertEqual(s["by_type"], {"choice": 2, "noul": 2, "score": 2})
            self.assertEqual(s["label_distribution"]["team:choice"], {"billing": 2})
        finally:
            os.unlink(path)

    def test_load_bad_json(self):
        fd, path = tempfile.mkstemp(suffix=".jsonl")
        with os.fdopen(fd, "w") as f:
            f.write("not json\n")
        try:
            with self.assertRaises(DatasetError):
                load_jsonl(path)
        finally:
            os.unlink(path)

    def test_split_deterministic(self):
        exs = [parse_example(dict(GOOD, state=f"s{i}"), f"t:{i}") for i in range(10)]
        tr1, ev1 = split(exs, 0.2, seed=7)
        tr2, ev2 = split(exs, 0.2, seed=7)
        self.assertEqual([e.state for e in tr1], [e.state for e in tr2])
        self.assertEqual(len(ev1), 2)
        self.assertEqual(len(tr1), 8)
        # no overlap
        self.assertFalse(set(e.state for e in tr1) & set(e.state for e in ev1))

    def test_label_coverage_warning(self):
        exs = [parse_example(GOOD, "t:1")]
        warns = check_label_coverage(exs)
        self.assertTrue(any("technical" in w for w in warns))
        self.assertTrue(any("no" in w for w in warns))


if __name__ == "__main__":
    unittest.main()
