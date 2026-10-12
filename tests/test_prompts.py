import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from decisiontune.dataset import parse_example
from decisiontune.prompts import answer_text, option_labels, parse_output, render, to_training_rows

EX = {
    "state": "Ticket: double charge.",
    "questions": [
        {"id": "team", "type": "choice", "question": "Which team?",
         "options": ["billing", "technical"], "criteria": {"billing": "money stuff"}},
        {"id": "urgent", "type": "noul", "question": "Urgent?"},
        {"id": "sev", "type": "score", "question": "Severity?", "levels": 3,
         "labels": ["low", "med", "high"]},
    ],
    "answers": {"team": "billing", "urgent": "yes", "sev": 3},
}


class TestPrompts(unittest.TestCase):
    def setUp(self):
        self.ex = parse_example(EX, "t:1")

    def test_option_labels(self):
        self.assertEqual(option_labels(self.ex.questions[0]), ["billing", "technical"])
        self.assertEqual(option_labels(self.ex.questions[1]), ["yes", "no"])
        self.assertEqual(option_labels(self.ex.questions[2]), ["1", "2", "3"])

    def test_render_contains_criteria(self):
        p = render(self.ex.questions[0], self.ex.state)
        self.assertIn("money stuff", p)
        self.assertIn("Ticket: double charge.", p)
        self.assertTrue(p.rstrip().endswith("Answer:"))

    def test_render_score_labels(self):
        p = render(self.ex.questions[2], self.ex.state)
        self.assertIn("1=low", p)
        self.assertIn("3=high", p)

    def test_parse_output_exact(self):
        q = self.ex.questions[0]
        best, probs = parse_output(q, "billing")
        self.assertEqual(best, "billing")
        self.assertAlmostEqual(sum(probs.values()), 1.0)

    def test_parse_output_mentions(self):
        q = self.ex.questions[0]
        best, _ = parse_output(q, "I think this is a technical issue")
        self.assertEqual(best, "technical")

    def test_parse_output_garbage_is_uniform(self):
        q = self.ex.questions[1]
        best, probs = parse_output(q, "???")
        self.assertAlmostEqual(probs["yes"], 0.5)
        self.assertAlmostEqual(probs["no"], 0.5)

    def test_training_rows(self):
        rows = to_training_rows(self.ex.state, self.ex.questions, self.ex.answers)
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0]["completion"], "billing")
        self.assertEqual(rows[2]["completion"], "3")
        # prompt/completion pair is self-consistent: the completion parses back
        best, _ = parse_output(self.ex.questions[0], rows[0]["completion"])
        self.assertEqual(best, "billing")


if __name__ == "__main__":
    unittest.main()
