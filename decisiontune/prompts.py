"""Prompt rendering for supervised fine-tuning and inference of decision models.

Each (state, question) pair becomes one training example: the model learns to
output the correct option. At serve time the same renderer is used, and the
decision head / scoring layer maps the model's output back to per-option
probabilities. Keeping one renderer for train and serve avoids train/serve skew.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

from .dataset import Question

SYSTEM = (
    "You are a decision model. Answer the typed question with exactly one of "
    "the allowed options and nothing else."
)


def option_labels(q: Question) -> List[str]:
    if q.type == "noul":
        return ["yes", "no"]
    if q.type == "choice":
        return list(q.options)
    return [str(i) for i in range(1, q.levels + 1)]


def render(q: Question, state: str) -> str:
    """Render a (state, question) pair as a training/inference prompt."""
    labels = option_labels(q)
    lines = [SYSTEM, "", f"State: {state}", "", f"Question: {q.question}"]
    if q.type == "choice":
        lines.append("Options:")
        for opt in labels:
            crit = q.criteria.get(opt)
            lines.append(f"  - {opt}" + (f": {crit}" if crit else ""))
    elif q.type == "score":
        if q.labels:
            lines.append("Scale: " + ", ".join(f"{i}={lab}" for i, lab in enumerate(q.labels, 1)))
        else:
            lines.append(f"Scale: 1 (lowest) to {q.levels} (highest)")
    elif q.type == "noul":
        lines.append("Answer with exactly: yes or no")
    lines.append("Answer:")
    return "\n".join(lines)


def answer_text(q: Question, answer) -> str:
    """Canonical target string the model is trained to emit."""
    return str(answer)


def parse_output(q: Question, text: str) -> Tuple[str, Dict[str, float]]:
    """Map raw model text back to (best_option, {option: score}).

    Real decision heads (Clef/Jev style) score options in one forward pass and
    return calibrated probabilities directly. For generative backends we do a
    simple, deterministic mapping: exact option match wins; otherwise the first
    option mentioned wins; otherwise uniform. The returned scores are raw
    (not calibrated) — calibration is measured and reported by the eval rig.
    """
    labels = option_labels(q)
    t = text.strip().lower()
    # exact match on its own line or the whole output
    for lab in labels:
        if t == lab.lower() or t.startswith(lab.lower() + "\n") or t.endswith("\n" + lab.lower()):
            return lab, {l: (1.0 if l == lab else 0.0) for l in labels}
    # first option mentioned anywhere
    for lab in labels:
        if lab.lower() in t:
            return lab, {l: (1.0 if l == lab else 0.0) for l in labels}
    uniform = 1.0 / len(labels)
    return labels[0], {l: uniform for l in labels}


def to_training_rows(state: str, questions: List[Question],
                     answers: Dict[str, object]) -> List[Dict[str, str]]:
    """Expand one JSONL example into one SFT row per question."""
    rows = []
    for q in questions:
        rows.append({
            "prompt": render(q, state),
            "completion": answer_text(q, answers[q.id]),
            "question_id": q.id,
            "question_type": q.type,
        })
    return rows
