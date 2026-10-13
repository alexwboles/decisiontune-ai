"""Dataset schema, validation, splitting, and statistics for decisiontune.

A training/eval example is one JSON object per line (JSONL):

    {
      "state": "Customer wrote: 'I was charged twice for my invoice #4821.'",
      "questions": [
        {"id": "team", "type": "choice", "question": "Which team should own this ticket?",
         "options": ["billing", "technical", "sales"],
         "criteria": {"billing": "payments, invoices, refunds",
                      "technical": "bugs, errors, crashes",
                      "sales": "pricing questions, upgrades"}},
        {"id": "urgent", "type": "noul", "question": "Is this urgent?"},
        {"id": "severity", "type": "score", "question": "How severe is the impact?",
         "levels": 5, "labels": ["trivial", "minor", "moderate", "major", "critical"]}
      ],
      "answers": {"team": "billing", "urgent": "yes", "severity": 4}
    }

Question types (mirroring the Jev-API shape used by Jev, Clef, SelfJev):
  noul   -> P(yes); answer is "yes" or "no"
  choice -> pick one of `options`; answer is one of the options
  score  -> ordinal scale 1..levels; answer is an int in range

Optional per-example `media`: a list of local image/audio/video paths, reserved
for multimodal decision models (e.g. Cloudflare Clef-omni). Text-only training
and serving ignore it; it is validated and round-tripped so datasets stay
forward-compatible.
"""
from __future__ import annotations

import json
import math
import random
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

QUESTION_TYPES = ("noul", "choice", "score")


@dataclass
class Question:
    id: str
    type: str
    question: str
    options: List[str] = field(default_factory=list)
    criteria: Dict[str, str] = field(default_factory=dict)
    levels: int = 0
    labels: List[str] = field(default_factory=list)


@dataclass
class Example:
    state: str
    questions: List[Question]
    answers: Dict[str, Any]
    source: str = ""  # "file.jsonl:line" for error messages
    media: List[str] = field(default_factory=list)  # optional media paths (Clef-omni)


class DatasetError(ValueError):
    pass


def parse_question(raw: Any, where: str) -> Question:
    if not isinstance(raw, dict):
        raise DatasetError(f"{where}: question must be an object")
    qid = raw.get("id")
    if not qid or not isinstance(qid, str):
        raise DatasetError(f"{where}: question needs a string 'id'")
    qtype = raw.get("type")
    if qtype not in QUESTION_TYPES:
        raise DatasetError(f"{where} question '{qid}': type must be one of {QUESTION_TYPES}")
    text = raw.get("question")
    if not text or not isinstance(text, str):
        raise DatasetError(f"{where} question '{qid}': needs a 'question' string")
    q = Question(id=qid, type=qtype, question=text)
    if qtype == "choice":
        opts = raw.get("options")
        if not isinstance(opts, list) or len(opts) < 2 or not all(isinstance(o, str) for o in opts):
            raise DatasetError(f"{where} question '{qid}': choice needs 'options' (list of >=2 strings)")
        if len(set(opts)) != len(opts):
            raise DatasetError(f"{where} question '{qid}': options must be unique")
        q.options = opts
        crit = raw.get("criteria") or {}
        if not isinstance(crit, dict):
            raise DatasetError(f"{where} question '{qid}': 'criteria' must be an object")
        q.criteria = {str(k): str(v) for k, v in crit.items()}
    elif qtype == "score":
        levels = raw.get("levels")
        if not isinstance(levels, int) or levels < 2:
            raise DatasetError(f"{where} question '{qid}': score needs integer 'levels' >= 2")
        q.levels = levels
        labels = raw.get("labels") or []
        if labels:
            if not isinstance(labels, list) or len(labels) != levels:
                raise DatasetError(f"{where} question '{qid}': 'labels' must match 'levels'")
            q.labels = [str(x) for x in labels]
    return q


def check_answer(q: Question, answer: Any, where: str) -> None:
    if q.type == "noul":
        if answer not in ("yes", "no"):
            raise DatasetError(f"{where} question '{q.id}': noul answer must be 'yes' or 'no'")
    elif q.type == "choice":
        if answer not in q.options:
            raise DatasetError(f"{where} question '{q.id}': answer {answer!r} not in options {q.options}")
    elif q.type == "score":
        if not isinstance(answer, int) or not (1 <= answer <= q.levels):
            raise DatasetError(f"{where} question '{q.id}': score answer must be int 1..{q.levels}")


def parse_example(raw: Any, where: str) -> Example:
    if not isinstance(raw, dict):
        raise DatasetError(f"{where}: example must be a JSON object")
    state = raw.get("state")
    if not state or not isinstance(state, str):
        raise DatasetError(f"{where}: example needs a non-empty 'state' string")
    qraw = raw.get("questions")
    if not isinstance(qraw, list) or not qraw:
        raise DatasetError(f"{where}: example needs a non-empty 'questions' list")
    questions = [parse_question(q, where) for q in qraw]
    ids = [q.id for q in questions]
    if len(set(ids)) != len(ids):
        raise DatasetError(f"{where}: duplicate question ids")
    answers = raw.get("answers")
    if not isinstance(answers, dict):
        raise DatasetError(f"{where}: example needs an 'answers' object")
    for q in questions:
        if q.id not in answers:
            raise DatasetError(f"{where}: missing answer for question '{q.id}'")
        check_answer(q, answers[q.id], where)
    extra = set(answers) - set(ids)
    if extra:
        raise DatasetError(f"{where}: answers for unknown questions: {sorted(extra)}")
    media = raw.get("media", [])
    if not isinstance(media, list) or not all(isinstance(m, str) for m in media):
        raise DatasetError(f"{where}: 'media' must be a list of path strings")
    return Example(state=state, questions=questions, answers=answers, source=where,
                   media=media)


def load_jsonl(path: str) -> List[Example]:
    examples: List[Example] = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as e:
                raise DatasetError(f"{path}:{i}: invalid JSON: {e}")
            examples.append(parse_example(raw, f"{path}:{i}"))
    if not examples:
        raise DatasetError(f"{path}: no examples found")
    return examples


def write_jsonl(path: str, examples: List[Example]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            doc: Dict[str, Any] = {
                "state": ex.state,
                "questions": [
                    {k: v for k, v in {
                        "id": q.id, "type": q.type, "question": q.question,
                        "options": q.options or None, "criteria": q.criteria or None,
                        "levels": q.levels or None, "labels": q.labels or None,
                    }.items() if v is not None}
                    for q in ex.questions
                ],
                "answers": ex.answers,
            }
            if ex.media:
                doc["media"] = ex.media
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")


def split(examples: List[Example], eval_fraction: float = 0.2,
          seed: int = 42) -> Tuple[List[Example], List[Example]]:
    """Stratified-ish split: shuffle with seed, hold out eval_fraction for eval."""
    if not 0 < eval_fraction < 1:
        raise DatasetError("eval_fraction must be between 0 and 1")
    idx = list(range(len(examples)))
    rng = random.Random(seed)
    rng.shuffle(idx)
    n_eval = max(1, round(len(examples) * eval_fraction))
    eval_idx = set(idx[:n_eval])
    train = [ex for i, ex in enumerate(examples) if i not in eval_idx]
    ev = [ex for i, ex in enumerate(examples) if i in eval_idx]
    return train, ev


def _state_tokens(s: str) -> set:
    return set(re.findall(r"[a-z0-9]+", s.lower()))


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def find_overlap(train: List[Example], ev: List[Example],
                 threshold: float = 0.8) -> List[Dict[str, Any]]:
    """Near-duplicate state texts across a train/eval split (eval leakage check).

    Returns [{eval, train, similarity}] for each eval example whose state has
    token-set Jaccard similarity >= threshold with some train example —
    those eval scores may be inflated by memorization, not learning.
    """
    train_toks = [(ex.source, _state_tokens(ex.state)) for ex in train]
    out = []
    for e in ev:
        et = _state_tokens(e.state)
        best_src, best_sim = "", 0.0
        for src, tt in train_toks:
            sim = _jaccard(et, tt)
            if sim > best_sim:
                best_sim, best_src = sim, src
        if best_sim >= threshold:
            out.append({"eval": e.source, "train": best_src,
                        "similarity": round(best_sim, 3)})
    out.sort(key=lambda d: -d["similarity"])
    return out


def stats(examples: List[Example]) -> Dict[str, Any]:
    n_q: Dict[str, int] = {}
    label_counts: Dict[str, Dict[str, int]] = {}
    state_chars = 0
    for ex in examples:
        state_chars += len(ex.state)
        for q in ex.questions:
            n_q[q.type] = n_q.get(q.type, 0) + 1
            key = f"{q.id}:{q.type}"
            label_counts.setdefault(key, {})
            ans = ex.answers[q.id]
            label_counts[key][str(ans)] = label_counts[key].get(str(ans), 0) + 1
    return {
        "examples": len(examples),
        "questions_total": sum(n_q.values()),
        "by_type": n_q,
        "avg_state_chars": round(state_chars / max(len(examples), 1), 1),
        "label_distribution": label_counts,
    }


def check_label_coverage(examples: List[Example]) -> List[str]:
    """Warn about labels that never appear (model can't learn what it never sees)."""
    warnings: List[str] = []
    seen: Dict[str, set] = {}
    possible: Dict[str, list] = {}
    for ex in examples:
        for q in ex.questions:
            key = f"{q.id}:{q.type}"
            seen.setdefault(key, set()).add(str(ex.answers[q.id]))
            if key not in possible:
                if q.type == "noul":
                    possible[key] = ["yes", "no"]
                elif q.type == "choice":
                    possible[key] = list(q.options)
                else:
                    possible[key] = [str(i) for i in range(1, q.levels + 1)]
    for key, opts in possible.items():
        missing = [o for o in opts if o not in seen.get(key, set())]
        if missing:
            warnings.append(f"question '{key}': labels never seen in data: {missing}")
    return warnings
