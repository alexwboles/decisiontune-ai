"""Eval rig: tuned decision endpoint vs a baseline on a held-out split.

For every (example, question) we query both endpoints and compute:
  accuracy      — argmax(probs) == label, per question type and overall
  agreement     — tuned argmax == baseline argmax (how often the tuned model
                  still agrees with the baseline it was distilled from)
  latency       — p50 / p95 / max per decision, per endpoint
  ece           — expected calibration error (10 bins) on predicted-class confidence
  brier         — mean squared error of the full probability vector vs one-hot label
  coverage@tau  — fraction of decisions with confidence >= tau that are correct

A baseline can be any Jev-API-compatible endpoint: an untuned base-model
server, a chat LLM wrapped as decisions, or a mock in tests.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .dataset import Example, Question
from .prompts import option_labels
from .serve import decide_remote


@dataclass
class EndpointResult:
    probs: Dict[str, float]
    latency_ms: float
    error: Optional[str] = None


@dataclass
class EvalConfig:
    tuned_url: str
    baseline_url: str
    tuned_key: Optional[str] = None
    baseline_key: Optional[str] = None
    timeout: float = 60.0
    coverage_threshold: float = 0.8


def _argmax(probs: Dict[str, float]) -> str:
    return max(probs.items(), key=lambda kv: (kv[1], kv[0]))[0]


def _ece(confidences: List[float], corrects: List[bool], bins: int = 10) -> float:
    if not confidences:
        return 0.0
    edges = [i / bins for i in range(bins + 1)]
    ece = 0.0
    n = len(confidences)
    for b in range(bins):
        lo, hi = edges[b], edges[b + 1]
        idx = [i for i, c in enumerate(confidences) if (lo < c <= hi) or (b == 0 and c == lo)]
        if not idx:
            continue
        acc = sum(1 for i in idx if corrects[i]) / len(idx)
        conf = sum(confidences[i] for i in idx) / len(idx)
        ece += (len(idx) / n) * abs(acc - conf)
    return ece


def _brier(probs: Dict[str, float], labels: List[str], truth: str) -> float:
    return sum((probs[l] - (1.0 if l == truth else 0.0)) ** 2 for l in labels) / len(labels)


def _pct(xs: List[float], p: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    return s[min(int(p * len(s)), len(s) - 1)]


def query_endpoint(url: str, key: Optional[str], ex: Example,
                   timeout: float) -> Tuple[Dict[str, EndpointResult], float]:
    t0 = time.perf_counter()
    try:
        resp = decide_remote(url, ex.state, ex.questions, api_key=key, timeout=timeout)
        total_ms = (time.perf_counter() - t0) * 1000
        answers = resp.get("answers", {})
        out = {}
        for q in ex.questions:
            probs = answers.get(q.id)
            if not isinstance(probs, dict):
                out[q.id] = EndpointResult({}, total_ms / max(len(ex.questions), 1),
                                           error="missing answer")
            else:
                out[q.id] = EndpointResult(
                    {str(k): float(v) for k, v in probs.items()},
                    total_ms / max(len(ex.questions), 1))
        return out, total_ms
    except Exception as e:
        total_ms = (time.perf_counter() - t0) * 1000
        return {q.id: EndpointResult({}, 0.0, error=str(e)[:200]) for q in ex.questions}, total_ms


@dataclass
class EvalReport:
    n_examples: int
    n_questions: int
    tuned: Dict
    baseline: Dict
    agreement: float
    per_type: Dict[str, Dict]
    errors: List[str] = field(default_factory=list)


def _endpoint_metrics(results: List[Tuple[Question, EndpointResult, str]]) -> Dict:
    """results: (question, endpoint_result, truth_label)."""
    accs, confs, briers, lats = [], [], [], []
    covered_correct = covered_total = 0
    for q, r, truth in results:
        if r.error or not r.probs:
            continue
        labels = option_labels(q)
        pred = _argmax(r.probs)
        correct = (pred == truth)
        accs.append(correct)
        conf = r.probs.get(pred, 0.0)
        confs.append(conf)
        briers.append(_brier(r.probs, labels, truth))
        lats.append(r.latency_ms)
    n = len(accs)
    return {
        "n": n,
        "accuracy": sum(accs) / n if n else 0.0,
        "ece": _ece(confs, accs),
        "brier": sum(briers) / n if n else 0.0,
        "latency_ms": {"p50": _pct(lats, 0.5), "p95": _pct(lats, 0.95), "max": _pct(lats, 1.0)},
    }


def run_eval(examples: List[Example], cfg: EvalConfig) -> EvalReport:
    tuned_rows: List[Tuple[Question, EndpointResult, str]] = []
    base_rows: List[Tuple[Question, EndpointResult, str]] = []
    per_type: Dict[str, Dict[str, list]] = {}
    agree = total = 0
    errors: List[str] = []
    n_questions = 0
    for ex in examples:
        t_res, _ = query_endpoint(cfg.tuned_url, cfg.tuned_key, ex, cfg.timeout)
        b_res, _ = query_endpoint(cfg.baseline_url, cfg.baseline_key, ex, cfg.timeout)
        for q in ex.questions:
            n_questions += 1
            truth = str(ex.answers[q.id])
            tr, br = t_res[q.id], b_res[q.id]
            if tr.error:
                errors.append(f"tuned {ex.source} {q.id}: {tr.error}")
            if br.error:
                errors.append(f"baseline {ex.source} {q.id}: {br.error}")
            tuned_rows.append((q, tr, truth))
            base_rows.append((q, br, truth))
            bucket = per_type.setdefault(q.type, {"tuned": [], "base": []})
            bucket["tuned"].append((q, tr, truth))
            bucket["base"].append((q, br, truth))
            if not tr.error and not br.error and tr.probs and br.probs:
                total += 1
                if _argmax(tr.probs) == _argmax(br.probs):
                    agree += 1
    per_type_metrics = {
        t: {"tuned": _endpoint_metrics(v["tuned"]), "baseline": _endpoint_metrics(v["base"])}
        for t, v in per_type.items()
    }
    return EvalReport(
        n_examples=len(examples), n_questions=n_questions,
        tuned=_endpoint_metrics(tuned_rows), baseline=_endpoint_metrics(base_rows),
        agreement=agree / total if total else 0.0,
        per_type=per_type_metrics, errors=errors[:20],
    )
