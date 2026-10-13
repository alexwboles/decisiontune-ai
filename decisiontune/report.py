"""Eval report rendering: text (terminal), Markdown, and a premium static HTML page."""
from __future__ import annotations

import html as _html

from .evaluate import EvalReport

HTML_CSS = """
:root{--ink:#16191d;--muted:#5d6672;--line:#dfe3e8;--bg:#f6f7f8;--card:#fff;
--accent:#0f766e;--accent-soft:#e3f2f0;--warn:#b45309;--bad:#b3261e}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Inter,Roboto,Helvetica,Arial,sans-serif}
.wrap{max-width:1020px;margin:0 auto;padding:36px 24px 64px}
header{border-bottom:2px solid var(--ink);padding-bottom:18px;margin-bottom:8px}
h1{font-size:24px;margin:0;letter-spacing:-.02em}h1 span{color:var(--accent)}
.meta{color:var(--muted);font-size:13px;margin:10px 0 26px}
h2{font-size:15px;text-transform:uppercase;letter-spacing:.07em;color:var(--muted);
margin:34px 0 12px}
.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}
.card{background:var(--card);border:1px solid var(--line);padding:18px}
.card .k{font-size:11px;text-transform:uppercase;letter-spacing:.08em;color:var(--muted);margin-bottom:8px}
.card .v{font-size:30px;font-weight:650;letter-spacing:-.02em}
.card .v small{font-size:14px;color:var(--muted);font-weight:400}
table{width:100%;border-collapse:collapse;background:var(--card);font-size:13.5px;margin-top:4px}
th{text-align:left;font-size:11px;text-transform:uppercase;letter-spacing:.06em;color:var(--muted);
border-bottom:1px solid var(--line);padding:10px 14px}
td{border-bottom:1px solid var(--line);padding:10px 14px}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}
.bar{height:8px;background:#edf0f2;position:relative;min-width:120px}
.bar i{position:absolute;left:0;top:0;bottom:0;background:var(--accent)}
.bar.base i{background:#9aa4ae}
.better{color:var(--accent);font-weight:650}.worse{color:var(--bad);font-weight:650}
.note{background:var(--accent-soft);border-left:3px solid var(--accent);padding:12px 16px;
font-size:13.5px;margin-top:24px}
footer{margin-top:40px;color:var(--muted);font-size:12px}
@media(max-width:700px){.grid{grid-template-columns:1fr}}
"""


def _fmt(x: float, pct: bool = False) -> str:
    return f"{x * 100:.1f}%" if pct else f"{x:.3f}"


def _cmp_row(label: str, tuned: float, base: float, pct: bool = False,
             lower_better: bool = False) -> str:
    cls = ""
    if abs(tuned - base) > 1e-9:
        tuned_better = tuned < base if lower_better else tuned > base
        cls = "better" if tuned_better else "worse"
    mark = "▲" if (tuned > base) != lower_better and abs(tuned - base) > 1e-9 else \
           ("▼" if abs(tuned - base) > 1e-9 else "–")
    return (f"<tr><td>{label}</td><td class=num>{_fmt(tuned, pct)}</td>"
            f"<td class=num>{_fmt(base, pct)}</td>"
            f"<td class='num {cls}'>{mark} {_fmt(tuned - base, pct)}</td></tr>")


def to_text(r: EvalReport, tuned_name: str = "tuned", base_name: str = "baseline") -> str:
    t, b = r.tuned, r.baseline
    L = [f"decisiontune eval — {r.n_examples} examples, {r.n_questions} questions",
         f"tuned: {tuned_name}   baseline: {base_name}", ""]
    L.append(f"{'metric':<22}{'tuned':>10}{'baseline':>10}{'delta':>10}")
    L.append("-" * 54)
    for label, key, pct, lb in [("accuracy", "accuracy", True, False),
                                ("ECE (calibration)", "ece", False, True),
                                ("Brier score", "brier", False, True),
                                ("latency p50 ms", None, False, True),
                                ("latency p95 ms", None, False, True)]:
        if key:
            tv, bv = t[key], b[key]
        else:
            k = "p50" if "p50" in label else "p95"
            tv, bv = t["latency_ms"][k], b["latency_ms"][k]
        L.append(f"{label:<22}{tv:>10.3f}{bv:>10.3f}{tv - bv:>+10.3f}")
    L += ["", f"tuned-vs-baseline agreement: {r.agreement:.1%}", ""]
    delta = t["accuracy"] - b["accuracy"]
    lo, hi = r.delta_ci
    L.append(f"accuracy delta: {delta:+.1%}  (paired bootstrap 95% CI: {lo:+.1%} to {hi:+.1%})")
    ds = r.delta_summary
    L.append(f"per-question: {ds.get('fixed', 0)} fixed, {ds.get('regressed', 0)} regressed, "
             f"{ds.get('still_correct', 0)} still right, {ds.get('still_wrong', 0)} still wrong")
    regs = [d for d in r.deltas if d["verdict"] == "regressed"][:5]
    if regs:
        L.append("top regressions (tuning broke these):")
        for d in regs:
            L.append(f"  {d['source']} {d['qid']}: truth={d['truth']} "
                     f"baseline={d['baseline_pred']} ({d['baseline_conf']:.0%}) -> "
                     f"tuned={d['tuned_pred']} ({d['tuned_conf']:.0%})")
    L.append("per question type:")
    for qtype, m in r.per_type.items():
        L.append(f"  {qtype}: tuned acc {m['tuned']['accuracy']:.1%} "
                 f"(n={m['tuned']['n']}) vs baseline {m['baseline']['accuracy']:.1%} "
                 f"(n={m['baseline']['n']})")
    if r.errors:
        L += ["", f"errors ({len(r.errors)} shown):"] + [f"  - {e}" for e in r.errors]
    return "\n".join(L)


def to_markdown(r: EvalReport, tuned_name: str = "tuned", base_name: str = "baseline") -> str:
    t, b = r.tuned, r.baseline
    L = [f"# decisiontune eval report",
         f"{r.n_examples} examples, {r.n_questions} questions — "
         f"tuned `{tuned_name}` vs baseline `{base_name}`", "",
         "| metric | tuned | baseline | delta |",
         "|---|---|---|---|"]
    for label, key, pct, lb in [("accuracy", "accuracy", True, False),
                                ("ECE", "ece", False, True),
                                ("Brier", "brier", False, True)]:
        L.append(f"| {label} | {_fmt(t[key], pct)} | {_fmt(b[key], pct)} | {_fmt(t[key] - b[key], pct)} |")
    L.append(f"| latency p50 | {t['latency_ms']['p50']:.1f} ms | {b['latency_ms']['p50']:.1f} ms | "
             f"{t['latency_ms']['p50'] - b['latency_ms']['p50']:+.1f} ms |")
    L.append(f"| latency p95 | {t['latency_ms']['p95']:.1f} ms | {b['latency_ms']['p95']:.1f} ms | "
             f"{t['latency_ms']['p95'] - b['latency_ms']['p95']:+.1f} ms |")
    L += ["", f"Tuned-vs-baseline agreement: **{r.agreement:.1%}**", ""]
    delta = t["accuracy"] - b["accuracy"]
    lo, hi = r.delta_ci
    L.append(f"Accuracy delta: **{delta:+.1%}** (paired bootstrap 95% CI: {lo:+.1%} to {hi:+.1%})")
    ds = r.delta_summary
    L.append(f"Per-question: **{ds.get('fixed', 0)}** fixed, **{ds.get('regressed', 0)}** regressed, "
             f"{ds.get('still_correct', 0)} still right, {ds.get('still_wrong', 0)} still wrong")
    regs = [d for d in r.deltas if d["verdict"] == "regressed"][:5]
    if regs:
        L += ["", "Top regressions (tuning broke these):", "",
              "| example | question | truth | baseline | tuned |",
              "|---|---|---|---|---|"]
        for d in regs:
            L.append(f"| {d['source']} | {d['qid']} | {d['truth']} | "
                     f"{d['baseline_pred']} ({d['baseline_conf']:.0%}) | "
                     f"{d['tuned_pred']} ({d['tuned_conf']:.0%}) |")
    return "\n".join(L)


def to_html(r: EvalReport, tuned_name: str = "tuned", base_name: str = "baseline") -> str:
    t, b = r.tuned, r.baseline

    def bar(v: float, base: bool = False) -> str:
        return f"<div class='bar{' base' if base else ''}'><i style='width:{min(v * 100, 100):.1f}%'></i></div>"

    cards = (
        f"<div class=card><div class=k>Accuracy — tuned</div><div class=v>{t['accuracy']:.1%}</div>{bar(t['accuracy'])}</div>"
        f"<div class=card><div class=k>Accuracy — baseline</div><div class=v>{b['accuracy']:.1%}</div>{bar(b['accuracy'], True)}</div>"
        f"<div class=card><div class=k>Agreement</div><div class=v>{r.agreement:.1%}</div>{bar(r.agreement)}</div>"
    )
    rows = (
        _cmp_row("Accuracy", t["accuracy"], b["accuracy"], pct=True)
        + _cmp_row("ECE (calibration error)", t["ece"], b["ece"], lower_better=True)
        + _cmp_row("Brier score", t["brier"], b["brier"], lower_better=True)
        + _cmp_row("Latency p50 (ms)", t["latency_ms"]["p50"], b["latency_ms"]["p50"], lower_better=True)
        + _cmp_row("Latency p95 (ms)", t["latency_ms"]["p95"], b["latency_ms"]["p95"], lower_better=True)
    )
    type_rows = "".join(
        f"<tr><td>{qt}</td><td class=num>{m['tuned']['accuracy']:.1%}</td>"
        f"<td class=num>{m['baseline']['accuracy']:.1%}</td>"
        f"<td class=num>{m['tuned']['n']} / {m['baseline']['n']}</td></tr>"
        for qt, m in r.per_type.items()
    )
    ds = r.delta_summary
    delta = t["accuracy"] - b["accuracy"]
    lo, hi = r.delta_ci
    change_cards = (
        f"<div class=card><div class=k>Fixed by tuning</div><div class=v>{ds.get('fixed', 0)}</div></div>"
        f"<div class=card><div class=k>Regressed by tuning</div><div class=v>{ds.get('regressed', 0)}</div></div>"
        f"<div class=card><div class=k>Accuracy delta (95% CI)</div>"
        f"<div class=v>{delta:+.1%}<small> {lo:+.1%} to {hi:+.1%}</small></div></div>"
    )
    regs = [d for d in r.deltas if d["verdict"] == "regressed"][:5]
    if regs:
        reg_rows = "".join(
            f"<tr><td>{_html.escape(str(d['source']))}</td><td>{_html.escape(str(d['qid']))}</td>"
            f"<td>{_html.escape(str(d['truth']))}</td>"
            f"<td class=num>{_html.escape(str(d['baseline_pred']))} ({d['baseline_conf']:.0%})</td>"
            f"<td class=num>{_html.escape(str(d['tuned_pred']))} ({d['tuned_conf']:.0%})</td></tr>"
            for d in regs
        )
        reg_section = ("<h2>Regressions — tuning broke these</h2>"
                       "<table><tr><th>Example</th><th>Question</th><th>Truth</th>"
                       "<th class=num>Baseline</th><th class=num>Tuned</th></tr>"
                       f"{reg_rows}</table>")
    else:
        reg_section = ("<div class=note>No regressions — tuning didn't break anything "
                       "the baseline got right.</div>")
    verdict = ("The tuned model beats the baseline on accuracy."
               if t["accuracy"] > b["accuracy"]
               else "The tuned model does not beat the baseline on accuracy — "
                    "check label quality and training data size.")
    return f"""<!doctype html><html><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>decisiontune — eval report</title><style>{HTML_CSS}</style></head><body>
<div class=wrap>
<header><h1>decisiontune <span>/ eval report</span></h1></header>
<p class=meta>{r.n_examples} examples · {r.n_questions} questions ·
tuned <b>{tuned_name}</b> vs baseline <b>{base_name}</b></p>
<div class=grid>{cards}</div>
<h2>Head-to-head</h2>
<table><tr><th>Metric</th><th class=num>Tuned</th><th class=num>Baseline</th><th class=num>Delta</th></tr>{rows}</table>
<h2>By question type</h2>
<table><tr><th>Type</th><th class=num>Tuned acc</th><th class=num>Baseline acc</th><th class=num>n (tuned/base)</th></tr>{type_rows}</table>
<h2>What changed</h2>
<div class=grid>{change_cards}</div>
{reg_section}
<div class=note>{verdict} Agreement between the two is {r.agreement:.1%} —
high agreement with higher tuned accuracy means the fine-tune kept the base
behavior and fixed its mistakes.</div>
<footer>Generated by decisiontune · free, self-hosted decision-model tuning</footer>
</div></body></html>"""
