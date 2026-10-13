"""Eval report rendering: text (terminal), Markdown, and a premium static HTML page."""
from __future__ import annotations

import html as _html
from datetime import datetime

from .evaluate import EvalReport

HTML_CSS = """
:root{
  --ink:#14161a; --ink2:#1e2229; --paper:#f7f7f5; --card:#ffffff;
  --line:#e3e3de; --line-strong:#c9c9c3;
  --text:#1b1e22; --muted:#5d646e; --faint:#8b919a;
  --amber:#d97f06; --amber-deep:#a85f04; --amber-soft:#fdf0da;
  --green:#1e7e4d; --green-soft:#e4f2e9;
  --red:#b3261e; --red-soft:#fbe7e5;
  --mono:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
  --sans:-apple-system,BlinkMacSystemFont,"Segoe UI",Inter,Roboto,Helvetica,Arial,sans-serif;
}
*{box-sizing:border-box}
body{margin:0;background:var(--paper);color:var(--text);font-family:var(--sans);
  font-size:15px;line-height:1.55}
/* ---- ink experiment header ---- */
.band{background:var(--ink);color:#eef0f3;border-bottom:4px solid var(--amber)}
.band .wrap{max-width:1080px;margin:0 auto;padding:34px 24px 28px}
.crumb{font-family:var(--mono);font-size:11px;letter-spacing:.22em;text-transform:uppercase;
  color:var(--amber);margin-bottom:10px}
.band h1{margin:0;font-size:30px;letter-spacing:-.02em;font-weight:700}
.band h1 .vs{color:#7c8592;font-weight:400;font-size:22px;margin:0 10px}
.band .meta{color:#9aa2ad;font-size:13px;margin-top:10px;font-family:var(--mono)}
.band .meta b{color:#eef0f3;font-weight:600}
/* ---- body ---- */
.wrap{max-width:1080px;margin:0 auto;padding:0 24px 64px}
h2{font-size:13px;text-transform:uppercase;letter-spacing:.09em;color:var(--muted);
  margin:38px 0 12px;padding-bottom:8px;border-bottom:2px solid var(--ink)}
/* verdict banner */
.verdict{margin:26px 0 0;padding:16px 20px;display:flex;gap:16px;align-items:center;
  border:1px solid var(--line-strong);border-left:6px solid var(--amber);background:var(--card)}
.verdict.win{border-left-color:var(--green)}
.verdict.loss{border-left-color:var(--red)}
.verdict .v-mark{font-family:var(--mono);font-size:11px;font-weight:700;letter-spacing:.14em;
  text-transform:uppercase;padding:4px 10px;color:#fff;background:var(--amber-deep)}
.verdict.win .v-mark{background:var(--green)}
.verdict.loss .v-mark{background:var(--red)}
.verdict .v-text{font-size:15px}
.verdict .v-text b{font-variant-numeric:tabular-nums}
.verdict .v-ci{color:var(--muted);font-size:13px;font-family:var(--mono)}
/* KPI cards */
.grid{display:grid;grid-template-columns:repeat(5,1fr);gap:12px;margin-top:18px}
.card{background:var(--card);border:1px solid var(--line);border-top:3px solid var(--amber);padding:16px}
.card .k{font-size:10.5px;text-transform:uppercase;letter-spacing:.08em;color:var(--muted);margin-bottom:8px}
.card .v{font-size:30px;font-weight:700;letter-spacing:-.02em;font-variant-numeric:tabular-nums}
.card .v small{font-size:13px;color:var(--muted);font-weight:400}
.card .d{margin-top:6px;font-size:12.5px;font-family:var(--mono)}
.up{color:var(--green);font-weight:700}.down{color:var(--red);font-weight:700}.flat{color:var(--faint)}
/* tables */
table{width:100%;border-collapse:collapse;background:var(--card);font-size:13.5px}
th{text-align:left;font-size:11px;text-transform:uppercase;letter-spacing:.06em;color:var(--muted);
  border-bottom:2px solid var(--ink);padding:10px 14px;background:#fbfbfa}
td{border-bottom:1px solid var(--line);padding:10px 14px;vertical-align:middle}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
tr:last-child td{border-bottom:none}
.dbar{display:inline-block;width:90px;height:8px;background:#eef0f1;position:relative;
  vertical-align:middle;margin-left:10px}
.dbar i{position:absolute;top:0;bottom:0;background:var(--green)}
.dbar i.neg{background:var(--red)}
.dbar i.zero{background:#b9bec5}
/* verdict pills */
.pill{font-family:var(--mono);font-size:10.5px;font-weight:700;letter-spacing:.08em;
  text-transform:uppercase;padding:2px 9px;border:1px solid}
.pill.fixed{color:var(--green);border-color:var(--green);background:var(--green-soft)}
.pill.regressed{color:var(--red);border-color:var(--red);background:var(--red-soft)}
.pill.still_correct{color:var(--muted);border-color:var(--line-strong)}
.pill.still_wrong{color:var(--amber-deep);border-color:var(--amber);background:var(--amber-soft)}
/* calibration bars */
.calib{display:grid;grid-template-columns:1fr 1fr;gap:12px}
.calib .cbox{background:var(--card);border:1px solid var(--line);padding:16px 18px}
.calib .cname{font-weight:700;font-size:14px;margin-bottom:10px}
.cbar{height:14px;background:#eef0f1;position:relative;margin:8px 0 4px}
.cbar i{position:absolute;left:0;top:0;bottom:0;background:var(--amber)}
.cbar.base i{background:#9aa4ae}
.clabel{font-size:12px;color:var(--muted);font-family:var(--mono)}
/* per-type grouped bars */
.trow{display:grid;grid-template-columns:110px 1fr 1fr 150px;gap:14px;align-items:center;
  background:var(--card);border:1px solid var(--line);border-bottom:none;padding:10px 14px;font-size:13px}
.trow:last-child{border-bottom:1px solid var(--line)}
.trow .tname{font-family:var(--mono);font-weight:700}
.tbar{display:block;height:10px;background:#eef0f1;position:relative}
.tbar i{position:absolute;left:0;top:0;bottom:0;background:var(--amber)}
.tbar.base i{background:#9aa4ae}
.tval{font-family:var(--mono);font-size:12px;text-align:right;font-variant-numeric:tabular-nums}
.thead{display:grid;grid-template-columns:110px 1fr 1fr 150px;gap:14px;padding:8px 14px;
  font-size:11px;text-transform:uppercase;letter-spacing:.06em;color:var(--muted)}
/* CI whisker */
.ci-track{position:relative;height:10px;background:#eef0f1;margin:10px 0 4px}
.ci-track .zero{position:absolute;top:-3px;bottom:-3px;width:2px;background:#7c8592}
.ci-track .rng{position:absolute;top:2px;bottom:2px;background:var(--ink)}
.ci-lab{display:flex;justify-content:space-between;font-family:var(--mono);font-size:11px;color:var(--muted)}
.ci-cap{text-align:center;font-family:var(--mono);font-size:10.5px;color:var(--faint);margin-top:2px}
.note{background:var(--amber-soft);border-left:4px solid var(--amber);padding:12px 16px;
  font-size:13.5px;margin-top:22px}
.note.good{background:var(--green-soft);border-left-color:var(--green)}
.err{background:var(--red-soft);border:1px solid var(--red);padding:12px 16px;font-size:13.5px;
  margin-top:12px}
.method{margin-top:34px;padding:16px 18px;border:1px solid var(--line);background:var(--card);
  font-size:12.5px;color:var(--muted)}
.method b{color:var(--text)}
footer{margin-top:28px;color:var(--faint);font-size:12px;font-family:var(--mono)}
.cap{font-size:12px;color:var(--faint);margin-top:8px}
.tscroll{overflow-x:auto;-webkit-overflow-scrolling:touch}
.tscroll table{min-width:640px}
@media(max-width:860px){.grid{grid-template-columns:repeat(2,1fr)}.calib{grid-template-columns:1fr}
  .trow,.thead{grid-template-columns:80px 1fr 90px}.trow .tbar.base,.thead span:nth-child(3){display:none}}
@media(prefers-reduced-motion:reduce){*{transition:none!important;animation:none!important}}
"""


def _fmt(x: float, pct: bool = False) -> str:
    return f"{x * 100:.1f}%" if pct else f"{x:.3f}"


def _delta_mark(tuned: float, base: float, lower_better: bool = False) -> tuple:
    """(css class, glyph) for a tuned-vs-baseline delta."""
    d = tuned - base
    if abs(d) <= 1e-9:
        return "flat", "–"
    better = d < 0 if lower_better else d > 0
    return ("up" if better else "down"), ("▲" if d > 0 else "▼")


def _delta_bar(d: float, scale: float, cls: str) -> str:
    """Inline bar for a delta value on a symmetric -scale..+scale axis.
    cls is the _delta_mark class (up/down/flat) so lower-better metrics
    color correctly."""
    w = min(abs(d) / scale * 50, 50) if scale > 0 else 0
    bcls = "zero" if cls == "flat" else ("neg" if cls == "down" else "")
    side = "right:50%" if d < 0 else "left:50%"
    return f"<span class=dbar><i class='{bcls}' style='{side};width:{w:.1f}%'></i></span>"


def _ci_whisker(lo: float, hi: float, delta: float) -> str:
    """Whisker plot of the bootstrap CI on a symmetric axis."""
    m = max(abs(lo), abs(hi), abs(delta), 0.05) * 1.15
    def pos(x: float) -> float:
        return (x / m * 50) + 50
    l, h, d = pos(lo), pos(hi), pos(delta)
    return (
        f"<div class=ci-track><span class=zero style='left:50%'></span>"
        f"<span class=rng style='left:{min(l, h):.1f}%;width:{abs(h - l):.1f}%'></span>"
        f"<span class=rng style='left:{d:.1f}%;width:2px;background:var(--amber)'></span></div>"
        f"<div class=ci-lab><span>{lo:+.1%}</span><span>{hi:+.1%}</span></div>"
        f"<div class=ci-cap>paired bootstrap 95% CI</div>"
    )


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


def _cmp_row(label: str, tuned: float, base: float, pct: bool = False,
             lower_better: bool = False, bar_scale: float = 0.2) -> str:
    cls, mark = _delta_mark(tuned, base, lower_better)
    d = tuned - base
    return (f"<tr><td>{label}</td><td class=num>{_fmt(tuned, pct)}</td>"
            f"<td class=num>{_fmt(base, pct)}</td>"
            f"<td class='num {cls}'>{mark} {_fmt(d, pct)}{_delta_bar(d, bar_scale, cls)}</td></tr>")


def _delta_table_row(d: dict) -> str:
    v = d["verdict"]
    pill_label = {"fixed": "fixed", "regressed": "regressed",
                  "still_correct": "still right", "still_wrong": "still wrong"}[v]
    return (
        f"<tr><td class=num>{_html.escape(str(d['source']))}</td>"
        f"<td class=num>{_html.escape(str(d['qid']))}</td>"
        f"<td>{_html.escape(str(d.get('qtype', '–')))}</td>"
        f"<td class=num>{_html.escape(str(d['truth']))}</td>"
        f"<td class=num>{_html.escape(str(d['baseline_pred']))} "
        f"<span class=dim>({_html.escape(f'{d['baseline_conf']:.0%}')})</span></td>"
        f"<td class=num>{_html.escape(str(d['tuned_pred']))} "
        f"<span class=dim>({_html.escape(f'{d['tuned_conf']:.0%}')})</span></td>"
        f"<td><span class='pill {v}'>{pill_label}</span></td></tr>"
    )


def to_html(r: EvalReport, tuned_name: str = "tuned", base_name: str = "baseline") -> str:
    t, b = r.tuned, r.baseline
    tn, bn = _html.escape(tuned_name), _html.escape(base_name)
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z")
    delta = t["accuracy"] - b["accuracy"]
    lo, hi = r.delta_ci
    ds = r.delta_summary

    wins = t["accuracy"] > b["accuracy"]
    vcls = "win" if wins else "loss"
    vword = "wins" if wins else "does not beat"
    verdict = (
        f"<div class='verdict {vcls}'><span class=v-mark>{'tuned wins' if wins else 'no win'}</span>"
        f"<div><div class=v-text>The tuned model <b>{vword}</b> the baseline on accuracy: "
        f"<b>{delta:+.1%}</b></div>"
        f"<div class=v-ci>paired bootstrap 95% CI {lo:+.1%} to {hi:+.1%} · "
        f"agreement {r.agreement:.1%}</div></div></div>"
    )

    dcls, dmark = _delta_mark(t["accuracy"], b["accuracy"])
    ecls, emark = _delta_mark(t["ece"], b["ece"], lower_better=True)
    cards = (
        f"<div class=card><div class=k>Tuned accuracy</div><div class=v>{t['accuracy']:.1%}</div>"
        f"<div class='d {dcls}'>{dmark} {_fmt(delta, True)} vs baseline</div></div>"
        f"<div class=card><div class=k>Accuracy delta</div><div class=v>{delta:+.1%}</div>"
        f"{_ci_whisker(lo, hi, delta)}</div>"
        f"<div class=card><div class=k>Agreement</div><div class=v>{r.agreement:.1%}</div>"
        f"<div class=d>argmax match rate</div></div>"
        f"<div class=card><div class=k>ECE — tuned</div><div class=v>{t['ece']:.3f}</div>"
        f"<div class='d {ecls}'>{emark} {_fmt(t['ece'] - b['ece'])} vs baseline</div></div>"
        f"<div class=card><div class=k>Brier — tuned</div><div class=v>{t['brier']:.3f}</div>"
        f"<div class=d>baseline {_fmt(b['brier'])}</div></div>"
    )

    rows = (
        _cmp_row("Accuracy", t["accuracy"], b["accuracy"], pct=True, bar_scale=0.25)
        + _cmp_row("ECE (calibration error)", t["ece"], b["ece"], lower_better=True, bar_scale=0.5)
        + _cmp_row("Brier score", t["brier"], b["brier"], lower_better=True, bar_scale=0.5)
        + _cmp_row("Latency p50 (ms)", t["latency_ms"]["p50"], b["latency_ms"]["p50"],
                   lower_better=True, bar_scale=20)
        + _cmp_row("Latency p95 (ms)", t["latency_ms"]["p95"], b["latency_ms"]["p95"],
                   lower_better=True, bar_scale=20)
    )

    calib = (
        f"<div class=calib>"
        f"<div class=cbox><div class=cname>{tn}</div>"
        f"<div class=cbar><i style='width:{min(t['ece'] * 100, 100):.1f}%'></i></div>"
        f"<div class=clabel>ECE {t['ece']:.3f} — lower is better calibrated</div></div>"
        f"<div class=cbox><div class=cname>{bn}</div>"
        f"<div class='cbar base'><i style='width:{min(b['ece'] * 100, 100):.1f}%'></i></div>"
        f"<div class=clabel>ECE {b['ece']:.3f} — lower is better calibrated</div></div>"
        f"</div>"
    )

    thead = ("<div class=thead><span>Type</span><span>Tuned accuracy</span>"
             "<span>Baseline accuracy</span><span>n (tuned / base)</span></div>")
    type_rows = "".join(
        f"<div class=trow><span class=tname>{_html.escape(qt)}</span>"
        f"<span><span class=tbar><i style='width:{m['tuned']['accuracy'] * 100:.1f}%'></i></span></span>"
        f"<span><span class='tbar base'><i style='width:{m['baseline']['accuracy'] * 100:.1f}%'></i></span></span>"
        f"<span class=tval>{m['tuned']['accuracy']:.1%} / {m['baseline']['accuracy']:.1%} "
        f"· {m['tuned']['n']} / {m['baseline']['n']}</span></div>"
        for qt, m in r.per_type.items()
    )

    change_cards = (
        f"<div class=card><div class=k>Fixed by tuning</div><div class=v>{ds.get('fixed', 0)}</div>"
        f"<div class=d>baseline wrong → tuned right</div></div>"
        f"<div class=card><div class=k>Regressed by tuning</div><div class=v>{ds.get('regressed', 0)}</div>"
        f"<div class=d>baseline right → tuned wrong</div></div>"
        f"<div class=card><div class=k>Still right</div><div class=v>{ds.get('still_correct', 0)}</div>"
        f"<div class=d>both correct</div></div>"
        f"<div class=card><div class=k>Still wrong</div><div class=v>{ds.get('still_wrong', 0)}</div>"
        f"<div class=d>both wrong</div></div>"
    )

    regs = [d for d in r.deltas if d["verdict"] == "regressed"][:10]
    if regs:
        reg_rows = "".join(_delta_table_row(d) for d in regs)
        reg_section = ("<h2>Regressions — tuning broke these</h2>"
                       "<div class=tscroll><table><tr><th class=num>Example</th><th class=num>Question</th>"
                       "<th>Type</th><th class=num>Truth</th><th class=num>Baseline</th>"
                       "<th class=num>Tuned</th><th>Verdict</th></tr>"
                       f"{reg_rows}</table></div>")
    else:
        reg_section = ("<div class='note good'>No regressions — tuning didn't break anything "
                       "the baseline got right.</div>")

    fixed = [d for d in r.deltas if d["verdict"] == "fixed"][:10]
    if fixed:
        fix_rows = "".join(_delta_table_row(d) for d in fixed)
        fix_section = ("<h2>Fixed — tuning repaired these</h2>"
                       "<div class=tscroll><table><tr><th class=num>Example</th><th class=num>Question</th>"
                       "<th>Type</th><th class=num>Truth</th><th class=num>Baseline</th>"
                       "<th class=num>Tuned</th><th>Verdict</th></tr>"
                       f"{fix_rows}</table></div>")
    else:
        fix_section = ""

    if r.errors:
        err_section = ("<h2>Errors</h2>" + "".join(
            f"<div class=err>{_html.escape(e)}</div>" for e in r.errors))
    else:
        err_section = ""

    method = (
        "<div class=method><b>How to read this.</b> Accuracy is the share of questions where the "
        "model's top-probability option matches the label. <b>ECE</b> (expected calibration error) "
        "measures whether stated confidence matches reality — lower is better. <b>Brier</b> scores "
        "the full probability distribution — lower is better. <b>Agreement</b> is how often both "
        "models pick the same option. The accuracy delta's 95% CI comes from a paired bootstrap "
        "over per-question correctness. High agreement plus higher tuned accuracy means the "
        "fine-tune kept base behavior and fixed its mistakes.</div>"
    )

    return f"""<!doctype html><html><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>decisiontune — eval report</title><style>{HTML_CSS}</style></head><body>
<div class=band><div class=wrap style="padding-bottom:28px">
<div class=crumb>decisiontune / eval report</div>
<h1>{tn}<span class=vs>vs</span>{bn}</h1>
<div class=meta><b>{r.n_examples}</b> examples · <b>{r.n_questions}</b> questions · {stamp}</div>
</div></div>
<div class=wrap>
{verdict}
<div class=grid>{cards}</div>
<h2>Head-to-head</h2>
<div class=tscroll><table><tr><th>Metric</th><th class=num>Tuned</th><th class=num>Baseline</th><th class=num>Delta</th></tr>{rows}</table></div>
<h2>Calibration</h2>
{calib}
<h2>By question type</h2>
{thead}{type_rows}
<h2>What changed</h2>
<div class=grid style="grid-template-columns:repeat(4,1fr)">{change_cards}</div>
{reg_section}
{fix_section}
{err_section}
{method}
<footer>Generated by decisiontune · free, self-hosted decision-model tuning</footer>
</div></body></html>"""
