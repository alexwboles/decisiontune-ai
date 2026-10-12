"""Jev-API-compatible serving: POST /v1/systemone with a state + typed
questions, get calibrated probabilities back. Stdlib only — no web framework.

Request:
    {"state": "...", "questions": [{"id": "q0", "type": "choice",
      "question": "...", "options": [...], "criteria": {...}}, ...]}
Response:
    {"answers": {"q0": {"billing": 0.82, "technical": 0.18}},
     "model": "clef-flash-9b+lora", "latency_ms": 42.1}

GET / shows a small status dashboard. GET /v1/models describes the backend.
"""
from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Dict, List, Optional
from urllib.parse import urlparse

from .backends import Backend
from .dataset import DatasetError, Question, parse_question

MAX_BODY = 2 * 1024 * 1024


class ServerStats:
    def __init__(self):
        self._lock = threading.Lock()
        self.started_at = time.time()
        self.requests = 0
        self.questions = 0
        self.latencies: List[float] = []
        self.recent: List[Dict] = []

    def record(self, n_questions: int, latency_ms: float, state: str,
               question_ids: List[str]) -> None:
        with self._lock:
            self.requests += 1
            self.questions += n_questions
            self.latencies.append(latency_ms)
            if len(self.latencies) > 1000:
                self.latencies = self.latencies[-1000:]
            self.recent.append({
                "at": time.strftime("%H:%M:%S"),
                "questions": question_ids,
                "state_preview": state[:80],
                "latency_ms": round(latency_ms, 1),
            })
            self.recent = self.recent[-20:]

    def snapshot(self) -> Dict:
        with self._lock:
            lat = sorted(self.latencies)
            def pct(p):
                return round(lat[min(int(p * len(lat)), len(lat) - 1)], 1) if lat else 0.0
            return {
                "uptime_s": round(time.time() - self.started_at, 1),
                "requests": self.requests,
                "questions": self.questions,
                "latency_ms": {"p50": pct(0.50), "p95": pct(0.95), "max": pct(1.0) if lat else 0.0},
                "recent": list(reversed(self.recent)),
            }


def _json(handler: BaseHTTPRequestHandler, status: int, obj: dict) -> None:
    body = json.dumps(obj).encode()
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


DASHBOARD_CSS = """
:root{--ink:#1a1d21;--muted:#5c6470;--line:#e2e5e9;--bg:#f7f8f9;--card:#ffffff;--accent:#0f766e;--accent-soft:#e6f4f2}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Inter,Roboto,sans-serif}
.wrap{max-width:960px;margin:0 auto;padding:32px 24px}
header{display:flex;justify-content:space-between;align-items:baseline;border-bottom:2px solid var(--ink);padding-bottom:16px;margin-bottom:24px}
h1{font-size:22px;margin:0;letter-spacing:-.02em}h1 span{color:var(--accent)}
.sub{color:var(--muted);font-size:13px}
.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:24px}
.card{background:var(--card);border:1px solid var(--line);padding:16px}
.card .k{font-size:11px;text-transform:uppercase;letter-spacing:.08em;color:var(--muted);margin-bottom:6px}
.card .v{font-size:26px;font-weight:650}
table{width:100%;border-collapse:collapse;background:var(--card);font-size:13px}
th{text-align:left;font-size:11px;text-transform:uppercase;letter-spacing:.06em;color:var(--muted);
border-bottom:1px solid var(--line);padding:10px 12px}
td{border-bottom:1px solid var(--line);padding:10px 12px}
.pill{display:inline-block;background:var(--accent-soft);color:var(--accent);font-weight:600;font-size:12px;padding:2px 8px}
code{background:#eef0f2;padding:1px 5px;font-size:12px}
@media(max-width:640px){.grid{grid-template-columns:repeat(2,1fr)}}
"""


def dashboard_html(model_name: str, snap: Dict) -> str:
    lat = snap["latency_ms"]
    rows = "".join(
        f"<tr><td>{r['at']}</td><td><code>{', '.join(r['questions'])}</code></td>"
        f"<td>{r['state_preview']}</td><td>{r['latency_ms']} ms</td></tr>"
        for r in snap["recent"]
    ) or "<tr><td colspan=4 style='color:#888'>No decisions yet — POST to /v1/systemone.</td></tr>"
    return f"""<!doctype html><html><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>decisiontune — server</title><style>{DASHBOARD_CSS}</style></head><body>
<div class=wrap>
<header><h1>decisiontune <span>/ server</span></h1>
<div class=sub>Jev-API-compatible decision endpoint</div></header>
<div class=grid>
<div class=card><div class=k>Model</div><div class=v style="font-size:15px">{model_name}</div></div>
<div class=card><div class=k>Decisions served</div><div class=v>{snap['questions']}</div></div>
<div class=card><div class=k>Latency p50 / p95</div><div class=v>{lat['p50']} / {lat['p95']}<span style="font-size:13px"> ms</span></div></div>
<div class=card><div class=k>Uptime</div><div class=v>{snap['uptime_s']}<span style="font-size:13px"> s</span></div></div>
</div>
<table><tr><th>Time</th><th>Questions</th><th>State</th><th>Latency</th></tr>{rows}</table>
<p class=sub style="margin-top:16px">POST <code>/v1/systemone</code> with
<code>{{"state": "...", "questions": [...]}}</code> — types: <span class=pill>noul</span>
<span class=pill>choice</span> <span class=pill>score</span></p>
</div></body></html>"""


class Handler(BaseHTTPRequestHandler):
    backend: Backend = None  # set by run_server
    model_name: str = "decisiontune"
    stats: ServerStats = None
    api_key: Optional[str] = None

    def log_message(self, *args):
        pass  # keep test output clean

    def _read_json(self):
        length = int(self.headers.get("Content-Length", 0))
        if length > MAX_BODY:
            raise ValueError("body too large")
        return json.loads(self.rfile.read(length) or b"{}")

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/v1/models":
            _json(self, 200, {"models": [self.backend.info()], "model_name": self.model_name})
        elif path == "/" or path == "/index.html":
            html = dashboard_html(self.model_name, self.stats.snapshot()).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(html)))
            self.end_headers()
            self.wfile.write(html)
        else:
            _json(self, 404, {"error": "not found"})

    def do_POST(self):
        if urlparse(self.path).path != "/v1/systemone":
            _json(self, 404, {"error": "not found"})
            return
        if self.api_key and self.headers.get("X-API-Key") != self.api_key:
            _json(self, 401, {"error": "bad api key"})
            return
        t0 = time.perf_counter()
        try:
            body = self._read_json()
        except Exception as e:
            _json(self, 400, {"error": f"invalid JSON: {e}"})
            return
        try:
            state = body.get("state")
            if not isinstance(state, str) or not state:
                raise DatasetError("request needs a non-empty 'state' string")
            qraw = body.get("questions")
            if not isinstance(qraw, list) or not qraw:
                raise DatasetError("request needs a non-empty 'questions' list")
            questions = [parse_question(q, "request") for q in qraw]
        except DatasetError as e:
            _json(self, 400, {"error": str(e)})
            return
        try:
            answers = self.backend.decide(state, questions)
        except Exception as e:
            _json(self, 500, {"error": f"backend failed: {e}"})
            return
        latency_ms = (time.perf_counter() - t0) * 1000
        self.stats.record(len(questions), latency_ms, state, [q.id for q in questions])
        _json(self, 200, {"answers": answers, "model": self.model_name,
                          "latency_ms": round(latency_ms, 2)})


def run_server(backend: Backend, model_name: str, host: str = "127.0.0.1",
               port: int = 8080, api_key: Optional[str] = None) -> ThreadingHTTPServer:
    # Bind per-server state on a fresh Handler subclass: class attributes on
    # Handler itself would leak across servers in the same process.
    class _Handler(Handler):
        pass
    _Handler.backend = backend
    _Handler.model_name = model_name
    _Handler.stats = ServerStats()
    _Handler.api_key = api_key
    return ThreadingHTTPServer((host, port), _Handler)


def decide_remote(base_url: str, state: str, questions: List[Question],
                  api_key: Optional[str] = None, timeout: float = 60.0) -> Dict:
    """Client for any Jev-API-compatible /v1/systemone endpoint (local or hosted)."""
    import urllib.request
    payload = json.dumps({
        "state": state,
        "questions": [
            {k: v for k, v in {
                "id": q.id, "type": q.type, "question": q.question,
                "options": q.options or None, "criteria": q.criteria or None,
                "levels": q.levels or None, "labels": q.labels or None,
            }.items() if v is not None}
            for q in questions
        ],
    }).encode()
    req = urllib.request.Request(base_url.rstrip("/") + "/v1/systemone",
                                 data=payload, method="POST",
                                 headers={"Content-Type": "application/json"})
    if api_key:
        req.add_header("X-API-Key", api_key)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())
