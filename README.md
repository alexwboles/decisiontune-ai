# decisiontune-ai

**Teach an open decision model your own decisions.** Free, self-hosted LoRA fine-tuning + eval rig for Jev-API-compatible decision models.

Cloudflare's Clef family (Clef / Clef-flash / Clef-omni) ships Apache-2.0 open weights — but Cloudflare's fine-tuning story is a hosted, hands-on RL service. There is no packaged free tool that lets you train *your* labels into these models and serve the result yourself. decisiontune is that tool:

- **CLI** that takes labeled decision examples (JSONL: state + typed questions + correct answers) and LoRA-fine-tunes a decision model (default target: Clef-flash 9B; any HuggingFace causal-LM id works)
- **Jev-API-compatible server** — `POST /v1/systemone` with `{state, questions}` → calibrated per-option probabilities. Drop-in for any Jev/Clef client (same endpoint shape SelfJev and Clef use)
- **Eval rig** — tuned model vs a baseline (untuned base, chat LLM, hosted API) on a held-out split: accuracy, agreement rate, latency p50/p95, calibration (ECE), Brier score, per-question fixed/regressed deltas, and a paired bootstrap 95% CI on the accuracy delta — printed as text, Markdown, and a static HTML report
- **Dogfood datasets** — guardrail allow/block (40 examples, including 24 HQ agent-action cases), ticket routing, lead qualification examples in `examples/`

Everything runs locally. No API keys, no usage bills.

> **Merged 2026-10-12:** this repo absorbed `clef-finetuner` (same-day sibling build). Its 24 HQ guardrail examples, `--min-accuracy` CI gate, GPU training doc, and `media[]` multimodal schema reservation now live here. The old repo is archived and points here.

## Quickstart (5 minutes, no GPU)

```bash
pip install decisiontune-ai        # stdlib only; no torch needed for the tour

# 1. Validate your labeled data
decisiontune validate examples/guardrail.jsonl

# 2. Split into train / eval
decisiontune split examples/guardrail.jsonl --train-out train.jsonl --eval-out eval.jsonl

# 3. Inspect the training plan (no GPU needed)
decisiontune train --data train.jsonl --dry-run

# 4. Serve two mock endpoints and compare them
decisiontune serve --mock hash --port 8080 &      # pretend "tuned" model
decisiontune serve --mock uniform --port 8081 &   # baseline
decisiontune eval --data eval.jsonl \
  --tuned http://127.0.0.1:8080 --baseline http://127.0.0.1:8081 \
  --html-out report.html --open
```

## Dataset format

One JSON object per line. Types mirror the Jev-API: `noul` (yes/no), `choice` (pick one), `score` (1..N ordinal).

```json
{"state": "Refund $49.99 — 4th request in 6 days, each just under the $50 auto-approve limit.",
 "questions": [
   {"id": "allow", "type": "noul", "question": "Allow automatically?"},
   {"id": "risk", "type": "score", "question": "Abuse risk?", "levels": 5,
    "labels": ["none", "low", "moderate", "high", "critical"]}],
 "answers": {"allow": "no", "risk": 4}}
```

`decisiontune validate` checks the schema, prints label distributions, and warns about labels that never appear in the data.

Optional per-example `media`: a list of local image/audio/video paths, reserved for multimodal decision models (Clef-omni). Text-only training and serving ignore it; it is validated and round-tripped.

```json
{"state": "Screenshot review: checkout button visible.",
 "media": ["shots/checkout.png"],
 "questions": [{"id": "ok", "type": "noul", "question": "Is the checkout flow intact?"}],
 "answers": {"ok": "yes"}}
```

## Real-model training (GPU machine)

You need one CUDA GPU (24 GB is enough for 4-bit QLoRA on the 9B model).

```bash
# 1. Install the training extras
pip install "decisiontune-ai[train]"

# 2. Log in to Hugging Face (Clef weights are gated Apache-2.0; accept the license once)
huggingface-cli login

# 3. Train a LoRA adapter on your labels (Clef-flash 9B, QLoRA 4-bit)
decisiontune train \
  --data train.jsonl \
  --model <clef-flash-hf-id> \
  --preset clef-flash \
  --out adapters/guardrail-v1

# 4. Serve the tuned model (Jev-API compatible)
decisiontune serve \
  --model <clef-flash-hf-id> \
  --adapter adapters/guardrail-v1 \
  --port 8080

# 5. Compare against the untuned base model on held-out data (exit 2 if below 0.7 accuracy)
decisiontune serve --model <clef-flash-hf-id> --port 8081 &
decisiontune eval --data eval.jsonl \
  --tuned http://127.0.0.1:8080 --tuned-name guardrail-v1 \
  --baseline http://127.0.0.1:8081 --baseline-name clef-flash-base \
  --min-accuracy 0.7 --html-out report.html --open
```

`--preset clef-flash|clef|clef-omni` fills in the known-good training recipe for that model family (explicit flags override the preset). See `docs/GPU_TRAINING.md` for the full GPU walkthrough.

Replace `<clef-flash-hf-id>` with the HuggingFace repo id of Cloudflare's Clef-flash 9B weights (Apache-2.0; check the Cloudflare blog/HuggingFace for the current id). Any causal-LM checkpoint works — the trainer renders each (state, question) pair with `decisiontune/prompts.py` and trains the model to emit the correct option; the server scores every option as a completion and soft-maxes the log-probs.

Practical notes:

- 50–200 labeled examples per question is a reasonable start; `validate` tells you the label mix.
- Keep a genuinely held-out eval split (`split --seed` is deterministic) — eval numbers only mean something on unseen data. `split` warns when an eval example near-duplicates a train example (eval leakage check).
- Gate production traffic on confidence: the HTML report shows calibration (ECE). If ECE is high, collect more labels before trusting thresholds.
- This is supervised fine-tuning (SFT/LoRA), the standard community recipe for adapting decision models (cf. JevBench's open LoRA entries). Cloudflare's own RL pipeline is proprietary; SFT gets you most of the way for classification-style decisions.

## API reference

`POST /v1/systemone`

```json
{
  "state": "Ticket #1042: double charge on invoice.",
  "questions": [
    {"id": "team", "type": "choice", "question": "Which team should own this ticket?",
     "options": ["billing", "technical", "sales"],
     "criteria": {"billing": "invoices, charges, refunds"}},
    {"id": "urgent", "type": "noul", "question": "Is this urgent?"}
  ]
}
```

```json
{
  "answers": {
    "team": {"billing": 0.91, "technical": 0.07, "sales": 0.02},
    "urgent": {"yes": 0.83, "no": 0.17}
  },
  "model": "clef-flash-9b+adapters/guardrail-v1",
  "latency_ms": 38.4
}
```

Also: `GET /v1/models` (backend info), `GET /` (status dashboard).

Point any Jev-compatible client at it (e.g. SelfJev-style `BASE_URL` env config) — the endpoint shape matches.

## Eval metrics

| Metric | What it tells you |
|---|---|
| Accuracy | argmax vs your labels, overall + per question type |
| Agreement | how often tuned and baseline pick the same option |
| Latency p50/p95 | per-decision wall time, per endpoint |
| ECE | expected calibration error — can you trust the probabilities as thresholds? |
| Brier | mean squared error of the full probability vector |

## Tests

```bash
bash test/smoke.sh   # unit tests (38) + full CLI pipeline on mock backends (13 checks)
```

The suite is hermetic: mock backends, no model downloads, no GPU, stdlib only.

## License

Apache-2.0. Built free, for free.
