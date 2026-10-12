"""decisiontune CLI: validate, split, train, serve, eval."""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import webbrowser

from . import __version__
from .dataset import check_label_coverage, load_jsonl, split, stats, write_jsonl
from .evaluate import EvalConfig, run_eval
from .report import to_html, to_markdown, to_text
from .train import TrainConfig, plan, train

DEFAULT_MODEL = "Clef-Flash-9B"  # HuggingFace id of Cloudflare's Clef-flash 9B

# Training recipes per model family (merged from clef-finetuner's SPEC).
# A preset fills in any train flag the user left at its default; explicit
# flags always win. Model ids below are the family names — set --model to the
# exact HuggingFace repo id (Cloudflare publishes them under Apache 2.0).
PRESETS = {
    "clef-flash": {"lora_r": 16, "lora_alpha": 32, "epochs": 3, "batch_size": 4,
                   "notes": "9B: 4-bit QLoRA fits a 24GB GPU."},
    "clef": {"lora_r": 16, "lora_alpha": 32, "epochs": 3, "batch_size": 2,
             "notes": "27B: needs ~48GB+ VRAM; full precision is out of scope."},
    "clef-omni": {"lora_r": 16, "lora_alpha": 32, "epochs": 3, "batch_size": 2,
                  "notes": "30B multimodal: text-only tuning works today; "
                           "media[] encoders follow the omni release helpers."},
}
_TRAIN_DEFAULTS = {"lora_r": 16, "lora_alpha": 32, "epochs": 3, "batch_size": 4}


def cmd_validate(args) -> int:
    examples = load_jsonl(args.data)
    s = stats(examples)
    print(f"{args.data}: {s['examples']} examples, {s['questions_total']} questions")
    print(f"  by type: {s['by_type']}")
    print(f"  avg state length: {s['avg_state_chars']} chars")
    for key, dist in s["label_distribution"].items():
        print(f"  {key}: {dist}")
    for w in check_label_coverage(examples):
        print(f"  WARNING: {w}")
    print("dataset OK")
    return 0


def cmd_split(args) -> int:
    examples = load_jsonl(args.data)
    train_ex, eval_ex = split(examples, eval_fraction=args.eval_fraction, seed=args.seed)
    write_jsonl(args.train_out, train_ex)
    write_jsonl(args.eval_out, eval_ex)
    print(f"train: {len(train_ex)} -> {args.train_out}")
    print(f"eval:  {len(eval_ex)} -> {args.eval_out}")
    return 0


def cmd_train(args) -> int:
    examples = load_jsonl(args.data)
    if args.preset:
        p = PRESETS[args.preset]
        for dest, default in _TRAIN_DEFAULTS.items():
            if getattr(args, dest) == default:
                setattr(args, dest, p[dest])
        print(f"preset '{args.preset}': {p['notes']}")
    cfg = TrainConfig(
        model_id=args.model, out_dir=args.out, lora_r=args.lora_r,
        lora_alpha=args.lora_alpha, learning_rate=args.lr, epochs=args.epochs,
        batch_size=args.batch_size, seed=args.seed,
        load_in_4bit=not args.no_4bit,
    )
    if args.dry_run:
        p = plan(examples, cfg)
        print(json.dumps(p, indent=2))
        return 0
    out = train(examples, cfg)
    print(f"adapter saved to {out}")
    print(f"serve it with: decisiontune serve --adapter {out} --model {cfg.model_id}")
    return 0


def _make_backend(args):
    from .backends import HFBackend, MockBackend
    if args.mock:
        mode, _, prob = args.mock.partition(":")
        return MockBackend(mode=mode or "first",
                           first_prob=float(prob) if prob else 0.8,
                           latency_ms=args.mock_latency), f"mock:{args.mock}"
    return HFBackend(model_id=args.model, adapter_path=args.adapter,
                     load_in_4bit=not args.no_4bit), \
        args.model + (f"+{args.adapter}" if args.adapter else "")


def cmd_serve(args) -> int:
    from .serve import run_server
    backend, model_name = _make_backend(args)
    server = run_server(backend, args.name or model_name,
                        host=args.host, port=args.port, api_key=args.api_key)
    url = f"http://{args.host}:{args.port}"
    print(f"serving {model_name} at {url}  (Jev-API: POST {url}/v1/systemone)")
    if args.open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    return 0


def cmd_eval(args) -> int:
    examples = load_jsonl(args.data)
    cfg = EvalConfig(tuned_url=args.tuned, baseline_url=args.baseline,
                     tuned_key=args.tuned_key, baseline_key=args.baseline_key,
                     timeout=args.timeout)
    report = run_eval(examples, cfg)
    text = to_text(report, args.tuned_name, args.baseline_name)
    print(text)
    if args.markdown_out:
        with open(args.markdown_out, "w") as f:
            f.write(to_markdown(report, args.tuned_name, args.baseline_name))
        print(f"\nmarkdown report -> {args.markdown_out}")
    if args.html_out:
        with open(args.html_out, "w") as f:
            f.write(to_html(report, args.tuned_name, args.baseline_name))
        print(f"html report -> {args.html_out}")
        if args.open:
            webbrowser.open("file://" + os.path.abspath(args.html_out))
    if args.min_accuracy is not None:
        acc = report.tuned["accuracy"]
        if acc < args.min_accuracy:
            print(f"tuned accuracy {acc:.3f} below --min-accuracy {args.min_accuracy}")
            return 2
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="decisiontune",
        description="Teach an open decision model your own decisions — "
                    "free, self-hosted LoRA fine-tuning + eval rig.")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    v = sub.add_parser("validate", help="validate a JSONL dataset")
    v.add_argument("data", help="dataset .jsonl file")
    v.set_defaults(fn=cmd_validate)

    s = sub.add_parser("split", help="split a dataset into train/eval")
    s.add_argument("data", help="dataset .jsonl file")
    s.add_argument("--train-out", default="train.jsonl")
    s.add_argument("--eval-out", default="eval.jsonl")
    s.add_argument("--eval-fraction", type=float, default=0.2)
    s.add_argument("--seed", type=int, default=42)
    s.set_defaults(fn=cmd_split)

    t = sub.add_parser("train", help="LoRA fine-tune a decision model")
    t.add_argument("--data", required=True, help="training .jsonl")
    t.add_argument("--model", default=DEFAULT_MODEL, help="base HF model id")
    t.add_argument("--preset", choices=sorted(PRESETS),
                   help="training recipe for a model family (explicit flags override)")
    t.add_argument("--out", default="adapters/run1", help="adapter output dir")
    t.add_argument("--lora-r", type=int, default=16)
    t.add_argument("--lora-alpha", type=int, default=32)
    t.add_argument("--lr", type=float, default=2e-4)
    t.add_argument("--epochs", type=int, default=3)
    t.add_argument("--batch-size", type=int, default=4)
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--no-4bit", action="store_true", help="disable 4-bit QLoRA")
    t.add_argument("--dry-run", action="store_true",
                   help="validate + print the training plan without touching torch")
    t.set_defaults(fn=cmd_train)

    sv = sub.add_parser("serve", help="serve a model as a Jev-API-compatible endpoint")
    sv.add_argument("--model", default=DEFAULT_MODEL)
    sv.add_argument("--adapter", default=None, help="PEFT LoRA adapter dir")
    sv.add_argument("--mock", default=None,
                    help="use mock backend: first[:prob] | uniform | hash (tests/demos)")
    sv.add_argument("--mock-latency", type=float, default=5.0)
    sv.add_argument("--name", default=None, help="model name reported by the API")
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=8080)
    sv.add_argument("--api-key", default=None)
    sv.add_argument("--open", action="store_true", help="open the dashboard in a browser")
    sv.set_defaults(fn=cmd_serve)

    e = sub.add_parser("eval", help="compare tuned vs baseline on a held-out split")
    e.add_argument("--data", required=True, help="eval .jsonl (with labels)")
    e.add_argument("--tuned", required=True, help="tuned endpoint base URL")
    e.add_argument("--baseline", required=True, help="baseline endpoint base URL")
    e.add_argument("--tuned-name", default="tuned")
    e.add_argument("--baseline-name", default="baseline")
    e.add_argument("--tuned-key", default=None)
    e.add_argument("--baseline-key", default=None)
    e.add_argument("--timeout", type=float, default=60.0)
    e.add_argument("--min-accuracy", type=float, default=None,
                   help="exit 2 if tuned accuracy is below this (CI gate)")
    e.add_argument("--markdown-out", default=None)
    e.add_argument("--html-out", default=None)
    e.add_argument("--open", action="store_true")
    e.set_defaults(fn=cmd_eval)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.fn(args)
    except Exception as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
