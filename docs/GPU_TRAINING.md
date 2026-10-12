# Running real LoRA training on a GPU machine

`decisiontune train --dry-run` prints the full training plan with zero GPU.
This doc covers the real thing: LoRA-tuning an open decision model
(Clef-flash 9B, Clef 27B, Clef-omni 30B, Strands Decider 2B, …) on your own
examples.

## What you need

- A machine with a CUDA GPU. 24 GB VRAM is enough for a 9B model at 4-bit
  quantization (the default); 41 GB+ for longer contexts or higher rank.
- Python 3.10+, ~30 GB disk for the base weights.
- The base weights from Hugging Face (Cloudflare publishes Clef/Clef-flash/
  Clef-omni under Apache 2.0 — set `--model` to the exact repo id).

## Steps

1. Validate and split your dataset (any machine):
   ```
   decisiontune validate data.jsonl
   decisiontune split data.jsonl --train-out train.jsonl --eval-out eval.jsonl
   ```

2. Preview the training plan:
   ```
   decisiontune train --data train.jsonl --preset clef-flash --dry-run
   ```

3. On the GPU machine:
   ```
   pip install "decisiontune-ai[train]"   # torch transformers peft datasets bitsandbytes
   decisiontune train --data train.jsonl --model <hf-repo-id> \
       --preset clef-flash --out adapters/guardrail-v1
   ```

4. Evaluate against a baseline, then serve:
   ```
   decisiontune serve --model <hf-repo-id> --adapter adapters/guardrail-v1 --port 8080 &
   decisiontune serve --model <hf-repo-id> --port 8081 &   # untuned baseline
   decisiontune eval --data eval.jsonl \
       --tuned http://127.0.0.1:8080 --baseline http://127.0.0.1:8081 \
       --min-accuracy 0.7 --html-out report.html
   ```

## Notes

- `--preset clef-flash|clef|clef-omni` fills in the known-good recipe for that
  family (rank, epochs, batch size); any explicit flag overrides the preset.
- Start with the preset defaults. If eval ECE is high, train longer before
  touching the rank.
- Mock backends (`decisiontune serve --mock …`) share the same eval and serve
  code paths — a pipeline that passes on mocks end-to-end will work unchanged
  with real weights.
- Multimodal (`media` field): the dataset schema accepts per-example media
  paths, reserved for Clef-omni-style tuning. Text-only tuning works today;
  media encoders follow the omni release helpers.
- Exit codes: `eval` returns 2 when tuned accuracy falls below
  `--min-accuracy`, so CI can gate on it.
