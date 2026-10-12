#!/usr/bin/env bash
# decisiontune end-to-end smoke test: full pipeline on mock backends.
# validate -> split -> train (dry-run) -> serve x2 -> eval -> reports.
set -euo pipefail
cd "$(dirname "$0")/.."

pass=0; fail=0
ok()   { echo "ok   $1"; pass=$((pass+1)); }
bad()  { echo "FAIL $1"; fail=$((fail+1)); }

echo "== unit tests =="
python3 -m unittest discover -s tests > /tmp/dt-unit.log 2>&1
if grep -q "^OK$" /tmp/dt-unit.log; then
  ok "unit tests"
else
  bad "unit tests"; tail -5 /tmp/dt-unit.log
fi

echo "== CLI pipeline =="
CLI="python3 -m decisiontune.cli"

$CLI validate examples/guardrail.jsonl > /tmp/dt-validate.log 2>&1 \
  && grep -q "dataset OK" /tmp/dt-validate.log && ok "validate guardrail" || bad "validate guardrail"
$CLI validate examples/ticket-routing.jsonl > /dev/null 2>&1 && ok "validate ticket-routing" || bad "validate ticket-routing"
$CLI validate examples/lead-qualification.jsonl > /dev/null 2>&1 && ok "validate lead-qualification" || bad "validate lead-qualification"

rm -rf /tmp/dt-e2e && mkdir -p /tmp/dt-e2e
$CLI split examples/guardrail.jsonl --train-out /tmp/dt-e2e/train.jsonl \
  --eval-out /tmp/dt-e2e/eval.jsonl --eval-fraction 0.25 --seed 1 > /dev/null 2>&1 \
  && [ "$(python3 -c "from decisiontune.dataset import load_jsonl; print(len(load_jsonl('/tmp/dt-e2e/eval.jsonl')))")" = "10" ] \
  && ok "split 30/10" || bad "split"

$CLI train --data /tmp/dt-e2e/train.jsonl --model FakeOrg/fake-9b --dry-run \
  > /tmp/dt-plan.json 2>&1 \
  && python3 -c "
import json
from decisiontune.dataset import load_jsonl
p = json.load(open('/tmp/dt-plan.json'))
expected = sum(len(ex.questions) for ex in load_jsonl('/tmp/dt-e2e/train.jsonl'))
assert p['train_rows'] == expected, (p['train_rows'], expected)
" && ok "train dry-run plan" || bad "train dry-run plan"

echo "== serve + eval (mock backends) =="
$CLI serve --mock hash --name tuned-mock --port 18081 > /tmp/dt-tuned.log 2>&1 &
TUNED_PID=$!
$CLI serve --mock uniform --name baseline-mock --port 18082 > /tmp/dt-base.log 2>&1 &
BASE_PID=$!
trap 'kill $TUNED_PID $BASE_PID 2>/dev/null || true' EXIT
sleep 1.5

RESP=$(curl -s -X POST http://127.0.0.1:18081/v1/systemone -H 'Content-Type: application/json' \
  -d '{"state":"Refund $49.99, 4th request in 6 days.","questions":[{"id":"allow","type":"noul","question":"Allow automatically?"}]}')
echo "$RESP" | grep -q '"allow"' && ok "POST /v1/systemone" || bad "POST /v1/systemone"
python3 -c "
import json
probs = json.loads('''$RESP''')['answers']['allow']
assert abs(sum(probs.values()) - 1.0) < 1e-6, probs
" && ok "probabilities sum to 1" || bad "probabilities sum to 1"
curl -s http://127.0.0.1:18081/ | grep -q "decisiontune" && ok "dashboard HTML" || bad "dashboard HTML"
curl -s http://127.0.0.1:18081/v1/models | grep -q '"mock"' && ok "GET /v1/models" || bad "GET /v1/models"

$CLI eval --data /tmp/dt-e2e/eval.jsonl \
  --tuned http://127.0.0.1:18081 --baseline http://127.0.0.1:18082 \
  --tuned-name tuned-mock --baseline-name baseline-mock \
  --markdown-out /tmp/dt-e2e/report.md --html-out /tmp/dt-e2e/report.html \
  > /tmp/dt-eval.txt 2>&1 \
  && grep -q "agreement" /tmp/dt-eval.txt && ok "eval report text" || bad "eval report text"
grep -q "| metric |" /tmp/dt-e2e/report.md && ok "eval report markdown" || bad "eval report markdown"
grep -q "eval report" /tmp/dt-e2e/report.html && ! grep -q "border-radius" /tmp/dt-e2e/report.html \
  && ok "eval report html (sharp edges)" || bad "eval report html"

kill $TUNED_PID $BASE_PID 2>/dev/null || true
trap - EXIT

echo ""
echo "== $pass passed, $fail failed =="
[ "$fail" = "0" ]
