# Decision models on Apple Silicon

The `decision` runtime serves **Decision-1.0-Kai-0.6B** through the Jev-compatible
`/v1/systemone` endpoint. Kai directly scores the candidates supplied with each question.
It returns learned candidate probabilities without generating text or parsing JSON.

## Install and run

Use Python 3.12 and an Apple Silicon Mac. The Decision extra uses Transformers 4.57.6;
the DiffusionGemma MLX extra uses Transformers 5. Install them in separate environments.

```sh
UV_PROJECT_ENVIRONMENT=.venv-decision uv sync --locked --extra decision --python 3.12
UV_PROJECT_ENVIRONMENT=.venv-decision uv run --extra decision \
  vllm-verifier --runtime decision --port 18080
```

The first start downloads approximately 2.3 GB into `~/.cache/diffusion-jev/decision/`.
Loading verifies the pinned manifest and every source/weight checksum before importing
upstream code. Model startup finishes before the HTTP listener becomes ready.

```sh
curl http://127.0.0.1:18080/readyz
curl http://127.0.0.1:18080/v1/systemone \
  -H 'Content-Type: application/json' --data-binary @examples/triage.json
```

Set `VERIFIER_API_KEY` before exposing the server on a non-loopback interface. Existing
TypeSafe SDK clients can use `base_url="http://127.0.0.1:18080"` and `model="jev-latest"`.
Responses identify `llm-semantic-router/Decision-1.0-Kai-0.6B`.

The runtime explicitly selects MPS/Metal and never silently falls back to CPU. For a CPU
reference run, set `VERIFIER_DECISION_DEVICE=cpu`. Docker Desktop cannot expose Metal;
run the Decision model on the Mac host. The standard Docker image remains the HTTP gateway.

## Execution and limits

A bounded FIFO queue collects up to eight requests over a configurable 2 ms window. Each
request is validated in full before its questions enter model execution. Questions are grouped
by type and sorted by input length, then split into physical batches bounded by both row count
and padded input tokens. Answers are restored to the original request and question order.

The model and all inference calls belong to one worker thread. Only one batch submission can
execute at a time. Cancelled queued requests are skipped; active Metal operations finish safely.
Shutdown rejects pending requests and drains the active operation. A malformed request does
not invalidate neighboring requests. A model execution failure returns no partial answers.

| Setting | Default | Meaning |
| --- | --- | --- |
| `VERIFIER_DECISION_DEVICE` | `mps` | `mps` or explicit `cpu` |
| `VERIFIER_DECISION_CACHE_DIR` | `~/.cache/diffusion-jev` | Materialized, verified model cache |
| `VERIFIER_DECISION_BATCH_SIZE` | `8` | Maximum questions per physical forward |
| `VERIFIER_DECISION_BATCH_TOKENS` | `4096` | Rows × longest padded sequence per forward |
| `VERIFIER_DECISION_BATCH_REQUESTS` | `8` | Requests collected per submission; maximum 8 |
| `VERIFIER_DECISION_BATCH_WAIT_MS` | `2` | Collection window; set 0 to disable intentional waiting |
| `VERIFIER_MAX_REQUESTS` | `32` | HTTP admission and pending queue capacity |
| `VERIFIER_REQUEST_TIMEOUT` | `60` | HTTP deadline in seconds |

Generation settings (`MAX_OUTPUT_TOKENS`, `TEMPERATURE`, `VALIDATION_RETRIES`,
`MAX_CONCURRENCY`) do not control this direct scoring runtime. Use the Decision batch settings.
The HTTP contract permits up to 64 questions per request. The complete **state + question +
candidates + special tokens must fit 1,024 tokens per question**; inputs are never truncated.
Choice requires 2–255 candidates and Score 2–10 levels. Text must be nonempty. Omitted
instructions use “Evaluate the candidates against the state.” Objects and arrays are serialized
as deterministic JSON. Choice names are included with their descriptions, following the
upstream System One adapter. Question IDs are bookkeeping and never enter tokens.

Noul uses the two native yes/no candidates, with optional custom descriptions. Choice returns
the maximum-probability candidate; Score returns the expected zero-based level. Their
`confidence` is the maximum candidate probability, matching Kai's readout. Probabilities are
**uncalibrated** and candidate order can affect predictions. Output token usage is zero;
input usage sums complete per-question sequences, including repeated state.

This runtime is text-only. Other Decision 1.0 checkpoints, CUDA/ROCm execution, distributed
workers, KV caching, and token-level continuous batching are not implemented here.

Measured on an Apple M5 with 24 GB unified memory; see the
[latency, throughput, parity and quality report](benchmarks/2026-09-23-decision/README.md).

## Reproduce measurements

The authored smoke workload includes English/Korean routing, negation, refund conditions,
and ordered urgency. It exercises the API and runtime; it is not a held-out quality benchmark.

```sh
.venv-decision/bin/python scripts/benchmark_decision.py \
  --dataset benchmarks/decision-smoke.jsonl --output /tmp/kai-mps.json --compare-cpu
```

The report records the pinned revision, environment, full responses, timing samples and optional
CPU probability differences. Loading and warmup are excluded from measured submissions.
CPU parity compares isolated FP32 questions with the batched MPS results. Existing output files
are never overwritten. For HTTP latency, use `scripts/benchmark.py` against the running server.

## Upstream and license

- [Decision 1.0 announcement](https://vllm-sr.ai/blog/decision-models/)
- [Kai repository](https://huggingface.co/llm-semantic-router/Decision-1.0-Kai-0.6B)
- Pinned revision: `7185f514f54b8f93c55998b1e8f9c5cc67f0d029`.
- Pinned native manifest SHA256: `c1bf07ab1c4c3fa1f819256d3de858d1ed87869bdfa663553280d7e78b88bee4`.

The adapter loads the release's unchanged `native.artifacts.load_export` and native forward
implementation. It uses its original FP32 weights, tokenizer, packing, attention and typed
heads. The AMD-only CLI/device gate is replaced by our MPS/CPU lifecycle and scheduler; the
model's arithmetic is not rewritten. No global module monkey patch or `trust_remote_code=True`
is used. The downloaded, hash-verified native Python modules are still executable upstream code.

Model files are not distributed in the wheel or image. Decision contributions use Apache 2.0;
retained upstream/tokenizer terms are documented in the downloaded `LICENSES`, `NOTICE` and
[licensing status](https://huggingface.co/llm-semantic-router/Decision-1.0-Kai-0.6B/blob/main/LICENSING_STATUS.md).
