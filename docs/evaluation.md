# Evaluation

## Local contract suite

`uv run pytest --cov=vllm_verifier` runs without a GPU. It checks typed arithmetic, rejection of
malformed outputs, retries and usage, HTTP errors, actual TypeSafe SDK parsing, image validation,
isolated prompts, shared concurrency, overload and cancellation. Mocked completions prove service
behavior, not model intelligence.

## Real engine smoke test

Docker protocol integration can be checked independently:

```sh
docker build -t vllm-verifier:compat .
uv run python scripts/check_container.py --image vllm-verifier:compat
```

The script launches a gateway and a deterministic HTTP fixture in an isolated Docker network,
uses real HTTP and the official SDK, and removes its containers/network afterwards. It checks
upstream authentication, concurrent requests, repair usage, overload/invalid-output errors and
vision observation. It does not run a real vLLM engine. This check also runs in container CI.

With the gateway and model loaded:

```sh
export VERIFIER_API_KEY=your-local-key
uv run python scripts/smoke.py --url http://127.0.0.1:8080
```

This deliberately rejects the separate demo server. A passing smoke test confirms the mixed typed-response
pipeline works on your engine; it does not establish accuracy, calibration or real-time deadlines.
For vision, run `uv run python examples/vision.py ./sample.png` and inspect `observation` as well
as the decision. Include images whose correct labels and visible text you already know.

## Benchmark

```sh
uv run python scripts/benchmark.py \
  --concurrency 4 --repeat 10 \
  --run-label 'GPU=<model>; vLLM=<image digest>; weights=<revision>' \
  --output benchmark-results-gpu.json
```

The report records completed/failed requests, successful-only p50/p95/p99 latency, successful
requests/sec, token totals, raw per-request rows, model names, errors and demo model markers.
Latency is measured from HTTP dispatch, excluding the script's work queue. Throughput includes
all elapsed worker time. Warm up separately and retain cold-start measurements separately.
Failed requests are counted and make the process exit nonzero, rather than disappearing from
results. Do not quote successful-only latency without the failure rate.

For labeled cases it also calculates Choice/Noul accuracy, Brier loss and Score mean absolute
error. Choice Brier loss sums across classes; Noul uses the scalar Bernoulli loss, so the combined
mean is workload-dependent and should not be compared across different task mixes.

`examples/evaluation.jsonl` contains **seven small wiring examples**, not a scientific benchmark.
Repeating them does not add independent quality evidence. Replace them with representative,
held-out labeled data, including ambiguous states, Korean text, long inputs, prompt injection,
near-duplicate labels, maximum choice counts and actual visual tasks. Example record:

```json
{"request":{"model":"jev-latest","state":"Refund please","questions":{"refund":{"type":"noul","instructions":"Refund requested?"}}},"expected":{"refund":true}}
```

Record gateway commit, all settings, engine image digest/version, model revision, GPU/VRAM, driver,
concurrency, question count, state length and workload. Compare with an autoregressive baseline
and Jev only under equivalent workloads. Diffusion's generation throughput advantage need not
translate to lower latency for very short decision outputs, especially with repair generations.

No GPU results are checked in because the model has not been run in this development environment.

## Question-count and concurrency sweep

```sh
uv run python scripts/benchmark_matrix.py \
  --url http://127.0.0.1:8080 \
  --questions 1 8 32 --concurrency 1 8 32 \
  --state-bytes 1024 --warmup 32 --requests 256 \
  --run-label 'hardware, image/model revisions, gateway settings' \
  --output-dir /tmp/verifier-matrix
```

This generates a repeated three-class Choice workload with an ASCII state, warms each cell
separately, and retains datasets, warmup results, measured raw rows and a summary. It is a closed-loop
load test, not a representative quality dataset or an arrival-rate/SLO benchmark. Warmup failures
stop the sweep; measured failures remain visible and cause a nonzero final exit code. Output paths
are reused on subsequent runs, so choose a new directory to preserve each experiment.

The same script can target a Jev-compatible service using `VERIFIER_API_KEY` and `--url`; doing so
sends requests and may incur provider charges. Use the same workload and concurrency for comparisons,
and record network location. Synthetic demo responses measure the whole HTTP fixture pipeline,
including Docker networking and the fixture server, not isolated gateway overhead or GPU inference.

See the [local synthetic baseline](benchmarks/2026-09-22/README.md) for measured results and limits.
