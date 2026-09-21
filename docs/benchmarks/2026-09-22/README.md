# Synthetic HTTP baseline — 2026-09-22

These are production gateway + synthetic demo-engine HTTP measurements, **not DiffusionGemma
inference results, Jev comparisons, or isolated gateway overhead**. No model weights were loaded.
All 2,304 measured requests succeeded; 288 additional requests warmed the pipeline.

## Environment and method

- Gateway source: `9024244db6c568283aa2a9da0e3e109c3da9834c`.
- Gateway image: `sha256:689043a122ea2391d4fa12ce08c8cacb86105f1efd317d096ef4447624120d13`.
- Demo image: `sha256:187a37578fece8da4d14116d3a308e4aa96663e6794ddef5df2d3af54f92baa2`.
- macOS ARM64 client; Docker Linux aarch64, 10 CPUs, 8,217,382,912 bytes allocated RAM.
- One gateway process, 8 upstream slots, 32 admitted requests, default timeouts/retry settings.
- Each request: 1,024 ASCII state bytes, repeated three-class Choice questions, no quality labels.
- Each cell: 32 warmup requests then 256 measured requests; same persistent client per phase.
- Closed-loop load; cells run sequentially, one trial per cell. Other host workloads were not isolated.
- Latency starts at HTTP dispatch, includes gateway queuing and synthetic upstream HTTP, and excludes
  the load generator queue. Percentiles use nearest rank; throughput counts whole requests, not questions.

| Questions | Concurrent requests | p50 ms | p95 ms | p99 ms | Requests/s | Errors |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | 2.62 | 3.88 | 4.91 | 376.5 | 0 |
| 1 | 8 | 7.97 | 12.16 | 14.39 | 968.2 | 0 |
| 1 | 32 | 38.81 | 44.37 | 49.75 | 238.0 | 0 |
| 8 | 1 | 6.10 | 7.28 | 8.82 | 161.1 | 0 |
| 8 | 8 | 41.94 | 52.39 | 85.16 | 180.5 | 0 |
| 8 | 32 | 188.29 | 201.06 | 205.26 | 168.9 | 0 |
| 32 | 1 | 20.71 | 23.51 | 27.57 | 47.6 | 0 |
| 32 | 8 | 162.64 | 181.80 | 195.32 | 48.4 | 0 |
| 32 | 32 | 672.18 | 717.56 | 721.47 | 47.2 | 0 |

Question count increases upstream call count and shared-slot contention. These observations support
profiling that path on a GPU deployment, but do not establish its share of real inference latency.
The synthetic server itself, HTTP connections, Docker networking and client scheduling affect results.
At 256 samples per cell, p99 is only a rough tail estimate; there are no confidence intervals.
The one-question/32-client cell includes a roughly one-second tail outlier above p99, reducing
whole-run throughput despite its modest p99. Raw rows retain that observation; it was not discarded.
No claims about classification accuracy or calibration can be made from uniform synthetic responses.

## Reproduce

Build the production and separate demo images and run them on a free local port:

```sh
VERIFIER_HTTP_PORT=18080 docker compose -p verifier-benchmark -f compose.demo.yaml up -d --build --wait
VERIFIER_API_KEY=local-demo-key uv run python scripts/benchmark_matrix.py \
  --url http://127.0.0.1:18080 --questions 1 8 32 --concurrency 1 8 32 \
  --state-bytes 1024 --warmup 32 --requests 256 \
  --run-label 'Synthetic upstream; record local hardware and image revisions' \
  --output-dir /tmp/verifier-matrix
VERIFIER_HTTP_PORT=18080 docker compose -p verifier-benchmark -f compose.demo.yaml down
```

[Summary](summary.json) and [compressed raw rows](raw-results.json.gz) retain measured and warmup
results. Dataset paths in the reports identify original temporary files; the matrix script recreates
the exact workload. Neither the production image nor package includes these artifacts.

TypeSafe advertises 193.6× faster workflows and shows a 114 ms example on its
[official site](https://typesafe.ai/). Those are vendor workflow claims, not measurements under this
workload. Actual comparison requires a running GPU inference endpoint and an authenticated Jev
endpoint, matching tasks/concurrency, and separate quality evaluation.
