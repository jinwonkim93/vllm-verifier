# Kai on Apple M5

Measurements from an Apple M5 Mac with 24 GB unified memory, macOS 26.6.2, Python 3.12.12,
PyTorch 2.14.0 and Transformers 4.57.6. Kai runs on MPS in FP32, with no CPU fallback,
quantization, response cache or text generation. Weight revision:
`7185f514f54b8f93c55998b1e8f9c5cc67f0d029`.

The [authored smoke workload](../../../benchmarks/decision-smoke.jsonl) contains eight English
and Korean support requests. Each has a routing Choice, a refund Noul, and a three-level urgency
Score. It is a small integration workload, not a held-out model-quality benchmark.

## Local inference

Each submission contains all eight requests / 24 decisions. Both configurations use the same
model, inputs, grouping by question type, and FP32 arithmetic. Physical batch size is the only
execution setting changed. One complete warmup precedes five measured submissions. Loading,
HTTP transport and request collection are excluded; tokenization, physical scheduling, device
transfer, model execution and typed readout are included.

| Maximum questions per physical batch | Median submission | Range | Output tokens |
| --- | ---: | ---: | ---: |
| 1 | 541.76 ms | 540.39–551.10 ms | 0 |
| 8 | 280.62 ms | 275.96–283.51 ms | 0 |

Batching reduces the median submission time by approximately **48% (1.93× speedup)** on this
workload. These are latencies for 24 decisions together, not single-decision latencies. Five
samples do not establish tail latency or a service-level objective. Model loading took about
20 seconds in these runs, including checksum and weight validation.

- [Physical batch size 1](mps-b1.json)
- [Physical batch size 8 and CPU parity](mps-b8.json)

The batched MPS probabilities differ from isolated CPU FP32 inference by at most
**0.000001371** across this workload. All top decisions agree; the preselected absolute
probability tolerance was 0.0001. This checks the port's numerical behavior, not correctness of
the model's judgments. No numerical identity across arbitrary hardware or inputs is promised.

## HTTP serving

The server uses physical batch size 8, a 4,096 padded-token budget, at most eight requests per
submission, a 2 ms collection window, and one inference owner. Each test warms the complete
workload once, then measures ten repetitions: 80 requests / 240 decisions. The latency includes
HTTP, queueing, tokenization and inference. Authentication was disabled on the loopback listener.

| Concurrent requests | Successful requests | Requests/s | Request p50 | Request p95 |
| --- | ---: | ---: | ---: | ---: |
| 1 | 80 / 80 | 14.00 | 70.82 ms | 78.02 ms |
| 8 | 80 / 80 | 26.25 | 285.74 ms | 415.17 ms |

Higher concurrency improves throughput but adds queueing latency. Each request contains three
questions; requests/s must not be presented as decisions/s. Reported p95 values are empirical
values from these 80-request samples, not long-duration production guarantees.

- [Concurrency 1](http-c1.json)
- [Concurrency 8](http-c8.json)

## Quality and limitations

Routing is correct on all eight examples; refund detection is correct on seven of eight at a
0.5 threshold: **15/16 classification decisions** overall. The missed refund request is
“My parcel never arrived. Please refund the order today.” Its refund probability is about 0.397.

Urgency Score is close to the middle level for all eight examples, with approximately
**0.748 mean absolute error on the 0–2 scale**. The CPU reference exhibits the same behavior.
The runtime does not adjust probabilities to fit the labels. This checkpoint is not established
as reliable for this urgency task, and this smoke set does not establish broader English or
Korean accuracy. Repeating the same examples ten times does not create 160 independent labels.

These measurements do not compare against hosted Jev. Previous DiffusionGemma results used
a different workload and output path and are not a controlled model comparison.

## Reproduction

From the repository root, install the Decision extra in its own environment as described in
[Decision setup](../../decision-models.md).

```sh
.venv-decision/bin/python scripts/benchmark_decision.py \
  --dataset benchmarks/decision-smoke.jsonl --batch-size 8 --compare-cpu \
  --output /tmp/kai-b8.json
.venv-decision/bin/python scripts/benchmark_decision.py \
  --dataset benchmarks/decision-smoke.jsonl --batch-size 1 --output /tmp/kai-b1.json
```

Start `vllm-verifier --runtime decision --port 18080` using that environment. Run each HTTP
measurement separately; change `--concurrency` from 1 to 8 for the second run.

```sh
.venv-decision/bin/python scripts/benchmark.py --url http://127.0.0.1:18080 \
  --dataset benchmarks/decision-smoke.jsonl --concurrency 1 --warmup 1 --repeat 10 \
  --run-label 'Kai FP32 on Apple M5; batch8 tokens4096 wait2ms' \
  --output /tmp/kai-http-c1.json
```

The actual Mac server also passed the real TypeSafe SDK example, the mixed-response smoke
check, and rejection of over-budget input while a neighboring valid request completed. The
Docker image passed the separate HTTP-fixture integration suite; that check does not execute
Kai or expose Metal inside Docker.
