# Apple M5 native DiffusionGemma measurements

Measured on 2026-09-23 using real local model weights and Metal execution. No synthetic completion
server or external inference API was used.

## Environment

- Apple M5, 24 GB unified memory, macOS 26.6.2.
- MLX 0.32.2, MLX-VLM 0.7.2, Transformers 5.17.0, Python 3.12.12.
- Model: `mlx-community/diffusiongemma-26B-A4B-it-4bit`.
- Weight/tokenizer revision: `a7a81407613811e8ba63af92ac0d852b809e191f`.
- One native sequence at a time; context limit 4,096; output limit 256; temperature zero.
- MLX allocator cache capped at 256 MiB; device-recommended allocation limit retained.
- Each configuration runs in a separate process, with one warmup then five measured evaluations.
- One short refund question, repeated unchanged. No quality or calibration benchmark is implied.

## Native results

| Maximum canvas tokens | Prefix cache | Median seconds | Min–max seconds | Peak MLX GB | Cached input tokens |
| --- | --- | --- | --- | --- | --- |
| 256 | False | 1.150 | 1.140–2.442 | 17.66 | [0] |
| 64 | False | 0.590 | 0.586–0.833 | 16.85 | [0] |
| 64 | True | 0.415 | 0.410–0.425 | 16.91 | [148] |

All 15 measured decisions returned `noul=1.0`, with zero repair generations. Timings include prompt
compilation, generation and verification, but exclude model loading and the explicit warmup.
Peak memory is MLX allocation, not total process or system memory. Full per-completion denoising,
canvas-work and token statistics are retained in the JSON reports.

The cached configuration repeats the exact same prompt after warmup. It does not measure cache
reuse between different questions or unseen states. Five samples on one simple prompt cannot
establish p95/p99, sustained throughput, general quality, or superiority to Jev. Smaller canvases
change generation behavior; broader held-out evaluation remains necessary.

## API and SDK checks

The Mac HTTP service also completed a mixed Choice/Score/Noul request (`examples/triage.json`) with
HTTP 200: billing was selected, urgency scored 1.9 on a 0–2 rubric, and refund probability was 1.0.
The first mixed request after startup took 11.625 seconds, including first-use runtime effects.
This three-question HTTP result is not directly comparable with the warmed single-question table.
The existing real-engine smoke script and official TypeSafe Python SDK example both passed against
the Mac service. See [the captured mixed response](http-mixed.json).

## Reproduce

From the repository root, after installing the Mac extra:

```sh
export HF_HOME="${HF_HOME:-$HOME/.cache/diffusion-jev/huggingface}"
uv run --extra mac python -m vllm_verifier.engine.cli \
  --runtime mlx --revision a7a81407613811e8ba63af92ac0d852b809e191f \
  --dataset docs/benchmarks/2026-09-23-mac/workload.jsonl \
  --output /tmp/mac-canvas256.json --canvas-tokens 256 \
  --output-tokens 256 --warmup 1 --repeat 5 --no-prefix-caching
```

Repeat in separate processes with `--canvas-tokens 64` and then with `--prefix-caching`, using new
output paths. Dataset paths in captured JSON identify the original temporary input; `workload.jsonl`
contains that exact input. Stop other model processes before running to avoid duplicate weight loads.
