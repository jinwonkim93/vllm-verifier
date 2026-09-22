# Apple Silicon execution

Run DiffusionGemma locally with MLX/Metal. The same Jev-compatible text API and native decision
compiler are available on macOS; no NVIDIA server is required. GPU inference runs directly on the
Mac host. The Linux Docker image remains an HTTP gateway and does not provide this Metal runtime.

Verified on an M5 with 24 GB memory: native typed generation, mixed Choice/Score/Noul HTTP
requests, and the official TypeSafe SDK. See [measurements and raw results](benchmarks/2026-09-23-mac/README.md).

## Install

Requires an Apple Silicon Mac and Python 3.11 or later:

```sh
export HF_HOME="${HF_HOME:-$HOME/.cache/diffusion-jev/huggingface}"
uv sync --locked --extra mac
```

The Mac extra pins MLX-VLM 0.7.2. The default checkpoint is
[`mlx-community/diffusiongemma-26B-A4B-it-4bit`](https://huggingface.co/mlx-community/diffusiongemma-26B-A4B-it-4bit),
approximately 16.5 GB of weights. The first run downloads it to the configured Hugging Face cache. Additional
memory is required for activations and KV state; a 24 GB machine has limited headroom. The runtime
uses Metal's recommended allocation limit and caps MLX's allocator cache at 256 MiB, rather than
raising the machine's wired-memory limits. These are allocator limits, not a guarantee against swap.

## Start the API

```sh
VERIFIER_MAX_CONCURRENCY=1 \
VERIFIER_MAX_REQUESTS=2 \
VERIFIER_MAX_OUTPUT_TOKENS=256 \
VERIFIER_REQUEST_TIMEOUT=300 \
uv run --extra mac vllm-verifier --runtime mlx --port 18080
```

The server binds to loopback by default. Set `VERIFIER_API_KEY` before exposing it on another
interface. Startup loads the model before accepting requests; `/readyz` then reports readiness.

```sh
curl http://127.0.0.1:18080/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{"model":"jev-latest","state":"Please refund my duplicate payment.","questions":{"refund":{"type":"noul","instructions":"Is a refund requested?"}}}'
```

Select another compatible DiffusionGemma MLX checkpoint with `--model`. Set `VERIFIER_MLX_REVISION`
to a checkpoint commit for reproducible loads. The default context budget is 4,096 tokens including
output (`VERIFIER_MLX_MAX_MODEL_LEN`). Over-budget inputs are rejected, not truncated silently.

Gemma may emit an explicit thought-channel wrapper even with thinking disabled. The runtime
separates a well-formed leading thought channel before strict JSON validation. It rejects malformed
channel framing and does not search arbitrary prose for JSON.

The Mac API currently supports text only. Vision requests return `unsupported_modality`.
Model loading and inference share one dedicated owning thread. Cancelling an HTTP request cannot
preempt a running Metal generation; that generation may finish before the next one starts. Shutdown
drains running work and cancels queued executor work. Keep admission limits low on memory-limited Macs.
The API does not retain a cross-request prefix cache.

## Native decision workloads

```sh
uv run --extra mac python -m vllm_verifier.engine.cli \
  --runtime mlx --dataset examples/evaluation.jsonl \
  --output /tmp/mac-decisions.json --output-tokens 256 \
  --warmup 0 --repeat 1 --no-prefix-caching
```

MLX-VLM's diffusion stream supports **one sequence at a time**. The runtime forces native batches
to one question; `execution: serial` in reports makes that explicit. vLLM batching speedups must not
be attributed to this path. The compiler still shares prompt structure and preserves question isolation.

Offline `--prefix-caching` enables an in-memory cache with two checkpoints and no disk tier. Reports
include actual cached-token counts from MLX, separate from the compiler's token-prefix overlap.
This cache is for a single trusted offline workload, not shared multi-tenant serving.

`--canvas-tokens 64` (or API `VERIFIER_MLX_CANVAS_TOKENS=64`) is an experimental shorter canvas
setting; the default maximum is 256. Compare latency **and task accuracy** against the default.
The runtime records canvas tokens, denoising steps, work tokens and peak MLX memory for every native
completion. These metrics include repair generations. Warmup metrics are excluded from measured runs.

This path still generates estimated probabilities as text. It is not a trained classification head,
calibrated Jev model, or custom Metal kernel implementation. See [native engine development](native-engine.md)
for the remaining optimization work.
