# Native decision engine

The native engine executes typed decision workloads directly through vLLM's Python runtime.
It compiles Jev-shaped requests into tokenized work items, admits bounded batches, and reconstructs
verified Choice, Score and Noul responses. No OpenAI HTTP server is involved in this execution path.

This is an **experimental offline execution engine**. The HTTP gateway remains a separate baseline;
its endpoints do not silently switch to the native engine. Native vLLM GPU execution still needs
validation on an NVIDIA host; the MLX path has separate real-Mac validation results. CPU tests validate the compiler,
admission, batching, repairs and runtime boundary using explicit test doubles.

For Apple Silicon, use the [Mac runtime](macos.md). It executes through MLX rather than vLLM
and currently supports one diffusion sequence at a time.

## Execution

```text
Jev request batch
  -> shared state serialization
  -> state-first prompts, actual model chat template and tokenization
  -> round-robin question work items
  -> bounded native vLLM generation batches
  -> typed verification, selective repair
  -> Jev-shaped response batch and execution statistics
```

Every question retains an independent sequence: questions are not combined in one model prompt.
All question types share the same system instruction and state prefix; question-specific material
and repair instructions follow it. This makes token-identical prefixes eligible for automatic
prefix caching. The runtime still generates probability text, which is structurally validated and
is **not calibrated classification probability**. There is no direct classification head yet.

The [official DiffusionGemma implementation](https://vllm-project.github.io/2026/06/10/diffusion-gemma)
supports prefix caching and native batched execution. Cache reuse depends on committed cache blocks,
scheduling and capacity: simultaneous cold requests do not imply one shared prefill. Reported
`shared_prefix_tokens` measures exact token overlap, not cache hits or compute saved.

## Run on a compatible GPU host

Use a vLLM installation with DiffusionGemma and model runner v2 support, following the
[official model recipe](https://recipes.vllm.ai/models/Google/diffusiongemma-26B-A4B-it.html).
Install this package into that environment; vLLM is deliberately not a dependency of the CPU gateway.
Model access and sufficient GPU memory must be available before execution.

```sh
pip install .
VLLM_USE_V2_MODEL_RUNNER=1 python -m vllm_verifier.engine.cli \
  --dataset examples/evaluation.jsonl \
  --output /tmp/native-batch.json \
  --mode batch --batch-questions 32 --batch-tokens 131072 \
  --output-tokens 2048 --warmup 1 --repeat 3
```

Add `--revision <model-commit>` to pin model and tokenizer weights. Use
`--tensor-parallel-size` for a supported multi-GPU configuration. The CLI disables thinking via the
chat template; it does not strip reasoning text after generation. Unexpected reasoning or malformed
outputs fail validation rather than being silently extracted.

`--batch-tokens` bounds the sum of input plus maximum output tokens admitted in one application
batch. It is a conservative work budget, not vLLM's per-step `max_num_batched_tokens`, a VRAM estimate,
or a bound on diffusion canvas allocations. `--batch-questions` bounds submitted sequences. The
vLLM runtime owns GPU scheduling inside each batch. Each application batch completes before the
next starts; this is not continuous admission from online clients.

Oversized jobs are rejected before generation. The default admission limit is 128 requests per
`evaluate()` call. Failed output validation retries only the affected question and counts every
attempt's token usage. Exhausted repairs fail the run without returning a partial success. Each
runtime must have one synchronous owner; this interface is not thread-safe.

## Measure the execution path

Compare fresh processes with identical model revisions, GPU settings and workload:

1. `--mode isolated --no-prefix-caching`: one question per native generation call.
2. `--mode batch --no-prefix-caching`: native batching without prefix reuse.
3. `--mode batch --prefix-caching`: native batching with cache reuse enabled.

Use different output paths; the CLI refuses to overwrite results. Repeat both `--warmup 0` cold
runs and warmed runs. Startup/model-loading time is excluded; measured elapsed time includes prompt
compilation, generation and verification. Generation time covers the synchronous runtime call, not
GPU kernel time alone. Token usage counts logical prompt tokens, including cached tokens.

Reports preserve each response, retries, batch sizes, token-identical prefix lengths and timings.
Use representative questions with different types and labels sharing a long state. Repeating the
same small dataset warms its complete prompts and is not proof of performance on unseen inputs.
Evaluate task accuracy and calibration separately; faster but less accurate decisions are not an
improvement. No GPU speedup result is currently claimed.

## Engine development milestones

| Milestone | Deliverable | Acceptance evidence |
| --- | --- | --- |
| Native execution | Token work items, bounded batches, shared prefixes, selective repairs | CPU contract tests implemented; real GPU gate pending |
| GPU baseline | Isolated/batched/cache-on comparisons; kernel and KV-cache profiling | Same workload and quality; raw results with GPU/runtime/weights revisions |
| Continuous admission | Async native runtime, per-request cancellation/deadlines, bounded fair queues | GPU overload and cancellation tests; no starvation or leaked work |
| Decision scoring | Compare candidate scoring or a trained decision head against generated probabilities | Verified model support; held-out accuracy and calibration; no unsupported logprob assumptions |
| Diffusion specialization | Explore canvas utilization, convergence criteria and decision-output representation | End-to-end latency gains without unacceptable quality loss |
| Scale-out | Worker lifecycle, native-engine metrics, multi-GPU replicas | Measured scaling efficiency and failure recovery |

Changes to the sampler, attention masks, or output head require model-level validation. They must
not be represented as completed optimizations by changing HTTP batching or prompt wording alone.

The explicit GPU integration gate is:

```sh
VERIFIER_TEST_GPU=1 pytest -q tests/test_native_gpu.py
```

It loads real weights and checks mixed typed generation. It is skipped in CPU CI and must pass
on the target runtime before native deployment. Passing it establishes execution compatibility,
not performance or task accuracy.
