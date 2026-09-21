# Architecture and inference semantics

The gateway is a small FastAPI service, with no PyTorch/vLLM import in the serving process.
The independently deployed inference server owns GPU memory, scheduling, prefix caching and
diffusion sampling. `Backend` in `backend.py` is the testable integration boundary.

## Inference path

Each question produces one prompt containing shared state and that question's content. IDs are
kept only in the application. Questions run concurrently, capped by one semaphore per gateway
process. Independent HTTP requests share the semaphore. An admission cap bounds the number of
in-progress request bodies and decisions. If a question fails, sibling tasks are cancelled and
no partial success envelope is returned. Timeouts include gateway queue time, repairs and vision
observation; body reading has a separate timeout. There is no external action executor.

DiffusionGemma denoises token blocks. That does not make arbitrary JSON schema decoding or
calibrated classification available. The adapter sends ordinary non-streaming chat requests,
disables thinking through chat-template kwargs, and does not depend on constrained decoding,
logprob support, speculative-decoding knobs or private vLLM APIs. Use a Gemma reasoning parser
on the engine so reasoning text is separated from content.

## Structural verifier

1. Require normal completion (`finish_reason=stop`). Reject truncation and tool-call finishes.
2. Parse one JSON object, optionally wrapped in one Markdown JSON fence. Reject duplicate keys,
   prose, trailing objects and non-finite probabilities. Never regex-extract an answer from reasoning.
3. Require exactly `probabilities: [number, ...]` (Choice/Score), or `noul: number`.
4. Check vector length, numeric types, range and sum. Only rounding error within 0.01 of one is
   normalized; zero-sum and materially wrong totals are rejected.
5. Derive named distributions, stable argmax Choice, expected Score, original legend and entropy
   confidence in code. A model cannot return an invented choice label or contradict its own score.
6. On invalid output, retry the same isolated question with a concise validation correction. Do not
   feed the untrusted failed output back. Default: one repair retry; all attempts count toward usage.

Upstream HTTP/protocol failures are not automatically retried, and there is no silent backend
fallback. Upstream overloaded responses become 529, connection/protocol/rejection errors 502,
and deadlines 504. Upstream bodies are not exposed. The operator can inspect vLLM's own logs.

## Probability semantics

The model writes estimated probabilities as text. These estimates are not token likelihoods,
empirical frequencies or calibrated Jev outputs. Structural validity is necessary but insufficient
for a correct decision. Confidence measures concentration, using normalized entropy; a confidently
wrong model can have confidence one. Lower diffusion temperature does not establish correctness
or determinism. The response header `X-Probability-Source` makes the source explicit.

Do not compare these confidence numbers directly against a Jev deployment's thresholds.
Evaluate accuracy, Brier score and task-specific error costs on held-out examples. Calibrate in
an application-specific layer if needed; this release does not include a fitted calibrator.

## Operational boundaries

- No prompt/response cache: avoids unintended cross-tenant reuse of private state.
- State and images are not logged by the gateway. Default Uvicorn access logs contain routes/status.
- Prometheus uses fixed route/status labels, avoiding user-controlled cardinality.
- Each ASGI worker has separate resource limits and metrics. Run one worker per replica; aggregate
  at Prometheus or a load balancer. Replicas require ingress-level global quotas if needed.
- `/healthz` only checks process liveness. `/readyz` checks that the configured model is advertised
  by the engine, not whether a real generation succeeds; use the live smoke script for that.
- Request cancellation closes HTTP work locally. Immediate GPU kernel cancellation is not guaranteed.

## Relationship to vLLM routers

[vLLM Router](https://github.com/vllm-project/router) forwards inference requests across workers.
It provides load balancing, cache-aware policies, worker discovery and prefill/decode routing.
This gateway instead translates typed decision requests into model prompts and verifies responses.
It currently targets one configured upstream URL and implements no worker selection or discovery.

[vLLM Semantic Router](https://github.com/vllm-project/semantic-router) uses request signals and
policies to select models and processing paths. The shared concept is a decision layer in front of
inference. This service exposes decisions to the caller; it does not route the caller's subsequent
inference or execute actions. Neither router is a drop-in replacement for the Jev API adapter.

For multiple inference replicas, the composition can be:

```text
Jev-compatible client -> vLLM Verifier -> vLLM Router -> vLLM workers
```

Set `VERIFIER_BASE_URL` to the router's OpenAI-compatible `/v1` endpoint. This composition is an
integration option, not a tested deployment in this repository. The router must advertise the
configured model through `/v1/models` for gateway readiness to pass.

## Latency limits

With Q questions, a valid response requires Q upstream completions; one repair per question can
raise that to 2Q under the default configuration. Every prompt includes the state. Prefix caching
may help if supported and enabled by the engine, but is not guaranteed by the gateway. Eight
shared inference slots are available by default, so concurrent requests compete for that capacity.
Vision adds an observation completion before any question can start.

For similar-duration completions without repairs, a single request with Q questions needs roughly
ceil(Q / available slots) waves of inference. Batching all questions into one prompt could reduce
calls, but would change isolation and failure semantics; it is not an equivalent optimization.
Diffusion token throughput alone therefore cannot establish Jev-like decision latency. Measure
short-output latency, quality, repairs and concurrency on the actual model before selecting a
backend or increasing the inference concurrency limit.
