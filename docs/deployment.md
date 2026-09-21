# Deployment

## Runtime split

The gateway runs on Python 3.11+ on CPU, including macOS. Real DiffusionGemma inference needs a
compatible vLLM installation and supported hardware. This repository does not ship weights.
The official vLLM recipe lists the `vllm/vllm-openai:gemma` image and H100/H200 deployment examples.
Check hardware capacity, model access and the model's own terms before downloading weights.

The GPU recipe below is provided from upstream documentation; it has **not** been executed in
this repository's macOS development environment. The `gemma` tag can change. Pin a tested digest
in `VLLM_IMAGE` for reproducibility and record the model revision separately.

## Docker Compose: GPU and gateway

For a local API demo without a GPU, use the standalone `compose.demo.yaml` file:

```sh
docker compose -f compose.demo.yaml up --build -d --wait
```

Its gateway calls the synthetic server in `examples/demo/`, built into a separate demo image.
Use `local-demo-key` unless `VERIFIER_API_KEY` is set. Stop with
`docker compose -f compose.demo.yaml down`. Demo outputs do not represent model inference.

The production Dockerfile pins multi-platform Python and uv image digests, checks `uv.lock`
with `--locked`, caches dependency downloads, and installs only production dependencies.
The `runtime` stage defaults to `VERIFIER_HOST=0.0.0.0` and requires an inbound key.
Override the `PYTHON_IMAGE` / `UV_IMAGE` build arguments when deliberately updating base images.

Create `.env` (do not commit it):

```dotenv
VERIFIER_API_KEY=your-long-random-local-key
VERIFIER_BASE_URL=http://engine:8000/v1
VERIFIER_MODEL=google/diffusiongemma-26B-A4B-it
VLLM_IMAGE=vllm/vllm-openai:gemma
# HF_TOKEN=your-token-if-model-access-requires-it
```

```sh
docker compose --profile gpu up --build -d
curl http://127.0.0.1:8080/readyz
```

The gateway starts before the model has necessarily loaded. `/readyz` returns 503 until the
expected model is advertised by vLLM. The engine's port is only on the Compose network. The
HTTP gateway is published on host loopback. NVIDIA Container Toolkit and an available Docker
GPU runtime are required. Default model context is conservatively set to 16K; it is not a
hardware-fit guarantee. Tune memory, context and sequence limits for your GPU.

For an existing remote engine, set `VERIFIER_BASE_URL=https://your-engine/v1` and run only
`docker compose up --build -d gateway`. Set `VERIFIER_UPSTREAM_API_KEY` if it requires auth.
Do not use `127.0.0.1` to refer to a different container: loopback is local to each container.
The checked-in `.env.example` targets a **non-containerized** gateway; change its URL for Compose.

## Standalone engine

A starting point adapted from the official recipe:

```sh
docker run --rm --gpus all --shm-size 16g \
  -p 127.0.0.1:8000:8000 \
  -e VLLM_USE_V2_MODEL_RUNNER=1 \
  -v model-cache:/root/.cache/huggingface \
  vllm/vllm-openai:gemma \
  --model google/diffusiongemma-26B-A4B-it \
  --host 0.0.0.0 --port 8000 \
  --max-model-len 16384 --max-num-seqs 8 \
  --gpu-memory-utilization 0.85 \
  --reasoning-parser gemma4 --limit-mm-per-prompt '{"image":4}'
```

Some versions require the V2 runner explicitly. Do not assume any older vLLM release supports
DiffusionGemma. A plain successful `/models` response is insufficient: run the mixed-request smoke
script. If the engine rejects the model or parser, use a build matching the upstream recipe.
No automatic structured-output feature probing is performed; the gateway uses ordinary JSON prompts.

## Gateway configuration

All settings use prefix `VERIFIER_`; environment overrides `.env`.

| Setting | Default | Purpose |
| --- | --- | --- |
| BASE_URL | http://127.0.0.1:8000/v1 | Trusted inference API root |
| MODEL | google/diffusiongemma-26B-A4B-it | Actual upstream model name |
| API_KEY | unset | Inbound gateway Bearer key |
| UPSTREAM_API_KEY | unset | Separate engine Bearer key |
| HOST / PORT | 127.0.0.1 / 8080 | CLI listen address |
| MAX_CONCURRENCY | 8 | Shared simultaneous inference calls per process |
| MAX_REQUESTS | 32 | In-progress decisions/body reads before 529 |
| REQUEST_TIMEOUT | 60 s | Decision deadline; also separate body read deadline |
| UPSTREAM_TIMEOUT | 30 s | HTTP connect/read/write/pool timeout |
| MAX_BODY_BYTES | 16777216 | Actual received request-body limit |
| MAX_OUTPUT_TOKENS | 2048 | Per completion output budget |
| VALIDATION_RETRIES | 1 | Extra generations for invalid output only |
| TEMPERATURE | 0 | Generation setting, not a confidence calibration knob |

In Docker, the CLI binds all container interfaces and requires a nonempty key. Compose passes
common settings explicitly; add additional environment entries for less common overrides.
The image runs as uid/gid 10001; Compose removes capabilities and makes its filesystem read-only.

## Operations

Use TLS and global per-client quotas at your ingress for multi-user use. Probes are public; all
other endpoints require the key if configured. `/metrics` exports request counts and latency.
The `X-Request-ID` header correlates API errors without storing prompts. No client-selected IDs
are accepted. SDK retries can multiply latency and token usage; tune retry policy deliberately.

Only successful API responses carry usage; if a request fails after GPU work, some work may have
been consumed without a usage response. Use vLLM metrics for overall engine accounting.

`VERIFIER_HTTP_PORT` sets the published host port in Compose (default 8080).
`VERIFIER_IMAGE_TAG` sets the locally built gateway image tag (default `local`).
