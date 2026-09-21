# vLLM Verifier

**Jev-compatible typed decisions powered by DiffusionGemma on vLLM.**

Send state and independent questions; receive Choice, Score and Noul answers through
`POST /v1/systemone`. Use an existing TypeSafe client with a different base URL, or plain HTTP.
An image extension observes images once, then evaluates questions against the resulting text.

This is an independent Apache-2.0 project, not TypeSafe's Jev model or an official vLLM component.
API compatibility does not imply equivalent accuracy, calibration, latency or cost.
Probabilities are model-generated estimates. The verifier checks structure and arithmetic,
not factual correctness. No benchmark or GPU inference result is claimed by this repository.

## Quick start (CPU, API wiring only)

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```sh
uv sync --locked
uv run vllm-verifier --demo
```

In another terminal:

```sh
curl http://127.0.0.1:8080/v1/systemone \
  -H 'Content-Type: application/json' \
  --data-binary @examples/triage.json
```

Demo mode returns uniform fixtures, labels every response as demo, and performs **no inference**.
Open [interactive API docs](http://127.0.0.1:8080/docs).

## Run with DiffusionGemma

1. Start a compatible vLLM server on a Linux NVIDIA GPU host using the [deployment guide](docs/deployment.md).
2. Configure the gateway:

```sh
cp .env.example .env
# Set VERIFIER_API_KEY to your own secret in .env.
# Set VERIFIER_BASE_URL to your vLLM server's /v1 URL.
uv run vllm-verifier
```

```sh
curl http://127.0.0.1:8080/v1/systemone \
  -H "Authorization: Bearer $VERIFIER_API_KEY" \
  -H 'Content-Type: application/json' \
  --data-binary @examples/triage.json
```

The shell variable in curl must be set separately; the server reads `.env` itself.
The default engine is `google/diffusiongemma-26B-A4B-it`. The request alias `jev-latest`
selects that locally configured model. Responses identify the actual configured model.
No requests are sent to TypeSafe. The gateway does not install vLLM or load GPU weights.

## TypeSafe Python SDK

```python
from typesafe_sdk import TypeSafeClient, Choice

with TypeSafeClient(api_key="your-local-key", base_url="http://127.0.0.1:8080") as client:
    result = client.system_one(
        model="jev-latest",
        state="I was charged twice. Please refund the extra payment.",
        questions={
            "team": Choice(
                instructions="Which team should handle this?",
                criteria={"billing": "Payments and refunds", "support": "Technical problems"},
            )
        },
    )
    print(result.answers["team"].choice)
```

SDK installation and executable examples: [examples](examples/).
See the [compatibility contract](docs/compatibility.md) for exact guarantees and differences.

## Architecture

```text
TypeSafe SDK / HTTP
        │
        ▼
Auth → bounded admission → request validation
        │
        ├─ /v1/vision/systemone → inline image validation → one image observation
        │
        ▼
Independent question prompts → shared concurrency limit → vLLM /chat/completions
        │
        ▼
Strict JSON + probability validation → bounded repair retry
        │
        ▼
Derived choice / score / confidence → Jev response
```

- Choice: complete distribution over supplied labels; stable argmax selection.
- Score: distribution over 2–10 levels, expected value and original legend.
- Noul: yes probability in `[0, 1]`.
- Each question gets only the shared state and its own instructions/criteria. Question IDs never enter prompts.
- Invalid, incomplete, non-finite and out-of-range outputs fail closed; never fabricate a successful decision.
- Token usage aggregates all question calls, repair attempts and image observation.
- Request deadlines, overload responses, health/readiness checks and Prometheus metrics.
- No automatic action execution: your application handles the returned decision.

Diffusion model support and structured-output support are different capabilities. The default
backend deliberately sends neither `response_format` nor `structured_outputs`. See
[architecture and probability semantics](docs/architecture.md).

## Development

```sh
uv sync --locked
uv run pytest --cov=vllm_verifier --cov-report=term-missing
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run python -m build
```

Tests use deterministic fake completions and the real TypeSafe Python SDK. They do not download
weights or prove model quality. Run [the live smoke test and benchmark](docs/evaluation.md)
on your target GPU before deployment. Dependencies are locked in `uv.lock`; the GPU image is
separately configurable and must be recorded with your benchmark results.

| Document | Purpose |
| --- | --- |
| [한국어 시작 가이드](docs/README.ko.md) | 구성, 실행, 검증 범위 |
| [Compatibility](docs/compatibility.md) | Jev contract and intentional differences |
| [Deployment](docs/deployment.md) | vLLM, Docker, configuration and operations |
| [Evaluation](docs/evaluation.md) | Live smoke tests, accuracy and latency measurement |
| [Contributing](CONTRIBUTING.md) | Contribution workflow |
| [Security](SECURITY.md) | Deployment boundaries and reporting |

Source references were reviewed on 2026-09-22. See [sources](docs/sources.md).
# vllm-verifier
