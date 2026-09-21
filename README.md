# vLLM Verifier

A Jev-compatible API for classification, scoring, and yes/no decisions, backed by
[DiffusionGemma](https://huggingface.co/google/diffusiongemma-26B-A4B-it) on vLLM.

Send text or structured data with a set of questions. Get typed answers your application can
use to route requests, rank items, or choose its next action. Existing TypeSafe Python clients
can connect by changing their base URL and API key.

- **Choice** selects an option and returns a distribution over all options.
- **Score** rates input against an ordered rubric and returns its expected score.
- **Noul** returns an estimated probability that a statement is true.
- **Vision** describes images once, then evaluates questions against the description.

The server validates model output, derives scores and choices from the returned probabilities,
and retries invalid generations within a bounded deadline. It includes authentication,
concurrency limits, health checks, and Prometheus metrics.

## Run with Docker

Clone the repository:

```sh
git clone https://github.com/jinwonkim93/vllm-verifier.git
cd vllm-verifier
```

Connect to a running vLLM server. Set the URL to an address reachable **from the container**:

```sh
export VERIFIER_API_KEY=your-local-api-key
export VERIFIER_BASE_URL=http://your-vllm-host:8000/v1
export VERIFIER_MODEL=google/diffusiongemma-26B-A4B-it
docker compose up --build -d gateway
```

For a vLLM server on the same Mac, use `http://host.docker.internal:8000/v1`.
`127.0.0.1` inside the container refers to the container itself.
If vLLM requires authentication, also set `VERIFIER_UPSTREAM_API_KEY`.

To run both services on a Linux NVIDIA GPU host instead:

```sh
export VERIFIER_API_KEY=your-local-api-key
export VERIFIER_BASE_URL=http://engine:8000/v1
docker compose --profile gpu up --build -d
```

Check readiness after the model loads:

```sh
curl http://127.0.0.1:8080/readyz
```

See [deployment](docs/deployment.md) for GPU requirements, image versions, and configuration.
The gateway image is CPU-only; vLLM serves the model in a separate process or container.

## Make a decision

```sh
curl http://127.0.0.1:8080/v1/systemone \
  -H "Authorization: Bearer $VERIFIER_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "jev-latest",
    "state": "I was charged twice. Please refund the extra payment.",
    "questions": {
      "team": {
        "type": "choice",
        "instructions": "Which team should handle this?",
        "criteria": {
          "billing": "Payments and refunds",
          "technical": "Software bugs"
        }
      }
    }
  }'
```

The response contains `model`, `answers`, and token `usage`. Each answer is keyed by its
question ID. A Choice answer includes `choice`, `probabilities`, and `confidence`.
The alias `jev-latest` selects the locally configured model; the response identifies that model.

### TypeSafe Python SDK

```sh
pip install 'typesafe-sdk>=0.7.1,<0.8'
```

```python
import os
from typesafe_sdk import Choice, TypeSafeClient

with TypeSafeClient(
    api_key=os.environ["VERIFIER_API_KEY"],
    base_url="http://127.0.0.1:8080",
) as client:
    result = client.system_one(
        state="I was charged twice. Please refund the extra payment.",
        questions={
            "team": Choice(
                instructions="Which team should handle this?",
                criteria={"billing": "Payments and refunds", "technical": "Software bugs"},
            )
        },
    )
    print(result.choices["team"].choice)
```

The SDK base URL ends at the host and port. The upstream `VERIFIER_BASE_URL` includes `/v1`.

## Try the API without a GPU

```sh
docker compose -f compose.demo.yaml up --build -d --wait
```

The [demo](examples/demo/README.md) connects the gateway to a separate synthetic HTTP server.
It returns uniform probabilities and `model: "demo-uniform"`; it does not classify text or
interpret images. Its default API key is `local-demo-key`, unless `VERIFIER_API_KEY` is set.
The production package and Docker image contain no demo backend.

```sh
docker compose -f compose.demo.yaml down
```

## Local development

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```sh
uv sync --locked
cp .env.example .env
# Configure your vLLM URL, model and API key in .env.
uv run vllm-verifier
```

```sh
uv run pytest --cov=vllm_verifier
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run python scripts/export_openapi.py --check
```

Build and test the production image against an isolated HTTP fixture:

```sh
docker build -t vllm-verifier:local .
uv run python scripts/check_container.py --image vllm-verifier:local
```

Application code lives in `src/`, tests and fixtures in `tests/`, and runnable examples in
`examples/`. The wheel and runtime image include only the application package and dependencies.

## Compatibility and limits

This independent project implements the Jev HTTP contract, not the Jev model or training method.
Probabilities are model-generated estimates, not calibrated Jev probabilities. Confidence is
computed from normalized entropy. Structural validation does not guarantee a correct judgment.
Evaluate thresholds on your own data before using decisions to automate actions.

Questions are evaluated independently. Adding questions can increase latency and token usage.
There is no guarantee of Jev-equivalent speed, accuracy, or cost. Real GPU inference performance
must be measured on your target vLLM build and hardware.

## Documentation

- [API compatibility](docs/compatibility.md)
- [Deployment and settings](docs/deployment.md)
- [Architecture and probability semantics](docs/architecture.md)
- [Evaluation](docs/evaluation.md)
- [한국어 가이드](docs/README.ko.md)
- [Contributing](CONTRIBUTING.md) · [Security](SECURITY.md) · [Apache-2.0 license](LICENSE)
