# vLLM Verifier

Typed classification, scoring, and yes/no decisions with a Jev-compatible HTTP API.
Run Decision Kai directly on Apple Silicon, use DiffusionGemma through MLX, or connect
to an external vLLM server.

Send text or structured data with a set of questions. Get typed answers your application can
use to route requests, rank items, or choose its next action. Existing TypeSafe Python clients
can connect by changing their base URL and API key.

- **Choice** selects an option and returns a distribution over all options.
- **Score** rates input against an ordered rubric and returns its expected score.
- **Noul** returns an estimated probability that a statement is true.
- **Vision** describes images once, then evaluates questions against the description.

The server includes authentication, bounded admission, health checks, and Prometheus metrics.
The Decision runtime scores candidates directly and batches questions across requests.
Generation backends validate JSON output and retry invalid generations within a bounded deadline.

## Native decision execution

The experimental [native engine](docs/native-engine.md) compiles typed questions into bounded
vLLM generation batches without an HTTP inference server. It preserves question isolation, places
shared state before question-specific tokens for prefix-cache reuse, and repairs only invalid
answers. A native CLI records batch sizes, token usage and timings for isolated/batched comparisons.

The vLLM generation path is currently offline and its GPU validation is pending. The
[Decision runtime](docs/decision-models.md) provides direct candidate heads and bounded online
request batching on Mac. Custom diffusion sampling and distributed serving remain future work.

## Run Decision Kai on an Apple Silicon Mac

```sh
git clone https://github.com/jinwonkim93/vllm-verifier.git
cd vllm-verifier
UV_PROJECT_ENVIRONMENT=.venv-decision uv sync --locked --extra decision --python 3.12
UV_PROJECT_ENVIRONMENT=.venv-decision uv run --extra decision \
  vllm-verifier --runtime decision --port 18080
```

Kai scores Choice, Noul and Score candidates without generating text. The first start downloads
approximately 2.3 GB. Each complete question, state and candidate set must fit 1,024 tokens.
See [Decision setup and scheduling](docs/decision-models.md) for settings, probability semantics,
CPU comparison and reproducible benchmarks. The [M5 measurements](docs/benchmarks/2026-09-23-decision/README.md)
include latency, throughput and observed quality limitations.

## Run DiffusionGemma on an Apple Silicon Mac

```sh
git clone https://github.com/jinwonkim93/vllm-verifier.git
cd vllm-verifier
export HF_HOME="${HF_HOME:-$HOME/.cache/diffusion-jev/huggingface}"
uv sync --locked --extra mac
VERIFIER_MAX_CONCURRENCY=1 VERIFIER_MAX_REQUESTS=2 \
VERIFIER_MAX_OUTPUT_TOKENS=256 VERIFIER_REQUEST_TIMEOUT=300 \
uv run --extra mac vllm-verifier --runtime mlx --port 18080
```

This loads the 4bit DiffusionGemma checkpoint directly through MLX/Metal. The first run downloads
approximately 16.5 GB of weights. See [Mac setup](docs/macos.md) for memory limits, native workload
execution and current text-only, sequential-generation support.

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
Kai returns learned candidate probabilities and maximum-probability confidence. Generation
backends return model-generated estimates and normalized-entropy confidence. Neither path
provides calibrated Jev probabilities. Structural validation does not guarantee a correct judgment.
Evaluate thresholds on your own data before using decisions to automate actions.

Questions are evaluated independently. Adding questions can increase latency and token usage.
There is no guarantee of Jev-equivalent speed, accuracy, or cost. Real GPU inference performance
must be measured on your target vLLM build and hardware.

## Documentation

- [Decision models on Mac](docs/decision-models.md)
- [API compatibility](docs/compatibility.md)
- [Deployment and settings](docs/deployment.md)
- [Architecture and probability semantics](docs/architecture.md)
- [Evaluation](docs/evaluation.md)
- [한국어 가이드](docs/README.ko.md)
- [Contributing](CONTRIBUTING.md) · [Security](SECURITY.md) · [Apache-2.0 license](LICENSE)
