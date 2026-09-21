# Contributing

Create an issue describing the problem or open a focused pull request with a reproduction.
For local setup, use `uv sync --locked`. Run Ruff, mypy and pytest as documented in README.
No GPU is needed for contract tests. GPU integration results must include the vLLM image digest,
model revision, hardware, request workload and raw measurements.

Keep inference behind the Backend protocol. Preserve question isolation, bounded resources,
actual usage accounting and fail-closed validation. Add regression tests for observable behavior.
Do not introduce a network fallback to a hosted service, a silent demo fallback, or claims of
calibration / performance without evaluation evidence. Do not include user state in logs or errors.

When changing the HTTP contract, update `docs/compatibility.md`, tests and the generated schema:
`uv run python scripts/export_openapi.py`. Commit `uv.lock` when dependencies change.
Do not edit generated OpenAPI by hand. Python 3.11–3.13 run in CI.

Contributions are provided under Apache-2.0. Be respectful and keep feedback specific and actionable.
