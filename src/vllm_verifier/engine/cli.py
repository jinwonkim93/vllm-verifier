"""Run and measure native typed decisions on a local GPU runtime."""

import argparse
import importlib.metadata
import json
import os
from dataclasses import asdict
from pathlib import Path

from ..models import SystemOneRequest
from .core import DecisionEngine, EngineConfig
from .native import VLLMRuntime


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="google/diffusiongemma-26B-A4B-it")
    parser.add_argument("--revision", help="Model/tokenizer revision for reproducible runs")
    parser.add_argument("--mode", choices=["batch", "isolated"], default="batch")
    parser.add_argument("--prefix-caching", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument("--batch-questions", type=int, default=32)
    parser.add_argument("--batch-tokens", type=int, default=131072)
    parser.add_argument("--output-tokens", type=int, default=2048)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    args = parser.parse_args()
    if args.repeat < 1 or args.warmup < 0 or args.tensor_parallel_size < 1:
        parser.error("repeat and tensor parallel size must be positive; warmup cannot be negative")
    config = EngineConfig(
        max_batch_questions=1 if args.mode == "isolated" else args.batch_questions,
        max_batch_tokens=args.batch_tokens,
        max_model_len=args.max_model_len,
        output_tokens=args.output_tokens,
        validation_retries=args.retries,
    )
    cases = [json.loads(line) for line in args.dataset.read_text().splitlines() if line.strip()]
    requests = [SystemOneRequest.model_validate(case["request"]) for case in cases]
    if not requests or len(requests) > config.max_requests:
        parser.error(f"dataset must contain 1..{config.max_requests} requests")
    if args.output.exists():
        parser.error("output already exists; choose a new path")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Must be set before importing vLLM. Preserve an explicit operator override.
    os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "1")
    runtime = VLLMRuntime(
        args.model,
        revision=args.revision,
        tokenizer_revision=args.revision,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        enable_prefix_caching=args.prefix_caching,
    )
    engine = DecisionEngine(runtime, args.model, config)
    for _ in range(args.warmup):
        engine.evaluate(requests)
    runs = []
    for _ in range(args.repeat):
        result = engine.evaluate(requests)
        runs.append(
            {
                "stats": asdict(result.stats),
                "responses": [r.model_dump() for r in result.responses],
            }
        )
    report = {
        "runtime": "native-vllm",
        "vllm_version": importlib.metadata.version("vllm"),
        "model": args.model,
        "revision": args.revision,
        "mode": args.mode,
        "prefix_caching": args.prefix_caching,
        "tensor_parallel_size": args.tensor_parallel_size,
        "config": config.model_dump(),
        "warmup_runs": args.warmup,
        "dataset": str(args.dataset),
        "probability_source": "uncalibrated-model-estimates",
        "runs": runs,
    }
    with args.output.open("x") as output:
        output.write(json.dumps(report, indent=2) + "\n")
    print(f"Wrote {len(runs)} native runs to {args.output}")


if __name__ == "__main__":
    main()
