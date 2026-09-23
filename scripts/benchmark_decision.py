"""Measure direct candidate inference, optionally checking CPU/Metal probability parity."""

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

from vllm_verifier.decision.artifact import MODEL, REVISION
from vllm_verifier.decision.runtime import KaiRuntime
from vllm_verifier.errors import ServiceError
from vllm_verifier.models import SystemOneRequest


def run(args):
    if args.output.exists():
        raise ValueError("Output already exists")
    cases = [json.loads(line) for line in args.dataset.read_text().splitlines() if line.strip()]
    requests = [SystemOneRequest.model_validate(case["request"]) for case in cases]
    if not 1 <= len(requests) <= 128 or sum(len(r.questions) for r in requests) > 512:
        raise ValueError("Dataset must fit 128 requests and 512 decisions")
    started = time.perf_counter()
    runtime = KaiRuntime(
        cache_dir=args.cache_dir,
        device=args.device,
        batch_size=args.batch_size,
        batch_tokens=args.batch_tokens,
    )
    load_seconds = time.perf_counter() - started

    def evaluate(batch):
        result = runtime.evaluate_batch(batch)
        for outcome in result:
            if isinstance(outcome, ServiceError):
                raise RuntimeError(f"{outcome.code}: {outcome.message}")
        return result

    for _ in range(args.warmup):
        evaluate(requests)
    seconds, results = [], []
    for _ in range(args.repeat):
        start = time.perf_counter()
        results = evaluate(requests)
        seconds.append(time.perf_counter() - start)
    correct, count, score_errors = 0, 0, []
    for case, result in zip(cases, results, strict=True):
        for key, expected in case.get("expected", {}).items():
            answer = result.answers[key]
            if answer.type == "score":
                score_errors.append(abs(answer.score - expected))
            else:
                prediction = answer.choice if answer.type == "choice" else answer.noul >= 0.5
                correct += prediction == expected
                count += 1
    report = {
        "model": MODEL,
        "revision": REVISION,
        "device": args.device,
        "dtype": "float32",
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": runtime.torch.__version__,
        "dataset": str(args.dataset),
        "batch_size": args.batch_size,
        "batch_tokens": args.batch_tokens,
        "requests_per_submission": len(requests),
        "decisions_per_submission": sum(len(r.questions) for r in requests),
        "load_seconds": load_seconds,
        "warmup": args.warmup,
        "repeat": args.repeat,
        "submission_seconds": seconds,
        "median_seconds": statistics.median(seconds),
        "classification": {"correct": correct, "total": count},
        "score_mae": statistics.mean(score_errors) if score_errors else None,
        "responses": [result.model_dump() for result in results],
        "measurement": "Local scoring includes tokenization, scheduling, GPU transfer and readout; "
        "excludes HTTP and model loading. No cached answers. Small authored smoke set, "
        "not a representative quality benchmark.",
    }
    if args.compare_cpu:
        if args.device != "mps":
            raise ValueError("--compare-cpu requires --device mps")
        runtime.model.to("cpu")
        runtime.device = "cpu"
        runtime.batch_size = 1
        cpu_results = evaluate(requests)
        drift = []
        same = True
        for gpu, cpu in zip(results, cpu_results, strict=True):
            for key, answer in gpu.answers.items():
                reference = cpu.answers[key]
                if answer.type == "noul":
                    drift.append(abs(answer.noul - reference.noul))
                    same &= (answer.noul >= 0.5) == (reference.noul >= 0.5)
                else:
                    drift.extend(
                        abs(p - reference.probabilities[k]) for k, p in answer.probabilities.items()
                    )
                    same &= max(answer.probabilities, key=answer.probabilities.get) == max(
                        reference.probabilities, key=reference.probabilities.get
                    )
        report["cpu_parity"] = {
            "reference": "Same pinned model and FP32 weights, isolated CPU questions",
            "max_probability_absolute_difference": max(drift),
            "same_top_decisions": bool(same),
            "tolerance": args.parity_tolerance,
            "passed": bool(same and max(drift) <= args.parity_tolerance),
            "responses": [r.model_dump() for r in cpu_results],
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            {k: v for k, v in report.items() if k not in {"responses", "cpu_parity"}}, indent=2
        )
    )
    if args.compare_cpu and not report["cpu_parity"]["passed"]:
        raise RuntimeError("CPU/Metal parity failed; inspect saved results")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path.home() / ".cache/diffusion-jev")
    parser.add_argument("--device", choices=["mps", "cpu"], default="mps")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--batch-tokens", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--compare-cpu", action="store_true")
    parser.add_argument("--parity-tolerance", type=float, default=0.0001)
    args = parser.parse_args()
    if args.repeat < 1 or args.warmup < 0:
        parser.error("repeat must be positive and warmup nonnegative")
    run(args)


if __name__ == "__main__":
    main()
