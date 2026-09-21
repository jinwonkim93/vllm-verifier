"""Measure real HTTP latency and optional labeled accuracy, without fabricating results."""

import argparse
import asyncio
import json
import math
import os
import time
from collections import Counter
from pathlib import Path

import httpx


def percentile(values, fraction):
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * fraction) - 1)] if ordered else None


async def run(args):
    cases = [json.loads(line) for line in args.dataset.read_text().splitlines() if line.strip()]
    if not cases:
        raise ValueError("dataset is empty")
    queue = asyncio.Queue()
    for _ in range(args.repeat):
        for case in cases:
            queue.put_nowait(case)
    rows = []
    headers = {"Authorization": "Bearer " + os.environ.get("VERIFIER_API_KEY", "")}
    async with httpx.AsyncClient(
        base_url=args.url.rstrip("/"), headers=headers, timeout=args.timeout
    ) as client:

        async def worker():
            while not queue.empty():
                case = queue.get_nowait()
                started = time.perf_counter()
                try:
                    response = await client.post("/v1/systemone", json=case["request"])
                    body = response.json()
                    row = {
                        "status": response.status_code,
                        "seconds": time.perf_counter() - started,
                        "demo": body.get("model") == "demo-uniform",
                    }
                    if response.is_success:
                        row.update({"model": body["model"], "usage": body["usage"]})
                        scores = []
                        for key, expected in case.get("expected", {}).items():
                            answer = body["answers"][key]
                            if answer["type"] == "choice":
                                distribution = answer["probabilities"]
                                scores.append(
                                    {
                                        "kind": "classification",
                                        "correct": answer["choice"] == expected,
                                        "brier": sum(
                                            (p - (k == expected)) ** 2
                                            for k, p in distribution.items()
                                        ),
                                    }
                                )
                            elif answer["type"] == "noul":
                                p = answer["noul"]
                                scores.append(
                                    {
                                        "kind": "classification",
                                        "correct": (p >= 0.5) == expected,
                                        "brier": (p - float(expected)) ** 2,
                                    }
                                )
                            else:
                                scores.append(
                                    {
                                        "kind": "score",
                                        "absolute_error": abs(answer["score"] - expected),
                                    }
                                )
                        row["scores"] = scores
                    else:
                        row["error"] = body.get("error", {}).get("code", "http_error")
                except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
                    row = {
                        "status": 0,
                        "seconds": time.perf_counter() - started,
                        "error": type(exc).__name__,
                    }
                rows.append(row)

        started = time.perf_counter()
        await asyncio.gather(*(worker() for _ in range(args.concurrency)))
        elapsed = time.perf_counter() - started
    successful = [row for row in rows if row["status"] == 200]
    latencies = [row["seconds"] for row in successful]
    decisions = [score for row in successful for score in row.get("scores", [])]
    classifications = [score for score in decisions if score["kind"] == "classification"]
    ordinal = [score for score in decisions if score["kind"] == "score"]
    report = {
        "metadata": {
            "dataset": str(args.dataset),
            "concurrency": args.concurrency,
            "repeat": args.repeat,
            "run_label": args.run_label,
            "models": sorted({row["model"] for row in successful}),
            "demo": any(row.get("demo") for row in rows),
        },
        "requests": len(rows),
        "successful": len(successful),
        "wall_seconds": elapsed,
        "successful_requests_per_second": len(successful) / elapsed,
        "successful_latency_seconds": {
            "p50": percentile(latencies, 0.5),
            "p95": percentile(latencies, 0.95),
            "p99": percentile(latencies, 0.99),
        },
        "errors": dict(
            Counter(row.get("error", "unknown") for row in rows if row["status"] != 200)
        ),
        "usage": {
            key: sum(row["usage"][key] for row in successful)
            for key in ("input_tokens", "output_tokens")
        },
        "classification_accuracy": (
            sum(s["correct"] for s in classifications) / len(classifications)
            if classifications
            else None
        ),
        "mean_brier": (
            sum(s["brier"] for s in classifications) / len(classifications)
            if classifications
            else None
        ),
        "score_mae": (
            sum(s["absolute_error"] for s in ordinal) / len(ordinal) if ordinal else None
        ),
        "rows": rows,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2))
    return 0 if len(successful) == len(rows) else 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8080")
    parser.add_argument("--dataset", type=Path, default=Path("examples/evaluation.jsonl"))
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=65)
    parser.add_argument("--run-label", default="unspecified hardware/runtime")
    parser.add_argument("--output", type=Path, default=Path("benchmark-results.json"))
    args = parser.parse_args()
    if args.concurrency < 1 or args.repeat < 1 or args.timeout <= 0:
        parser.error("concurrency, repeat and timeout must be positive")
    raise SystemExit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
