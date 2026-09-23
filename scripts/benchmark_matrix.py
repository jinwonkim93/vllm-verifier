"""Sweep HTTP load and question count; synthetic workloads do not measure accuracy."""

import argparse
import asyncio
import contextlib
import io
import json
from pathlib import Path

from benchmark import run


async def matrix(args):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for count in args.questions:
        dataset = args.output_dir / f"q{count}.jsonl"
        dataset.write_text(
            json.dumps(
                {
                    "request": {
                        "model": args.model,
                        "state": "x" * args.state_bytes,
                        "questions": {
                            f"q{i}": {
                                "type": "choice",
                                "instructions": "Classify the state.",
                                "criteria": {"a": "A", "b": "B", "c": "C"},
                            }
                            for i in range(count)
                        },
                    }
                }
            )
            + "\n"
        )
        for concurrency in args.concurrency:
            for phase, repeat in (("warmup", args.warmup), ("measured", args.requests)):
                output = args.output_dir / f"q{count}-c{concurrency}-{phase}.json"
                config = argparse.Namespace(
                    dataset=dataset,
                    repeat=repeat,
                    warmup=0,
                    concurrency=concurrency,
                    url=args.url,
                    timeout=args.timeout,
                    run_label=args.run_label,
                    output=output,
                )
                with contextlib.redirect_stdout(io.StringIO()):
                    result = await run(config)
                if phase == "warmup" and result:
                    raise RuntimeError(f"Warmup failed; inspect {output}")
                if phase == "measured":
                    report = json.loads(output.read_text())
                    report.pop("rows")
                    report.update(
                        questions_per_request=count,
                        state_bytes=args.state_bytes,
                        warmup_requests=args.warmup,
                    )
                    summaries.append(report)
                    print(json.dumps(report), flush=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    return int(any(r["successful"] != r["requests"] for r in summaries))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8080")
    parser.add_argument("--model", default="jev-latest")
    parser.add_argument("--questions", type=int, nargs="+", default=[1, 8, 32])
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 8, 32])
    parser.add_argument("--state-bytes", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=32)
    parser.add_argument("--requests", type=int, default=256)
    parser.add_argument("--timeout", type=float, default=65)
    parser.add_argument("--run-label", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not all(1 <= q <= 64 for q in args.questions):
        parser.error("questions must be between 1 and 64")
    if min(*args.concurrency, args.state_bytes, args.warmup, args.requests, args.timeout) <= 0:
        parser.error("concurrency, state bytes, warmup, requests and timeout must be positive")
    raise SystemExit(asyncio.run(matrix(args)))


if __name__ == "__main__":
    main()
