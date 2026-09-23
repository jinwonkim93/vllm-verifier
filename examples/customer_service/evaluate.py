"""Evaluate authored catalog examples against a real running classifier."""

import argparse
import asyncio
import json
import statistics
from pathlib import Path

from .catalog import INTENTS
from .classifier import SystemOneClassifier


async def run(url: str, output: Path, limit: int):
    classifier = SystemOneClassifier(url)
    rows = []
    try:
        for intent in list(INTENTS.values())[:limit]:
            result = await classifier.classify(intent.example, "")
            rows.append(
                {
                    "expected": intent.id,
                    "input": intent.example,
                    "predicted": result.intent,
                    "candidates": result.candidates,
                    "trace": result.trace,
                }
            )
            print(f"{len(rows):3} {intent.id}: {result.intent}", flush=True)
    finally:
        await classifier.close()
        output.parent.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(
            output.write_text,
            json.dumps(
                {
                    "dataset": "Authored catalog examples; not a held-out benchmark",
                    "count": len(rows),
                    "correct": sum(r["predicted"] == r["expected"] for r in rows),
                    "clarifications": sum(r["predicted"] is None for r in rows),
                    "median_seconds": statistics.median(r["trace"]["seconds"] for r in rows)
                    if rows
                    else None,
                    "rows": rows,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine-url", default="http://127.0.0.1:18080")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()
    asyncio.run(run(args.engine_url, args.output, args.limit))
