"""Export the API contract without starting an inference client."""

import argparse
import json
from pathlib import Path

from vllm_verifier.app import create_app
from vllm_verifier.config import Settings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    target = Path(__file__).resolve().parents[1] / "docs/openapi.json"
    schema = create_app(Settings(_env_file=None)).openapi()
    rendered = json.dumps(schema, indent=2, ensure_ascii=False) + "\n"
    if args.check:
        if not target.exists() or target.read_text() != rendered:
            raise SystemExit("OpenAPI is stale; run scripts/export_openapi.py")
    else:
        target.write_text(rendered)


if __name__ == "__main__":
    main()
