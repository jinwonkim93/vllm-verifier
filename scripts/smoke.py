"""Require a real engine and validate a mixed decision response over HTTP."""

import argparse
import json
import math
import os
from pathlib import Path

import httpx

from vllm_verifier.models import SystemOneResponse

parser = argparse.ArgumentParser()
parser.add_argument("--url", default="http://127.0.0.1:8080")
args = parser.parse_args()
with httpx.Client(
    base_url=args.url,
    timeout=65,
    headers={"Authorization": "Bearer " + os.environ.get("VERIFIER_API_KEY", "")},
) as client:
    client.get("/readyz").raise_for_status()
    payload = json.loads(Path("examples/triage.json").read_text())
    response = client.post("/v1/systemone", json=payload)
    response.raise_for_status()
    if response.json().get("model") == "demo-uniform":
        raise SystemExit("Refusing demo result: this command requires real inference")
    result = SystemOneResponse.model_validate(response.json())
    assert set(result.answers) == set(payload["questions"])
    for key, answer in result.answers.items():
        assert answer.type == payload["questions"][key]["type"]
        if answer.type != "noul":
            assert math.isclose(sum(answer.probabilities.values()), 1.0, abs_tol=1e-9)
    print(result.model_dump_json(indent=2))
