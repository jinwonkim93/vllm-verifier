"""Test the built gateway against an HTTP fixture in another Docker container.

Requires the dev dependencies and Docker. Does not download or validate model weights.
Every container/network created by this script is removed on exit.
"""

import argparse
import base64
import io
import json
import subprocess
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
from PIL import Image
from typesafe_sdk import Choice, Noul, RetryPolicy, Score, TypeSafeClient

ROOT = Path(__file__).resolve().parents[1]


def docker(*args, check=True):
    result = subprocess.run(["docker", *args], capture_output=True, text=True, check=check)
    return result.stdout.strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default="vllm-verifier:compat")
    args = parser.parse_args()
    prefix = "verifier-check-" + uuid.uuid4().hex[:12]
    engine, gateway = prefix + "-engine", prefix + "-gateway"
    hardening = [
        "--read-only",
        "--tmpfs",
        "/tmp:size=64m",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges:true",
    ]
    try:
        docker("network", "create", prefix)
        docker(
            "run",
            "-d",
            "--name",
            engine,
            "--network",
            prefix,
            *hardening,
            "--mount",
            f"type=bind,src={ROOT / 'tests/container/fake_engine.py'},dst=/fixture.py,readonly",
            "--entrypoint",
            "python",
            args.image,
            "/fixture.py",
        )
        docker(
            "run",
            "-d",
            "--name",
            gateway,
            "--network",
            prefix,
            *hardening,
            "-p",
            "127.0.0.1::8080",
            "-e",
            "VERIFIER_API_KEY=fixture-gateway-key",
            "-e",
            "VERIFIER_UPSTREAM_API_KEY=fixture-upstream-key",
            "-e",
            f"VERIFIER_BASE_URL=http://{engine}:8000/v1",
            "-e",
            "VERIFIER_MODEL=fixture-model",
            args.image,
        )
        port = docker("port", gateway, "8080").rsplit(":", 1)[1]
        url = f"http://127.0.0.1:{port}"
        with httpx.Client(base_url=url, timeout=10) as client:
            for _ in range(60):
                try:
                    if client.get("/readyz").status_code == 200:
                        break
                except httpx.HTTPError:
                    pass
                time.sleep(0.5)
            else:
                raise AssertionError("Gateway/fixture failed readiness")
            assert client.post("/v1/systemone", content="bad").status_code == 401
            client.headers["Authorization"] = "Bearer fixture-gateway-key"
            payload = json.loads((ROOT / "examples/triage.json").read_text())
            with ThreadPoolExecutor(max_workers=8) as pool:
                results = list(
                    pool.map(lambda _: client.post("/v1/systemone", json=payload), range(24))
                )
            assert all(r.status_code == 200 for r in results)
            assert all(r.json()["model"] == "fixture-model" for r in results)
            assert all(
                r.json()["usage"] == {"input_tokens": 30, "output_tokens": 15} for r in results
            )
            assert len({r.headers["x-request-id"] for r in results}) == 24
            question = {"q": {"type": "noul", "instructions": "True?"}}
            for state, status in [("repair", 200), ("invalid", 502), ("overloaded", 529)]:
                response = client.post(
                    "/v1/systemone",
                    json={
                        "model": "jev-latest",
                        "state": state,
                        "questions": question,
                    },
                )
                assert response.status_code == status, response.text
                if state == "repair":
                    assert response.json()["usage"] == {"input_tokens": 20, "output_tokens": 10}
            buffer = io.BytesIO()
            Image.new("RGB", (2, 2), "red").save(buffer, format="PNG")
            payload["images"] = [
                "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()
            ]
            response = client.post("/v1/vision/systemone", json=payload)
            assert response.status_code == 200, response.text
            assert response.json()["observation"].startswith("FIXTURE:")
            assert response.json()["usage"] == {"input_tokens": 40, "output_tokens": 20}
            assert client.get("/metrics").status_code == 200
        with TypeSafeClient(
            api_key="fixture-gateway-key", base_url=url, retry=RetryPolicy(max_retries=0)
        ) as sdk:
            result = sdk.system_one(
                state="fixture",
                questions={
                    "choice": Choice(criteria={"a": None, "b": None}),
                    "score": Score(criteria=["low", "high"]),
                    "noul": Noul(instructions="True?"),
                },
            )
            assert result.choices["choice"].choice == "a"
            assert result.scores["score"].score == 0
            assert result.nouls["noul"].noul == 0.8
            assert "jev-latest" in {m.name for m in sdk.models.list().models}
        assert docker("exec", gateway, "python", "-c", "import os; print(os.getuid())") == "10001"
        print(
            "PASS: Docker HTTP fixture integration, SDK, auth, 24 concurrent requests, "
            "repair/usage, failure mapping, vision, metrics and non-root execution."
        )
        print("This is a protocol test, not a real vLLM/DiffusionGemma inference test.")
    except BaseException:
        for name in [gateway, engine]:
            print(docker("logs", name, check=False))
        raise
    finally:
        docker("rm", "-f", gateway, engine, check=False)
        docker("network", "rm", prefix, check=False)


if __name__ == "__main__":
    main()
