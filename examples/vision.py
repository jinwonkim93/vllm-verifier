"""Usage: uv run python examples/vision.py ./photo.png"""

import argparse
import base64
import mimetypes
import os
from pathlib import Path

import httpx

parser = argparse.ArgumentParser()
parser.add_argument("image", type=Path)
args = parser.parse_args()
mime = mimetypes.guess_type(args.image.name)[0]
if mime not in {"image/png", "image/jpeg", "image/webp"}:
    parser.error("Use a PNG, JPEG or WebP file")
encoded = base64.b64encode(args.image.read_bytes()).decode()
response = httpx.post(
    os.environ.get("VERIFIER_URL", "http://127.0.0.1:8080") + "/v1/vision/systemone",
    headers={"Authorization": "Bearer " + os.environ.get("VERIFIER_API_KEY", "")},
    json={
        "model": "jev-latest",
        "state": "Inspect this product photo.",
        "images": [f"data:{mime};base64,{encoded}"],
        "questions": {"damage": {"type": "noul", "instructions": "Is visible damage present?"}},
    },
    timeout=65,
)
response.raise_for_status()
print(response.json())
