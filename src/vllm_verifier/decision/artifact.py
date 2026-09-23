"""Load the reviewed Kai export without global imports or runtime monkey patches."""

import hashlib
import importlib
import importlib.util
import json
import sys
from pathlib import Path, PurePosixPath
from typing import Any

MODEL = "llm-semantic-router/Decision-1.0-Kai-0.6B"
REVISION = "7185f514f54b8f93c55998b1e8f9c5cc67f0d029"
MANIFEST_SHA256 = "c1bf07ab1c4c3fa1f819256d3de858d1ed87869bdfa663553280d7e78b88bee4"


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_artifact(root: Path) -> None:
    """Verify the trusted manifest and every payload before importing Python."""
    if digest(root / "MANIFEST.json") != MANIFEST_SHA256:
        raise ValueError("Kai manifest does not match the reviewed release")
    manifest = json.loads((root / "MANIFEST.json").read_text())
    for name, ref in manifest["files"].items():
        rel = PurePosixPath(name)
        if rel.is_absolute() or ".." in rel.parts or "\\" in name:
            raise ValueError("Unsafe artifact path")
        path = root / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Missing materialized artifact: {name}")
        if path.stat().st_size != ref["bytes"] or digest(path) != ref["sha256"]:
            raise ValueError(f"Artifact checksum mismatch: {name}")
    actual = {
        p.relative_to(root).as_posix()
        for p in root.rglob("*")
        if p.is_file() and "__pycache__" not in p.parts and p.name != "MANIFEST.json"
    }
    if actual != set(manifest["files"]):
        raise ValueError("Unexpected artifact files")


def load_artifact(directory: Path) -> Any:
    root = directory / "native"
    verify_artifact(root)
    namespace = "_verifier_kai_" + REVISION
    if namespace not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            namespace, root / "__init__.py", submodule_search_locations=[str(root)]
        )
        if spec is None or spec.loader is None:
            raise RuntimeError("Cannot import verified Kai runtime")
        package = importlib.util.module_from_spec(spec)
        sys.modules[namespace] = package
        try:
            spec.loader.exec_module(package)
        except BaseException:
            del sys.modules[namespace]
            raise
    return importlib.import_module(namespace + ".artifacts")


def download_artifact(cache_dir: Path) -> Path:
    from huggingface_hub import snapshot_download

    directory = cache_dir / "decision" / REVISION
    manifest_path = directory / "native" / "MANIFEST.json"
    if manifest_path.is_file() and digest(manifest_path) == MANIFEST_SHA256:
        files = json.loads(manifest_path.read_text())["files"]
        if all((directory / "native" / name).is_file() for name in files):
            # load_artifact still verifies all bytes; cached starts require no network.
            return directory
    snapshot_download(
        MODEL,
        revision=REVISION,
        local_dir=directory,
        allow_patterns=["native/*", "LICENSE*", "NOTICE*", "*TERMS*", "LICENSING_STATUS.md"],
    )
    return directory
