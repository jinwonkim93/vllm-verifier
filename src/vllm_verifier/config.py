from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="VERIFIER_", env_file=".env", extra="ignore", validate_assignment=True
    )

    runtime: Literal["http", "mlx", "decision"] = "http"
    decision_device: Literal["mps", "cpu"] = "mps"
    decision_cache_dir: Path = Path.home() / ".cache" / "diffusion-jev"
    decision_batch_size: int = Field(default=8, ge=1, le=32)
    decision_batch_tokens: int = Field(default=4096, ge=1024, le=32768)
    decision_batch_requests: int = Field(default=8, ge=1, le=8)
    decision_batch_wait_ms: float = Field(default=2, ge=0, le=100)
    mlx_revision: str | None = None
    mlx_canvas_tokens: int = Field(default=256, ge=1, le=256)
    mlx_max_model_len: int = Field(default=4096, ge=257)
    base_url: str = "http://127.0.0.1:8000/v1"
    model: str = "google/diffusiongemma-26B-A4B-it"
    api_key: SecretStr | None = None
    upstream_api_key: SecretStr | None = None
    request_timeout: float = Field(default=60, gt=0, le=600)
    upstream_timeout: float = Field(default=30, gt=0, le=600)
    max_concurrency: int = Field(default=8, ge=1, le=1024)
    max_requests: int = Field(default=32, ge=1, le=4096)
    max_body_bytes: int = Field(default=16 * 1024 * 1024, ge=1024)
    max_output_tokens: int = Field(default=2048, ge=256, le=16384)
    validation_retries: int = Field(default=1, ge=0, le=3)
    temperature: float = Field(default=0.0, ge=0, le=2)
    host: str = "127.0.0.1"
    port: int = Field(default=8080, ge=1, le=65535)

    @field_validator("api_key", "upstream_api_key", mode="before")
    @classmethod
    def empty_key(cls, value: str | SecretStr | None) -> str | SecretStr | None:
        return None if value == "" else value

    @field_validator("base_url")
    @classmethod
    def validate_url(cls, value: str) -> str:
        url = urlsplit(value)
        if url.scheme not in {"http", "https"} or not url.hostname:
            raise ValueError("base_url must be an HTTP(S) URL")
        if url.username or url.password or url.query or url.fragment:
            raise ValueError("base_url must not contain credentials, query or fragment")
        return value.rstrip("/")
