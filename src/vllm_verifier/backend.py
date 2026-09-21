import asyncio
import json
from dataclasses import dataclass
from typing import Any, Protocol

import httpx
from pydantic import ValidationError

from .config import Settings
from .errors import ServiceError
from .models import Usage


@dataclass
class Completion:
    text: str
    usage: Usage
    finish_reason: str


class Backend(Protocol):
    async def complete(self, messages: list[dict[str, Any]]) -> Completion: ...

    async def ready(self) -> bool: ...

    async def close(self) -> None: ...


class VLLMBackend:
    def __init__(self, settings: Settings, transport: httpx.AsyncBaseTransport | None = None):
        self.settings = settings
        headers = {}
        if settings.upstream_api_key:
            headers["Authorization"] = "Bearer " + settings.upstream_api_key.get_secret_value()
        self.client = httpx.AsyncClient(
            base_url=settings.base_url + "/",
            headers=headers,
            timeout=settings.upstream_timeout,
            limits=httpx.Limits(max_connections=settings.max_concurrency + 2),
            transport=transport,
            trust_env=False,
        )

    async def complete(self, messages: list[dict[str, Any]]) -> Completion:
        body = {
            "model": self.settings.model,
            "messages": messages,
            "max_tokens": self.settings.max_output_tokens,
            "temperature": self.settings.temperature,
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        raw = await self._request_json("POST", "chat/completions", body)
        try:
            choice = raw["choices"][0]
            content = choice["message"]["content"]
            finish = choice["finish_reason"]
            if not isinstance(content, str) or not isinstance(finish, str):
                raise ValueError("missing completion")
            usage = Usage(
                input_tokens=raw["usage"]["prompt_tokens"],
                output_tokens=raw["usage"]["completion_tokens"],
            )
            return Completion(content, usage, finish)
        except (ValueError, KeyError, IndexError, TypeError, ValidationError) as exc:
            raise ServiceError(
                502, "upstream_protocol_error", "Invalid inference server response"
            ) from exc

    async def _request_json(
        self, method: str, path: str, body: dict[str, Any] | None = None
    ) -> Any:
        try:
            async with self.client.stream(method, path, json=body) as response:
                if response.status_code in {429, 503, 529}:
                    raise ServiceError(529, "upstream_overloaded", "Inference server is overloaded")
                if not response.is_success:
                    raise ServiceError(
                        502, "upstream_error", "Inference server rejected the request"
                    )
                data = bytearray()
                async for chunk in response.aiter_bytes(chunk_size=65536):
                    data.extend(chunk)
                    if len(data) > 2 * 1024 * 1024:
                        raise ServiceError(
                            502, "upstream_response_too_large", "Response exceeded limit"
                        )
            return json.loads(data)
        except httpx.TimeoutException as exc:
            raise ServiceError(504, "upstream_timeout", "Inference server timed out") from exc
        except httpx.HTTPError as exc:
            raise ServiceError(
                502, "upstream_unavailable", "Inference server is unavailable"
            ) from exc
        except (ValueError, RecursionError) as exc:
            raise ServiceError(
                502, "upstream_protocol_error", "Invalid inference server response"
            ) from exc

    async def ready(self) -> bool:
        try:
            async with asyncio.timeout(3):
                body = await self._request_json("GET", "models")
                return any(item["id"] == self.settings.model for item in body["data"])
        except (ServiceError, ValueError, KeyError, TypeError, TimeoutError):
            return False

    async def close(self) -> None:
        await self.client.aclose()
