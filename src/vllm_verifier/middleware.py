import asyncio
import hmac
import time
import uuid

from prometheus_client import Counter, Histogram
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from .config import Settings


class BoundaryMiddleware:
    """Authenticate before reading bodies, bound admission and body reads, add request IDs."""

    def __init__(self, app: ASGIApp, settings: Settings, requests: Counter, latency: Histogram):
        self.app = app
        self.settings = settings
        self.requests = requests
        self.latency = latency
        self.active = 0

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        started = time.monotonic()
        status = 500
        request_id = uuid.uuid4().hex
        scope.setdefault("state", {})["request_id"] = request_id
        path = scope["path"]
        route = (
            path
            if path in {"/v1/systemone", "/v1/vision/systemone", "/healthz", "/readyz", "/metrics"}
            else "other"
        )

        async def send_response(message: Message) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
                message["headers"].append((b"x-request-id", request_id.encode()))
                message["headers"].append(
                    (b"x-probability-source", b"uncalibrated-model-estimates")
                )
            await send(message)

        async def reject(code: int, name: str, message: str) -> None:
            headers = {"Retry-After": "1"} if code == 529 else None
            response = JSONResponse(
                {"error": {"code": name, "message": message, "request_id": request_id}},
                status_code=code,
                headers=headers,
            )
            await response(scope, receive, send_response)

        admitted = False
        try:
            if self.settings.api_key and path not in {"/healthz", "/readyz"}:
                headers = dict(scope["headers"])
                expected = ("Bearer " + self.settings.api_key.get_secret_value()).encode()
                if not hmac.compare_digest(headers.get(b"authorization", b""), expected):
                    await reject(401, "unauthorized", "Missing or invalid Bearer token")
                    return
            if path in {"/v1/systemone", "/v1/vision/systemone"}:
                if self.active >= self.settings.max_requests:
                    await reject(529, "overloaded", "Request capacity exceeded")
                    return
                self.active += 1
                admitted = True
                body = bytearray()
                try:
                    async with asyncio.timeout(self.settings.request_timeout):
                        while True:
                            message = await receive()
                            if message["type"] == "http.disconnect":
                                return
                            body.extend(message.get("body", b""))
                            if len(body) > self.settings.max_body_bytes:
                                await reject(413, "body_too_large", "Request body exceeded limit")
                                return
                            if not message.get("more_body", False):
                                break
                except TimeoutError:
                    await reject(408, "body_timeout", "Request body read timed out")
                    return
                delivered = False

                async def replay() -> Message:
                    nonlocal delivered
                    if not delivered:
                        delivered = True
                        return {"type": "http.request", "body": bytes(body), "more_body": False}
                    return await receive()

                await self.app(scope, replay, send_response)
            else:
                await self.app(scope, receive, send_response)
        finally:
            if admitted:
                self.active -= 1
            self.requests.labels(route, str(status)).inc()
            self.latency.labels(route).observe(time.monotonic() - started)
