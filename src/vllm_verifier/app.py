import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from fastapi.security import HTTPBearer
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Histogram,
    generate_latest,
)

from . import __version__
from .backend import Backend, VLLMBackend
from .config import Settings
from .errors import ServiceError
from .middleware import BoundaryMiddleware
from .models import SystemOneRequest, SystemOneResponse, VisionRequest, VisionResponse
from .service import DecisionService


def create_app(settings: Settings | None = None, backend: Backend | None = None) -> FastAPI:
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        active_backend: Backend
        if backend is not None:
            active_backend = backend
        elif settings.runtime == "mlx":
            from .engine.mlx_backend import MLXBackend

            local_backend = MLXBackend(settings)
            await local_backend.start()
            active_backend = local_backend
        else:
            active_backend = VLLMBackend(settings)
        app.state.service = DecisionService(settings, active_backend)
        try:
            yield
        finally:
            await active_backend.close()

    app = FastAPI(
        title="vLLM Verifier",
        version=__version__,
        lifespan=lifespan,
        description="Jev-compatible decisions with uncalibrated model estimates.",
    )
    registry = CollectorRegistry()
    auth = [Depends(HTTPBearer(auto_error=False))]
    count = Counter(
        "verifier_requests_total", "HTTP requests", ["route", "status"], registry=registry
    )
    latency = Histogram("verifier_request_seconds", "HTTP latency", ["route"], registry=registry)
    app.add_middleware(BoundaryMiddleware, settings=settings, requests=count, latency=latency)

    @app.exception_handler(ServiceError)
    async def service_error(request: Request, exc: ServiceError) -> JSONResponse:
        return JSONResponse(
            {
                "error": {
                    "code": exc.code,
                    "message": exc.message,
                    "request_id": request.state.request_id,
                }
            },
            status_code=exc.status,
            headers={"Retry-After": "1"} if exc.status == 529 else None,
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        # Do not echo state, images, secrets or non-finite input values in validation errors.
        details = [{"loc": e["loc"], "type": e["type"]} for e in exc.errors()]
        return JSONResponse(
            {
                "error": {
                    "code": "validation_error",
                    "message": "Request failed validation",
                    "details": details,
                    "request_id": request.state.request_id,
                }
            },
            status_code=422,
        )

    @app.get("/healthz", openapi_extra={"security": []})
    async def health() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    @app.get("/readyz", openapi_extra={"security": []})
    async def readiness(request: Request) -> JSONResponse:
        ready = await request.app.state.service.backend.ready()
        return JSONResponse(
            {"status": "ready" if ready else "not_ready"}, status_code=200 if ready else 503
        )

    @app.get("/metrics", include_in_schema=False)
    async def metrics() -> Response:
        return Response(generate_latest(registry), headers={"Content-Type": CONTENT_TYPE_LATEST})

    @app.get("/v1/models", dependencies=auth)
    async def models(request: Request) -> dict[str, Any]:
        actual = request.app.state.service.model_name
        return {
            "models": [
                {
                    "name": name,
                    "description": f"vLLM Verifier adapter for {actual}; uncalibrated estimates. "
                    "release_date describes this adapter, not the model weights.",
                    "release_date": "2026-09-22",
                }
                for name in dict.fromkeys(["jev-latest", "diffusion-jev", actual])
            ]
        }

    @app.post("/v1/systemone", response_model=SystemOneResponse, dependencies=auth)
    async def system_one(body: SystemOneRequest, request: Request) -> SystemOneResponse:
        try:
            async with asyncio.timeout(settings.request_timeout):
                result: SystemOneResponse = await request.app.state.service.evaluate(body)
                return result
        except TimeoutError as exc:
            raise ServiceError(504, "request_timeout", "Decision deadline exceeded") from exc

    @app.post("/v1/vision/systemone", response_model=VisionResponse, dependencies=auth)
    async def vision_one(body: VisionRequest, request: Request) -> VisionResponse:
        try:
            async with asyncio.timeout(settings.request_timeout):
                result: VisionResponse = await request.app.state.service.evaluate_vision(body)
                return result
        except TimeoutError as exc:
            raise ServiceError(504, "request_timeout", "Decision deadline exceeded") from exc

    return app
