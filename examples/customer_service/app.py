"Loopback-only customer-service demo with isolated, bounded in-memory sessions."

import argparse
import asyncio
import copy
import os
import secrets
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from .catalog import DATA
from .classifier import Classifier, SystemOneClassifier
from .dialogue import Conversation, Dialogue

ROOT = Path(__file__).parent
COOKIE = "customer_service_session"


@dataclass
class Session:
    state: Conversation = field(default_factory=Conversation)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    accessed: float = field(default_factory=time.monotonic)


class Message(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    text: str = Field(min_length=1, max_length=400)
    intent: str | None = Field(default=None, max_length=64)


def create_app(
    classifier: Classifier | None = None,
    engine_url: str = "http://127.0.0.1:18080",
    model: str = "jev-latest",
    max_sessions: int = 256,
    ttl: float = 3600,
) -> FastAPI:
    router = classifier or SystemOneClassifier(engine_url, os.getenv("VERIFIER_API_KEY", ""), model)
    dialogue = Dialogue(router)
    sessions: dict[str, Session] = {}

    @asynccontextmanager
    async def lifespan(app):
        yield
        await router.close()

    app = FastAPI(title="Customer Service Lab", lifespan=lifespan)
    app.state.sessions = sessions

    @app.middleware("http")
    async def boundary(request: Request, call_next):
        # The local demo accepts requests from its own browser origin only.
        origin = request.headers.get("origin")
        if origin and origin != f"{request.url.scheme}://{request.headers.get('host', '')}":
            return Response("Cross-origin request rejected", status_code=403)
        if request.method == "POST":
            data = bytearray()
            async for chunk in request.stream():
                data.extend(chunk)
                if len(data) > 8192:
                    return Response("Request too large", status_code=413)
            request._body = bytes(data)
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self'; "
            "img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'"
        )
        return response

    def session(request: Request, response: Response) -> Session:
        now = time.monotonic()
        for key, value in list(sessions.items()):
            if now - value.accessed > ttl and not value.lock.locked():
                del sessions[key]
        token = request.cookies.get(COOKIE)
        if token not in sessions:
            if len(sessions) >= max_sessions:
                raise HTTPException(503, "세션이 가득 찼어요. 잠시 후 다시 시도해주세요.")
            token = secrets.token_urlsafe(32)
            sessions[token] = Session()
            response.set_cookie(COOKIE, token, httponly=True, samesite="strict", max_age=int(ttl))
        result = sessions[token]
        result.accessed = now
        return result

    @app.get("/")
    async def home():
        return FileResponse(ROOT / "static/index.html")

    @app.get("/api/catalog")
    async def catalog():
        return DATA

    @app.get("/api/session")
    async def get_session(request: Request, response: Response):
        current = session(request, response)
        async with current.lock:
            return {
                "messages": current.state.messages,
                "state": current.state.snapshot(),
                "trace": current.state.trace,
            }

    @app.post("/api/reset")
    async def reset(request: Request, response: Response):
        current = session(request, response)
        async with current.lock:
            current.state = Conversation()
        return {"messages": [], "state": current.state.snapshot(), "trace": {}}

    @app.post("/api/chat")
    async def chat(body: Message, request: Request, response: Response):
        current = session(request, response)
        try:
            async with asyncio.timeout(100):
                async with current.lock:
                    # A failed model request cannot partially advance a conversation.
                    working = copy.deepcopy(current.state)
                    result = await dialogue.turn(working, body.text, body.intent)
                    current.state = working
                    current.accessed = time.monotonic()
                    return result
        except (httpx.HTTPError, TimeoutError) as exc:
            raise HTTPException(
                503,
                (
                    "분류 엔진에 연결하지 못했어요. Kai 또는 "
                    "DiffusionGemma 서버를 확인해주세요. 대화 상태는 "
                    "유지돼요."
                ),
            ) from exc
        except ValueError as exc:
            raise HTTPException(
                422, "입력 또는 분류 응답을 확인할 수 없어요. 짧게 다시 입력해주세요."
            ) from exc

    app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")
    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine-url", default="http://127.0.0.1:18080")
    parser.add_argument("--model", default="jev-latest")
    parser.add_argument("--port", type=int, default=18090)
    args = parser.parse_args()
    uvicorn.run(
        create_app(engine_url=args.engine_url, model=args.model), host="127.0.0.1", port=args.port
    )


if __name__ == "__main__":
    main()
