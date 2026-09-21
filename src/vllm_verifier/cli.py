import argparse

import uvicorn

from .app import create_app
from .config import Settings


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve a Jev-compatible decision API")
    parser.add_argument("--host")
    parser.add_argument("--port", type=int, choices=range(1, 65536), metavar="PORT")
    args = parser.parse_args()
    settings = Settings()
    if args.host is not None:
        settings.host = args.host
    if args.port is not None:
        settings.port = args.port
    if settings.api_key is None and settings.host not in {"127.0.0.1", "localhost", "::1"}:
        parser.error("Set VERIFIER_API_KEY before binding to a non-loopback interface")
    uvicorn.run(create_app(settings), host=settings.host, port=settings.port)


if __name__ == "__main__":
    main()
