"""Standalone synthetic OpenAI-compatible server. No model or image inference."""

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODEL = "demo-uniform"


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def reply(self, status, body):
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path != "/v1/models":
            self.reply(404, {"error": "not found"})
            return
        self.reply(200, {"data": [{"id": MODEL}]})

    def do_POST(self):
        if self.path != "/v1/chat/completions":
            self.reply(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 16 * 1024 * 1024:
                self.reply(413, {"error": "body too large or empty"})
                return
            body = json.loads(self.rfile.read(length))
            if body["model"] != MODEL:
                self.reply(400, {"error": "unknown model"})
                return
            content = body["messages"][-1]["content"]
            if isinstance(content, list):
                answer = "DEMO: image contents were not evaluated."
            else:
                question = json.loads(content)["question"]
                if question["type"] == "noul":
                    answer = json.dumps({"noul": 0.5})
                else:
                    count = len(question["criteria"])
                    answer = json.dumps({"probabilities": [1 / count] * count})
            self.reply(
                200,
                {
                    "choices": [{"message": {"content": answer}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 0, "completion_tokens": 0},
                },
            )
        except (ValueError, KeyError, IndexError, TypeError, ZeroDivisionError):
            self.reply(400, {"error": "invalid request"})


if __name__ == "__main__":
    ThreadingHTTPServer(("0.0.0.0", 8000), Handler).serve_forever()
