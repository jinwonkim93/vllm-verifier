"""HTTP protocol fixture only. No model is loaded and no AI inference occurs."""

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODEL = "fixture-model"


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

    def authorized(self):
        if self.headers.get("Authorization") != "Bearer fixture-upstream-key":
            self.reply(401, {"error": "wrong upstream credential"})
            return False
        return True

    def do_GET(self):
        if self.authorized():
            self.reply(200, {"data": [{"id": MODEL}]})

    def do_POST(self):
        if not self.authorized():
            return
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path != "/v1/chat/completions" or body["model"] != MODEL:
            self.reply(400, {"error": "wrong path/model"})
            return
        if "response_format" in body or "structured_outputs" in body:
            self.reply(400, {"error": "unsupported diffusion constraint"})
            return
        content = body["messages"][-1]["content"]
        if isinstance(content, list):
            answer = "FIXTURE: a red square. No image inference performed."
        else:
            prompt = json.loads(content)
            state, question = prompt["state"], prompt["question"]
            if state == "overloaded":
                self.reply(503, {"error": "fixture overload"})
                return
            if state == "invalid" or (
                state == "repair" and "failed validation" not in body["messages"][0]["content"]
            ):
                answer = "invalid JSON"
            elif question["type"] == "noul":
                answer = json.dumps({"noul": 0.8})
            else:
                count = len(question["criteria"])
                answer = json.dumps({"probabilities": [1.0] + [0.0] * (count - 1)})
        self.reply(
            200,
            {
                "choices": [{"message": {"content": answer}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            },
        )


if __name__ == "__main__":
    ThreadingHTTPServer(("0.0.0.0", 8000), Handler).serve_forever()
