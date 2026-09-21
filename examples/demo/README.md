# Local API demo

```sh
docker compose -f compose.demo.yaml up --build -d --wait
curl http://127.0.0.1:8080/v1/systemone \
  -H 'Authorization: Bearer local-demo-key' \
  -H 'Content-Type: application/json' --data-binary @examples/triage.json
```

Run these commands from the repository root. If `VERIFIER_API_KEY` is set in your environment
or `.env`, use that key. Set `VERIFIER_HTTP_PORT` to change the default port 8080.

The gateway calls a separate HTTP server that returns uniform probabilities and the model name
`demo-uniform`. Text and images are not evaluated. Do not use these outputs to measure model
quality or inference latency. The demo server is not included in the application package or
Docker image. It has its own Dockerfile and runs in a separate container with no published port.

```sh
docker compose -f compose.demo.yaml down
```
