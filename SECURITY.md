# Security

Do not include credentials, private prompts or images in public issues. For sensitive reports,
use your repository host's private vulnerability-reporting facility when enabled, or contact
the deployment maintainer privately. This local repository does not define a public security inbox.

The gateway binds to loopback by default. Its CLI requires VERIFIER_API_KEY for a non-loopback bind.
Library users calling create_app directly must enforce their own network/auth policy. Use TLS,
per-client rate limiting and request-size limits at an ingress proxy for shared deployments.
Only healthz and readyz bypass authentication. API docs and metrics require a key when configured.

Do not expose the vLLM port to untrusted clients. It is a separate service with independent
credentials and limits. Configure its URL only from trusted deployment settings. The gateway
ignores HTTP proxy environment variables and does not follow upstream redirects.

The image extension accepts checked inline PNG/JPEG/WebP data only. It does not fetch client URLs.
Image contents and user state remain untrusted model input. Prompt injection can still change
model judgments: structural verification is not semantic verification or an authorization system.
No commands, payments, webhooks or tool calls are executed by this project.

Resource limits apply per gateway process, not globally across replicas. Upstream GPU cancellation
is ultimately controlled by vLLM; closing a request does not prove a kernel stopped immediately.
