# Jev compatibility

Target: TypeSafe's HTTP documentation reviewed 2026-09-22 and the installed official
`typesafe-sdk==0.7.1` wire models. This is protocol interoperability, not a Jev model clone.

| Surface | Supported behavior |
| --- | --- |
| `POST /v1/systemone` | `model`, `state`, `questions` → `model`, `answers`, `usage` |
| State | String, JSON object or array; nested JSON data retained |
| Questions | Choice, Score and Noul, mixed in one request |
| Instructions | String/object/array; omitted or null also accepted to match SDK wire schema |
| Choice criteria | 1–255 named options; string/object/array/null descriptions |
| Score criteria | 1–10 ordered descriptions; use at least 2 for a meaningful scale |
| Noul criteria | Optional object with optional `true` and `false` descriptions |
| Choice response | `type`, `choice`, full `probabilities`, `confidence` |
| Score response | `type`, expected-value `score`, `probabilities`, `legend`, `confidence` |
| Noul response | `type`, `noul` only |
| Question IDs | Preserved as response keys, never sent to the model |
| `GET /v1/models` | SDK model-list envelope with names, descriptions and adapter release date |
| Authentication | `Authorization: Bearer <local key>` when configured |
| Python SDK | Real SDK serializer and parser tested, including structured score legends |
| Other clients | HTTP contract available; JS/TS SDK has not been tested |

The prose reference requires instructions and recommends at least two Score levels. The SDK's
wire schema allows absent instructions and one level. We accept that broader SDK input; a
single-level Score necessarily returns zero and confidence one.

## Deliberate differences

- `jev-latest` and `diffusion-jev` are aliases for the operator's configured model. An arbitrary
  `jev-*` version is not accepted. Responses identify the actual configured model, never pretend
  the weights are Jev. The separate demo server advertises `demo-uniform`.
- Probabilities are generated estimates, not Jev's learned calibrated probabilities, and not token
  logprobs. `confidence = 1 - H(p)/ln(K)` (singleton: 1). TypeSafe does not publish the exact
  confidence formula in the reviewed reference, so numeric equivalence is not claimed.
- Questions run as independent model calls under a shared concurrency limit. More questions
  can increase latency and repeat input tokens. There is no near-constant-latency promise.
- Maximum 64 questions; question IDs and choice labels are nonempty and at most 256 characters.
  Unknown fields and non-finite JSON numbers are rejected. Body limit defaults to 16 MiB.
- Extra extension: `/v1/vision/systemone`. Jev's text API does not accept image content.
- Error bodies have a stable local `error: {code, message, request_id}` envelope. Validation
  includes field locations/types but does not echo input. Byte-for-byte TypeSafe error parity
  is not promised. Standard statuses 401/422 and overload status 529 are used; 413, 502 and 504
  additionally describe local resource limits and inference failures.
- `/v1/models` release dates refer to this adapter release, not the upstream model weights.
- Streaming, hosted account/billing APIs, fine-tuning and Jev training/calibration are out of scope.

## SDK migration

Install `typesafe-sdk` separately in your application. Change only the API root and key:

```python
TypeSafeClient(api_key="gateway-key", base_url="http://127.0.0.1:8080")
```

The SDK root must **not** end in `/v1`: the SDK appends `/v1/systemone`.
The gateway's `VERIFIER_BASE_URL`, by contrast, **must** point to the vLLM `/v1` API root.
SDK retries can retry failed full requests; turn them off for controlled latency measurements.
Re-evaluate decision thresholds with labeled application data before migrating model providers.

## Vision extension

Use the normal request plus `images`, a list of 1–4 inline `data:image/...;base64,...` strings.
PNG, JPEG and WebP are checked for declared format, integrity and a 16 MP / 8 MiB per-image limit.
The total encoded request still must fit the gateway's body limit. Remote URLs are rejected.

The response adds `observation` to the ordinary response. One vision call produces that text;
each question then sees `{"context": <original state>, "image_observation": <observation>}`.
Errors during observation abort the entire request. Observation may omit visual details, so this
path should be evaluated on image-specific tasks. It is not a video stream or an OCR guarantee.
