# Source notes

Reviewed 2026-09-22. Primary documentation and installed SDK source guided the implementation.
The linked social posts supplied the use case; inaccessible video/social content was not treated
as an API specification.

- [TypeSafe HTTP API](https://docs.typesafe.ai/api): endpoint, envelopes and error status conventions.
- [Choice](https://docs.typesafe.ai/primitives/choice), [Score](https://docs.typesafe.ai/primitives/score),
  [Noul](https://docs.typesafe.ai/primitives/noul): decision types and arithmetic semantics.
- [State](https://docs.typesafe.ai/concepts/state): text/structured state; no original image input.
- [Confidence](https://docs.typesafe.ai/confidence): confidence is distribution-derived; no exact
  formula is published on this page. Our entropy formula is a local design choice.
- [Official Python SDK](https://docs.typesafe.ai/sdk/python): package and client interface.
  Installed `typesafe-sdk==0.7.1` wire models also permit omitted instructions and singleton Score.
- [vLLM DiffusionGemma implementation](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/diffusion_gemma.py):
  native model support, multimodal architecture and sampler integration.
- [Official vLLM recipe](https://github.com/vllm-project/recipes/blob/main/models/Google/diffusiongemma-26B-A4B-it.yaml):
  model ID, GPU image and deployment options. The image tag is mutable.
- [vLLM integration article](https://vllm-project.github.io/2026/06/10/diffusion-gemma):
  block diffusion differs from ordinary left-to-right decoding; generation throughput is not
  evidence of this gateway's decision latency.
- [Structured-output support tracking](https://github.com/vllm-project/vllm/issues/45572):
  diffusion schema decoding has historically been unsupported. The gateway does not depend on it.

No proprietary Jev weights, training method, SDK code or vLLM kernels are copied into this project.
Runtime dependencies and model weights retain their respective upstream licenses.
