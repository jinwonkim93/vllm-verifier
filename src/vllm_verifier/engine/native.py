"""Optional vLLM runtime; install in a compatible GPU environment."""

import importlib
from typing import Any

from ..backend import Completion
from ..models import Usage
from .core import WorkItem


class VLLMRuntime:
    def __init__(self, model: str, **engine_args: Any):
        try:
            vllm = importlib.import_module("vllm")
        except ImportError as exc:
            raise RuntimeError(
                "Native execution requires a compatible vLLM GPU installation"
            ) from exc
        self.sampling_params = vllm.SamplingParams
        self.llm = vllm.LLM(model=model, **engine_args)
        self.tokenizer = self.llm.get_tokenizer()

    def encode(self, messages: list[dict[str, Any]]) -> list[int]:
        result = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        if not isinstance(result, list) or any(type(token) is not int for token in result):
            raise RuntimeError("Tokenizer did not return a list of token IDs")
        return result

    def generate(self, jobs: list[WorkItem]) -> list[Completion]:
        outputs = self.llm.generate(
            [{"prompt_token_ids": list(job.token_ids)} for job in jobs],
            sampling_params=[
                self.sampling_params(temperature=0, max_tokens=job.output_tokens, n=1)
                for job in jobs
            ],
            use_tqdm=False,
        )
        if len(outputs) != len(jobs):
            raise RuntimeError("Native vLLM returned a different number of outputs")
        completions = []
        for job, output in zip(jobs, outputs, strict=True):
            if output.prompt_token_ids != list(job.token_ids):
                raise RuntimeError("Native vLLM output does not match the submitted prompt")
            if len(output.outputs) != 1 or output.prompt_token_ids is None:
                raise RuntimeError("Unexpected native vLLM output shape")
            candidate = output.outputs[0]
            completions.append(
                Completion(
                    candidate.text,
                    Usage(
                        input_tokens=len(output.prompt_token_ids),
                        output_tokens=len(candidate.token_ids),
                    ),
                    candidate.finish_reason or "unfinished",
                )
            )
        return completions
