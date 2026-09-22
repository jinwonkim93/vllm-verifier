"""DiffusionGemma execution on Apple Silicon through the pinned MLX-VLM runtime."""

import importlib
import platform
from typing import Any

from ..backend import Completion
from ..models import Usage
from .core import WorkItem
from .gemma import final_content

DEFAULT_MODEL = "mlx-community/diffusiongemma-26B-A4B-it-4bit"


class MLXRuntime:
    """Single-owner runtime. MLX-VLM diffusion currently executes one sequence at a time."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        revision: str | None = None,
        canvas_tokens: int = 256,
        prefix_caching: bool = False,
        max_model_len: int = 4096,
    ):
        if platform.system() != "Darwin" or platform.machine() != "arm64":
            raise RuntimeError("MLX execution requires an Apple Silicon Mac")
        if not 1 <= canvas_tokens <= 256 or max_model_len < 257:
            raise ValueError("Invalid canvas or context budget")
        self.mx = importlib.import_module("mlx.core")
        if not self.mx.metal.is_available():
            raise RuntimeError("Metal GPU is not available")
        self.mx.set_cache_limit(256 * 1024 * 1024)
        self.mx.set_memory_limit(self.mx.device_info()["max_recommended_working_set_size"])
        vlm = importlib.import_module("mlx_vlm")
        self.model, self.processor = vlm.load(model, revision=revision)
        if self.model.config.model_type != "diffusion_gemma":
            raise ValueError("The MLX runtime currently supports diffusion_gemma models only")
        self.tokenizer = getattr(self.processor, "tokenizer", self.processor)
        self.stream = importlib.import_module(
            "mlx_vlm.generate.diffusion"
        ).stream_diffusion_generate
        self.canvas_tokens = canvas_tokens
        self.max_model_len = max_model_len
        self.apc = None
        if prefix_caching:
            self.apc = importlib.import_module("mlx_vlm.apc").APCManager(
                num_blocks=128,
                block_size=64,
                overrides={"checkpoint_entries": 2, "checkpoint_interval_tokens": 256},
            )
        self.stats: list[dict[str, Any]] = []

    def encode(self, messages: list[dict[str, Any]]) -> list[int]:
        if any(not isinstance(message.get("content"), str) for message in messages):
            raise ValueError("The Mac decision runtime currently accepts text only")
        result = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        if not isinstance(result, list) or any(type(token) is not int for token in result):
            raise RuntimeError("Tokenizer did not return token IDs")
        return result

    def complete_tokens(self, token_ids: list[int], output_tokens: int) -> Completion:
        if len(token_ids) + output_tokens > self.max_model_len:
            raise ValueError("Prompt plus output budget exceeds the Mac context limit")
        self.mx.reset_peak_memory()
        parts = []
        terminal = None
        stream = self.stream(
            self.model,
            self.processor,
            self.tokenizer,
            self.mx.array([token_ids], dtype=self.mx.int32),
            None,
            None,
            max_tokens=output_tokens,
            skip_special_token_ids=[],
            temperature=0,
            diffusion_max_canvas_length=self.canvas_tokens,
            apc_manager=self.apc,
        )
        try:
            for result in stream:
                if not result.is_draft:
                    parts.append(result.text)
                if result.finish_reason is not None:
                    terminal = result
        finally:
            stream.close()
        if terminal is None:
            raise RuntimeError("MLX generation ended without a terminal result")
        self.stats.append(
            {
                key: getattr(terminal, key)
                for key in (
                    "cached_tokens",
                    "diffusion_canvas_tokens",
                    "diffusion_denoising_steps",
                    "diffusion_work_tokens",
                    "peak_memory",
                    "prompt_tps",
                    "generation_tps",
                )
            }
        )
        text = "".join(parts)
        try:
            text = final_content(text)
        except ValueError:
            # Preserve malformed framing for the strict verifier and bounded repair path.
            # Never recover an arbitrary JSON substring from an invalid response.
            pass
        return Completion(
            text,
            Usage(input_tokens=terminal.prompt_tokens, output_tokens=terminal.generation_tokens),
            terminal.finish_reason,
        )

    def generate(self, jobs: list[WorkItem]) -> list[Completion]:
        return [self.complete_tokens(list(job.token_ids), job.output_tokens) for job in jobs]
