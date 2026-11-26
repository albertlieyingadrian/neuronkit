"""Judges: anything with `complete(system, prompt, max_tokens) -> str`.

neuronkit's judge steps take any object with that method, which is circuitcheck's `LLM` protocol, so circuitcheck's
`AnthropicLLM`, `OpenAILLM`, `CachedLLM` and `ScriptedLLM` all work. This module adds an OpenAI-compatible client with
no SDK dependency (vLLM, OpenRouter and most local servers speak that API), and a helper to run many calls at once.
"""

from __future__ import annotations

import json
import os
import urllib.request
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Protocol


class LLM(Protocol):
    def complete(self, system: str, prompt: str, max_tokens: int = 1024) -> str: ...


@dataclass
class OpenAICompatLLM:
    """A chat model behind an OpenAI-compatible `/chat/completions` endpoint.

    >>> judge = OpenAICompatLLM("https://openrouter.ai/api/v1", "qwen/qwen3-32b", api_key_env="OPENROUTER_API_KEY")
    """

    base_url: str
    model: str
    api_key_env: str | None = "OPENAI_API_KEY"
    temperature: float = 0.0
    timeout: float = 120.0
    extra: dict = field(default_factory=dict)
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def complete(self, system: str, prompt: str, max_tokens: int = 1024) -> str:
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": self.temperature,
            **self.extra,
        }
        req = urllib.request.Request(
            self.base_url.rstrip("/") + "/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        key = os.environ.get(self.api_key_env) if self.api_key_env else None
        if key:
            req.add_header("Authorization", f"Bearer {key}")
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            out = json.loads(resp.read())
        self.calls += 1
        usage = out.get("usage") or {}
        self.prompt_tokens += int(usage.get("prompt_tokens", 0))
        self.completion_tokens += int(usage.get("completion_tokens", 0))
        return out["choices"][0]["message"].get("content") or ""


@dataclass
class FunctionLLM:
    """Answers with `fn(system, prompt)`: for tests and offline demos."""

    fn: Callable[[str, str], str]
    model: str = "function"

    def complete(self, system: str, prompt: str, max_tokens: int = 1024) -> str:
        return self.fn(system, prompt)


def complete_many(llm, items: Sequence[tuple[str, str, int]], workers: int = 8) -> list[str]:
    """Run (system, prompt, max_tokens) calls. Uses `llm.complete_many(items)` when the judge batches natively."""
    native = getattr(llm, "complete_many", None)
    if native is not None:
        return list(native(items))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(lambda it: llm.complete(it[0], it[1], it[2]), items))
