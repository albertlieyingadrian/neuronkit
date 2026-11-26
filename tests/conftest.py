"""Tiny random models (dense Qwen2 and OLMoE MoE) and a character tokenizer: everything runs on CPU in seconds."""

from __future__ import annotations

import pytest
import torch


class CharTokenizer:
    """One token per character (ASCII code mod 128). Enough for neuronkit's text helpers."""

    pad_token_id = 0
    eos_token_id = 1
    vocab_size = 128

    def __call__(self, text, add_special_tokens=False, **kw):
        if isinstance(text, list):
            return {"input_ids": [[ord(c) % 128 for c in t] for t in text]}
        return {"input_ids": [ord(c) % 128 for c in text]}

    def decode(self, ids, **kw):
        return "".join(chr(int(i)) for i in ids)

    def convert_ids_to_tokens(self, ids):
        return [chr(int(i)) for i in ids]


TEXTS = [
    "the cat sat on the mat. the dog sat on the log.",
    "Numbers: 1 2 3 4 5 6 7 8 9 10 and more numbers 11 12.",
    "def f(x): return x * 2  # a tiny python function",
    "Paris is the capital of France; Rome is the capital of Italy.",
    "She sells sea shells by the sea shore, said the teacher.",
    "In 1969 the first humans walked on the moon in July.",
    "Water boils at 100 degrees Celsius at sea level pressure.",
    "A B C D E F G H I J K L M N O P Q R S T U V W X Y Z",
]


def _qwen2():
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(0)
    cfg = Qwen2Config(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
        tie_word_embeddings=False,
    )
    cfg._attn_implementation = "eager"
    return Qwen2ForCausalLM(cfg).eval()


def _olmoe():
    from transformers import OlmoeConfig, OlmoeForCausalLM

    torch.manual_seed(0)
    cfg = OlmoeConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=16,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=4,
        num_experts=6,
        num_experts_per_tok=2,
        max_position_embeddings=256,
    )
    cfg._attn_implementation = "eager"
    return OlmoeForCausalLM(cfg).eval()


@pytest.fixture(scope="session")
def tok():
    return CharTokenizer()


@pytest.fixture(scope="session")
def texts():
    return TEXTS


@pytest.fixture(scope="session", params=["dense", "moe"])
def model(request):
    return _qwen2() if request.param == "dense" else _olmoe()


@pytest.fixture(scope="session")
def dense_model():
    return _qwen2()


@pytest.fixture(scope="session")
def moe_model():
    return _olmoe()
