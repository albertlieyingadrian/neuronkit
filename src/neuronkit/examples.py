"""Top-activating examples of a neuron pole, as text excerpts with per-token activations."""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass, field

import torch
from torch import nn

from neuronkit.core import Neuron, batches, run_capture


@dataclass(order=True)
class Firing:
    """One firing: the document, the token position and the (signed) activation there."""

    value: float
    doc: int
    pos: int


@dataclass
class Excerpt:
    """A window of tokens around a position, with the neuron's activation on each token.

    `acts` are raw signed activations (NaN where an MoE neuron was not routed). `pole_acts(sign)` gives the
    activations seen from one pole: sign * act, clipped at 0.
    """

    tokens: list[str]
    acts: list[float]
    doc: int = -1
    start: int = 0
    focus: int = -1  # index within the window of the firing this excerpt was built around
    meta: dict = field(default_factory=dict)

    def pole_acts(self, sign: int) -> list[float]:
        return [max(0.0, sign * a) if not math.isnan(a) else 0.0 for a in self.acts]

    def text(self) -> str:
        return "".join(self.tokens)

    def to_record(self, sign: int = 1):
        """A circuitcheck `Record` (tokens and this pole's activations)."""
        from circuitcheck.labels import Record

        return Record(tokens=self.tokens, acts=self.pole_acts(sign), source=f"doc{self.doc}:{self.start}")


def pole_values(acts: torch.Tensor, sign: int) -> torch.Tensor:
    """sign * acts with NaN (not routed) as -inf, so it is never a top firing."""
    v = sign * acts.float()
    return torch.nan_to_num(v, nan=-math.inf)


def top_firings(acts_per_doc: Sequence[torch.Tensor], k: int, sign: int = 1, per_doc: int = 3) -> list[Firing]:
    """The k strongest firings of one pole over documents, at most `per_doc` from any one document."""
    cands: list[Firing] = []
    for d, a in enumerate(acts_per_doc):
        v = pole_values(a, sign)
        if v.numel() == 0:
            continue
        top = torch.topk(v, min(per_doc, v.numel()))
        for val, pos in zip(top.values.tolist(), top.indices.tolist(), strict=True):
            if val > 0 and math.isfinite(val):
                cands.append(Firing(sign * val, d, pos))
    cands.sort(key=lambda f: -sign * f.value)
    return cands[:k]


def firings_above(acts_per_doc: Sequence[torch.Tensor], threshold: float, sign: int = 1) -> list[Firing]:
    """Every firing beyond a threshold on one pole (sign * act > sign * threshold)."""
    out = []
    for d, a in enumerate(acts_per_doc):
        v = pole_values(a, sign)
        for pos in torch.nonzero(v > sign * threshold).flatten().tolist():
            out.append(Firing(float(a[pos]), d, pos))
    return out


def excerpt(
    tokenizer, token_ids: Sequence[int], acts: torch.Tensor, pos: int, before: int = 24, after: int = 8, doc: int = -1
) -> Excerpt:
    """The window [pos - before, pos + after] of one document."""
    lo, hi = max(0, pos - before), min(len(token_ids), pos + after + 1)
    toks = [tokenizer.decode([t]) for t in token_ids[lo:hi]]
    return Excerpt(toks, acts[lo:hi].tolist(), doc=doc, start=lo, focus=pos - lo)


def random_windows(
    tokenizer,
    tokens: Sequence[Sequence[int]],
    acts_per_doc: Sequence[torch.Tensor],
    n: int,
    width: int = 32,
    seed: int = 0,
) -> list[Excerpt]:
    """`n` random windows (uniform document, then uniform start), for negatives and baselines."""
    rng = random.Random(seed)
    out = []
    docs = [d for d in range(len(tokens)) if len(tokens[d]) >= width]
    for _ in range(n):
        d = rng.choice(docs)
        s = rng.randrange(0, len(tokens[d]) - width + 1)
        toks = [tokenizer.decode([t]) for t in tokens[d][s : s + width]]
        out.append(Excerpt(toks, acts_per_doc[d][s : s + width].tolist(), doc=d, start=s))
    return out


def top_examples(
    model: nn.Module,
    tokenizer,
    texts: Sequence[str],
    neuron: Neuron,
    k: int = 20,
    sign: int = 1,
    per_doc: int = 3,
    max_len: int = 256,
    batch_size: int = 8,
    before: int = 24,
    after: int = 8,
) -> list[Excerpt]:
    """Run the corpus, then return the k strongest firings of one pole of `neuron` as excerpts."""
    acts, toks = run_capture(model, batches(tokenizer, texts, max_len, batch_size), [neuron])
    fir = top_firings(acts[neuron], k, sign, per_doc)
    return [excerpt(tokenizer, toks[f.doc], acts[neuron][f.doc], f.pos, before, after, doc=f.doc) for f in fir]
