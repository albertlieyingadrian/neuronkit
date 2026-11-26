"""How much does the model lose when most neuron firings are removed? Loss and KL per document, per condition."""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field

import torch
from torch import nn

from neuronkit._arch import hook_sites
from neuronkit.core import Batch, forward


@dataclass
class CurvePoint:
    name: str
    loss: list[float] = field(default_factory=list)  # mean next-token loss per document
    kl: list[float] = field(default_factory=list)  # mean KL(base || condition) per document
    n_tokens: list[int] = field(default_factory=list)

    @property
    def mean_loss(self) -> float:
        return _wmean(self.loss, self.n_tokens)

    @property
    def mean_kl(self) -> float:
        return _wmean(self.kl, self.n_tokens)


def _wmean(x: list[float], w: list[int]) -> float:
    tot = sum(w)
    return sum(a * b for a, b in zip(x, w, strict=True)) / tot if tot else float("nan")


def site_condition(fn) -> Callable[[nn.Module], contextlib.AbstractContextManager[None]]:
    """Wrap a site callback (`band_fn`, `topk_fn`, ...) as a condition for `sparsity_curve`."""
    return lambda model: hook_sites(model, fn)


def _token_stats(logits: torch.Tensor, batch: Batch, base_logp: torch.Tensor | None):
    """Per-document mean loss and KL over real next-token positions."""
    ids = batch.input_ids.to(logits.device)
    mask = batch.attention_mask.to(logits.device)[:, 1:].float()
    logp = torch.log_softmax(logits[:, :-1].float(), -1)
    nll = -logp.gather(-1, ids[:, 1:, None])[..., 0]
    n = mask.sum(1).clamp(min=1)
    loss = (nll * mask).sum(1) / n
    kl = None
    if base_logp is not None:
        kl_tok = (base_logp.exp() * (base_logp - logp)).sum(-1)
        kl = (kl_tok * mask).sum(1) / n
    return loss, kl, logp, mask.sum(1)


def sparsity_curve(
    model: nn.Module,
    batch_iter: Iterable[Batch],
    conditions: Mapping[str, Callable[[nn.Module], contextlib.AbstractContextManager[None]]],
) -> dict[str, CurvePoint]:
    """Run every batch once intact ("base") and once under each condition.

    `conditions` maps a name to a function `model -> context manager`, for example
    `{"top-7.8%": lambda m: neuronkit.keep_topk(m, frac=0.078), "p97/3": lambda m: neuronkit.ablate_band(m, pct)}`.
    Returns name -> CurvePoint with per-document loss and KL(base || condition); "base" has KL 0. The base pass runs
    through neuronkit's hooks with no edit, so KL measures the edit and not a numerical difference between code paths
    (it matters for MoE models in bf16).
    """
    points = {"base": CurvePoint("base")} | {name: CurvePoint(name) for name in conditions}
    for batch in batch_iter:
        with hook_sites(model, lambda a, c: None):  # same code path as the edits (see describe._window_logp)
            out = forward(model, batch)
        loss, _, base_logp, n = _token_stats(out.logits, batch, None)
        points["base"].loss += loss.tolist()
        points["base"].kl += [0.0] * len(loss)
        points["base"].n_tokens += n.long().tolist()
        for name, make in conditions.items():
            with make(model):
                out = forward(model, batch)
            loss, kl, _, n = _token_stats(out.logits, batch, base_logp)
            assert kl is not None
            points[name].loss += loss.tolist()
            points[name].kl += kl.tolist()
            points[name].n_tokens += n.long().tolist()
        del base_logp
    return points
