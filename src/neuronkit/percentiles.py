"""Per-neuron activation percentiles over a corpus, and the sparsity edits built on them.

Each neuron gets its own distribution: every firing on the corpus (MoE: only tokens routed to its expert, values
router-weighted). Percentiles are exact over what was collected. Memory is tokens x neurons per layer, so collect a
few layers per pass with `layers_per_pass` (each pass re-reads the corpus).
"""

from __future__ import annotations

import contextlib
import math
import re
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import nn

from neuronkit._arch import Call, flat_index, hook_sites, sites
from neuronkit.core import Batch, forward

DEFAULT_QS = (0.1, 1, 3, 5, 10, 25, 50, 75, 90, 95, 97, 99, 99.9)


@dataclass
class Percentiles:
    """`values[layer]` is (n_q, n_neurons) for the percentiles `qs` (in %). NaN for a neuron that never fired."""

    qs: list[float]
    values: dict[int, torch.Tensor] = field(default_factory=dict)
    mean: dict[int, torch.Tensor] = field(default_factory=dict)
    min: dict[int, torch.Tensor] = field(default_factory=dict)
    max: dict[int, torch.Tensor] = field(default_factory=dict)
    count: dict[int, torch.Tensor] = field(default_factory=dict)
    n_tokens: int = 0

    def at(self, layer: int, q: float) -> torch.Tensor:
        """The q-th percentile of every neuron of `layer`, shape (n_neurons,)."""
        for i, qq in enumerate(self.qs):
            if math.isclose(qq, q):
                return self.values[layer][i]
        raise KeyError(f"percentile {q} was not computed (have {self.qs})")

    def save(self, path: str | Path) -> None:
        torch.save(
            {
                "qs": self.qs,
                "values": self.values,
                "mean": self.mean,
                "min": self.min,
                "max": self.max,
                "count": self.count,
                "n_tokens": self.n_tokens,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> Percentiles:
        return cls(**torch.load(path, weights_only=False))


def quantiles(x: torch.Tensor, qs: Sequence[float]) -> torch.Tensor:
    """Linear-interpolation quantiles along dim 0 (numpy's default method), for any number of rows."""
    n = x.shape[0]
    if n == 0:
        return x.new_full((len(qs), *x.shape[1:]), float("nan"))
    s, _ = torch.sort(x, dim=0)
    out = []
    for q in qs:
        pos = (q / 100) * (n - 1)
        lo, hi = math.floor(pos), math.ceil(pos)
        w = pos - lo
        out.append(s[lo] * (1 - w) + s[hi] * w)
    return torch.stack(out)


def percentiles(
    model: nn.Module,
    batch_source: Callable[[], Iterable[Batch]],
    qs: Sequence[float] = DEFAULT_QS,
    layers: Sequence[int] | None = None,
    layers_per_pass: int | None = None,
    store_device: str | torch.device = "cpu",
    store_dtype: torch.dtype | None = None,
    sort_device: str | torch.device | None = None,
    chunk: int = 2048,
) -> Percentiles:
    """Per-neuron percentiles of every firing on the corpus.

    `batch_source` returns a fresh iterable of `Batch`es (it is called once per pass), for example
    `lambda: neuronkit.batches(tok, texts)`. Padding tokens are ignored. `store_dtype=torch.float16` halves the memory
    (values are rounded to 3 significant digits); `sort_device="cuda"` sorts on the GPU.
    """
    site_list = sites(model)
    layers = list(range(len(site_list))) if layers is None else list(layers)
    per = layers_per_pass or len(layers)
    result = Percentiles(qs=list(qs))
    for g in range(0, len(layers), per):
        group = layers[g : g + per]
        dense_vals: dict[int, list[torch.Tensor]] = {lay: [] for lay in group}
        moe_vals: dict[tuple[int, int], list[torch.Tensor]] = {}
        mask_holder: dict[str, torch.Tensor] = {}

        def fn(acts: torch.Tensor, call: Call) -> None:
            valid_tok = mask_holder["mask"].reshape(-1).to(acts.device).bool()
            if call.kind == "dense":
                dense_vals[call.layer].append(
                    acts.reshape(-1, acts.shape[-1])[valid_tok].detach().to(store_device, store_dtype or acts.dtype)
                )
                return
            assert call.expert_index is not None
            site = site_list[call.layer]
            idx = call.expert_index
            for e in torch.unique(idx).tolist():
                if not 0 <= e < site.n_experts:
                    continue
                tok, pos = torch.where((idx == e) & valid_tok[:, None])
                if len(tok):
                    moe_vals.setdefault((call.layer, e), []).append(
                        acts[tok, pos].detach().to(store_device, store_dtype or acts.dtype)
                    )

        n_tok = 0
        with hook_sites(model, fn, group):
            for batch in batch_source():
                mask_holder["mask"] = batch.attention_mask
                forward(model, batch)
                n_tok += int(batch.attention_mask.sum())
        result.n_tokens = n_tok
        for lay in group:
            site = site_list[lay]
            vals = torch.full((len(qs), site.n_neurons), float("nan"))
            mean, mn, mx = (torch.full((site.n_neurons,), float("nan")) for _ in range(3))
            cnt = torch.zeros(site.n_neurons, dtype=torch.long)
            if site.kind == "dense":
                pieces = [(torch.arange(site.n_neurons), torch.cat(dense_vals.pop(lay)))]
            else:
                pieces = []
                for e in range(site.n_experts):
                    got = moe_vals.pop((lay, e), None)
                    if got:
                        pieces.append((torch.arange(e * site.n_units, (e + 1) * site.n_units), torch.cat(got)))
            for cols, x in pieces:
                for c0 in range(0, x.shape[1], chunk):
                    xc = x[:, c0 : c0 + chunk].to(sort_device or x.device).float()
                    cc = cols[c0 : c0 + chunk]
                    vals[:, cc] = quantiles(xc, qs).cpu()
                    mean[cc], mn[cc], mx[cc] = xc.mean(0).cpu(), xc.min(0).values.cpu(), xc.max(0).values.cpu()
                    cnt[cc] = xc.shape[0]
            result.values[lay], result.mean[lay], result.min[lay] = vals, mean, mn
            result.max[lay], result.count[lay] = mx, cnt
    return result


# ---------------------------------------------------------------- band edits

_KEEP = re.compile(r"^\s*>\s*p([\d.]+)\s*\|\s*<\s*p([\d.]+)\s*$")


def parse_keep(keep: str) -> tuple[float, float]:
    """ ">p97|<p3" -> (97, 3): keep firings above p97 or below p3."""
    m = _KEEP.match(keep)
    if not m:
        raise ValueError(f"keep must look like '>p97|<p3', got {keep!r}")
    hi, lo = float(m.group(1)), float(m.group(2))
    if not lo < hi:
        raise ValueError(f"the upper percentile must be above the lower one: {keep!r}")
    return hi, lo


def _per_slot(t: torch.Tensor, call: Call, acts: torch.Tensor, units: int) -> torch.Tensor:
    """Per-neuron values (n_neurons,) aligned to acts: dense (..., N), MoE (T, k, I)."""
    t = t.to(acts.device, acts.dtype)
    return t if call.kind == "dense" else t[flat_index(call, units)]


def band_fn(pct: Percentiles, keep: str = ">p97|<p3", mode: str = "zero"):
    """Site callback: firings strictly between the two percentiles are replaced (by zero, or by the neuron's mean);
    the tails are kept."""
    hi_q, lo_q = parse_keep(keep)
    if mode not in ("zero", "mean"):
        raise ValueError("mode is 'zero' or 'mean'")

    def fn(acts: torch.Tensor, call: Call) -> torch.Tensor | None:
        if call.layer not in pct.values:
            return None
        units = acts.shape[-1]
        hi = _per_slot(pct.at(call.layer, hi_q), call, acts, units)
        lo = _per_slot(pct.at(call.layer, lo_q), call, acts, units)
        inside = (acts > lo) & (acts < hi)  # NaN thresholds compare False: never-fired neurons are kept
        if call.kind == "moe":
            assert call.expert_index is not None
            inside = inside & (call.expert_index >= 0)[..., None]
        if mode == "zero":
            repl = torch.zeros_like(acts)
        else:
            repl = torch.nan_to_num(_per_slot(pct.mean[call.layer], call, acts, units)).expand_as(acts)
        return torch.where(inside, repl, acts)

    return fn


def topk_fn(frac: float | None = None, k: int | None = None, mode: str = "top", seed: int = 0):
    """Site callback for per-token layer top-K: keep the K firings with the largest |activation| on each token
    (MoE: among the routed neurons, router-weighted) and zero the rest.

    mode="top" keeps the strongest K; "bottom" zeroes the strongest K and keeps the rest; "random" keeps K chosen
    uniformly. Give `frac` (of the firings present on a token) or `k`.
    """
    if (frac is None) == (k is None):
        raise ValueError("give exactly one of frac and k")
    if mode not in ("top", "bottom", "random"):
        raise ValueError("mode is top, bottom or random")
    gen: dict[str, torch.Generator] = {}

    def fn(acts: torch.Tensor, call: Call) -> torch.Tensor:
        shape = acts.shape
        flat = acts.reshape(-1, shape[-1]) if call.kind == "dense" else acts.reshape(shape[0], -1)
        n = flat.shape[1]
        kk = k if k is not None else max(1, round(frac * n))  # type: ignore[operator]
        kk = min(kk, n)
        if mode == "random":
            g = gen.setdefault(str(flat.device), torch.Generator(device=flat.device).manual_seed(seed))
            score = torch.rand(flat.shape, device=flat.device, generator=g)
        else:
            score = flat.abs().float()
        idx = score.topk(kk, dim=1).indices
        keep = torch.zeros_like(flat, dtype=torch.bool).scatter_(1, idx, True)
        if mode == "bottom":
            keep = ~keep
        return torch.where(keep, flat, torch.zeros_like(flat)).reshape(shape)

    return fn


@contextlib.contextmanager
def ablate_band(model: nn.Module, pct: Percentiles, keep: str = ">p97|<p3", mode: str = "zero") -> Iterator[None]:
    """Zero (or mean-replace) every firing inside each neuron's central band; keep only the tails.

    >>> with ablate_band(model, pct, keep=">p97|<p3"):
    ...     out = model(**inputs)
    """
    with hook_sites(model, band_fn(pct, keep, mode), layers=sorted(pct.values)):
        yield


@contextlib.contextmanager
def keep_topk(
    model: nn.Module, frac: float | None = None, k: int | None = None, mode: str = "top", seed: int = 0
) -> Iterator[None]:
    """Per-token layer top-K sparsity on every MLP layer (see `topk_fn`)."""
    with hook_sites(model, topk_fn(frac, k, mode, seed)):
        yield
