"""Use neuronkit's percentiles and band edits from TransformerLens or nnsight.

neuronkit's own hooks work on any Hugging Face model. If you already work in TransformerLens or nnsight, these helpers
apply the same per-neuron band ablation there. Neither library is imported until you call them.

TransformerLens: the neuron is `blocks.{l}.mlp.hook_post` (shape batch, pos, d_mlp).

    pct = tl_percentiles(model, tokens)
    logits = model.run_with_hooks(tokens, fwd_hooks=tl_band_hooks(pct, keep=">p97|<p3"))

nnsight: the neuron is the input of `model.model.layers[l].mlp.down_proj`.

    with model.trace(prompt):
        nnsight_ablate_band(model, pct, keep=">p97|<p3")
        logits = model.lm_head.output.save()
"""

from __future__ import annotations

from collections.abc import Callable

import torch

from neuronkit.percentiles import DEFAULT_QS, Percentiles, parse_keep, quantiles


def band_tensor(
    acts: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor, mode: str = "zero", mean: torch.Tensor | None = None
) -> torch.Tensor:
    """Replace values strictly between lo and hi (per neuron, the last dimension) by zero or by `mean`."""
    lo = lo.to(acts.device, acts.dtype)
    hi = hi.to(acts.device, acts.dtype)
    inside = (acts > lo) & (acts < hi)
    if mode == "zero":
        repl = torch.zeros_like(acts)
    else:
        assert mean is not None
        repl = torch.nan_to_num(mean.to(acts.device, acts.dtype)).expand_as(acts)
    return torch.where(inside, repl, acts)


def tl_band_hooks(pct: Percentiles, keep: str = ">p97|<p3", mode: str = "zero") -> list[tuple[str, Callable]]:
    """TransformerLens fwd_hooks that keep only each neuron's tails, at every layer in `pct`."""
    hi_q, lo_q = parse_keep(keep)
    hooks = []
    for layer in sorted(pct.values):
        lo, hi, mean = pct.at(layer, lo_q), pct.at(layer, hi_q), pct.mean[layer]

        def hook(acts, hook=None, lo=lo, hi=hi, mean=mean):
            return band_tensor(acts, lo, hi, mode, mean)

        hooks.append((f"blocks.{layer}.mlp.hook_post", hook))
    return hooks


def tl_percentiles(model, tokens: torch.Tensor, qs=DEFAULT_QS, batch_size: int = 8) -> Percentiles:
    """Per-neuron percentiles from a HookedTransformer over a (n, seq) token tensor (every position counts)."""
    n_layers = model.cfg.n_layers
    names = {f"blocks.{lay}.mlp.hook_post": lay for lay in range(n_layers)}
    vals: dict[int, list[torch.Tensor]] = {lay: [] for lay in range(n_layers)}
    for s in range(0, tokens.shape[0], batch_size):
        _, cache = model.run_with_cache(tokens[s : s + batch_size], names_filter=lambda n: n in names)
        for name, lay in names.items():
            vals[lay].append(cache[name].reshape(-1, cache[name].shape[-1]).float().cpu())
    out = Percentiles(qs=list(qs))
    for lay, parts in vals.items():
        x = torch.cat(parts)
        out.values[lay] = quantiles(x, qs)
        out.mean[lay], out.min[lay], out.max[lay] = x.mean(0), x.min(0).values, x.max(0).values
        out.count[lay] = torch.full((x.shape[1],), x.shape[0])
    out.n_tokens = int(tokens.numel())
    return out


def nnsight_ablate_band(model, pct: Percentiles, keep: str = ">p97|<p3", mode: str = "zero") -> None:
    """Inside an nnsight `model.trace(...)` context: band-ablate the input of every down projection in `pct`."""
    hi_q, lo_q = parse_keep(keep)
    layers = model.model.layers
    for layer in sorted(pct.values):
        down = layers[layer].mlp.down_proj
        down.input = band_tensor(down.input, pct.at(layer, lo_q), pct.at(layer, hi_q), mode, pct.mean[layer])
