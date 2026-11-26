"""Neurons, batches of text, capturing activations and editing them."""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass

import torch
from torch import nn

from neuronkit._arch import Call, compose, flat_index, hook_sites, sites


@dataclass(frozen=True, order=True)
class Neuron:
    """One MLP neuron: its layer and its flat index in that layer (MoE: expert * units_per_expert + unit)."""

    layer: int
    index: int

    def __str__(self) -> str:
        return f"L{self.layer}/N{self.index}"

    @classmethod
    def parse(cls, s: str) -> Neuron:
        a, b = s.replace("L", "").replace("N", "").split("/")
        return cls(int(a), int(b))


@dataclass(frozen=True)
class Pole:
    """The positive (+1) or negative (-1) end of a neuron."""

    neuron: Neuron
    sign: int

    def __str__(self) -> str:
        return f"{self.neuron}{'+' if self.sign > 0 else '-'}"


@dataclass
class Batch:
    input_ids: torch.Tensor  # (B, S)
    attention_mask: torch.Tensor  # (B, S), 1 for real tokens
    doc_ids: list[int]


def batches(
    tokenizer, texts: Sequence[str], max_len: int = 256, batch_size: int = 8, doc_offset: int = 0
) -> Iterator[Batch]:
    """Tokenise texts (no special tokens), cut each to `max_len` tokens, right-pad, and yield batches."""
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    for start in range(0, len(texts), batch_size):
        chunk = texts[start : start + batch_size]
        ids = [tokenizer(t, add_special_tokens=False)["input_ids"][:max_len] or [pad] for t in chunk]
        width = max(len(x) for x in ids)
        input_ids = torch.full((len(ids), width), pad, dtype=torch.long)
        mask = torch.zeros((len(ids), width), dtype=torch.long)
        for i, x in enumerate(ids):
            input_ids[i, : len(x)] = torch.tensor(x)
            mask[i, : len(x)] = 1
        yield Batch(input_ids, mask, list(range(doc_offset + start, doc_offset + start + len(chunk))))


def model_device(model: nn.Module) -> torch.device:
    return next(model.parameters()).device


def forward(model: nn.Module, batch: Batch, **kw):
    dev = model_device(model)
    with torch.no_grad():
        return model(
            input_ids=batch.input_ids.to(dev), attention_mask=batch.attention_mask.to(dev), use_cache=False, **kw
        )


def dense_view(acts: torch.Tensor, call: Call, n_neurons: int, units: int, fill: float = float("nan")) -> torch.Tensor:
    """Site activations as (batch, seq, n_neurons). MoE slots go to their flat index; unrouted neurons read `fill`."""
    if call.kind == "dense":
        return acts
    assert call.expert_index is not None
    t = acts.shape[0]
    out = acts.new_full((t, n_neurons), fill)
    valid = (call.expert_index >= 0) & (call.expert_index * units < n_neurons)  # (T, k)
    flat = flat_index(call, units)  # (T, k, I)
    vals = torch.where(valid[..., None], acts, torch.full_like(acts, fill))
    # write invalid slots first so a real slot sharing their (clamped) index wins
    order = torch.argsort(valid.to(torch.int8), dim=1, stable=True)
    flat = torch.gather(flat, 1, order[..., None].expand_as(flat))
    vals = torch.gather(vals, 1, order[..., None].expand_as(vals))
    out.scatter_(1, flat.reshape(t, -1), vals.reshape(t, -1))
    b, s = call.batch_shape
    return out.reshape(b, s, n_neurons)


@contextlib.contextmanager
def capture(
    model: nn.Module, layers: Sequence[int] | None = None, neurons: Sequence[Neuron] | None = None
) -> Iterator[dict[int, list[torch.Tensor]]]:
    """Record neuron activations during forward passes.

    Yields a dict layer -> list of tensors, one per forward pass, shaped (batch, seq, n): every neuron of the layer,
    or only the listed `neurons` of that layer (in the order given). MoE neurons that are not routed on a token read
    NaN, and MoE values are router-weighted.
    """
    site_list = sites(model)
    want: dict[int, list[int]] = {}
    if neurons is not None:
        for n in neurons:
            want.setdefault(n.layer, []).append(n.index)
        layers = sorted(want)
    layers = list(range(len(site_list))) if layers is None else list(layers)
    store: dict[int, list[torch.Tensor]] = {layer: [] for layer in layers}

    def fn(acts: torch.Tensor, call: Call) -> None:
        s = site_list[call.layer]
        full = dense_view(acts.detach(), call, s.n_neurons, s.n_units)
        if call.layer in want:
            full = full[..., torch.tensor(want[call.layer], device=full.device)]
        store[call.layer].append(full.float().cpu())

    with hook_sites(model, fn, layers):
        yield store


def run_capture(
    model: nn.Module, batch_iter: Iterable[Batch], neurons: Sequence[Neuron]
) -> tuple[dict[Neuron, list[torch.Tensor]], list[list[int]]]:
    """Activations of `neurons` on every document: neuron -> list of 1-D tensors (one per document, real tokens
    only), plus each document's token ids."""
    out: dict[Neuron, list[torch.Tensor]] = {n: [] for n in neurons}
    tokens: list[list[int]] = []
    order: dict[int, list[Neuron]] = {}
    for n in neurons:
        order.setdefault(n.layer, []).append(n)
    for batch in batch_iter:
        with capture(model, neurons=[n for ns in order.values() for n in ns]) as store:
            forward(model, batch)
        lens = batch.attention_mask.sum(1).tolist()
        for i, n_tok in enumerate(lens):
            tokens.append(batch.input_ids[i, :n_tok].tolist())
        for layer, ns in order.items():
            a = store[layer][0]  # (B, S, len(ns))
            for j, n in enumerate(ns):
                for i, n_tok in enumerate(lens):
                    out[n].append(a[i, :n_tok, j].clone())
    return out, tokens


# ---------------------------------------------------------------- edits

Value = float | torch.Tensor | Callable[[torch.Tensor], torch.Tensor]


def _value(v: Value, cur: torch.Tensor, moe: bool = False) -> torch.Tensor:
    """The new activation: a constant, a per-token tensor shaped (batch, seq), or a function of the current one."""
    if callable(v):
        return v(cur)
    if isinstance(v, torch.Tensor):
        v = v.to(cur.device, cur.dtype)
        return v.reshape(-1, 1).expand_as(cur) if moe else v.reshape(cur.shape)
    return torch.full_like(cur, float(v))


def neuron_edit(values: dict[Neuron, Value], positions: torch.Tensor | None = None):
    """A site callback that sets listed neurons to a value, a per-token (batch, seq) tensor of values, or a function
    of their current activation.

    `positions`: None for every token, or a (batch, seq) boolean mask of where to edit. MoE neurons are edited only
    where they are routed (their value is router-weighted, so set it on that scale).
    """
    by_layer: dict[int, list[tuple[int, Value]]] = {}
    for n, v in values.items():
        by_layer.setdefault(n.layer, []).append((n.index, v))

    def fn(acts: torch.Tensor, call: Call) -> torch.Tensor | None:
        if call.layer not in by_layer:
            return None
        acts = acts.clone()
        if call.kind == "dense":
            for idx, v in by_layer[call.layer]:
                cur = acts[..., idx]
                new = _value(v, cur)
                if positions is not None:
                    new = torch.where(positions.to(cur.device).bool(), new, cur)
                acts[..., idx] = new.to(acts.dtype)
            return acts
        units = acts.shape[-1]
        assert call.expert_index is not None
        for idx, v in by_layer[call.layer]:
            e, u = divmod(idx, units)
            slot = call.expert_index == e  # (T, k)
            if positions is not None:
                slot = slot & positions.to(slot.device).bool().reshape(-1, 1)
            cur = acts[..., u]
            new = _value(v, cur, moe=True)
            acts[..., u] = torch.where(slot, new.to(acts.dtype), cur)
        return acts

    return fn


@contextlib.contextmanager
def edit(model: nn.Module, *fns) -> Iterator[None]:
    """Apply site callbacks (`neuron_edit`, `band_fn`, `topk_fn`, or your own) while the context is open."""
    with hook_sites(model, compose(*fns)):
        yield


@contextlib.contextmanager
def ablate(
    model: nn.Module, neurons: Iterable[Neuron], value: float = 0.0, positions: torch.Tensor | None = None
) -> Iterator[None]:
    """Set neurons to `value` (default zero) on every token, or only on the masked positions."""
    with edit(model, neuron_edit({n: value for n in neurons}, positions)):
        yield
