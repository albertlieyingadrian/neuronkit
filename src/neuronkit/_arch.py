"""Find the MLP neurons of a Hugging Face model and expose them through one hook point per layer.

A *site* is one MLP layer. Every intervention in neuronkit goes through a single callback per site:

    fn(acts, call) -> new acts or None

- Dense MLPs (Llama, Qwen2/3, Mistral, OLMo, Gemma, GPT-2): `acts` is the input of the down projection, shape
  (batch, seq, n_neurons). A forward pre-hook on `down_proj` (or `c_proj`) delivers it.
- Fused mixture-of-experts blocks (OLMoE, Qwen3-MoE, Mixtral in transformers >= 5: an `experts` module holding
  `gate_up_proj` [E, 2I, H] and `down_proj` [E, H, I]): the experts' forward is replaced, for the duration of the
  context, by an equivalent one that computes every routed neuron first. `acts` is then (tokens, top_k, I), already
  multiplied by the router weight (the "router-weighted activation"), and `call.expert_index` (tokens, top_k) says
  which expert each slot holds. A neuron's flat index within its layer is `expert * I + unit`.

Unrouted tokens are not firings: they never appear in `acts`.
"""

from __future__ import annotations

import contextlib
import types
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn


@dataclass
class Call:
    """What a site callback is told about one forward pass through one MLP layer."""

    layer: int
    kind: str  # "dense" or "moe"
    batch_shape: tuple[int, int]  # (batch, seq)
    expert_index: torch.Tensor | None = None  # (tokens, top_k), MoE only
    router_weight: torch.Tensor | None = None  # (tokens, top_k), MoE only
    gate: torch.Tensor | None = None  # pre-activations, only with keep_preacts=True
    up: torch.Tensor | None = None


@dataclass
class Site:
    """One MLP layer and how to read its weights."""

    layer: int
    kind: str
    module: nn.Module  # dense: the MLP; moe: the fused experts module
    n_neurons: int  # neurons in the layer (MoE: experts x units)
    n_experts: int = 1
    n_units: int = 0  # MoE: neurons per expert
    top_k: int = 0
    norm: nn.Module | None = None  # the norm in front of the MLP
    attn: nn.Module | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    def gate_rows(self, flat: torch.Tensor | None = None) -> torch.Tensor:
        """Gate weight rows, shape (n, H), for flat neuron indices (all when None)."""
        return self._rows("gate", flat)

    def up_rows(self, flat: torch.Tensor | None = None) -> torch.Tensor:
        return self._rows("up", flat)

    def down_cols(self, flat: torch.Tensor | None = None) -> torch.Tensor:
        """The direction each neuron writes into the residual stream, as rows (n, H)."""
        if self.kind == "dense":
            w = _linear_weight(down_module(self.module)).T  # (N, H)
            return w if flat is None else w[flat]
        d = self.module.down_proj  # (E, H, I)
        if flat is None:
            return d.permute(0, 2, 1).reshape(-1, d.shape[1])
        e, u = flat // self.n_units, flat % self.n_units
        return d[e, :, u]

    def _rows(self, which: str, flat: torch.Tensor | None) -> torch.Tensor:
        if self.kind == "dense":
            m = self.module
            if hasattr(m, "gate_proj"):
                w = _linear_weight(m.gate_proj if which == "gate" else m.up_proj)
            elif which == "up" and hasattr(m, "c_fc"):
                w = _linear_weight(m.c_fc)
            else:
                raise ValueError(f"layer {self.layer} has no {which} projection")
            return w if flat is None else w[flat]
        g = self.module.gate_up_proj  # (E, 2I, H)
        off = 0 if which == "gate" else self.n_units
        if flat is None:
            return g[:, off : off + self.n_units].reshape(-1, g.shape[-1])
        e, u = flat // self.n_units, flat % self.n_units
        return g[e, off + u]

    @property
    def gated(self) -> bool:
        return self.kind == "moe" or hasattr(self.module, "gate_proj")


def _linear_weight(m: nn.Module) -> torch.Tensor:
    """(out, in) weight for nn.Linear, and for GPT-2's Conv1D (stored transposed)."""
    w = m.weight
    return w.T if type(m).__name__ == "Conv1D" else w


def down_module(mlp: nn.Module) -> nn.Module:
    for name in ("down_proj", "c_proj", "fc2", "dense_4h_to_h", "w2"):
        if hasattr(mlp, name):
            return getattr(mlp, name)
    raise ValueError(f"no down projection on {type(mlp).__name__}")


def decoder_layers(model: nn.Module) -> nn.ModuleList:
    for path in (
        "model.layers",
        "model.model.layers",
        "model.language_model.layers",  # multimodal wrappers (e.g. Qwen3.5/3.6 MoE ...ForConditionalGeneration)
        "model.model.language_model.layers",
        "language_model.model.layers",
        "transformer.h",
        "gpt_neox.layers",
        "model.decoder.layers",
    ):
        obj: Any = model
        for part in path.split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                break
        if isinstance(obj, nn.ModuleList):
            return obj
    raise ValueError(f"cannot find the decoder layers of {type(model).__name__}")


def _is_fused_experts(m: Any) -> bool:
    g = getattr(m, "gate_up_proj", None)
    return isinstance(g, torch.Tensor) and g.dim() == 3 and isinstance(getattr(m, "down_proj", None), torch.Tensor)


def sites(model: nn.Module) -> list[Site]:
    """One `Site` per decoder layer's MLP (dense or fused MoE). Cached on the model."""
    cache = model.__dict__.get("_neuronkit_sites")
    if cache is not None:
        return cache
    out = []
    for i, layer in enumerate(decoder_layers(model)):
        mlp = getattr(layer, "mlp", None)
        if mlp is None:
            raise ValueError(f"layer {i} has no .mlp")
        norm = getattr(layer, "post_attention_layernorm", None) or getattr(layer, "ln_2", None)
        attn = getattr(layer, "self_attn", None) or getattr(layer, "attn", None)
        experts = getattr(mlp, "experts", None)
        if experts is not None and _is_fused_experts(experts):
            e, two_i, _ = experts.gate_up_proj.shape
            units = two_i // 2
            router = getattr(mlp, "gate", None)
            top_k = int(getattr(router, "top_k", 0) or getattr(model.config, "num_experts_per_tok", 0))
            out.append(Site(i, "moe", experts, e * units, e, units, top_k, norm, attn))
        elif experts is not None:
            raise NotImplementedError(f"layer {i}: {type(experts).__name__} is not a fused experts module")
        else:
            n = _linear_weight(down_module(mlp)).shape[1]
            out.append(Site(i, "dense", mlp, n, norm=norm, attn=attn))
    model.__dict__["_neuronkit_sites"] = out
    return out


# ---------------------------------------------------------------- the hook point

SiteFn = Callable[[torch.Tensor, Call], "torch.Tensor | None"]


def _moe_forward(site: Site, fn: SiteFn, keep_preacts: bool):
    def forward(self, hidden_states: torch.Tensor, top_k_index: torch.Tensor, top_k_weights: torch.Tensor):
        t, k = top_k_index.shape
        units = site.n_units
        acts = hidden_states.new_zeros(t, k, units)
        gates = hidden_states.new_zeros(t, k, units) if keep_preacts else None
        ups = hidden_states.new_zeros(t, k, units) if keep_preacts else None
        where = {}
        for e in torch.unique(top_k_index).tolist():
            if not 0 <= e < site.n_experts:  # padding sentinel used by expert parallelism
                continue
            tok, pos = torch.where(top_k_index == e)
            where[e] = (tok, pos)
            gate, up = nn.functional.linear(hidden_states[tok], self.gate_up_proj[e]).chunk(2, dim=-1)
            acts[tok, pos] = (self.act_fn(gate) * up).to(acts.dtype)
            if gates is not None and ups is not None:
                gates[tok, pos] = gate.to(acts.dtype)
                ups[tok, pos] = up.to(acts.dtype)
        valid = (top_k_index >= 0) & (top_k_index < site.n_experts)
        weighted = acts * (top_k_weights * valid).to(acts.dtype)[..., None]
        shape = self.__dict__.get("_nk_shape", (1, t))
        new = fn(weighted, Call(site.layer, "moe", shape, top_k_index, top_k_weights, gates, ups))
        if new is not None:
            weighted = new
        out = torch.zeros_like(hidden_states)
        for e, (tok, pos) in where.items():
            out.index_add_(0, tok, nn.functional.linear(weighted[tok, pos], self.down_proj[e]).to(out.dtype))
        return out

    return types.MethodType(forward, site.module)


@contextlib.contextmanager
def hook_sites(
    model: nn.Module, fn: SiteFn, layers: Sequence[int] | None = None, keep_preacts: bool = False
) -> Iterator[None]:
    """Route the neuron activations of every listed MLP layer through `fn` while the context is open."""
    all_sites = sites(model)
    chosen = all_sites if layers is None else [all_sites[i] for i in layers]
    handles: list[Any] = []
    patched: list[tuple[nn.Module, Any]] = []
    try:
        for s in chosen:
            if s.kind == "dense":
                store: dict[str, torch.Tensor] = {}
                if keep_preacts and hasattr(s.module, "gate_proj"):
                    handles.append(s.module.gate_proj.register_forward_hook(_stash(store, "gate")))
                    handles.append(s.module.up_proj.register_forward_hook(_stash(store, "up")))

                def pre(mod, args, kwargs, s=s, store=store):
                    x = args[0]
                    call = Call(s.layer, "dense", (x.shape[0], x.shape[1]), gate=store.get("gate"), up=store.get("up"))
                    new = fn(x, call)
                    return None if new is None else ((new, *args[1:]), kwargs)

                handles.append(down_module(s.module).register_forward_pre_hook(pre, with_kwargs=True))
            else:

                def shape_pre(mod, args, kwargs, s=s):
                    x = args[0] if args else kwargs["hidden_states"]
                    s.module.__dict__["_nk_shape"] = (x.shape[0], x.shape[1])

                handles.append(_parent_of(model, s.module).register_forward_pre_hook(shape_pre, with_kwargs=True))
                patched.append((s.module, s.module.__dict__.get("forward")))
                s.module.forward = _moe_forward(s, fn, keep_preacts)
        yield
    finally:
        for h in handles:
            h.remove()
        for mod, old in patched:
            if old is None:
                mod.__dict__.pop("forward", None)
            else:
                mod.forward = old


def _stash(store: dict, key: str):
    def hook(mod, args, out):
        store[key] = out

    return hook


def _parent_of(model: nn.Module, child: nn.Module) -> nn.Module:
    for m in model.modules():
        for c in m.children():
            if c is child:
                return m
    raise ValueError("module not found in model")


def compose(*fns: SiteFn) -> SiteFn:
    """Apply several site callbacks in order."""

    def fn(acts: torch.Tensor, call: Call) -> torch.Tensor | None:
        changed = False
        for f in fns:
            new = f(acts, call)
            if new is not None:
                acts, changed = new, True
        return acts if changed else None

    return fn


def flat_index(call: Call, units: int) -> torch.Tensor:
    """MoE: the flat neuron index (expert * I + unit) of every slot of `acts`, shape (tokens, top_k, I)."""
    assert call.expert_index is not None
    ar = torch.arange(units, device=call.expert_index.device)
    return call.expert_index.clamp(min=0)[..., None] * units + ar
