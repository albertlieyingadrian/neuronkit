"""The output side of a neuron: what it writes, read through a lens.

A SwiGLU neuron adds a * w to the residual stream, where w is its column of the down projection, so each sign of
the neuron is a context-gated steering vector (+w or -w). Following "Returning to the Neuron Basis for
Interpretability" (https://alignment.anthropic.com/2026/neuron-basis-theory/), the two signs are read separately and
a sign counts as used only when it occurs with the gate open (gate pre-activation > 0; SiLU dips slightly below zero
for negative inputs, so a negative activation with a closed gate is not a real negative write).

Lenses (scores over the vocabulary for the direction sign * w):
- "logit": the unembedding with the final norm gain folded in, applied to w directly. Cheap; it reads early and
  middle layers poorly, because later layers transform what a neuron writes.
- "jacobian": the unembedding applied to J_L w, where J_L is the Jacobian of the final residual stream with respect to
  the residual stream after the neuron's layer, at the same token, averaged over reference texts. neuronkit
  estimates J_L w by a finite difference (add eps * w after the layer at the last token, measure the change in the
  final residual there), which needs only two forward passes per text and no stored Jacobian. Unlike the published
  J-lens, token scores are not divided by the norm of each token's J-lens direction, so very frequent tokens can rank
  a little higher. The published J-lens files cover other models; use them when your model is among them.

The detector of the applications post scores a token by the cosine between the residual stream and a neuron's
output vector (`write_cosine`); `concept_neurons` picks neurons zero-shot, with no labels, by how well their readout
matches a list of words.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import nn

from neuronkit._arch import Call, decoder_layers, hook_sites, sites
from neuronkit.core import Neuron, model_device


@dataclass
class Readout:
    neuron: Neuron
    sign: int
    lens: str
    tokens: list[str]
    scores: list[float]

    def __str__(self) -> str:
        s = "+" if self.sign > 0 else "-"
        return f"{self.neuron}{s} [{self.lens}] " + ", ".join(
            f"{t!r} {v:+.2f}" for t, v in zip(self.tokens, self.scores, strict=True)
        )


def output_vector(model: nn.Module, neuron: Neuron) -> torch.Tensor:
    """The neuron's write direction w (its down-projection column), float32, shape (hidden,)."""
    s = sites(model)[neuron.layer]
    return s.down_cols(torch.tensor([neuron.index]))[0].detach().float()


def _final_norm(model: nn.Module) -> nn.Module:
    from neuronkit.subcircuit import _final_norm as fn

    return fn(model)


def _unembed(model: nn.Module) -> torch.Tensor:
    """(vocab, hidden) unembedding with the final norm's gain folded in."""
    head = model.get_output_embeddings().weight.detach().float()
    norm = _final_norm(model)
    g = getattr(norm, "weight", None)
    return head * g.detach().float()[None, :] if g is not None else head


def jacobian_direction(
    model: nn.Module, tokenizer, layer: int, direction: torch.Tensor, texts: Sequence[str], eps: float = 1e-2
) -> torch.Tensor:
    """Average over texts of J_L @ direction at the last token, by finite difference (see the module docstring)."""
    layers = decoder_layers(model)
    norm = _final_norm(model)
    dev = model_device(model)
    d = direction.to(dev)
    scale = eps * (float(d.norm()) or 1.0)
    out = []
    for t in texts:
        ids = torch.tensor([tokenizer(t, add_special_tokens=False)["input_ids"]], device=dev)
        finals = []
        for add in (False, True):
            store: dict[str, torch.Tensor] = {}

            def bump(mod, args, output, add=add):
                if not add:
                    return None
                h = output[0] if isinstance(output, tuple) else output
                h = h.clone()
                h[:, -1] = h[:, -1] + (scale / float(d.norm())) * d.to(h.dtype)
                return (h, *output[1:]) if isinstance(output, tuple) else h

            def grab(mod, args):
                store["h"] = args[0][0, -1].detach().float()

            h1 = layers[layer].register_forward_hook(bump)
            h2 = norm.register_forward_pre_hook(grab)
            try:
                with torch.no_grad():
                    model(input_ids=ids, use_cache=False)
            finally:
                h1.remove()
                h2.remove()
            finals.append(store["h"])
        out.append((finals[1] - finals[0]) * (float(d.norm()) / scale))
    return torch.stack(out).mean(0).cpu()


def readout(
    model: nn.Module,
    tokenizer,
    neuron: Neuron,
    sign: int = 1,
    lens: str = "logit",
    k: int = 10,
    texts: Sequence[str] | None = None,
) -> Readout:
    """The k tokens the neuron's sign * w raises most under the lens ("logit", or "jacobian" with reference texts)."""
    w = sign * output_vector(model, neuron)
    if lens == "logit":
        v = w
    elif lens == "jacobian":
        if not texts:
            raise ValueError("the jacobian lens needs reference texts")
        v = jacobian_direction(model, tokenizer, neuron.layer, w, texts)
    else:
        raise ValueError("lens is 'logit' or 'jacobian'")
    scores = _unembed(model).to(v.device) @ v.float()
    top = torch.topk(scores, k)
    toks = [tokenizer.decode([i]) for i in top.indices.tolist()]
    return Readout(neuron, sign, lens, toks, [float(x) for x in top.values])


def concept_neurons(
    model: nn.Module,
    tokenizer,
    layers: Sequence[int],
    words: Sequence[str],
    k: int = 20,
    lens_vectors: dict[int, torch.Tensor] | None = None,
) -> list[tuple[Neuron, int, float]]:
    """Zero-shot: the k signed neurons (over `layers`) whose logit-lens readout best matches `words`.

    Score = cosine between the neuron's lens vector over the vocabulary (W_U' (sign * w)) and an indicator over the
    words' first tokens, centred. No labels are used. `lens_vectors` may pass a precomputed (n_neurons, hidden)
    matrix per layer (for example Jacobian-lens directions) instead of the raw output vectors.
    """
    U = _unembed(model)
    ids = sorted({tokenizer(w, add_special_tokens=False)["input_ids"][0] for w in words})
    target = torch.zeros(U.shape[0])
    target[ids] = 1.0
    target -= target.mean()
    target /= target.norm()
    # cos(U v, t) = (U^T t) . v / ||U v||; compute in two products to avoid a vocab x neurons matrix
    ut = U.T @ target  # (hidden,)
    out: list[tuple[Neuron, int, float]] = []
    for layer in layers:
        W = lens_vectors[layer] if lens_vectors is not None else sites(model)[layer].down_cols().detach().float()
        num = W @ ut
        UtU = U.T @ U  # (hidden, hidden)
        den = torch.sqrt(torch.einsum("nh,hk,nk->n", W, UtU, W)).clamp(min=1e-12)
        cos = num / den
        for sign in (1, -1):
            sc = sign * cos
            top = torch.topk(sc, min(k, len(sc)))
            out += [(Neuron(layer, int(i)), sign, float(v)) for v, i in zip(top.values, top.indices, strict=True)]
    out.sort(key=lambda x: -x[2])
    return out[:k]


def write_cosine(resid: torch.Tensor, model: nn.Module, picks: Sequence[tuple[Neuron, int]]) -> torch.Tensor:
    """The zero-shot detector score: mean over picked signed neurons of cos(residual, sign * w).

    `resid` is (rows, hidden), read after the neurons' layer (all picks should share it)."""
    ws = torch.stack([s * output_vector(model, n) for n, s in picks])
    ws = ws / ws.norm(dim=1, keepdim=True)
    r = resid.float() / resid.float().norm(dim=1, keepdim=True).clamp(min=1e-12)
    return (r @ ws.T).mean(1)


def gate_open(model: nn.Module, input_ids: torch.Tensor, neurons: Sequence[Neuron]) -> dict[Neuron, torch.Tensor]:
    """Whether each neuron's gate is open (gate pre-activation > 0) at each token, shape (batch, seq).

    For an MoE neuron, False where it is not routed. Use it with the sign of the activation to count a sign as used
    only with the gate open."""
    want: dict[int, list[Neuron]] = {}
    for n in neurons:
        want.setdefault(n.layer, []).append(n)
    out: dict[Neuron, torch.Tensor] = {}

    def fn(acts: torch.Tensor, call: Call) -> None:
        if call.layer not in want or call.gate is None:
            return
        s = sites(model)[call.layer]
        for n in want[call.layer]:
            if call.kind == "dense":
                out[n] = (call.gate[..., n.index] > 0).cpu()
            else:
                e, u = divmod(n.index, s.n_units)
                assert call.expert_index is not None
                slot = call.expert_index == e
                g = (call.gate[..., u] > 0) & slot
                out[n] = g.any(-1).reshape(call.batch_shape).cpu()

    with hook_sites(model, fn, layers=sorted(want), keep_preacts=True), torch.no_grad():
        model(input_ids=input_ids.to(model_device(model)), use_cache=False)
    return out
