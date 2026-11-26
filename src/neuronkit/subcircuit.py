"""The local wiring of one firing: what drove the neuron there, and what its write changes.

For a neuron n at layer L and token t, with the norm in front of the MLP frozen (its scale computed from the real
residual stream x and treated as a constant), the gate and up pre-activations are linear in the residual stream:

    gate = w_gate . (gamma * x / rms(x)),   x = embedding + sum of every earlier write at t

so each earlier component's write splits the gate (and the up) exactly into contributions: the embedding, each
attention head of layers <= L, and each MLP neuron of layers < L (MoE: the routed ones). `completeness` checks that
the parts add up to the real pre-activation. Gate and up contributions are combined to first order into the
neuron's activation a = act(gate) * up (times the router weight in an MoE):

    da = act'(gate) * up * d_gate + act(gate) * d_up

With attention weights (model loaded with attn_implementation="eager"), each head's contribution is further split by
the source token it attended to (exact for that last hop: z_h(t) = sum_s A[h, t, s] v_h(s)), so `by_source` says
which tokens of the prompt drove the firing through attention.

Downstream, the neuron's write a * W_down[:, n] is read, to first order and with frozen norms, by:
- later neurons at the same token (their activation change);
- later attention heads, through their values (how much their output at later tokens moves, weighted by how much
  those tokens attend to t);
- the next-token logits directly (the write through the final norm and the unembedding).

This is direct-path attribution: effects that pass through several components in sequence are not traced.
"""

from __future__ import annotations

import html
from collections.abc import Sequence
from dataclasses import dataclass, field

import torch
from torch import nn

from neuronkit._arch import Call, decoder_layers, hook_sites, sites
from neuronkit.core import Neuron, model_device


@dataclass
class Contribution:
    kind: str  # "embed", "head", "neuron", "bias"
    layer: int
    index: int
    gate: float
    up: float
    act: float  # first-order contribution to the neuron's activation

    @property
    def key(self) -> str:
        if self.kind == "embed":
            return "embed"
        if self.kind == "bias":
            return f"bias{self.layer}"
        return f"{'H' if self.kind == 'head' else 'N'}{self.layer}.{self.index}"


@dataclass
class SourceToken:
    pos: int
    token: str
    act: float  # summed first-order contribution of everything read from this position


@dataclass
class Reader:
    kind: str  # "neuron" or "head"
    layer: int
    index: int
    effect: float  # neuron: first-order activation change; head: size of its output change

    @property
    def key(self) -> str:
        return f"{'H' if self.kind == 'head' else 'N'}{self.layer}.{self.index}"


@dataclass
class Subcircuit:
    neuron: Neuron
    position: int
    token: str
    activation: float
    gate: float
    up: float
    completeness: float  # (sum of gate contributions) / real gate pre-activation; 1.0 when exact
    upstream: list[Contribution]
    by_source: list[SourceToken]
    readers: list[Reader]
    logits_up: list[tuple[str, float]]
    logits_down: list[tuple[str, float]]
    tokens: list[str] = field(default_factory=list)

    def profile(self, k: int = 200) -> dict[str, float]:
        """The k largest upstream contributions to the activation, keyed by component: a vector to cluster on."""
        top = sorted(self.upstream, key=lambda c: -abs(c.act))[:k]
        return {c.key: c.act for c in top}

    def to_text(self, k: int = 8) -> str:
        lines = [
            f"{self.neuron} at token {self.position} ({self.token!r}): activation {self.activation:+.3f} "
            f"(gate {self.gate:+.3f}, up {self.up:+.3f}; parts add up to {self.completeness:.0%} of the gate)",
            "  drivers (contribution to the activation):",
        ]
        for c in sorted(self.upstream, key=lambda c: -abs(c.act))[:k]:
            lines.append(f"    {c.key:<10} {c.act:+.3f}")
        if self.by_source:
            lines.append("  source tokens (through attention, plus the token itself):")
            for s in sorted(self.by_source, key=lambda s: -abs(s.act))[:k]:
                lines.append(f"    [{s.pos:>3}] {s.token!r:<16} {s.act:+.3f}")
        lines.append("  readers:")
        for r in sorted(self.readers, key=lambda r: -abs(r.effect))[:k]:
            lines.append(f"    {r.key:<10} {r.effect:+.3f}")
        lines.append("  next-token logits pushed up:   " + ", ".join(f"{t!r} {v:+.2f}" for t, v in self.logits_up[:k]))
        lines.append(
            "  next-token logits pushed down: " + ", ".join(f"{t!r} {v:+.2f}" for t, v in self.logits_down[:k])
        )
        return "\n".join(lines)

    def to_html(self, k: int = 8) -> str:
        def rows(items):
            return "".join(f"<tr><td>{html.escape(a)}</td><td>{b:+.3f}</td></tr>" for a, b in items)

        src = ""
        if self.tokens and self.by_source:
            peak = max(abs(s.act) for s in self.by_source) or 1.0
            score = {s.pos: s.act for s in self.by_source}
            spans = []
            for i, t in enumerate(self.tokens):
                a = score.get(i, 0.0) / peak
                col = f"rgba(214,96,0,{abs(a):.2f})" if a > 0 else f"rgba(40,100,200,{abs(a):.2f})"
                style = f"background:{col}" + (";outline:2px solid #000" if i == self.position else "")
                spans.append(f'<span style="{style}">{html.escape(t)}</span>')
            src = f"<p style='font-family:monospace;white-space:pre-wrap'>{''.join(spans)}</p>"
        up = sorted(self.upstream, key=lambda c: -abs(c.act))[:k]
        rd = sorted(self.readers, key=lambda r: -abs(r.effect))[:k]
        return (
            f"<div class='subcircuit'><h4>{html.escape(str(self.neuron))} at {self.position} "
            f"({html.escape(repr(self.token))}): {self.activation:+.3f}</h4>{src}"
            f"<table><tr><th>driver</th><th>effect</th></tr>{rows([(c.key, c.act) for c in up])}</table>"
            f"<table><tr><th>reader</th><th>effect</th></tr>{rows([(r.key, r.effect) for r in rd])}</table>"
            f"<table><tr><th>logit up</th><th></th></tr>{rows(self.logits_up[:k])}</table>"
            f"<table><tr><th>logit down</th><th></th></tr>{rows(self.logits_down[:k])}</table></div>"
        )


def _rms_eps(norm: nn.Module) -> float:
    for name in ("variance_epsilon", "eps", "epsilon"):
        v = getattr(norm, name, None)
        if isinstance(v, float):
            return v
    return 1e-6


def _attn_dims(model: nn.Module) -> tuple[int, int, int]:
    cfg = model.config
    n_heads = cfg.num_attention_heads
    n_kv = getattr(cfg, "num_key_value_heads", None) or n_heads
    head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // n_heads
    return n_heads, n_kv, head_dim


def _act_grad(fn, g: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """act(g) and act'(g), elementwise."""
    g = g.detach().float().requires_grad_(True)
    with torch.enable_grad():
        y = fn(g)
        (d,) = torch.autograd.grad(y.sum(), g)
    return y.detach(), d


class Trace:
    """One forward pass of one text with everything subcircuits need cached. Ask it for many subcircuits."""

    def __init__(self, model: nn.Module, input_ids: torch.Tensor | Sequence[int], tokenizer=None):
        self.model = model
        self.tok = tokenizer
        ids = torch.as_tensor(input_ids, dtype=torch.long).reshape(1, -1)
        self.ids = ids[0].tolist()
        self.sites = sites(model)
        layers = decoder_layers(model)
        n = len(layers)
        self.n_heads, self.n_kv, self.head_dim = _attn_dims(model)
        self.resid_mlp: list[torch.Tensor | None] = [None] * n  # input of the norm before the MLP, (S, H)
        self.resid_attn: list[torch.Tensor | None] = [None] * n  # input of the norm before attention
        self.z: list[torch.Tensor | None] = [None] * n  # o_proj input, (S, heads*dh)
        self.v: list[torch.Tensor | None] = [None] * n  # v_proj output, (S, kv*dh)
        self.attn: list[torch.Tensor | None] = [None] * n  # (heads, S, S) when available
        self.acts: list[torch.Tensor | None] = [None] * n  # dense (S, N); MoE (S, k, I) router-weighted
        self.gate: list[torch.Tensor | None] = [None] * n
        self.up: list[torch.Tensor | None] = [None] * n
        self.expert: list[torch.Tensor | None] = [None] * n
        self.router: list[torch.Tensor | None] = [None] * n
        self.embed: torch.Tensor | None = None
        self.final: torch.Tensor | None = None
        handles = []

        def keep(store: list, i: int, which: str):
            def pre(mod, args, kwargs):
                x = args[0] if args else kwargs["hidden_states"]
                store[i] = x[0].detach().float()

            def post(mod, args, out):
                o = out[0] if isinstance(out, tuple) else out
                store[i] = o[0].detach().float()

            return pre if which == "pre" else post

        for i, layer in enumerate(layers):
            s = self.sites[i]
            if s.norm is not None:
                handles.append(s.norm.register_forward_pre_hook(keep(self.resid_mlp, i, "pre"), with_kwargs=True))
            ln1 = getattr(layer, "input_layernorm", None)
            if ln1 is not None:
                handles.append(ln1.register_forward_pre_hook(keep(self.resid_attn, i, "pre"), with_kwargs=True))
            attn = s.attn
            if attn is not None and hasattr(attn, "o_proj"):
                handles.append(attn.o_proj.register_forward_pre_hook(keep(self.z, i, "pre"), with_kwargs=True))
                handles.append(attn.v_proj.register_forward_hook(keep(self.v, i, "post")))

                def attn_post(mod, args, kwargs, out, i=i):
                    if isinstance(out, tuple) and len(out) > 1 and isinstance(out[1], torch.Tensor):
                        self.attn[i] = out[1][0].detach().float()

                handles.append(attn.register_forward_hook(attn_post, with_kwargs=True))
        emb = model.get_input_embeddings()
        handles.append(emb.register_forward_hook(lambda m, a, o: setattr(self, "embed", o[0].detach().float())))
        final_norm = _final_norm(model)
        self.final_norm = final_norm
        handles.append(
            final_norm.register_forward_pre_hook(lambda m, a: setattr(self, "final", a[0][0].detach().float()))
        )

        def site_fn(acts: torch.Tensor, call: Call) -> None:
            i = call.layer
            self.acts[i] = (
                acts.detach().float().reshape(-1, *acts.shape[-1:]) if call.kind == "dense" else acts.detach().float()
            )
            if call.gate is not None:
                self.gate[i] = call.gate.detach().float().reshape(self.acts[i].shape)
                self.up[i] = call.up.detach().float().reshape(self.acts[i].shape)  # type: ignore[union-attr]
            if call.kind == "moe":
                self.expert[i], self.router[i] = call.expert_index, call.router_weight.detach().float()  # type: ignore[union-attr]

        try:
            with hook_sites(model, site_fn, keep_preacts=True), torch.no_grad():
                eager = getattr(model.config, "_attn_implementation", None) == "eager"
                extra = {"output_attentions": True} if eager else {}
                model(input_ids=ids.to(model_device(model)), use_cache=False, **extra)
        finally:
            for h in handles:
                h.remove()
        self.tokens = [self._decode(t) for t in self.ids]

    def _decode(self, t: int) -> str:
        return self.tok.decode([t]) if self.tok is not None else str(t)

    # ---- the pieces ----
    def _norm_vec(self, norm: nn.Module, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """d(w . norm(x)) / dx with the norm's scale frozen: gamma * w / rms(x)."""
        rms = torch.sqrt(x.pow(2).mean() + _rms_eps(norm))
        gamma = norm.weight.detach().float() if getattr(norm, "weight", None) is not None else torch.ones_like(x)
        return gamma * w / rms

    def activation(self, neuron: Neuron, pos: int) -> float:
        s = self.sites[neuron.layer]
        a = self.acts[neuron.layer]
        assert a is not None
        if s.kind == "dense":
            return float(a[pos, neuron.index])
        slot = self._slot(neuron, pos)
        return float("nan") if slot is None else float(a[pos, slot, neuron.index % s.n_units])

    def _slot(self, neuron: Neuron, pos: int) -> int | None:
        s = self.sites[neuron.layer]
        e = neuron.index // s.n_units
        ex = self.expert[neuron.layer]
        assert ex is not None
        hit = torch.nonzero(ex[pos] == e).flatten()
        return int(hit[0]) if len(hit) else None

    def _write_dot(self, layer: int, pos: int, v: torch.Tensor) -> list[tuple[str, int, int, float]]:
        """Each component of `layer` at `pos`, dotted with v: heads (and o_proj bias) then neurons."""
        out = []
        s = self.sites[layer]
        attn = s.attn
        z = self.z[layer]
        if attn is not None and z is not None:
            wo = attn.o_proj.weight.detach().float()  # (H, heads*dh)
            m = (wo.T @ v).reshape(self.n_heads, self.head_dim)
            c = (z[pos].reshape(self.n_heads, self.head_dim) * m).sum(-1)
            out += [("head", layer, h, float(c[h])) for h in range(self.n_heads)]
            if getattr(attn.o_proj, "bias", None) is not None:
                out.append(("bias", layer, 0, float(attn.o_proj.bias.detach().float() @ v)))
        return out

    def _neuron_dots(self, layer: int, pos: int, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(flat neuron ids, write . v) for every neuron of `layer` firing at `pos`."""
        s = self.sites[layer]
        a = self.acts[layer]
        assert a is not None
        if s.kind == "dense":
            w = s.down_cols().detach().float()  # (N, H)
            return torch.arange(s.n_neurons), a[pos] * (w @ v)
        ex = self.expert[layer]
        assert ex is not None
        ids, vals = [], []
        for j, e in enumerate(ex[pos].tolist()):
            if not 0 <= e < s.n_experts:
                continue
            d = s.module.down_proj[e].detach().float()  # (H, I)
            ids.append(torch.arange(e * s.n_units, (e + 1) * s.n_units))
            vals.append(a[pos, j] * (d.T @ v))
        if not ids:
            return torch.zeros(0, dtype=torch.long), torch.zeros(0)
        return torch.cat(ids), torch.cat(vals)

    def subcircuit(self, neuron: Neuron, pos: int | None = None, k_up: int = 200, k_read: int = 50) -> Subcircuit:
        L = neuron.layer
        s = self.sites[L]
        if not s.gated:
            raise NotImplementedError("subcircuit needs a gated (SwiGLU) MLP")
        if pos is None:
            acts = self.acts[L]
            assert acts is not None
            if s.kind == "dense":
                pos = int(acts[:, neuron.index].abs().argmax())
            else:
                vals = torch.tensor(
                    [
                        abs(self.activation(neuron, p)) if self._slot(neuron, p) is not None else -1.0
                        for p in range(len(self.ids))
                    ]
                )
                pos = int(vals.argmax())
        x = self.resid_mlp[L]
        assert x is not None and s.norm is not None
        flat = torch.tensor([neuron.index])
        wg = s.gate_rows(flat)[0].detach().float()
        wu = s.up_rows(flat)[0].detach().float()
        vg = self._norm_vec(s.norm, x[pos], wg)
        vu = self._norm_vec(s.norm, x[pos], wu)
        g_real = float(vg @ x[pos])
        u_real = float(vu @ x[pos])
        act_fn = s.module.act_fn
        g_t = torch.tensor([g_real])
        ag, dg = _act_grad(act_fn, g_t)
        w_r = 1.0
        if s.kind == "moe":
            slot = self._slot(neuron, pos)
            if slot is None:
                raise ValueError(f"{neuron} is not routed at token {pos}")
            router = self.router[L]
            assert router is not None
            w_r = float(router[pos, slot])
        kg, ku = float(dg[0]) * u_real * w_r, float(ag[0]) * w_r

        comps: list[Contribution] = []
        emb = self.embed
        assert emb is not None
        cg, cu = float(vg @ emb[pos]), float(vu @ emb[pos])
        comps.append(Contribution("embed", -1, 0, cg, cu, kg * cg + ku * cu))
        total_gate = cg
        n_layer, n_id, n_g, n_u = [], [], [], []
        for lay in range(L + 1):
            dg_ = self._write_dot(lay, pos, vg)
            du_ = self._write_dot(lay, pos, vu)
            for (kind, ll, idx, a), (_, _, _, b) in zip(dg_, du_, strict=True):
                comps.append(Contribution(kind, ll, idx, a, b, kg * a + ku * b))
                total_gate += a
            if lay < L:
                ids, gvals = self._neuron_dots(lay, pos, vg)
                _, uvals = self._neuron_dots(lay, pos, vu)
                n_layer.append(torch.full_like(ids, lay))
                n_id.append(ids)
                n_g.append(gvals.cpu())
                n_u.append(uvals.cpu())
        if n_id:
            lays, ids, gv, uv = torch.cat(n_layer), torch.cat(n_id), torch.cat(n_g), torch.cat(n_u)
            av = kg * gv + ku * uv
            total_gate += float(gv.sum())
            self._last_neuron_act_sum = float(av.sum())
            for i in torch.topk(av.abs(), min(k_up, len(av))).indices.tolist():
                comps.append(
                    Contribution("neuron", int(lays[i]), int(ids[i]), float(gv[i]), float(uv[i]), float(av[i]))
                )
        else:
            self._last_neuron_act_sum = 0.0
        completeness = total_gate / g_real if abs(g_real) > 1e-9 else float("nan")
        comps.sort(key=lambda c: -abs(c.act))

        by_source = self._by_source(L, pos, vg, vu, kg, ku, comps)
        activation = self.activation(neuron, pos)

        # ---- downstream ----
        d = s.down_cols(flat)[0].detach().float() * activation  # the write (activation already router-weighted)
        readers = self._readers(L, pos, d)[:k_read]
        up, down = self._logit_effect(pos, d)
        return Subcircuit(
            neuron,
            pos,
            self.tokens[pos],
            activation,
            g_real,
            u_real,
            completeness,
            comps[:k_up],
            by_source,
            readers,
            up,
            down,
            tokens=self.tokens,
        )

    def _by_source(self, L, pos, vg, vu, kg, ku, comps) -> list[SourceToken]:
        if any(self.attn[lay] is None or self.v[lay] is None for lay in range(L + 1)):
            return []
        per = torch.zeros(len(self.ids), device=vg.device)
        group = self.n_heads // self.n_kv
        for lay in range(L + 1):
            attn = self.sites[lay].attn
            wo = attn.o_proj.weight.detach().float()  # type: ignore[union-attr]
            mg = (wo.T @ vg).reshape(self.n_heads, self.head_dim)
            mu = (wo.T @ vu).reshape(self.n_heads, self.head_dim)
            v = self.v[lay].reshape(len(self.ids), self.n_kv, self.head_dim)  # type: ignore[union-attr]
            a = self.attn[lay][:, pos, :]  # type: ignore[index]  # (heads, S)
            for h in range(self.n_heads):
                vh = v[:, h // group, :]  # (S, dh)
                per += a[h] * (kg * (vh @ mg[h]) + ku * (vh @ mu[h]))
        local = sum(c.act for c in comps if c.kind in ("embed", "bias")) + self._last_neuron_act_sum
        per[pos] += local
        return [SourceToken(i, self.tokens[i], float(per[i])) for i in range(len(self.ids)) if i <= pos]

    def _readers(self, L: int, pos: int, d: torch.Tensor) -> list[Reader]:
        out: list[Reader] = []
        layers = decoder_layers(self.model)
        group = self.n_heads // self.n_kv
        for m in range(L + 1, len(self.sites)):
            s = self.sites[m]
            # heads of layer m read the write through their values
            ln1 = getattr(layers[m], "input_layernorm", None)
            xa = self.resid_attn[m]
            if ln1 is not None and xa is not None and s.attn is not None and self.attn[m] is not None:
                rms = torch.sqrt(xa[pos].pow(2).mean() + _rms_eps(ln1))
                dn = ln1.weight.detach().float() * d / rms
                dv = (s.attn.v_proj.weight.detach().float() @ dn).reshape(self.n_kv, self.head_dim)
                wo = s.attn.o_proj.weight.detach().float().reshape(-1, self.n_heads, self.head_dim)
                share = self.attn[m][:, pos:, pos].sum(-1)  # type: ignore[index]  # (heads,)
                for h in range(self.n_heads):
                    size = torch.linalg.vector_norm(wo[:, h, :] @ dv[h // group])
                    out.append(Reader("head", m, h, float(size * share[h])))
            # neurons of layer m at the same token
            x = self.resid_mlp[m]
            g, u = self.gate[m], self.up[m]
            if x is None or g is None or u is None or s.norm is None:
                continue
            rms = torch.sqrt(x[pos].pow(2).mean() + _rms_eps(s.norm))
            dn = s.norm.weight.detach().float() * d / rms
            if s.kind == "dense":
                ids = torch.arange(s.n_neurons)
                dg = s.gate_rows().detach().float() @ dn
                du = s.up_rows().detach().float() @ dn
                gg, uu, wr = g[pos], u[pos], torch.ones(s.n_neurons, device=dn.device)
            else:
                ex, router = self.expert[m], self.router[m]
                assert ex is not None and router is not None
                idl, dgl, dul, ggl, uul, wrl = [], [], [], [], [], []
                for j, e in enumerate(ex[pos].tolist()):
                    if not 0 <= e < s.n_experts:
                        continue
                    gu = s.module.gate_up_proj[e].detach().float()
                    idl.append(torch.arange(e * s.n_units, (e + 1) * s.n_units))
                    dgl.append(gu[: s.n_units] @ dn)
                    dul.append(gu[s.n_units :] @ dn)
                    ggl.append(g[pos, j])
                    uul.append(u[pos, j])
                    wrl.append(torch.full((s.n_units,), float(router[pos, j]), device=dn.device))
                if not idl:
                    continue
                ids, dg, du = torch.cat(idl), torch.cat(dgl), torch.cat(dul)
                gg, uu, wr = torch.cat(ggl), torch.cat(uul), torch.cat(wrl)
            ag, dag = _act_grad(s.module.act_fn, gg)
            eff = (dag * uu * dg + ag * du) * wr
            top = torch.topk(eff.abs(), min(64, len(eff))).indices
            out += [Reader("neuron", m, int(ids[i]), float(eff[i])) for i in top.tolist()]
        out.sort(key=lambda r: -abs(r.effect))
        return out

    def _logit_effect(self, pos: int, d: torch.Tensor, k: int = 10):
        head = self.model.get_output_embeddings()
        if head is None or self.final is None:
            return [], []
        norm = self.final_norm
        rms = torch.sqrt(self.final[pos].pow(2).mean() + _rms_eps(norm))
        dn = norm.weight.detach().float() * d / rms
        delta = head.weight.detach().float() @ dn
        up = torch.topk(delta, k)
        down = torch.topk(-delta, k)
        return (
            [(self._decode(i), float(v)) for v, i in zip(up.values.tolist(), up.indices.tolist(), strict=True)],
            [(self._decode(i), -float(v)) for v, i in zip(down.values.tolist(), down.indices.tolist(), strict=True)],
        )

    def logit_effect_on(self, neuron: Neuron, pos: int, token_ids: Sequence[int]) -> list[float]:
        """Direct effect of the neuron's write at `pos` on chosen next-token logits (e.g. the answer letters)."""
        s = self.sites[neuron.layer]
        d = s.down_cols(torch.tensor([neuron.index]))[0].detach().float() * self.activation(neuron, pos)
        head = self.model.get_output_embeddings()
        norm = self.final_norm
        assert self.final is not None
        rms = torch.sqrt(self.final[pos].pow(2).mean() + _rms_eps(norm))
        dn = norm.weight.detach().float() * d / rms
        w = head.weight.detach().float()[torch.tensor(list(token_ids))]
        return (w @ dn).tolist()


def _final_norm(model: nn.Module) -> nn.Module:
    for path in (
        "model.norm",
        "model.model.norm",
        "model.language_model.norm",
        "model.model.language_model.norm",
        "transformer.ln_f",
        "model.final_layernorm",
    ):
        obj = model
        for part in path.split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                break
        if isinstance(obj, nn.Module):
            return obj
    raise ValueError("cannot find the final norm")


def trace(model: nn.Module, tokenizer, text: str | Sequence[int]) -> Trace:
    """Cache one forward pass of `text` (a string or token ids) for subcircuit queries."""
    ids = tokenizer(text, add_special_tokens=False)["input_ids"] if isinstance(text, str) else list(text)
    return Trace(model, ids, tokenizer)


def subcircuit(
    model: nn.Module, neuron: Neuron, example: str | Sequence[int], tokenizer=None, position: int | None = None
) -> Subcircuit:
    """The local wiring of `neuron` on one example, at `position` (default: its strongest firing there).

    >>> sc = subcircuit(model, Neuron(12, 345), "The capital of France is", tokenizer=tok)
    >>> print(sc.to_text())
    """
    if isinstance(example, str):
        if tokenizer is None:
            raise ValueError("pass tokenizer= with a text example")
        ids = tokenizer(example, add_special_tokens=False)["input_ids"]
    else:
        ids = list(example)
    return Trace(model, ids, tokenizer).subcircuit(neuron, position)


__all__ = ["Contribution", "Reader", "SourceToken", "Subcircuit", "Trace", "subcircuit", "trace"]
