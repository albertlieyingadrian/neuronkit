from __future__ import annotations

import math

import torch

import neuronkit as nk
from neuronkit._arch import sites


def _strong(model, tok, text, layer):
    tr = nk.trace(model, tok, text)
    s = sites(model)[layer]
    a = tr.acts[layer]
    if s.kind == "dense":
        pos, idx = divmod(int(a.abs().argmax()), a.shape[1])
        return tr, nk.Neuron(layer, idx), pos
    t, slot, unit = (int(x) for x in torch.unravel_index(a.abs().argmax(), a.shape))
    e = int(tr.expert[layer][t, slot])
    return tr, nk.Neuron(layer, e * s.n_units + unit), t


def test_parts_add_up(model, tok, texts):
    for layer in (1, 2):
        tr, n, pos = _strong(model, tok, texts[3], layer)
        sc = tr.subcircuit(n, pos, k_up=10_000)
        assert math.isclose(sc.completeness, 1.0, rel_tol=1e-3), sc.completeness
        assert math.isclose(sum(c.up for c in sc.upstream), sc.up, rel_tol=1e-3, abs_tol=1e-4)
        s = sites(model)[layer]
        a = float(s.module.act_fn(torch.tensor(sc.gate))) * sc.up
        if s.kind == "moe":
            a *= float(tr.router[layer][pos, tr._slot(n, pos)])
        assert math.isclose(a, sc.activation, rel_tol=1e-3, abs_tol=1e-5)


def test_by_source_sums_to_upstream(model, tok, texts):
    tr, n, pos = _strong(model, tok, texts[0], 2)
    sc = tr.subcircuit(n, pos, k_up=10_000)
    assert sc.by_source, "eager attention gives per-source attribution"
    total_src = sum(s.act for s in sc.by_source)
    total_up = sum(c.act for c in sc.upstream)
    assert math.isclose(total_src, total_up, rel_tol=1e-3, abs_tol=1e-4)
    assert all(s.pos <= pos for s in sc.by_source)


def test_first_order_matches_ablation_direction(dense_model, tok, texts):
    """Zeroing the top upstream neuron at that token moves the target the way its contribution says."""
    tr, n, pos = _strong(dense_model, tok, texts[5], 2)
    sc = tr.subcircuit(n, pos)
    top = next(c for c in sc.upstream if c.kind == "neuron")
    ids = torch.tensor([tr.ids])
    mask = torch.zeros_like(ids, dtype=torch.bool)
    mask[0, pos] = True
    with (
        nk.capture(dense_model, neurons=[n]) as st,
        nk.ablate(dense_model, [nk.Neuron(top.layer, top.index)], positions=mask),
        torch.no_grad(),
    ):
        dense_model(ids)
    after = float(st[n.layer][0][0, pos, 0])
    assert (after - sc.activation) * top.act <= 1e-6


def test_readers_and_logits(model, tok, texts):
    tr, n, pos = _strong(model, tok, texts[1], 0)
    sc = tr.subcircuit(n, pos)
    assert sc.readers and all(r.layer > 0 for r in sc.readers)
    assert len(sc.logits_up) == 10 and sc.logits_up[0][1] >= sc.logits_up[-1][1]
    text = sc.to_text()
    assert str(n) in text and "readers" in text
    assert "<table>" in sc.to_html()
    assert 0 < len(sc.profile(k=20)) <= 20


def test_subcircuit_function_default_position(dense_model, tok, texts):
    sc = nk.subcircuit(dense_model, nk.Neuron(1, 5), texts[2], tokenizer=tok)
    tr = nk.trace(dense_model, tok, texts[2])
    assert sc.position == int(tr.acts[1][:, 5].abs().argmax())
