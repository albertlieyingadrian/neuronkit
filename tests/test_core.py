from __future__ import annotations

import math

import numpy as np
import pytest
import torch

import neuronkit as nk
from neuronkit._arch import compose, hook_sites, sites


def _ids(tok, text):
    return torch.tensor([tok(text)["input_ids"]])


def test_sites_found(model):
    s = sites(model)
    assert len(s) == 3
    if s[0].kind == "moe":
        assert s[0].n_experts == 6 and s[0].n_units == 16 and s[0].n_neurons == 96 and s[0].top_k == 2
    else:
        assert s[0].n_neurons == 48


def test_identity_hook_keeps_logits(model, tok, texts):
    ids = _ids(tok, texts[0])
    with torch.no_grad():
        ref = model(ids).logits
        with hook_sites(model, lambda a, c: a.clone()):
            got = model(ids).logits
    assert torch.allclose(ref, got, atol=1e-5)
    with torch.no_grad():  # the original forward is restored afterwards
        assert torch.allclose(model(ids).logits, ref)


def test_capture_shapes_and_routing(model, tok, texts):
    ids = _ids(tok, texts[1])
    with nk.capture(model) as store, torch.no_grad():
        model(ids)
    a = store[0][0]
    s = sites(model)[0]
    assert a.shape == (1, ids.shape[1], s.n_neurons)
    if s.kind == "moe":
        routed = (~torch.isnan(a)).sum(-1)
        assert torch.all(routed == s.top_k * s.n_units)
    else:
        assert not torch.isnan(a).any()


def test_capture_selected_neurons(model, tok, texts):
    ns = [nk.Neuron(1, 3), nk.Neuron(1, 7), nk.Neuron(2, 0)]
    acts, toks = nk.run_capture(model, nk.batches(tok, texts[:3], batch_size=2), ns)
    assert len(toks) == 3 and all(len(acts[n]) == 3 for n in ns)
    assert acts[ns[0]][0].shape == (len(texts[0]),)


def test_percentiles_match_numpy(dense_model, tok, texts):
    pct = nk.percentiles(dense_model, lambda: nk.batches(tok, texts, batch_size=3), qs=(3, 50, 97), layers_per_pass=2)
    with nk.capture(dense_model, layers=[1]) as store:
        for b in nk.batches(tok, texts, batch_size=3):
            nk.core.forward(dense_model, b)
    vals = []
    for b, a in zip(nk.batches(tok, texts, batch_size=3), store[1], strict=True):
        vals.append(a.reshape(-1, a.shape[-1])[b.attention_mask.reshape(-1).bool()])
    x = torch.cat(vals).numpy()
    expect = np.percentile(x, [3, 50, 97], axis=0)
    assert np.allclose(pct.values[1].numpy(), expect, atol=1e-5)
    assert pct.n_tokens == sum(len(t) for t in texts)


def test_moe_percentiles_count_only_routed(moe_model, tok, texts):
    pct = nk.percentiles(moe_model, lambda: nk.batches(tok, texts, batch_size=4), qs=(3, 97))
    s = sites(moe_model)[0]
    total = sum(len(t) for t in texts)
    assert int(pct.count[0].sum()) == total * s.top_k * s.n_units


def _after(model, ids, fn):
    """Neuron activations as the down projection sees them, after the edit `fn`."""
    seen = {}

    def grab(a, call):
        s = sites(model)[call.layer]
        seen.setdefault(call.layer, nk.core.dense_view(a.detach(), call, s.n_neurons, s.n_units))

    with hook_sites(model, compose(fn, grab)), torch.no_grad():
        model(ids)
    return seen


def test_band_ablation_keeps_only_tails(model, tok, texts):
    pct = nk.percentiles(model, lambda: nk.batches(tok, texts, batch_size=4), qs=(3, 97))
    ids = _ids(tok, texts[2])
    seen = _after(model, ids, nk.band_fn(pct, ">p97|<p3"))
    for layer, a in seen.items():
        hi, lo = pct.at(layer, 97), pct.at(layer, 3)
        a = a[0]
        nz = (a != 0) & ~torch.isnan(a)
        ok = (a >= hi) | (a <= lo)
        assert torch.all(ok[nz])
        assert nz.float().mean() < 0.5


def test_parse_keep():
    assert nk.parse_keep(">p97|<p3") == (97.0, 3.0)
    with pytest.raises(ValueError):
        nk.parse_keep("p97")
    with pytest.raises(ValueError):
        nk.parse_keep(">p3|<p97")


@pytest.mark.parametrize("mode", ["top", "bottom", "random"])
def test_topk_counts(model, tok, texts, mode):
    ids = _ids(tok, texts[3])
    seen = _after(model, ids, nk.topk_fn(k=5, mode=mode))
    s = sites(model)[0]
    present = s.top_k * s.n_units if s.kind == "moe" else s.n_neurons
    nz = ((seen[0] != 0) & ~torch.isnan(seen[0])).sum(-1)
    expect = 5 if mode != "bottom" else present - 5
    assert torch.all(nz <= expect) and torch.all(nz >= expect - 1)


def test_topk_keeps_largest(dense_model, tok, texts):
    ids = _ids(tok, texts[3])
    raw = _after(dense_model, ids, lambda a, c: None)[0][0]
    kept = _after(dense_model, ids, nk.topk_fn(k=4))[0][0]
    for t in range(raw.shape[0]):
        top = set(raw[t].abs().topk(4).indices.tolist())
        assert set(torch.nonzero(kept[t]).flatten().tolist()) <= top


def test_ablate_changes_output_and_restores(model, tok, texts):
    ids = _ids(tok, texts[4])
    with torch.no_grad():
        ref = model(ids).logits
        s = sites(model)
        ns = [nk.Neuron(1, i) for i in range(s[1].n_neurons)]
        with nk.ablate(model, ns):
            abl = model(ids).logits
        again = model(ids).logits
    assert not torch.allclose(ref, abl)
    assert torch.allclose(ref, again)


def test_sparsity_curve(model, tok, texts):
    pts = nk.sparsity_curve(
        model,
        nk.batches(tok, texts, batch_size=4),
        {"all": lambda m: nk.keep_topk(m, frac=1.0), "top10": lambda m: nk.keep_topk(m, frac=0.1)},
    )
    assert math.isclose(pts["all"].mean_kl, 0.0, abs_tol=1e-6)
    assert pts["top10"].mean_kl > 0
    assert len(pts["base"].loss) == len(texts)


def test_top_examples(dense_model, tok, texts):
    ex = nk.top_examples(dense_model, tok, texts, nk.Neuron(1, 2), k=5, sign=1)
    assert len(ex) == 5
    peaks = [e.acts[e.focus] for e in ex]
    assert peaks == sorted(peaks, reverse=True)
    neg = nk.top_examples(dense_model, tok, texts, nk.Neuron(1, 2), k=3, sign=-1)
    assert all(e.acts[e.focus] < 0 for e in neg)
