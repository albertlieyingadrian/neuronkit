from __future__ import annotations

import torch

import neuronkit as nk
from neuronkit._arch import sites
from neuronkit.readout import concept_neurons, gate_open, output_vector, readout, write_cosine


def test_logit_readout_finds_planted_token(dense_model, tok):
    s = sites(dense_model)[1]
    n = nk.Neuron(1, 4)
    head = dense_model.get_output_embeddings().weight
    gain = dense_model.model.norm.weight
    target = tok("Z")["input_ids"][0]
    with torch.no_grad():
        old = s.module.down_proj.weight[:, 4].clone()
        s.module.down_proj.weight[:, 4] = 5 * head[target] / gain
    try:
        r = readout(dense_model, tok, n, sign=1, lens="logit", k=3)
        assert r.tokens[0] == "Z"
        r_neg = readout(dense_model, tok, n, sign=-1, lens="logit", k=3)
        assert "Z" not in r_neg.tokens
        picks = concept_neurons(dense_model, tok, [1], ["Z"], k=1)
        assert picks[0][:2] == (n, 1)
    finally:
        with torch.no_grad():
            s.module.down_proj.weight[:, 4] = old


def test_jacobian_lens_is_logit_lens_at_the_last_layer(dense_model, tok, texts):
    """After the last layer only the final norm remains; its scale does not change the ranking much, so the top token
    agrees and the direction is close (cosine > 0.9)."""
    last = len(sites(dense_model)) - 1
    n = nk.Neuron(last, 3)
    from neuronkit.readout import jacobian_direction

    w = output_vector(dense_model, n)
    v = jacobian_direction(dense_model, tok, last, w, texts[:3])
    assert torch.nn.functional.cosine_similarity(v, w, dim=0) > 0.9
    a = readout(dense_model, tok, n, lens="logit", k=1).tokens
    b = readout(dense_model, tok, n, lens="jacobian", k=1, texts=texts[:3]).tokens
    assert a == b


def test_write_cosine_and_gate_open(model, tok, texts):
    n = nk.Neuron(1, 2)
    w = output_vector(model, n)
    resid = torch.stack([w, -w, torch.randn_like(w)])
    sc = write_cosine(resid, model, [(n, 1)])
    assert sc[0] > 0.99 and sc[1] < -0.99
    ids = torch.tensor([tok(texts[0])["input_ids"]])
    g = gate_open(model, ids, [n])[n]
    assert g.shape == ids.shape and g.dtype == torch.bool
