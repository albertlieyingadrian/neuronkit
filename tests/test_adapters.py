"""The TransformerLens and nnsight adapters agree with neuronkit's own band ablation (skipped without the extras)."""

from __future__ import annotations

import pytest
import torch

import neuronkit as nk
from neuronkit.adapters import band_tensor, tl_band_hooks, tl_percentiles


def test_band_tensor():
    a = torch.tensor([[-3.0, -0.5, 0.0, 0.5, 3.0]])
    out = band_tensor(a, torch.full((5,), -1.0), torch.full((5,), 1.0))
    assert out.tolist() == [[-3.0, 0.0, 0.0, 0.0, 3.0]]


def test_transformer_lens_matches_neuronkit(dense_model, tok, texts):
    """TransformerLens 4 (TransformerBridge) around the same HF model: same percentiles, same band-ablated logits."""
    pytest.importorskip("transformer_lens")
    from transformer_lens.model_bridge import TransformerBridge

    bridge = TransformerBridge.boot_transformers("Qwen/Qwen2.5-0.5B", hf_model=dense_model)
    tokens = torch.tensor([tok(t)["input_ids"][:40] for t in texts if len(t) >= 40])
    pct_tl = tl_percentiles(bridge, tokens, qs=(3, 97))
    batch = nk.Batch(tokens, torch.ones_like(tokens), list(range(len(tokens))))
    pct_nk = nk.percentiles(dense_model, lambda: [batch], qs=(3, 97))
    for layer in pct_nk.values:
        assert torch.allclose(pct_tl.values[layer], pct_nk.values[layer], atol=1e-4)
    with torch.no_grad():
        with nk.ablate_band(dense_model, pct_nk):
            ref = dense_model(tokens).logits
        got = bridge.run_with_hooks(tokens, fwd_hooks=tl_band_hooks(pct_tl))
    assert torch.allclose(ref, got, atol=1e-4)


def test_nnsight_matches_neuronkit(dense_model, tok, texts):
    nnsight = pytest.importorskip("nnsight")
    pct = nk.percentiles(dense_model, lambda: nk.batches(tok, texts, batch_size=4), qs=(3, 97))
    ids = torch.tensor([tok(texts[0])["input_ids"]])
    with torch.no_grad(), nk.ablate_band(dense_model, pct, ">p97|<p3"):
        ref = dense_model(ids).logits
    from neuronkit.adapters import nnsight_ablate_band

    wrapped = nnsight.NNsight(dense_model)
    with wrapped.trace(ids):
        nnsight_ablate_band(wrapped, pct, ">p97|<p3")
        out = wrapped.output.save()
    logits = out.logits if hasattr(out, "logits") else out[0]
    assert torch.allclose(ref, logits, atol=1e-5)
