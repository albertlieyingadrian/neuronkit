"""Describe / predict / simulate / write and modes, with a scripted judge (no network)."""

from __future__ import annotations

import json
import math

import pytest

import neuronkit as nk
from neuronkit import describe as D
from neuronkit import modes as M
from neuronkit.llm import FunctionLLM, complete_many

pytest.importorskip("circuitcheck")


def _judge(system: str, prompt: str) -> str:
    if system == D.DESCRIBE_SYSTEM:
        return '```json\n{"description": "fires on the letter e"}\n```'
    if system == D.PREDICT_SYSTEM:
        n = int(prompt.split("There are exactly ")[1].split()[0])
        return json.dumps({"probs": [0.9 if i % 2 else 0.1 for i in range(n)]})
    if "simulate" in system:
        n = prompt.count("\n") - 2
        return json.dumps({"acts": [5] * max(n, 1)})
    if "write" in system.lower():
        return json.dumps({"texts": ["eeee eee", "free bee"]})
    if "modes" in system:
        return json.dumps({"modes": [{"name": "letters", "groups": [0, 1]}]})
    return "{}"


def test_parsers():
    assert D.parse_description('{"description": " x "}') == "x"
    assert D.parse_description("no json here") == "no json here"
    assert D.parse_probs('{"probs": [0.2, 1.5]}', 2) == [0.2, 1.0]
    assert D.parse_probs('{"probs": [0.2]}', 2) is None
    assert D.parse_probs('{"probs": {"1": 0.3, "2": 0.9, "3": 0.1}}', 2) == [0.3, 0.9]
    assert D.parse_probs('{"probs": {"1": 0.3}}', 2) is None
    assert D.parse_levels('{"acts": [1, 20]}', 3) == [1.0, 10.0, 0.0]
    assert D.parse_texts('{"texts": ["a", "", 3]}') == ["a"]
    assert D.balanced_accuracy([0.9, 0.1], [1, 0]) == 1.0
    assert D.auroc([0.1, 0.9], [1, 0]) == 0.0


def test_score_description_runs_end_to_end(dense_model, tok, texts):
    n = nk.Neuron(1, 2)
    pos = nk.top_examples(dense_model, tok, texts, n, k=4)
    acts, toks = nk.run_capture(dense_model, nk.batches(tok, texts), [n])
    neg = nk.random_windows(tok, toks, acts[n], 4, width=16)
    windows = [(toks[e.doc], e) for e in pos[:2]]
    score = D.score_description(
        dense_model, tok, n, 1, "fires on the letter e", FunctionLLM(_judge), pos, neg, windows, a_ref=1.0,
        reference_max=100.0, n_write=2,
    )  # fmt: skip
    assert score.unparsed == 0
    assert 0.0 <= score.predict_accuracy <= 1.0
    assert len(score.windows) == 2 and all(w.ablate >= 0 for w in score.windows)
    assert math.isfinite(score.recovered)
    assert score.write_beats_reference is False


def test_patch_kl_exact_values_recover_everything(dense_model, tok, texts):
    """Patching in the neuron's own activations (as levels) recovers all the ablation damage."""
    n = nk.Neuron(2, 7)
    acts, toks = nk.run_capture(dense_model, nk.batches(tok, texts[:1]), [n])
    a = acts[n][0]
    a_ref = float(a.abs().max())
    sign = 1
    start, width = 5, 12
    levels = [10 * float(x) / a_ref for x in a[start : start + width]]  # may be negative: patch_kl takes them as-is
    k = D.patch_kl(dense_model, n, sign, toks[0], start, levels, a_ref)
    assert k.ablate > 0
    assert k.simulated < 1e-6 * max(1.0, k.ablate) + 1e-9


def test_modes_pipeline(dense_model, tok, texts):
    tr = nk.trace(dense_model, tok, texts[0])
    n = nk.Neuron(2, 3)
    profs = [tr.subcircuit(n, p).profile(50) for p in range(5, 25)]
    labels, sil = M.cluster_subcircuits(profs)
    assert len(labels) == 20
    groups = {g: [] for g in set(labels)}
    ex = nk.top_examples(dense_model, tok, texts, n, k=len(labels))
    for e, g in zip(ex, labels, strict=False):
        groups[g].append(e)
    system, prompt = M.modes_prompt(groups, 1)
    mp = M.parse_modes(FunctionLLM(_judge).complete(system, prompt), sorted(groups))
    assert set(mp) == set(groups)
    shares = M.mode_shares(labels, [1.0] * len(labels), mp)
    assert math.isclose(sum(shares.damage_share), 1.0)
    assert shares.n_modes >= 1


def test_mode_shares_single_mode():
    s = M.mode_shares([0, 0, 1, 1], [9.0, 9.0, 0.5, 0.5], {0: "a", 1: "b"})
    assert s.names[0] == "a" and s.single_mode(0.9)
    assert M.parse_modes("nonsense", [0]) is None
    assert M.parse_modes('{"modes": [{"name": "x", "groups": [0]}]}', [0, 1]) == {0: "x", 1: "group 1"}


def test_complete_many_uses_threads():
    llm = FunctionLLM(lambda s, p: p.upper())
    assert complete_many(llm, [("s", "a", 5), ("s", "b", 5)]) == ["A", "B"]
