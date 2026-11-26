"""Does a neuron's description hold up? Describe, then predict, simulate and write.

The three tests, after the "Neurons as an Interpretable Basis" study (Goodfire, 2026):

- **predict**: given the description, a judge says which held-out excerpts make the neuron fire in its tail (beyond
  p97, or below p3 for the negative pole) and which are random text. Scored as balanced accuracy and AUROC.
- **simulate**: the judge predicts a 0-10 activation for every token of held-out windows; those predictions are
  patched in for the real activations (scaled to the pole's p99.9 magnitude) and compared with zero-ablating the
  neuron on the same tokens. Recovered = 1 - KL(base || simulated) / KL(base || ablated), as a ratio of sums over
  the windows: 1.0 means the description carries everything the neuron's activations carry there; 0.0 means
  no better than deleting the neuron.
- **write**: the judge writes texts meant to fire the pole as hard as possible; success when one beats the pole's
  maximum over the reference corpus.

Every step is a prompt builder plus a parser, so many neurons can be judged in one batch (`*_prompt`, `parse_*`), and
`score_description` runs the whole thing for one neuron. The excerpt format and the detection and simulation
prompts are circuitcheck's (`pip install "neuronkit[labels]"`), so labels from either tool are scored the same way.
"""

from __future__ import annotations

import json
import math
import random
import re
from collections.abc import Sequence
from dataclasses import dataclass, field

import torch
from torch import nn

from neuronkit.core import Neuron, ablate, edit, neuron_edit
from neuronkit.examples import Excerpt
from neuronkit.llm import complete_many


def _cc():
    try:
        from circuitcheck.labels import autointerp
    except ImportError as e:  # pragma: no cover
        raise ImportError('the judge steps use circuitcheck: pip install "neuronkit[labels]"') from e
    return autointerp


_COMMENT = re.compile(r"//[^\n]*|/\*.*?\*/", re.DOTALL)


def parse_json(text: str) -> dict:
    """The first JSON object in a reply. Tolerates code fences, Python-style quotes and // comments (some judges
    annotate every number)."""
    cc = _cc()
    try:
        return cc._json(text)
    except ValueError:
        return cc._json(_COMMENT.sub("", text))


JSON_ONLY = "\n\nReply with the JSON object only: no comments, no explanation."


DESCRIBE_SYSTEM = (
    "You describe one neuron of a language model. You see excerpts of text; the tokens where the neuron fires are "
    "wrapped as <<token>>(a), with a its strength from 1 to 10. Some excerpts may come from different groups (modes). "
    "Write a description of at most 40 words that would let someone predict, for new text, where the neuron fires "
    "and how strongly: the kind of token, its context, and what usually comes next. If there are clearly distinct "
    'modes, name each briefly. Answer with JSON only: {"description": "..."}'
)

WRITE_SYSTEM = (
    "You write text to make one neuron of a language model fire as strongly as possible. Given the neuron's "
    "description, write {n} different short texts (each under 60 words) that you expect to contain its strongest "
    'possible firing. Answer with JSON only: {{"texts": ["...", "..."]}}'
)


def describe_prompt(excerpts: Sequence[Excerpt], sign: int = 1, modes: Sequence[str] | None = None) -> tuple[str, str]:
    cc = _cc()
    body = "\n".join(f"{i + 1}. {cc.format_record(e.to_record(sign))}" for i, e in enumerate(excerpts))
    extra = ""
    if modes:
        extra = "\n\nGroups found among its firings:\n" + "\n".join(f"- {m}" for m in modes)
    return DESCRIBE_SYSTEM, f"Excerpts:\n{body}{extra}"


def parse_description(text: str) -> str:
    try:
        return str(parse_json(text).get("description", "")).strip()
    except ValueError:
        return text.strip()[:400]


def predict_prompt(
    description: str, positives: Sequence[Excerpt], negatives: Sequence[Excerpt], seed: int = 0
) -> tuple[str, str, list[int]]:
    """circuitcheck's detection prompt over a shuffled mix; returns (system, prompt, truth in prompt order)."""
    cc = _cc()
    items = [(e, 1) for e in positives] + [(e, 0) for e in negatives]
    random.Random(seed).shuffle(items)
    body = "\n".join(f"{i + 1}. {cc.format_record(e.to_record(1), marks=False)}" for i, (e, _) in enumerate(items))
    prompt = f"Label: {description}\n\nThere are exactly {len(items)} excerpts.\n\nExcerpts:\n{body}{JSON_ONLY}"
    return cc.DETECT_SYSTEM, prompt, [y for _, y in items]


def parse_probs(text: str, n: int) -> list[float] | None:
    try:
        raw = parse_json(text).get("probs", [])
    except ValueError:
        return None
    if not isinstance(raw, list) or len(raw) != n:
        return None
    try:
        return [min(1.0, max(0.0, float(p))) for p in raw]
    except (TypeError, ValueError):
        return None


def simulate_prompt(description: str, window: Excerpt) -> tuple[str, str]:
    cc = _cc()
    body = "\n".join(f"{i}: {cc._clean(t)!r}" for i, t in enumerate(window.tokens))
    return cc.SIMULATE_SYSTEM, f"Label: {description}\n\nTokens:\n{body}{JSON_ONLY}"


def parse_levels(text: str, n: int) -> list[float] | None:
    try:
        raw = parse_json(text).get("acts", [])
    except ValueError:
        return None
    if not isinstance(raw, list):
        return None
    try:
        vals = [min(10.0, max(0.0, float(a))) for a in raw]
    except (TypeError, ValueError):
        return None
    if len(vals) < n:
        vals += [0.0] * (n - len(vals))
    return vals[:n]


def write_prompt(description: str, n: int = 5) -> tuple[str, str]:
    return WRITE_SYSTEM.format(n=n), f"Neuron description: {description}"


def parse_texts(text: str) -> list[str]:
    try:
        raw = parse_json(text).get("texts", [])
    except ValueError:
        return []
    return [str(t) for t in raw if isinstance(t, str) and t.strip()] if isinstance(raw, list) else []


# ---------------------------------------------------------------- scores


def balanced_accuracy(probs: Sequence[float], truth: Sequence[int]) -> float:
    pos = [p >= 0.5 for p, y in zip(probs, truth, strict=True) if y]
    neg = [p < 0.5 for p, y in zip(probs, truth, strict=True) if not y]
    parts = [sum(x) / len(x) for x in (pos, neg) if x]
    return sum(parts) / len(parts) if parts else float("nan")


def auroc(scores: Sequence[float], truth: Sequence[int]) -> float:
    pos = [s for s, y in zip(scores, truth, strict=True) if y]
    neg = [s for s, y in zip(scores, truth, strict=True) if not y]
    if not pos or not neg:
        return float("nan")
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))


@dataclass
class WindowKL:
    ablate: float
    simulated: float


def _window_logp(model, ids, mask, edit_fn=None):
    with torch.no_grad():
        if edit_fn is None:
            lg = model(input_ids=ids, use_cache=False).logits
        else:
            with edit(model, edit_fn):
                lg = model(input_ids=ids, use_cache=False).logits
    return torch.log_softmax(lg.float()[0][mask[0]], -1)


def patch_kl(
    model: nn.Module,
    neuron: Neuron,
    sign: int,
    token_ids: Sequence[int],
    start: int,
    levels: Sequence[float],
    a_ref: float,
) -> WindowKL:
    """KL(base || ablated) and KL(base || simulated) over the window [start, start + len(levels)) of one document.

    The neuron is zeroed (ablated) or set to sign * level/10 * a_ref (simulated) on the window's tokens only; the
    document before the window is kept as context.
    """
    dev = next(model.parameters()).device
    ids = torch.tensor([list(token_ids)], device=dev)
    mask = torch.zeros_like(ids, dtype=torch.bool)
    mask[0, start : start + len(levels)] = True
    base = _window_logp(model, ids, mask)
    vals = torch.zeros(ids.shape, dtype=torch.float32)
    vals[0, start : start + len(levels)] = torch.tensor([sign * float(s) / 10 * a_ref for s in levels])
    abl = _window_logp(model, ids, mask, neuron_edit({neuron: 0.0}, positions=mask))
    sim = _window_logp(model, ids, mask, neuron_edit({neuron: vals}, positions=mask))

    def kl(lp):
        return float((base.exp() * (base - lp)).sum(-1).mean())

    return WindowKL(kl(abl), kl(sim))


def recovered(kls: Sequence[WindowKL]) -> float:
    """1 - sum KL(simulated) / sum KL(ablated) over windows."""
    a = sum(k.ablate for k in kls)
    return 1 - sum(k.simulated for k in kls) / a if a > 0 else float("nan")


@dataclass
class DescriptionScore:
    neuron: str
    sign: int
    description: str
    predict_accuracy: float
    predict_auroc: float
    recovered: float
    write_best: float | None = None
    write_beats_reference: bool | None = None
    windows: list[WindowKL] = field(default_factory=list)
    unparsed: int = 0

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        d["windows"] = [w.__dict__ for w in self.windows]
        return d


def score_description(
    model: nn.Module,
    tokenizer,
    neuron: Neuron,
    sign: int,
    description: str,
    llm,
    positives: Sequence[Excerpt],
    negatives: Sequence[Excerpt],
    sim_windows: Sequence[tuple[Sequence[int], Excerpt]],
    a_ref: float,
    reference_max: float | None = None,
    n_write: int = 5,
    seed: int = 0,
) -> DescriptionScore:
    """Predict, simulate (with damage recovery) and write, for one pole and one description.

    `sim_windows` pairs each window with its document's token ids (the window's `start` indexes into them).
    `reference_max` is the pole's strongest firing on a reference corpus (sign * activation), for the writing test.
    """
    unparsed = 0
    system, prompt, truth = predict_prompt(description, positives, negatives, seed)
    probs = parse_probs(llm.complete(system, prompt, 1200), len(truth))
    if probs is None:
        unparsed += 1
        acc, auc = float("nan"), float("nan")
    else:
        acc, auc = balanced_accuracy(probs, truth), auroc(probs, truth)
    items = [(*simulate_prompt(description, w), 600) for _, w in sim_windows]
    replies = complete_many(llm, items)
    kls = []
    for (doc_ids, w), rep in zip(sim_windows, replies, strict=True):
        lv = parse_levels(rep, len(w.tokens))
        if lv is None:
            unparsed += 1
            continue
        kls.append(patch_kl(model, neuron, sign, doc_ids, w.start, lv, a_ref))
    best = None
    if n_write:
        texts = parse_texts(llm.complete(*write_prompt(description, n_write), 800))
        for t in texts:
            peak = max_firing(model, tokenizer, neuron, sign, t)
            best = peak if best is None else max(best, peak)
    return DescriptionScore(
        str(neuron),
        sign,
        description,
        acc,
        auc,
        recovered(kls),
        best,
        (best > reference_max) if best is not None and reference_max is not None else None,
        kls,
        unparsed,
    )


def max_firing(model: nn.Module, tokenizer, neuron: Neuron, sign: int, text: str, max_tokens: int = 128) -> float:
    """The pole's strongest firing (sign * activation) on a text; -inf when an MoE neuron is never routed."""
    from neuronkit.core import capture

    ids = tokenizer(text, add_special_tokens=False)["input_ids"][:max_tokens]
    if not ids:
        return -math.inf
    dev = next(model.parameters()).device
    with capture(model, neurons=[neuron]) as st, torch.no_grad():
        model(input_ids=torch.tensor([ids], device=dev), use_cache=False)
    a = st[neuron.layer][0][0, :, 0]
    return float(torch.nan_to_num(sign * a, nan=-math.inf).max())


__all__ = [
    "DescriptionScore",
    "WindowKL",
    "auroc",
    "balanced_accuracy",
    "describe_prompt",
    "max_firing",
    "parse_description",
    "parse_json",
    "parse_levels",
    "parse_probs",
    "parse_texts",
    "patch_kl",
    "predict_prompt",
    "recovered",
    "score_description",
    "simulate_prompt",
    "write_prompt",
    "ablate",
    "json",
]
