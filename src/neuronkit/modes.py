"""Modes of a neuron pole: cluster its firings by their subcircuits, let a judge name and merge the clusters, and
split the neuron's ablation damage over the modes.

1. `cluster_subcircuits`: each firing's upstream profile (`Subcircuit.profile()`), TF-IDF-weighted across the pole's
   firings and L2-normalised, clustered with average-linkage cosine clustering. The number of clusters is chosen by
   silhouette (one cluster when no split reaches `min_silhouette`). The clusters stand for subcircuits: firings that
   are driven by the same upstream wiring.
2. `modes_prompt` / `parse_modes`: a judge sees a few excerpts per cluster and groups clusters that share a semantic
   category into named modes.
3. `mode_shares`: each mode's share of the firings and of the ablation damage (`firing_damage`). A pole is
   single-mode when one mode carries at least 90% of the damage.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from neuronkit.examples import Excerpt

MODES_SYSTEM = (
    "You group the firings of one neuron of a language model into modes. A mode is a semantic category of contexts "
    "where it fires (for example 'closing quotation marks', 'numbers in dates'). You see numbered groups of "
    "excerpts; tokens where the neuron fires are wrapped as <<token>>(a), a from 1 to 10. Groups that share a "
    "category belong to the same mode; most neurons have one or two modes. Answer with JSON only: "
    '{"modes": [{"name": "<= 8 words", "groups": [0, 2]}, ...]}, using every group number exactly once.'
)


def cluster_subcircuits(
    profiles: Sequence[Mapping[str, float]], max_k: int = 8, min_silhouette: float = 0.1
) -> tuple[list[int], float]:
    """Cluster firings by their upstream profiles. Returns (cluster per firing, best silhouette)."""
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import silhouette_score

    n = len(profiles)
    if n < 4:
        return [0] * n, float("nan")
    keys = sorted({k for p in profiles for k in p})
    col = {k: i for i, k in enumerate(keys)}
    X = np.zeros((n, len(keys)))
    for i, p in enumerate(profiles):
        for k, v in p.items():
            X[i, col[k]] = v
    df = (X != 0).sum(0)
    X = X * np.log(n / np.maximum(df, 1))
    X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    best, best_s = [0] * n, -1.0
    for k in range(2, min(max_k, n - 1) + 1):
        lab = AgglomerativeClustering(n_clusters=k, metric="cosine", linkage="average").fit_predict(X)
        if len(set(lab)) < 2:
            continue
        s = float(silhouette_score(X, lab, metric="cosine"))
        if s > best_s:
            best, best_s = [int(x) for x in lab], s
    return (best, best_s) if best_s >= min_silhouette else ([0] * n, best_s)


def modes_prompt(groups: Mapping[int, Sequence[Excerpt]], sign: int = 1, per_group: int = 8) -> tuple[str, str]:
    from neuronkit.describe import _cc

    cc = _cc()
    parts = []
    for g, exs in sorted(groups.items()):
        body = "\n".join(f"  - {cc.format_record(e.to_record(sign))}" for e in list(exs)[:per_group])
        parts.append(f"Group {g}:\n{body}")
    return MODES_SYSTEM, "\n\n".join(parts)


def parse_modes(text: str, groups: Sequence[int]) -> dict[int, str] | None:
    """group -> mode name; None when the reply is unusable. Groups the judge left out become their own mode."""
    from neuronkit.describe import parse_json

    try:
        raw = parse_json(text).get("modes", [])
    except ValueError:
        return None
    if not isinstance(raw, list):
        return None
    out: dict[int, str] = {}
    for m in raw:
        if not isinstance(m, dict):
            continue
        name = str(m.get("name", "")).strip() or "unnamed"
        for g in m.get("groups", []) or []:
            try:
                g = int(g)
            except (TypeError, ValueError):
                continue
            if g in groups and g not in out:
                out[g] = name
    for g in groups:
        out.setdefault(g, f"group {g}")
    return out


@dataclass
class ModeShares:
    names: list[str]
    firing_share: list[float]
    damage_share: list[float]

    @property
    def n_modes(self) -> int:
        return len(self.names)

    @property
    def dominant_share(self) -> float:
        return max(self.damage_share) if self.damage_share else float("nan")

    def single_mode(self, threshold: float = 0.9) -> bool:
        return self.dominant_share >= threshold


def mode_shares(clusters: Sequence[int], damage: Sequence[float], mode_of: Mapping[int, str]) -> ModeShares:
    """Each mode's share of firings and of the summed ablation damage."""
    names = sorted(set(mode_of[c] for c in clusters))
    tot_d = sum(max(d, 0.0) for d in damage) or 1.0
    fs, ds = [], []
    for nm in names:
        idx = [i for i, c in enumerate(clusters) if mode_of[c] == nm]
        fs.append(len(idx) / len(clusters))
        ds.append(sum(max(damage[i], 0.0) for i in idx) / tot_d)
    order = np.argsort(ds)[::-1]
    return ModeShares([names[i] for i in order], [fs[i] for i in order], [ds[i] for i in order])
