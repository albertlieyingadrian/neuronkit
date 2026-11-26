from __future__ import annotations

import numpy as np

import neuronkit as nk
from neuronkit.monitor import neuron_monitor, tail_features


def test_monitor_finds_the_planted_neuron():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 50))
    y = (X[:, 7] > 0.3).astype(int)
    neurons = [nk.Neuron(3, i) for i in range(50)]
    m = neuron_monitor(X, y, neurons)
    assert m.top(1)[0][0] == nk.Neuron(3, 7)
    assert m.n_used < 50
    s = m.score(X)
    assert np.corrcoef(s, y)[0, 1] > 0.6
    assert max(m.cv.values()) > 0.9


def test_tail_features_and_grouped_cv():
    X = np.array([[0.0, 5.0], [10.0, -5.0], [np.nan, 0.0]])
    t = tail_features(X, np.array([-1.0, -1.0]), np.array([1.0, 1.0]))
    assert t.tolist() == [[0, 1], [1, -1], [0, 0]]
    rng = np.random.default_rng(1)
    X = rng.normal(size=(120, 10))
    y = (X[:, 2] > 1.0).astype(int)
    groups = np.repeat(np.arange(30), 4)
    lo, hi = np.full(10, -1.0), np.full(10, 1.0)
    m = neuron_monitor(X, y, [nk.Neuron(0, i) for i in range(10)], groups=groups, features="tails", lo=lo, hi=hi)
    assert m.top(1)[0][0] == nk.Neuron(0, 2)
