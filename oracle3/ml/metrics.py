"""Scoring rules and calibration error, with event-clustered bootstrap intervals."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

EPS = 1e-6


def brier(p: np.ndarray, y: np.ndarray) -> float:
    p, y = np.asarray(p, float), np.asarray(y, float)
    return float(np.mean((p - y) ** 2))


def log_loss(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    y = np.asarray(y, float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def ece(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    """Expected calibration error with equal-width probability bins: sum_b (n_b / n) |mean p_b - mean y_b|."""
    p, y = np.asarray(p, float), np.asarray(y, float)
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.any():
            total += m.sum() * abs(p[m].mean() - y[m].mean())
    return float(total / len(p))


def reliability(
    p: np.ndarray, y: np.ndarray, n_bins: int = 10
) -> list[dict[str, float]]:
    p, y = np.asarray(p, float), np.asarray(y, float)
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    out = []
    for b in range(n_bins):
        m = bins == b
        if m.any():
            out.append(
                {
                    'bin': b,
                    'lo': b / n_bins,
                    'hi': (b + 1) / n_bins,
                    'n': int(m.sum()),
                    'mean_prediction': float(p[m].mean()),
                    'outcome_rate': float(y[m].mean()),
                }
            )
    return out


def score(p: np.ndarray, y: np.ndarray) -> dict[str, float]:
    return {
        'brier': brier(p, y),
        'log_loss': log_loss(p, y),
        'ece': ece(p, y),
        'ece_15': ece(p, y, 15),
        'n': int(len(y)),
        'mean_prediction': float(np.mean(p)),
        'outcome_rate': float(np.mean(y)),
    }


def cluster_bootstrap(
    groups: np.ndarray,
    stat: Callable[[np.ndarray], float],
    n_boot: int = 1000,
    seed: int = 0,
) -> tuple[float, float]:
    """95% percentile interval of ``stat(rows)`` when whole groups (events) are resampled with replacement."""
    rng = np.random.default_rng(seed)
    groups = np.asarray(groups)
    uniq, inverse = np.unique(groups, return_inverse=True)
    members = [[] for _ in range(len(uniq))]
    for row, g in enumerate(inverse):
        members[g].append(row)
    members_arr = [np.asarray(m) for m in members]
    vals = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        rows = np.concatenate([members_arr[i] for i in pick])
        vals[b] = stat(rows)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))
