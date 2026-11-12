"""Event-grouped, purged walk-forward cross-validation.

Every contract of an event belongs to one group, and a group is never split
between training and testing. Groups are ordered by their first prediction time
and cut into ``n_splits + 1`` consecutive blocks; fold ``k`` tests on block
``k + 1``. Its training set holds only groups whose last outcome was known at
least ``embargo`` seconds before the test block's first prediction, so no
training label comes from after (or just before) the test period begins.
Groups from earlier blocks that resolved too late are purged, not moved.

The same rule cuts each training window again for tuning and calibration
(``inner_walk_forward``): its groups are ordered and cut into ``n_blocks + 1``
blocks, and block ``j`` is predicted by models fitted only on earlier groups that
resolved ``embargo`` seconds before it. Those out-of-fold predictions choose the
hyperparameters and fit the isotonic map, so both use training data only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Fold:
    k: int
    train: np.ndarray  # row positions
    test: np.ndarray
    test_start: float
    test_end: float
    purged_groups: int


def _group_table(
    t: np.ndarray, resolve: np.ndarray, groups: np.ndarray
) -> pd.DataFrame:
    df = pd.DataFrame({'t': t, 'resolve': resolve, 'g': groups})
    return (
        df.groupby('g', sort=False)
        .agg(start=('t', 'min'), end=('resolve', 'max'))
        .sort_values(['start'])
        .reset_index()
    )


def purged_walk_forward(
    t: np.ndarray,
    resolve: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 5,
    embargo: float = 86400.0,
    min_train_groups: int = 50,
) -> list[Fold]:
    t, resolve, groups = (
        np.asarray(t, float),
        np.asarray(resolve, float),
        np.asarray(groups),
    )
    table = _group_table(t, resolve, groups)
    blocks = np.array_split(np.arange(len(table)), n_splits + 1)
    folds: list[Fold] = []
    row_group = pd.Series(np.arange(len(groups))).groupby(groups).apply(np.asarray)
    for k in range(1, n_splits + 1):
        test_groups = table.iloc[blocks[k]]
        test_start = float(test_groups['start'].min())
        test_end = float(test_groups['start'].max())
        earlier = table.iloc[np.concatenate(blocks[:k])]
        ok = earlier['end'] < test_start - embargo
        train_groups = earlier[ok]
        if len(train_groups) < min_train_groups:
            continue
        train = np.sort(np.concatenate([row_group[g] for g in train_groups['g']]))
        test = np.sort(np.concatenate([row_group[g] for g in test_groups['g']]))
        folds.append(Fold(k, train, test, test_start, test_end, int((~ok).sum())))
    return folds


def inner_split(
    t: np.ndarray,
    resolve: np.ndarray,
    groups: np.ndarray,
    idx: np.ndarray,
    inner_fraction: float = 0.2,
    embargo: float = 86400.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Purged chronological (fit, validation) split of the training rows ``idx``."""
    idx = np.asarray(idx)
    table = _group_table(
        np.asarray(t)[idx], np.asarray(resolve)[idx], np.asarray(groups)[idx]
    )
    n_val = max(1, int(round(inner_fraction * len(table))))
    val_groups = table.iloc[len(table) - n_val :]
    val_start = float(val_groups['start'].min())
    fit_groups = table.iloc[: len(table) - n_val]
    fit_groups = fit_groups[fit_groups['end'] < val_start - embargo]
    g = np.asarray(groups)[idx]
    fit = idx[np.isin(g, fit_groups['g'].to_numpy())]
    val = idx[np.isin(g, val_groups['g'].to_numpy())]
    return fit, val


def inner_walk_forward(
    t: np.ndarray,
    resolve: np.ndarray,
    groups: np.ndarray,
    idx: np.ndarray,
    n_blocks: int = 3,
    embargo: float = 86400.0,
    min_fit_groups: int = 20,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Purged expanding-window (fit, validation) pairs inside the training rows ``idx``."""
    idx = np.asarray(idx)
    g_all = np.asarray(groups)[idx]
    table = _group_table(np.asarray(t)[idx], np.asarray(resolve)[idx], g_all)
    blocks = np.array_split(np.arange(len(table)), n_blocks + 1)
    pairs = []
    for j in range(1, n_blocks + 1):
        val_groups = table.iloc[blocks[j]]
        if val_groups.empty:
            continue
        start = float(val_groups['start'].min())
        earlier = table.iloc[np.concatenate(blocks[:j])]
        fit_groups = earlier[earlier['end'] < start - embargo]
        if len(fit_groups) < min_fit_groups:
            continue
        pairs.append(
            (
                idx[np.isin(g_all, fit_groups['g'].to_numpy())],
                idx[np.isin(g_all, val_groups['g'].to_numpy())],
            )
        )
    return pairs
