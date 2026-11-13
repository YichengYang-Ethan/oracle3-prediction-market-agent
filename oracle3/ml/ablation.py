"""Where does the gain come from? LightGBM on nested feature sets, same folds and procedure as the benchmark.

python -m oracle3.ml.ablation --snapshots S.parquet --out evidence/ml_calibration/ablation.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from oracle3.ml.benchmark import apply_isotonic, fit_pipeline
from oracle3.ml.cv import inner_walk_forward, purged_walk_forward
from oracle3.ml.features import FEATURES
from oracle3.ml.metrics import brier, cluster_bootstrap, ece
from oracle3.ml.models import LEARNERS, predict

SUBSETS: dict[str, list[str]] = {
    'price_only': ['price', 'logit_price', 'extremity'],
    'price_venue_category': ['price', 'logit_price', 'extremity', 'is_kalshi']
    + [f for f in FEATURES if f.startswith('cat_')],
    'all_features': list(FEATURES),
}


def run(
    df: pd.DataFrame,
    n_splits: int = 5,
    embargo: float = 86400.0,
    inner_blocks: int = 3,
    n_boot: int = 300,
) -> dict[str, Any]:
    y = df['label'].to_numpy(int)
    t, resolve, groups = (
        df['t'].to_numpy(float),
        df['resolve_ts'].to_numpy(float),
        df['event_id'].to_numpy(),
    )
    folds = purged_walk_forward(t, resolve, groups, n_splits=n_splits, embargo=embargo)
    learner = next(lr for lr in LEARNERS if lr.name == 'lightgbm')
    rows = np.concatenate([f.test for f in folds])
    market, yt, g = df['price'].to_numpy(float)[rows], y[rows], groups[rows]
    out: dict[str, Any] = {
        'market': {'brier': brier(market, yt), 'ece': ece(market, yt)}
    }
    for name, cols in SUBSETS.items():
        X = df.loc[:, cols].astype(float)
        raw_parts, iso_parts = [], []
        for fold in folds:
            pairs = inner_walk_forward(
                t, resolve, groups, fold.train, inner_blocks, embargo
            )
            est, iso, _ = fit_pipeline(learner, X, y, fold.train, pairs, [], fold.k)
            raw = predict(est, X.iloc[fold.test])
            raw_parts.append(raw)
            iso_parts.append(apply_isotonic(iso, raw))
        for tag, p in (
            ('raw', np.concatenate(raw_parts)),
            ('isotonic', np.concatenate(iso_parts)),
        ):

            def rel(r: np.ndarray, p: np.ndarray = p) -> float:
                return 1.0 - brier(p[r], yt[r]) / brier(market[r], yt[r])

            out[f'{name}/{tag}'] = {
                'features': cols,
                'brier': brier(p, yt),
                'ece': ece(p, yt),
                'brier_reduction_pct': 100.0 * (1.0 - brier(p, yt) / brier(market, yt)),
                'brier_reduction_pct_ci95': [
                    100.0 * v for v in cluster_bootstrap(g, rel, n_boot)
                ],
                'ece_reduction_pct': 100.0 * (1.0 - ece(p, yt) / ece(market, yt)),
            }
            print(
                name,
                tag,
                round(out[f'{name}/{tag}']['brier_reduction_pct'], 2),
                flush=True,
            )
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument('--snapshots', required=True, type=Path)
    ap.add_argument('--out', required=True, type=Path)
    a = ap.parse_args(argv)
    res = run(pd.read_parquet(a.snapshots))
    a.out.write_text(json.dumps(res, indent=2))


if __name__ == '__main__':
    main()
