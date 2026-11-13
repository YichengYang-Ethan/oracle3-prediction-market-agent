"""Out-of-sample benchmark of eight learners and two ensembles against market prices and the paper's model.

For every outer fold of the event-grouped, purged walk-forward split:

1. the training window is cut again into purged, expanding inner walk-forward blocks, each
   predicted by models fitted only on earlier training events;
2. each learner's hyperparameters are chosen by the Brier score of those out-of-fold
   predictions (LightGBM's number of trees by early stopping on each inner block);
3. an isotonic map is fitted to the chosen configuration's out-of-fold predictions, i.e. on
   training folds only, and kept only if it passes a check on the same data: fitted on the
   earlier inner blocks, it must lower the Brier score of the later ones (the refitted model
   is stronger than the inner ones, so a map that only suits the inner models is dropped);
4. the chosen configuration is refitted on the whole training window and, when kept, the
   isotonic map applied to its test predictions;
5. the Wang baselines and the isotonic recalibration of the market price are fitted on all
   training rows; the market baseline needs no fitting;
6. two ensembles combine the learners: the mean of their recalibrated probabilities, and a
   logistic regression stacked on their log-odds, fitted to the inner out-of-fold predictions.

Scores are computed on the test rows only and pooled across folds. Intervals resample whole
events. The model behind the MCP tool is then fitted the same way on all rows.

    python -m oracle3.ml.benchmark --snapshots S.parquet --evidence DIR --bundle DIR
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from oracle3.ml.cv import inner_walk_forward, purged_walk_forward
from oracle3.ml.features import FEATURES, KALSHI_ONLY
from oracle3.ml.metrics import brier, cluster_bootstrap, ece, reliability, score
from oracle3.ml.models import (
    LEARNERS,
    SERVABLE,
    Learner,
    WangBaseline,
    design,
    fit_learner,
    predict,
)

ISO_CLIP = 1e-4
# Learners within this relative Brier margin of the best servable one count as tied; among tied
# learners the bundle takes the first in SERVE_PREFERENCE (smallest file, fastest to score).
SERVE_TIE = 0.001
SERVE_PREFERENCE = (
    'lightgbm',
    'lightgbm_monotone',
    'logistic',
    'spline_logistic',
    'mlp',
    'random_forest',
)
BASELINES = ('market', 'market_isotonic', 'wang', 'wang_hier')
ENSEMBLES = ('ensemble_mean', 'stacked')


def fit_isotonic(p: np.ndarray, y: np.ndarray) -> Any:
    from sklearn.isotonic import IsotonicRegression

    return IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds='clip').fit(p, y)


def apply_isotonic(iso: Any, p: np.ndarray) -> np.ndarray:
    return np.clip(iso.predict(p), ISO_CLIP, 1 - ISO_CLIP)


def tune(
    learner: Learner,
    X: pd.DataFrame,
    y: np.ndarray,
    pairs: list[tuple[np.ndarray, np.ndarray]],
    log: list[dict[str, Any]],
    fold: int,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    """Best configuration by out-of-fold Brier over the inner blocks; returns it with its OOF predictions."""
    best: tuple[float, dict[str, Any], np.ndarray, np.ndarray] | None = None
    val_rows = np.concatenate([val for _, val in pairs])
    for params in learner.grid:
        t0 = time.time()
        oof, trees = [], []
        for fit_idx, val_idx in pairs:
            est, used = fit_learner(
                learner,
                params,
                X.iloc[fit_idx],
                y[fit_idx],
                X.iloc[val_idx],
                y[val_idx],
            )
            oof.append(predict(est, X.iloc[val_idx]))
            trees.append(used.get('n_estimators'))
        pv = np.concatenate(oof)
        b = brier(pv, y[val_rows])
        used = dict(params)
        if learner.early_stopping:
            used['n_estimators'] = int(np.median([n for n in trees if n]))
        log.append(
            {
                'fold': fold,
                'model': learner.name,
                'params': _jsonable(used),
                'oof_brier': b,
                'oof_ece': ece(pv, y[val_rows]),
                'oof_rows': int(len(val_rows)),
                'seconds': round(time.time() - t0, 1),
            }
        )
        if best is None or b < best[0]:
            best = (b, used, pv, val_rows)
    assert best is not None
    return best[1], best[2], best[3]


def isotonic_helps(
    oof: np.ndarray, y_oof: np.ndarray, block_sizes: list[int]
) -> tuple[bool, dict[str, float]]:
    """Does isotonic recalibration, fitted on earlier inner blocks, lower the Brier score of later ones?

    Uses only training-window out-of-fold predictions. Later blocks are predicted by models fitted on
    more data, as the refitted model is, so the check mirrors how the map will be used.
    """
    edges = np.cumsum([0, *block_sizes])
    raw, cal, ys = [], [], []
    for j in range(1, len(block_sizes)):
        lo, hi = int(edges[j]), int(edges[j + 1])
        if lo == 0 or hi == lo:
            continue
        iso = fit_isotonic(oof[:lo], y_oof[:lo])
        raw.append(oof[lo:hi])
        cal.append(apply_isotonic(iso, oof[lo:hi]))
        ys.append(y_oof[lo:hi])
    if not ys:
        return False, {}
    yv = np.concatenate(ys)
    b_raw, b_iso = brier(np.concatenate(raw), yv), brier(np.concatenate(cal), yv)
    return b_iso < b_raw, {'inner_brier_raw': b_raw, 'inner_brier_isotonic': b_iso}


def fit_pipeline_oof(
    learner: Learner,
    X: pd.DataFrame,
    y: np.ndarray,
    train_idx: np.ndarray,
    pairs: list[tuple[np.ndarray, np.ndarray]],
    log: list[dict[str, Any]],
    fold: int,
) -> tuple[Any, Any, dict[str, Any], np.ndarray, np.ndarray]:
    """Tune, fit the isotonic map on the inner out-of-fold predictions, refit on the whole window."""
    params, oof, oof_rows = tune(learner, X, y, pairs, log, fold)
    iso = fit_isotonic(oof, y[oof_rows])
    est, _ = fit_learner(learner, dict(params), X.iloc[train_idx], y[train_idx])
    return est, iso, params, oof, oof_rows


def recalibrate(
    learner: Learner,
    oof: np.ndarray,
    y_oof: np.ndarray,
    pairs: list[tuple[np.ndarray, np.ndarray]],
    log: list[dict[str, Any]],
    fold: int,
) -> bool:
    """Decide on training data whether this learner's isotonic map is kept, and log why."""
    keep, check = isotonic_helps(oof, y_oof, [len(v) for _, v in pairs])
    log.append(
        {
            'fold': fold,
            'model': learner.name,
            'recalibration': 'isotonic' if keep else 'none',
            **check,
        }
    )
    return keep


def fit_pipeline(
    learner: Learner,
    X: pd.DataFrame,
    y: np.ndarray,
    train_idx: np.ndarray,
    pairs: list[tuple[np.ndarray, np.ndarray]],
    log: list[dict[str, Any]],
    fold: int,
) -> tuple[Any, Any, dict[str, Any]]:
    est, iso, params, _, _ = fit_pipeline_oof(
        learner, X, y, train_idx, pairs, log, fold
    )
    return est, iso, params


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, float), ISO_CLIP, 1 - ISO_CLIP)
    return np.log(p / (1 - p))


def fit_stacker(oof: dict[str, np.ndarray], y: np.ndarray) -> Any:
    """Logistic regression on the learners' out-of-fold log-odds (training folds only)."""
    from sklearn.linear_model import LogisticRegression

    Z = np.column_stack([_logit(oof[name]) for name in sorted(oof)])
    return LogisticRegression(C=1.0, max_iter=1000).fit(Z, y)


def apply_stacker(stacker: Any, raw: dict[str, np.ndarray]) -> np.ndarray:
    Z = np.column_stack([_logit(raw[name]) for name in sorted(raw)])
    return np.clip(stacker.predict_proba(Z)[:, 1], ISO_CLIP, 1 - ISO_CLIP)


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj


def run_cv(
    df: pd.DataFrame, n_splits: int, embargo: float, inner_blocks: int
) -> tuple[pd.DataFrame, list, list]:
    X = design(df)
    y = df['label'].to_numpy(int)
    t, resolve, groups = (
        df['t'].to_numpy(float),
        df['resolve_ts'].to_numpy(float),
        df['event_id'].to_numpy(),
    )
    folds = purged_walk_forward(t, resolve, groups, n_splits=n_splits, embargo=embargo)
    preds: list[pd.DataFrame] = []
    tuning: list[dict[str, Any]] = []
    fold_info: list[dict[str, Any]] = []
    for fold in folds:
        pairs = inner_walk_forward(
            t, resolve, groups, fold.train, inner_blocks, embargo
        )
        oof_rows = int(sum(len(v) for _, v in pairs))
        test = df.iloc[fold.test]
        out = pd.DataFrame({'row': fold.test, 'fold': fold.k})
        out['market'] = test['price'].to_numpy(float)
        train_df = df.iloc[fold.train]
        out['market_isotonic'] = apply_isotonic(
            fit_isotonic(train_df['price'].to_numpy(float), y[fold.train]),
            out['market'].to_numpy(float),
        )
        out['wang'] = WangBaseline().fit(train_df).predict(test)
        out['wang_hier'] = WangBaseline(hierarchical=True).fit(train_df).predict(test)
        chosen = {}
        oof_by: dict[str, np.ndarray] = {}
        raw_by: dict[str, np.ndarray] = {}
        stack_rows: np.ndarray | None = None
        for learner in LEARNERS:
            t0 = time.time()
            est, iso, params, oof, oof_at = fit_pipeline_oof(
                learner, X, y, fold.train, pairs, tuning, fold.k
            )
            assert stack_rows is None or np.array_equal(stack_rows, oof_at)
            stack_rows = oof_at
            keep_iso = recalibrate(learner, oof, y[oof_at], pairs, tuning, fold.k)
            raw = predict(est, X.iloc[fold.test])
            oof_by[learner.name], raw_by[learner.name] = oof, raw
            out[f'{learner.name}_raw'] = raw
            out[f'{learner.name}_isotonic'] = apply_isotonic(iso, raw)
            out[learner.name] = (
                out[f'{learner.name}_isotonic']
                if keep_iso
                else np.clip(raw, ISO_CLIP, 1 - ISO_CLIP)
            )
            chosen[learner.name] = {
                'params': _jsonable(params),
                'recalibration': 'isotonic' if keep_iso else 'none',
                'seconds': round(time.time() - t0, 1),
            }
            print(
                f'[cv] fold {fold.k} {learner.name} done in {time.time() - t0:.0f}s params={params}',
                flush=True,
            )
        out['ensemble_mean'] = out[[lr.name for lr in LEARNERS]].mean(axis=1)
        assert stack_rows is not None
        out['stacked'] = apply_stacker(fit_stacker(oof_by, y[stack_rows]), raw_by)
        preds.append(out)
        fold_info.append(
            {
                'fold': fold.k,
                'train_rows': int(len(fold.train)),
                'inner_blocks': len(pairs),
                'inner_oof_rows': oof_rows,
                'test_rows': int(len(fold.test)),
                'train_events': int(df.iloc[fold.train]['event_id'].nunique()),
                'test_events': int(test['event_id'].nunique()),
                'test_contracts': int(test['market_id'].nunique()),
                'test_start': _iso(fold.test_start),
                'test_end': _iso(fold.test_end),
                'purged_events': fold.purged_groups,
                'chosen': chosen,
            }
        )
    return pd.concat(preds, ignore_index=True), tuning, fold_info


def _iso(ts: float) -> str:
    return pd.Timestamp(ts, unit='s', tz='UTC').strftime('%Y-%m-%dT%H:%M:%SZ')


# Each learner: recalibration chosen on training data (the method), plus both variants for reference.
MAIN_METHODS = (*BASELINES, *(lr.name for lr in LEARNERS), *ENSEMBLES)
METHODS = (
    *MAIN_METHODS,
    *(f'{lr.name}_raw' for lr in LEARNERS),
    *(f'{lr.name}_isotonic' for lr in LEARNERS),
)


def summarize(df: pd.DataFrame, preds: pd.DataFrame, n_boot: int) -> dict[str, Any]:
    rows = preds['row'].to_numpy()
    y = df['label'].to_numpy(int)[rows]
    groups = df['event_id'].to_numpy()[rows]
    pooled = {m: score(preds[m].to_numpy(float), y) for m in METHODS}
    per_fold = {}
    for k, part in preds.groupby('fold'):
        yk = df['label'].to_numpy(int)[part['row'].to_numpy()]
        per_fold[int(k)] = {m: score(part[m].to_numpy(float), yk) for m in METHODS}
    mkt = preds['market'].to_numpy(float)
    vs_market: dict[str, Any] = {}
    for m in METHODS:
        if m == 'market':
            continue
        if (
            m not in MAIN_METHODS
        ):  # variants: point estimates only, to keep the bootstrap affordable
            vs_market[m] = {
                'brier_reduction_pct': 100.0
                * (1.0 - pooled[m]['brier'] / pooled['market']['brier']),
                'ece_reduction_pct': 100.0
                * (1.0 - pooled[m]['ece'] / pooled['market']['ece']),
                'folds_beating_market_on_brier': int(
                    sum(
                        per_fold[k][m]['brier'] < per_fold[k]['market']['brier']
                        for k in per_fold
                    )
                ),
                'folds': len(per_fold),
            }
            continue
        pm = preds[m].to_numpy(float)

        def d_brier(r: np.ndarray, pm: np.ndarray = pm) -> float:
            return brier(pm[r], y[r]) - brier(mkt[r], y[r])

        def rel_brier(r: np.ndarray, pm: np.ndarray = pm) -> float:
            return 1.0 - brier(pm[r], y[r]) / brier(mkt[r], y[r])

        def rel_ece(r: np.ndarray, pm: np.ndarray = pm) -> float:
            return 1.0 - ece(pm[r], y[r]) / ece(mkt[r], y[r])

        vs_market[m] = {
            'brier_reduction_pct': 100.0
            * (1.0 - pooled[m]['brier'] / pooled['market']['brier']),
            'brier_reduction_pct_ci95': [
                100.0 * v for v in cluster_bootstrap(groups, rel_brier, n_boot)
            ],
            'brier_difference': pooled[m]['brier'] - pooled['market']['brier'],
            'brier_difference_ci95': list(cluster_bootstrap(groups, d_brier, n_boot)),
            'ece_reduction_pct': 100.0
            * (1.0 - pooled[m]['ece'] / pooled['market']['ece']),
            'ece_reduction_pct_ci95': [
                100.0 * v for v in cluster_bootstrap(groups, rel_ece, n_boot)
            ],
            'folds_beating_market_on_brier': int(
                sum(
                    per_fold[k][m]['brier'] < per_fold[k]['market']['brier']
                    for k in per_fold
                )
            ),
            'folds': len(per_fold),
        }
    learners = [lr.name for lr in LEARNERS]
    best = min(learners, key=lambda m: pooled[m]['brier'])
    best_ece = min(learners, key=lambda m: pooled[m]['ece'])
    best_servable = min(SERVABLE, key=lambda m: pooled[m]['brier'])
    tied = [
        m
        for m in SERVABLE
        if pooled[m]['brier'] <= pooled[best_servable]['brier'] * (1 + SERVE_TIE)
    ]
    served = next(m for m in SERVE_PREFERENCE if m in tied)
    candidates = [*learners, *ENSEMBLES]
    best_overall = min(candidates, key=lambda m: pooled[m]['brier'])
    sub = df.iloc[rows].reset_index(drop=True)
    sub['liquidity'] = liquidity_bucket(sub)
    breakdown: dict[str, dict[str, Any]] = {}
    for key in ('platform', 'category', 'liquidity'):
        breakdown[key] = {}
        for val, part in sub.groupby(key):
            idx = part.index.to_numpy()
            breakdown[key][str(val)] = {
                m: score(preds[m].to_numpy(float)[idx], y[idx])
                for m in dict.fromkeys(('market', 'wang', best, served))
            }
    return {
        'pooled': pooled,
        'per_fold': per_fold,
        'vs_market': vs_market,
        'best_by_brier': best,
        'best_by_ece': best_ece,
        'best_servable_by_brier': best_servable,
        'served': served,
        'served_rule': f'best servable learner, or the first of {list(SERVE_PREFERENCE)} '
        f'within {100 * SERVE_TIE:.1f}% of its Brier score',
        'best_overall_by_brier': best_overall,
        'any_learner_beats_market_on_brier': any(
            pooled[m]['brier'] < pooled['market']['brier'] for m in learners
        ),
        'reliability': {
            m: reliability(preds[m].to_numpy(float), y)
            for m in dict.fromkeys(('market', 'wang', best, served))
        },
        'breakdown': breakdown,
    }


def liquidity_bucket(df: pd.DataFrame) -> pd.Series:
    """Quote quality at prediction time: Kalshi by spread, Polymarket by whether the price moved in the last day."""
    spread = df['spread'].to_numpy(float)
    moved = df['hours_since_change'].to_numpy(float) <= 24.0
    kalshi = df['platform'].to_numpy() == 'kalshi'
    out = np.where(
        kalshi,
        np.where(
            spread <= 0.05,
            'kalshi: spread <= 5c',
            np.where(spread <= 0.20, 'kalshi: spread 5-20c', 'kalshi: spread > 20c'),
        ),
        np.where(
            moved,
            'polymarket: price moved in last 24h',
            'polymarket: no move in last 24h',
        ),
    )
    return pd.Series(out, index=df.index)


def dataset_summary(df: pd.DataFrame) -> dict[str, Any]:
    by_platform = df.groupby('platform').agg(
        snapshots=('label', 'size'),
        contracts=('market_id', 'nunique'),
        events=('event_id', 'nunique'),
        yes_rate=('label', 'mean'),
    )
    by_cat = df.groupby('category').agg(
        snapshots=('label', 'size'), contracts=('market_id', 'nunique')
    )
    return {
        'snapshots': int(len(df)),
        'contracts': int(df['market_id'].nunique()),
        'events': int(df['event_id'].nunique()),
        'by_platform': {
            k: {
                kk: (float(vv) if kk == 'yes_rate' else int(vv)) for kk, vv in v.items()
            }
            for k, v in by_platform.to_dict('index').items()
        },
        'by_category': {
            k: {kk: int(vv) for kk, vv in v.items()}
            for k, v in by_cat.to_dict('index').items()
        },
        'prediction_time_range': [_iso(df['t'].min()), _iso(df['t'].max())],
        'features': list(FEATURES),
        'kalshi_only_features': list(KALSHI_ONLY),
        'snapshot_fractions_of_scheduled_life': [0.25, 0.5, 0.75],
    }


def git_commit() -> str:
    try:
        return subprocess.run(
            ['git', 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:  # noqa: BLE001
        return 'unknown'


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument('--snapshots', required=True, type=Path)
    ap.add_argument(
        '--evidence',
        required=True,
        type=Path,
        help='directory for results.json and the report',
    )
    ap.add_argument(
        '--predictions',
        type=Path,
        help='where to write per-snapshot test predictions (parquet)',
    )
    ap.add_argument('--bundle', type=Path, help='directory for the served model bundle')
    ap.add_argument('--n-splits', type=int, default=5)
    ap.add_argument('--embargo-hours', type=float, default=24.0)
    ap.add_argument('--inner-blocks', type=int, default=3)
    ap.add_argument('--n-boot', type=int, default=1000)
    a = ap.parse_args(argv)
    t0 = time.time()
    df = pd.read_parquet(a.snapshots)
    preds, tuning, folds = run_cv(
        df, a.n_splits, a.embargo_hours * 3600.0, a.inner_blocks
    )
    summary = summarize(df, preds, a.n_boot)
    a.evidence.mkdir(parents=True, exist_ok=True)
    if a.predictions:
        a.predictions.parent.mkdir(parents=True, exist_ok=True)
        out = preds.copy()
        out['snapshot_id'] = df['snapshot_id'].to_numpy()[out['row'].to_numpy()]
        out['label'] = df['label'].to_numpy()[out['row'].to_numpy()]
        out.drop(columns=['row']).to_parquet(a.predictions, index=False)
    bundle_info = None
    if a.bundle:
        from oracle3.ml.serve import train_bundle

        bundle_info = train_bundle(
            df,
            summary['served']
            if summary['any_learner_beats_market_on_brier']
            else min(SERVABLE, key=lambda m: summary['pooled'][m]['ece']),
            a.bundle,
            a.inner_blocks,
            a.embargo_hours * 3600.0,
            summary,
        )
    results = {
        'generated_at': _iso(time.time()),
        'git_commit': git_commit(),
        'snapshots_sha256': hashlib.sha256(a.snapshots.read_bytes()).hexdigest(),
        'design': {
            'cv': 'event-grouped purged walk-forward, expanding window',
            'n_splits': a.n_splits,
            'embargo_hours': a.embargo_hours,
            'inner_blocks': a.inner_blocks,
            'tuning_metric': 'out-of-fold Brier over the inner walk-forward blocks',
            'calibration': 'isotonic, fitted on the inner out-of-fold predictions',
            'bootstrap': f'{a.n_boot} event-cluster resamples, 95% percentile intervals',
        },
        'dataset': dataset_summary(df),
        'folds': folds,
        'results': summary,
        'tuning_log': tuning,
        'served_bundle': bundle_info,
        'runtime_seconds': round(time.time() - t0, 1),
    }
    (a.evidence / 'results.json').write_text(json.dumps(_jsonable(results), indent=2))
    from oracle3.ml.report import render

    (a.evidence / 'REPORT.md').write_text(
        render(json.loads((a.evidence / 'results.json').read_text()))
    )
    print(
        json.dumps(
            {
                'best_by_brier': summary['best_by_brier'],
                'best_by_ece': summary['best_by_ece'],
                'beats_market': summary['any_learner_beats_market_on_brier'],
                'market': summary['pooled']['market'],
                'best': summary['pooled'][summary['best_by_brier']],
                'vs_market_best': summary['vs_market'][summary['best_by_brier']],
            },
            indent=2,
        )
    )


if __name__ == '__main__':
    main()
