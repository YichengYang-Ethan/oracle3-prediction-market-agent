"""Render ``results.json`` from :mod:`oracle3.ml.benchmark` as a Markdown report.

python -m oracle3.ml.report evidence/ml_calibration/results.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

LEARNER_LABELS = {
    'logistic': 'Penalized logistic',
    'spline_logistic': 'Spline GAM (penalized logistic on spline bases)',
    'random_forest': 'Random forest',
    'lightgbm': 'LightGBM',
    'lightgbm_monotone': 'LightGBM, monotone in price',
    'xgboost': 'XGBoost',
    'catboost': 'CatBoost',
    'mlp': 'MLP',
}
LABELS = {
    'market': 'Market price',
    'market_isotonic': 'Market price + isotonic (price-only recalibration)',
    'wang': 'Wang (paper), λ per venue',
    'wang_hier': 'Wang (paper), hierarchical λ',
    **LEARNER_LABELS,  # recalibration (isotonic or none) chosen per fold on training data
    **{f'{k}_raw': f'{v}, no recalibration' for k, v in LEARNER_LABELS.items()},
    **{f'{k}_isotonic': f'{v}, always isotonic' for k, v in LEARNER_LABELS.items()},
    'ensemble_mean': 'Ensemble: mean of the eight learners',
    'stacked': 'Ensemble: logistic stack of the eight learners',
}
ORDER = (
    'market',
    'market_isotonic',
    'wang',
    'wang_hier',
    *LEARNER_LABELS,
    'ensemble_mean',
    'stacked',
)


def _pct(x: float) -> str:
    return f'{x:+.1f}%'


def _ci(ci: list[float]) -> str:
    return f'[{ci[0]:+.1f}%, {ci[1]:+.1f}%]'


def _data(res: dict[str, Any]) -> list[str]:
    d = res['dataset']
    plat = d['by_platform']
    lines = [
        '## Data',
        '',
        f'{d["contracts"]:,} resolved contracts ({plat.get("kalshi", {}).get("contracts", 0):,} Kalshi, '
        f'{plat.get("polymarket", {}).get("contracts", 0):,} Polymarket) in {d["events"]:,} events, '
        f"{d['snapshots']:,} prediction snapshots at 25%, 50% and 75% of each contract's scheduled life "
        f'(kept only while the market still traded). Prediction times span {d["prediction_time_range"][0]} '
        f'to {d["prediction_time_range"][1]}.',
        '',
        '| Venue | Contracts | Events | Snapshots | YES rate |',
        '|---|---:|---:|---:|---:|',
    ]
    for k, v in plat.items():
        lines.append(
            f'| {k} | {v["contracts"]:,} | {v["events"]:,} | {v["snapshots"]:,} | {v["yes_rate"]:.3f} |'
        )
    lines += [
        '',
        'Features (all observed at or before the prediction time): '
        + ', '.join(f'`{f}`' for f in d['features'])
        + '. Spread and volume features exist only for Kalshi; Polymarket rows carry them as missing.',
        '',
    ]
    return lines


def _design(res: dict[str, Any]) -> list[str]:
    des = res['design']
    lines = [
        '## Validation design',
        '',
        f"- {des['cv']}: {des['n_splits']} folds. An event's contracts are never split between training and "
        f'test; a training event must have resolved at least {des["embargo_hours"]:.0f} hours before the test '
        'block begins (later-resolving events are purged).',
        f'- Inside each training window, {des["inner_blocks"]} purged inner walk-forward blocks are each '
        'predicted by models fitted only on earlier training events. Hyperparameters are chosen by '
        f'{des["tuning_metric"]}; the three boosted-tree libraries early-stop on each inner block.',
        '- The ensembles combine the eight learners: the mean of their recalibrated probabilities, and a '
        'logistic regression on their log-odds fitted to the same inner out-of-fold predictions.',
        f'- {des["calibration"]}; the chosen configuration is then refitted on the full training window.',
        f'- Intervals: {des["bootstrap"]}.',
        '',
        '| Fold | Test window (prediction time) | Train events | Test events | Test contracts | Test snapshots | Purged |',
        '|---:|---|---:|---:|---:|---:|---:|',
    ]
    for f in res['folds']:
        lines.append(
            f'| {f["fold"]} | {f["test_start"][:10]} – {f["test_end"][:10]} | {f["train_events"]:,} | '
            f'{f["test_events"]:,} | {f["test_contracts"]:,} | {f["test_rows"]:,} | {f["purged_events"]:,} |'
        )
    return lines


def _scores(r: dict[str, Any]) -> list[str]:
    pooled, vs = r['pooled'], r['vs_market']
    lines = [
        '',
        '## Out-of-sample scores (all test folds pooled)',
        '',
        '| Method | Brier | Log loss | ECE (10 bins) | Brier vs market | 95% CI | ECE vs market | 95% CI | '
        'Folds better on Brier |',
        '|---|---:|---:|---:|---:|---|---:|---|---:|',
    ]
    for m in ORDER:
        if m not in pooled:
            continue
        s = pooled[m]
        if m == 'market':
            lines.append(
                f'| {LABELS[m]} | {s["brier"]:.5f} | {s["log_loss"]:.4f} | {s["ece"]:.4f} | — | — | — | — | — |'
            )
            continue
        v = vs[m]
        lines.append(
            f'| {LABELS[m]} | {s["brier"]:.5f} | {s["log_loss"]:.4f} | {s["ece"]:.4f} | '
            f'{_pct(v["brier_reduction_pct"])} | {_ci(v["brier_reduction_pct_ci95"])} | '
            f'{_pct(v["ece_reduction_pct"])} | {_ci(v["ece_reduction_pct_ci95"])} | '
            f'{v["folds_beating_market_on_brier"]}/{v["folds"]} |'
        )
    lines += [
        '',
        'Positive percentages are reductions relative to the market price (lower Brier and ECE are better). '
        "Each learner's recalibration (isotonic or none) is chosen per fold on training data; both variants "
        'are in the recalibration table below.',
        '',
    ]
    best, best_ece = r['best_by_brier'], r['best_by_ece']
    served = r.get('best_servable_by_brier', best)
    overall = r.get('best_overall_by_brier', best)
    if r['any_learner_beats_market_on_brier']:
        for title, m in dict.fromkeys(
            (
                ('Best single learner', best),
                ('Served by `model_probability`', served),
                ('Best of all methods', overall),
            )
        ):
            if title != 'Best single learner' and m == best:
                continue
            v = vs[m]
            lines += [
                f'**{title}: {LABELS[m]}.** Out-of-sample Brier {pooled[m]["brier"]:.5f} vs '
                f'{pooled["market"]["brier"]:.5f} for market prices ({_pct(v["brier_reduction_pct"])}, 95% CI '
                f'{_ci(v["brier_reduction_pct_ci95"])}); ECE {pooled[m]["ece"]:.4f} vs '
                f'{pooled["market"]["ece"]:.4f} ({_pct(v["ece_reduction_pct"])}, 95% CI '
                f'{_ci(v["ece_reduction_pct_ci95"])}).',
                '',
            ]
    else:
        v = vs[best_ece]
        lines += [
            f'**No learner beat market prices on Brier score.** The largest calibration gain came from '
            f'{LABELS[best_ece]}: ECE {pooled[best_ece]["ece"]:.4f} vs {pooled["market"]["ece"]:.4f} '
            f'({_pct(v["ece_reduction_pct"])}, 95% CI {_ci(v["ece_reduction_pct_ci95"])}).',
            '',
        ]
    cols = [
        m
        for m in dict.fromkeys(
            ('market', 'market_isotonic', 'wang_hier', best, served, overall)
        )
        if m in pooled
    ]
    lines += [
        '## Per fold (Brier)',
        '',
        '| Fold | ' + ' | '.join(LABELS[m] for m in cols) + ' |',
        '|---:|' + '---:|' * len(cols),
    ]
    for k, s in sorted(r['per_fold'].items(), key=lambda kv: int(kv[0])):
        lines.append(
            f'| {k} | ' + ' | '.join(f'{s[m]["brier"]:.5f}' for m in cols) + ' |'
        )
    return lines


def _recalibration(res: dict[str, Any]) -> list[str]:
    pooled = res['results']['pooled']
    learners = [k for k in LEARNER_LABELS if f'{k}_isotonic' in pooled]
    if not learners:
        return []
    lines = [
        '',
        '## Recalibration check (Brier, test folds)',
        '',
        'Isotonic recalibration is kept for a learner in a fold only if, fitted on the earlier inner blocks of '
        'the training window, it lowers the Brier score of the later ones.',
        '',
        '| Learner | No recalibration | Always isotonic | As chosen on training data | Folds that kept isotonic |',
        '|---|---:|---:|---:|---:|',
    ]
    for k in learners:
        kept = sum(
            f['chosen'].get(k, {}).get('recalibration') == 'isotonic'
            for f in res['folds']
        )
        lines.append(
            f'| {LEARNER_LABELS[k]} | {pooled[f"{k}_raw"]["brier"]:.5f} | '
            f'{pooled[f"{k}_isotonic"]["brier"]:.5f} | {pooled[k]["brier"]:.5f} | '
            f'{kept}/{len(res["folds"])} |'
        )
    return lines


def _breakdowns(r: dict[str, Any]) -> list[str]:
    best = r.get('best_servable_by_brier', r['best_by_brier'])  # the served model
    lines: list[str] = []
    for key in ('platform', 'category', 'liquidity'):
        if key not in r['breakdown']:
            continue
        lines += [
            '',
            f'## By {key} (test folds)',
            '',
            f'| {key.title()} | Snapshots | Market Brier | Wang Brier | {LABELS[best]} Brier | Market ECE | '
            f'{LABELS[best]} ECE |',
            '|---|---:|---:|---:|---:|---:|---:|',
        ]
        for val, s in sorted(
            r['breakdown'][key].items(), key=lambda kv: -kv[1]['market']['n']
        ):
            lines.append(
                f'| {val} | {s["market"]["n"]:,} | {s["market"]["brier"]:.5f} | {s["wang"]["brier"]:.5f} | '
                f'{s[best]["brier"]:.5f} | {s["market"]["ece"]:.4f} | {s[best]["ece"]:.4f} |'
            )
    lines += [
        '',
        '## Reliability (test folds, 10 equal-width bins)',
        '',
        f'| Bin | Market: mean price | outcome rate | n | {LABELS[best]}: mean prediction | outcome rate | n |',
        '|---|---:|---:|---:|---:|---:|---:|',
    ]
    mk = {b['bin']: b for b in r['reliability']['market']}
    bl = {b['bin']: b for b in r['reliability'][best]}
    for b in range(10):
        a, c = mk.get(b), bl.get(b)
        cells = [
            f'{a["mean_prediction"]:.3f} | {a["outcome_rate"]:.3f} | {a["n"]:,}'
            if a
            else '— | — | 0',
            f'{c["mean_prediction"]:.3f} | {c["outcome_rate"]:.3f} | {c["n"]:,}'
            if c
            else '— | — | 0',
        ]
        lines.append(f'| {b / 10:.1f}–{(b + 1) / 10:.1f} | ' + ' | '.join(cells) + ' |')
    return lines


LIMITS = [
    '## Limits',
    '',
    '- Contracts were sampled per series (evenly over the window) and capped per category and per event, so the mix is '
    "not the venues' volume mix; scores are for this sample.",
    '- Polymarket history is only available at 3-hour grain for older markets, so Polymarket contracts scheduled to '
    'live less than a day are excluded, and its spread and volume history are not public.',
    '- Snapshots of one event (and of one contract) are correlated; intervals resample whole events.',
    "- Kalshi contracts come from the venue's public event listings, which leave out events from before its move to "
    '`KX` tickers, and most Kalshi series began in late 2025 or 2026; the Kalshi part of the sample is therefore '
    'concentrated in 2026, while the Polymarket part spans the whole window.',
    '- Scheduled end dates are read as the venues report them after resolution; a date that was revised while the '
    'market traded would place that snapshot using the revised date.',
    '- A lower Brier score than the market price is a statement about calibration on resolved contracts, not about '
    'tradable profit after fees, spreads and limited depth.',
    '',
]


def _ablation(ab: dict[str, Any] | None) -> list[str]:
    if not ab:
        return []
    names = {
        'price_only': 'Price only (price, logit, extremity)',
        'price_venue_category': '+ venue and category',
        'all_features': '+ time, price-path and liquidity features (all)',
    }
    lines = [
        '',
        '## Ablation: LightGBM on nested feature sets (same folds and procedure)',
        '',
        '| Features | Brier vs market (no recalibration) | 95% CI | Brier vs market (isotonic) | 95% CI | '
        'ECE vs market (no recalibration) | ECE vs market (isotonic) |',
        '|---|---:|---|---:|---|---:|---:|',
    ]
    for key, label in names.items():
        raw, iso = ab.get(f'{key}/raw'), ab.get(f'{key}/isotonic')
        if not raw or not iso:
            continue
        lines.append(
            f'| {label} | {_pct(raw["brier_reduction_pct"])} | {_ci(raw["brier_reduction_pct_ci95"])} | '
            f'{_pct(iso["brier_reduction_pct"])} | {_ci(iso["brier_reduction_pct_ci95"])} | '
            f'{_pct(raw["ece_reduction_pct"])} | {_pct(iso["ece_reduction_pct"])} |'
        )
    lines += [
        '',
        'Recalibrating the price alone recovers part of the gain; venue, category, timing, price-path and '
        'liquidity features add the rest. The isotonic step, fitted on inner out-of-fold predictions of models '
        'trained on less data, lowered the scores slightly in every row.',
        '',
    ]
    return lines


def render(res: dict[str, Any], ablation: dict[str, Any] | None = None) -> str:
    lines = [
        '# Calibrated probabilities for Kalshi and Polymarket contracts: out-of-sample benchmark',
        '',
        f'Generated {res["generated_at"]} from commit `{res["git_commit"][:10]}`; snapshot table '
        f'SHA-256 `{res["snapshots_sha256"][:16]}…`. Every number below is computed on test folds only.',
        '',
    ]
    lines += (
        _data(res)
        + _design(res)
        + _scores(res['results'])
        + _recalibration(res)
        + _breakdowns(res['results'])
        + _ablation(ablation)
    )
    if res.get('served_bundle'):
        sb = res['served_bundle']
        lines += [
            '',
            '## Served model',
            '',
            f'`model_probability` (MCP tool) serves {LABELS[sb["model"]]}, the best learner whose library is '
            'in the base `oracle3[ml]` install, refitted on all snapshots with the same inner tuning and '
            f'recalibration check (recalibration: {sb.get("recalibration", "isotonic")}); parameters '
            f'`{json.dumps(sb["params"])}`. File hashes are recorded in its manifest and checked on load.',
            '',
        ]
    return '\n'.join(lines + [''] + LIMITS)


def main(argv: list[str] | None = None) -> None:
    path = Path((argv or sys.argv[1:])[0])
    res = json.loads(path.read_text())
    ab_path = path.with_name('ablation.json')
    ablation = json.loads(ab_path.read_text()) if ab_path.exists() else None
    out = path.with_name('REPORT.md')
    out.write_text(render(res, ablation))
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
