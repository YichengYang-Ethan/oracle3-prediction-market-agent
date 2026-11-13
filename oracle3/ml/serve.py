"""The served model: training the bundle, loading it, and scoring a live market.

A bundle is a directory with ``manifest.json`` (features, hyperparameters, data range,
out-of-sample scores and file hashes), the fitted learner (``model.txt`` for LightGBM,
``model.joblib`` otherwise) and ``isotonic.json``. Only load bundles you produced or
that ship with this package: a joblib file is a pickle.

Live scoring rebuilds the training features at the current time from the venue's
public API: the same bar width as in training, the same displayed-price rule, the
scheduled end date and the same category scheme.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from oracle3.ml.collect import MAX_BARS, POLY_BUCKET_S, bar_period, iso_ts
from oracle3.ml.cv import inner_walk_forward
from oracle3.ml.features import FEATURES, Bars, displayed_prices, snapshot_features
from oracle3.ml.taxonomy import kalshi_category, polymarket_category

DEFAULT_BUNDLE = Path(__file__).with_name('bundle')


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _version(pkg: str) -> str:
    try:
        return metadata.version(pkg)
    except metadata.PackageNotFoundError:
        return 'not installed'


def train_bundle(
    df: pd.DataFrame,
    model_name: str,
    out: Path,
    inner_blocks: int,
    embargo: float,
    cv_summary: dict[str, Any],
) -> dict[str, Any]:
    """Fit ``model_name`` on all snapshots the way each CV fold was fitted, and save it."""
    from oracle3.ml.benchmark import fit_isotonic, tune
    from oracle3.ml.models import LEARNERS, design, fit_learner

    learner = next(lr for lr in LEARNERS if lr.name == model_name)
    X, y = design(df), df['label'].to_numpy(int)
    t, resolve, groups = (
        df['t'].to_numpy(float),
        df['resolve_ts'].to_numpy(float),
        df['event_id'].to_numpy(),
    )
    pairs = inner_walk_forward(
        t, resolve, groups, np.arange(len(df)), inner_blocks, embargo
    )
    log: list[dict[str, Any]] = []
    params, oof, oof_rows = tune(learner, X, y, pairs, log, fold=0)
    iso = fit_isotonic(oof, y[oof_rows])
    est, _ = fit_learner(learner, params, X, y)
    out.mkdir(parents=True, exist_ok=True)
    for old in ('model.txt', 'model.joblib'):
        (out / old).unlink(missing_ok=True)
    if model_name == 'lightgbm':
        model_file = out / 'model.txt'
        est.booster_.save_model(str(model_file))
    else:
        import joblib

        model_file = out / 'model.joblib'
        joblib.dump(est, model_file, compress=3)
    iso_file = out / 'isotonic.json'
    iso_file.write_text(
        json.dumps(
            {
                'x': [float(v) for v in iso.X_thresholds_],
                'y': [float(v) for v in iso.y_thresholds_],
            }
        )
    )
    pooled = cv_summary['pooled']
    manifest = {
        'name': 'oracle3-calibrated-probability',
        'version': 1,
        'model': model_name,
        'params': params,
        'features': list(FEATURES),
        'trained_on': {
            'snapshots': int(len(df)),
            'contracts': int(df['market_id'].nunique()),
            'events': int(df['event_id'].nunique()),
            'contracts_by_platform': {
                k: int(v)
                for k, v in df.groupby('platform')['market_id'].nunique().items()
            },
            'prediction_times': [
                pd.Timestamp(df['t'].min(), unit='s', tz='UTC').isoformat(),
                pd.Timestamp(df['t'].max(), unit='s', tz='UTC').isoformat(),
            ],
        },
        'out_of_sample': {
            'model': pooled[model_name],
            'market': pooled['market'],
            'wang': pooled['wang'],
            'vs_market': cv_summary['vs_market'][model_name],
        },
        'files': {
            model_file.name: _sha256(model_file),
            iso_file.name: _sha256(iso_file),
        },
        'libraries': {p: _version(p) for p in ('scikit-learn', 'lightgbm', 'numpy')},
        'created_at': pd.Timestamp.now(tz='UTC').isoformat(),
    }
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2, default=str))
    return {
        'dir': str(out),
        'model': model_name,
        'params': params,
        'files': manifest['files'],
    }


@dataclass
class Bundle:
    manifest: dict[str, Any]
    model: Any
    iso_x: np.ndarray
    iso_y: np.ndarray

    def predict(self, row: dict[str, float]) -> tuple[float, float]:
        """(calibrated probability, uncalibrated learner output) for one feature row."""
        X = pd.DataFrame(
            [[row.get(f, float('nan')) for f in FEATURES]],
            columns=list(FEATURES),
            dtype=float,
        )
        if self.manifest['model'] == 'lightgbm':
            raw = float(self.model.predict(X.to_numpy())[0])
        else:
            raw = float(self.model.predict_proba(X)[0, 1])
        cal = float(np.interp(raw, self.iso_x, self.iso_y))
        return min(max(cal, 1e-4), 1 - 1e-4), raw


_cache: dict[str, Bundle] = {}


def load_bundle(path: str | Path | None = None) -> Bundle:
    """Load (once) the bundle at ``path``, ``$ORACLE3_ML_BUNDLE`` or the one shipped with the package."""
    root = Path(path or os.environ.get('ORACLE3_ML_BUNDLE') or DEFAULT_BUNDLE)
    key = str(root.resolve())
    if key in _cache:
        return _cache[key]
    manifest_path = root / 'manifest.json'
    if not manifest_path.exists():
        raise FileNotFoundError(
            f'no model bundle at {root}; train one with python -m oracle3.ml.benchmark --bundle'
        )
    manifest = json.loads(manifest_path.read_text())
    for name, digest in manifest['files'].items():
        if _sha256(root / name) != digest:
            raise ValueError(
                f'{root / name} does not match the hash recorded in its manifest'
            )
    if manifest['model'] == 'lightgbm':
        import lightgbm as lgb

        model = lgb.Booster(model_file=str(root / 'model.txt'))
    else:
        import joblib

        model = joblib.load(root / 'model.joblib')
    iso = json.loads((root / 'isotonic.json').read_text())
    bundle = Bundle(
        manifest, model, np.asarray(iso['x'], float), np.asarray(iso['y'], float)
    )
    _cache[key] = bundle
    return bundle


# ── live features ────────────────────────────────────────────────────────


def _f(value: Any) -> float | None:
    try:
        return None if value in (None, '') else float(value)
    except (TypeError, ValueError):
        return None


def _display(bid: float | None, ask: float | None, last: float | None) -> float | None:
    out = displayed_prices(
        np.array([np.nan if bid is None else bid]),
        np.array([np.nan if ask is None else ask]),
        np.array([np.nan if last is None else last]),
    )[0]
    return float(out) if np.isfinite(out) else None


async def kalshi_live(
    client: Any, get: Any, base: str, ticker: str, now: float
) -> dict[str, Any]:
    m = (await get(client, f'{base}/markets/{ticker}')).get('market') or {}
    if not m:
        raise ValueError(f'unknown Kalshi market {ticker}')
    if m.get('status') not in ('active', 'open'):
        raise ValueError(
            f'Kalshi market {ticker} is {m.get("status")}; the model scores open markets only'
        )
    series = ticker.split('-', 1)[0]
    try:
        cat = kalshi_category(
            ((await get(client, f'{base}/series/{series}')).get('series') or {}).get(
                'category'
            )
        )
    except Exception:  # noqa: BLE001  (category is optional context; keep scoring)
        cat = 'other'
    open_ts = iso_ts(m.get('open_time'))
    sched_end = iso_ts(m.get('expected_expiration_time')) or iso_ts(m.get('close_time'))
    if not open_ts or not sched_end or sched_end <= open_ts:
        raise ValueError(
            f'Kalshi market {ticker} has no usable open and scheduled end times'
        )
    period = bar_period(open_ts, sched_end)
    lo = int(max(open_ts - 60 * period, now - MAX_BARS * 60 * period))
    candles = (
        await get(
            client,
            f'{base}/series/{series}/markets/{ticker}/candlesticks',
            {'start_ts': lo, 'end_ts': int(now), 'period_interval': period},
        )
    ).get('candlesticks') or []
    from oracle3.ml.collect import _kalshi_bar

    rows = [_kalshi_bar(c) for c in candles]
    rows = [r for r in rows if r[0] <= now]
    bars = Bars.from_rows(
        [r[0] for r in rows],
        [r[1] for r in rows],
        [r[2] for r in rows],
        [r[3] if r[3] is not None else r[4] for r in rows],
        [r[5] for r in rows],
    )
    bid = _f(m.get('yes_bid_dollars'))
    ask = _f(m.get('yes_ask_dollars'))
    last = _f(m.get('last_price_dollars'))
    price = _display(
        bid if bid and bid > 0 else None,
        ask if ask and ask < 1 else None,
        last if last else None,
    )
    row = snapshot_features(
        bars, open_ts, sched_end, now, 'kalshi', cat, current_price=price
    )
    if row is not None:
        vol = _f(m.get('volume_fp')) or _f(m.get('volume'))
        if vol is not None:
            row['log_volume_to_date'] = math.log1p(max(vol, 0.0))
        if bid is not None or ask is not None:
            row['spread'] = (ask if ask and ask < 1 else 1.0) - (
                bid if bid and bid > 0 else 0.0
            )
    return {
        'title': m.get('title'),
        'category': cat,
        'market_price': price,
        'features': row,
        'scheduled_end': m.get('expected_expiration_time'),
        'bar_seconds': 60 * period,
    }


async def polymarket_live(
    client: Any, get: Any, gamma: str, data_api: str, market_id: str, now: float
) -> dict[str, Any]:
    m = await get(client, f'{gamma}/markets/{market_id}', {'include_tag': 'true'})
    if not m:
        raise ValueError(f'unknown Polymarket market {market_id}')
    if m.get('closed') or not m.get('active', True):
        raise ValueError(
            f'Polymarket market {market_id} is closed; the model scores open markets only'
        )
    tokens = m.get('clobTokenIds')
    tokens = json.loads(tokens) if isinstance(tokens, str) else (tokens or [])
    if len(tokens) != 2:
        raise ValueError(f'Polymarket market {market_id} is not a two-outcome market')
    tags: list[str] = [
        str(t['label'])
        for t in m.get('tags') or []
        if isinstance(t, dict) and t.get('label')
    ]
    for ev in m.get('events') or []:
        try:
            full = await get(client, f'{gamma}/events/{ev["id"]}')
            tags += [t.get('label') for t in full.get('tags') or [] if t.get('label')]
        except Exception:  # noqa: BLE001
            pass
    cat = polymarket_category(tags, m.get('question') or '')
    open_ts = iso_ts(m.get('startDate')) or iso_ts(m.get('createdAt'))
    sched_end = iso_ts(m.get('endDate'))
    if not open_ts or not sched_end or sched_end <= open_ts:
        raise ValueError(
            f'Polymarket market {market_id} has no usable start and end dates'
        )
    hist = await get(
        client,
        f'{data_api}/v2/prices-history',
        {
            'token_id': tokens[0],
            'interval': 'max',
            'bucket_seconds': POLY_BUCKET_S,
            'limit': 10000,
        },
    )
    pts = [
        p for p in hist.get('data') or [] if int(p.get('resolution_seconds') or 0) > 0
    ]
    pts = [p for p in pts if int(p['timestamp']) + int(p['resolution_seconds']) <= now]
    bars = Bars.from_prices(
        [int(p['timestamp']) + int(p['resolution_seconds']) for p in pts],
        [float(p['price']) for p in pts],
    )
    bid, ask = _f(m.get('bestBid')), _f(m.get('bestAsk'))
    price = _display(bid, ask, _f(m.get('lastTradePrice')))
    row = snapshot_features(
        bars, open_ts, sched_end, now, 'polymarket', cat, current_price=price
    )
    return {
        'title': m.get('question'),
        'category': cat,
        'market_price': price,
        'features': row,
        'scheduled_end': m.get('endDate'),
        'bar_seconds': POLY_BUCKET_S,
    }


async def score_market(
    venue: str, market_id: str, bundle_path: str | Path | None = None
) -> dict[str, Any]:
    """Model probability for one open market, computed from its public data right now."""
    from oracle3.mcp_server import venues

    bundle = load_bundle(bundle_path)
    now = time.time()
    async with venues._client() as client:
        if venue == 'kalshi':
            live = await kalshi_live(
                client, venues._get, venues.KALSHI_API, market_id, now
            )
        else:
            live = await polymarket_live(
                client,
                venues._get,
                venues.GAMMA_API,
                'https://data-api.polymarket.com',
                market_id,
                now,
            )
    if live['features'] is None or live['market_price'] is None:
        raise ValueError(f'{venue} market {market_id} has no usable price yet')
    prob, raw = bundle.predict(live['features'])
    oos = bundle.manifest.get('out_of_sample', {})
    return {
        'venue': venue,
        'market_id': market_id,
        'title': live['title'],
        'category': live['category'],
        'model_probability': round(prob, 4),
        'market_price': round(live['market_price'], 4),
        'model_minus_market': round(prob - live['market_price'], 4),
        'uncalibrated_model_output': round(raw, 4),
        'as_of': pd.Timestamp(now, unit='s', tz='UTC').isoformat(),
        'scheduled_end': live['scheduled_end'],
        'model': bundle.manifest['model'],
        'trained_through': bundle.manifest['trained_on']['prediction_times'][1],
        'out_of_sample': {k: oos.get(k) for k in ('vs_market',)},
        'features': {
            k: (
                None
                if v is None or (isinstance(v, float) and math.isnan(v))
                else round(float(v), 6)
            )
            for k, v in live['features'].items()
        },
        'note': (
            'A calibrated probability from historical patterns in market prices, timing and liquidity. '
            'It is not a trading signal; check resolution rules and fees before acting.'
        ),
    }
