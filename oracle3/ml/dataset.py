"""Turn the raw collection cache into one table of prediction snapshots.

Each row is one contract at one prediction time (25%, 50% or 75% of its
scheduled life, kept only while the market still traded), with the features of
:mod:`oracle3.ml.features`, the realized outcome and the bookkeeping needed for
event-grouped, purged cross-validation:

    platform, market_id, event_id, series, category, t (prediction time),
    resolve_ts (when the outcome became known), label

    python -m oracle3.ml.dataset --cache DIR --out snapshots.parquet
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from oracle3.ml.collect import iso_ts
from oracle3.ml.features import Bars, snapshot_features


def kalshi_rows(record: dict[str, Any]) -> list[dict[str, Any]]:
    m, e = record['market'], record['event']
    if m.get('result') not in ('yes', 'no') or not record.get('bars'):
        return []
    open_ts = iso_ts(m.get('open_time'))
    sched_end = iso_ts(m.get('expected_expiration_time')) or iso_ts(m.get('close_time'))
    resolve = iso_ts(m.get('settlement_ts')) or iso_ts(m.get('close_time'))
    if not (open_ts and sched_end and resolve):
        return []
    rows = record['bars']
    bars = Bars.from_rows(
        ts=[r[0] for r in rows],
        bid=[r[1] for r in rows],
        ask=[r[2] for r in rows],
        last=[r[3] if r[3] is not None else r[4] for r in rows],
        volume=[r[5] for r in rows],
    )
    out = []
    for t in record.get('snapshot_times') or []:
        f = snapshot_features(
            bars, open_ts, sched_end, t, 'kalshi', e.get('category', 'other')
        )
        if f is None:
            continue
        out.append(
            {
                'platform': 'kalshi',
                'market_id': m['ticker'],
                'event_id': f'K:{m["event_ticker"]}',
                'series': e.get('series_ticker') or m['ticker'].split('-', 1)[0],
                'category': e.get('category', 'other'),
                'title': m.get('title') or e.get('title'),
                't': t,
                'open_ts': open_ts,
                'sched_end': sched_end,
                'resolve_ts': resolve,
                'label': 1 if m['result'] == 'yes' else 0,
                'bar_seconds': 60 * (record.get('period') or 0),
                **f,
            }
        )
    return out


def polymarket_rows(record: dict[str, Any]) -> list[dict[str, Any]]:
    m, e = record['market'], record['event']
    points = [
        p for p in record.get('points') or [] if p[2] > 0
    ]  # drop terminal and settlement ticks
    if not points:
        return []
    open_ts = iso_ts(m.get('startDate')) or iso_ts(m.get('createdAt'))
    sched_end = iso_ts(m.get('endDate'))
    resolve = iso_ts(m.get('closedTime')) or sched_end
    if not (open_ts and sched_end and resolve):
        return []
    bars = Bars.from_prices(
        ts=[p[0] + p[2] for p in points], prices=[p[1] for p in points]
    )
    out = []
    for t in record.get('snapshot_times') or []:
        f = snapshot_features(
            bars, open_ts, sched_end, t, 'polymarket', record.get('category', 'other')
        )
        if f is None:
            continue
        out.append(
            {
                'platform': 'polymarket',
                'market_id': str(m['id']),
                'event_id': f'P:{e["id"]}',
                'series': e.get('series') or f'P:{e["id"]}',
                'category': record.get('category', 'other'),
                'title': m.get('question'),
                't': t,
                'open_ts': open_ts,
                'sched_end': sched_end,
                'resolve_ts': resolve,
                'label': int(m['label']),
                'bar_seconds': int(record.get('bucket') or 0),
                **f,
            }
        )
    return out


def build(cache: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in sorted((cache / 'kalshi' / 'candles').glob('*.json')):
        rows += kalshi_rows(json.loads(path.read_text()))
    for path in sorted((cache / 'polymarket' / 'history').glob('*.json')):
        rows += polymarket_rows(json.loads(path.read_text()))
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.sort_values(['t', 'market_id']).reset_index(drop=True)
    df['snapshot_id'] = (
        df['market_id'] + '@' + df['t'].round().astype('int64').astype(str)
    )
    # A prediction made after the outcome was known is not a forecast.
    df = df[df['t'] < df['resolve_ts']].reset_index(drop=True)
    for c in ('label', 'is_kalshi'):
        df[c] = df[c].astype(np.int8 if c == 'label' else float)
    return df


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument('--cache', required=True, type=Path)
    ap.add_argument('--out', required=True, type=Path)
    a = ap.parse_args(argv)
    df = build(a.cache)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(a.out, index=False)
    print(
        f'{len(df):,} snapshots from {df["market_id"].nunique():,} contracts in {df["event_id"].nunique():,} events'
    )
    print(
        df.groupby('platform')
        .agg(
            snapshots=('label', 'size'),
            contracts=('market_id', 'nunique'),
            yes_rate=('label', 'mean'),
        )
        .to_string()
    )


if __name__ == '__main__':
    main()
