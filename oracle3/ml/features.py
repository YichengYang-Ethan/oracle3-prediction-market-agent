"""Leakage-safe features of one contract at one prediction time.

The same function serves training and live use. It receives a price history in
which every bar has been completely observed by its timestamp (a bar's
timestamp is the end of its interval) and reads only bars stamped at or before
the prediction time. Nothing about the outcome, the final price, the actual
closing time or later trading enters a feature.

Market price follows the convention venues display: the bid/ask midpoint when
the spread is at most 10 cents, otherwise the last traded price. A missing bid
counts as 0 and a missing ask as 1.

Time features use the *scheduled* end (Kalshi ``expected_expiration_time``,
Polymarket ``endDate``), which is known when the market opens; the actual
closing time can move earlier when the outcome happens early, so using it would
leak the outcome.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from oracle3.ml.taxonomy import CATEGORIES

WIDE_SPREAD = 0.10
PRICE_FLOOR = 1e-3

NUMERIC_FEATURES = (
    'price',
    'logit_price',
    'extremity',
    'log_hours_to_end',
    'log_life_hours',
    'elapsed_fraction',
    'dp_life05',
    'dp_life20',
    'dp_1h',
    'dp_24h',
    'vol_logit_life20',
    'hours_since_change',
    'spread',
    'log_volume_to_date',
    'log_volume_life10',
    'active_share_life10',
)
CATEGORY_FEATURES = tuple(f'cat_{c}' for c in CATEGORIES)
FEATURES = (*NUMERIC_FEATURES, 'is_kalshi', *CATEGORY_FEATURES)
# Observed only on Kalshi (Polymarket's public history has neither depth nor time-stamped volume).
KALSHI_ONLY = (
    'spread',
    'log_volume_to_date',
    'log_volume_life10',
    'active_share_life10',
)


@dataclass
class Bars:
    """A price history whose rows are complete by ``ts`` (seconds, ascending)."""

    ts: np.ndarray
    bid: np.ndarray  # best YES bid at bar end, NaN when unknown
    ask: np.ndarray  # best YES ask at bar end, NaN when unknown
    last: np.ndarray  # last traded price known at bar end, NaN when unknown
    volume: np.ndarray  # contracts traded within the bar, NaN when unknown
    price: np.ndarray  # displayed price at bar end (filled by ``displayed_prices``)

    @classmethod
    def from_rows(
        cls,
        ts: list[float],
        bid: list[Any],
        ask: list[Any],
        last: list[Any],
        volume: list[Any],
    ) -> Bars:
        def arr(values: list[Any]) -> np.ndarray:
            return np.array(
                [np.nan if v is None else float(v) for v in values], dtype=float
            )

        order = np.argsort(np.asarray(ts, dtype=float), kind='stable')
        bars = cls(
            np.asarray(ts, dtype=float)[order],
            arr(bid)[order],
            arr(ask)[order],
            arr(last)[order],
            arr(volume)[order],
            np.full(len(ts), np.nan),
        )
        bars.price = displayed_prices(bars.bid, bars.ask, bars.last)
        return bars

    @classmethod
    def from_prices(cls, ts: list[float], prices: list[float]) -> Bars:
        """A history that only carries a displayed price (Polymarket)."""
        n = len(ts)
        order = np.argsort(np.asarray(ts, dtype=float), kind='stable')
        nan = np.full(n, np.nan)
        return cls(
            np.asarray(ts, dtype=float)[order],
            nan.copy(),
            nan.copy(),
            nan.copy(),
            nan.copy(),
            np.asarray(prices, dtype=float)[order],
        )

    def __len__(self) -> int:
        return len(self.ts)


def displayed_prices(bid: np.ndarray, ask: np.ndarray, last: np.ndarray) -> np.ndarray:
    b = np.where(np.isfinite(bid) & (bid > 0), bid, 0.0)
    a = np.where(np.isfinite(ask) & (ask > 0) & (ask < 1), ask, 1.0)
    has_quote = (np.isfinite(bid) & (bid > 0)) | (
        np.isfinite(ask) & (ask > 0) & (ask < 1)
    )
    mid = (b + a) / 2.0
    wide = (a - b) > WIDE_SPREAD + 1e-12
    out = np.where(wide & np.isfinite(last), last, mid)
    return np.where(has_quote | np.isfinite(last), out, np.nan)


def _logit(p: float) -> float:
    q = min(max(p, PRICE_FLOOR), 1 - PRICE_FLOOR)
    return math.log(q / (1 - q))


def _price_at(bars: Bars, upto: int, when: float) -> float:
    """Displayed price of the last bar complete by ``when`` among the first ``upto`` bars."""
    j = int(np.searchsorted(bars.ts[:upto], when, side='right')) - 1
    if j < 0:
        return float('nan')
    p = bars.price[j]
    return float(p) if np.isfinite(p) else float('nan')


def snapshot_features(
    bars: Bars,
    open_ts: float,
    sched_end: float,
    t: float,
    platform: str,
    category: str,
    current_price: float | None = None,
) -> dict[str, float] | None:
    """Feature row at prediction time ``t``; None when no usable price is known by then.

    ``current_price`` (live use) replaces the last bar's displayed price with a
    fresher quote observed at ``t``.
    """
    if not (sched_end > open_ts) or t < open_ts:
        return None
    upto = int(np.searchsorted(bars.ts, t, side='right'))
    known = bars.price[:upto]
    finite = np.flatnonzero(np.isfinite(known))
    if current_price is None and len(finite) == 0:
        return None
    if current_price is not None:
        p = float(current_price)
        last_ts = t
    else:
        i = int(finite[-1])
        p = float(known[i])
        last_ts = float(bars.ts[i])
    if not (0.0 <= p <= 1.0):
        return None
    life = sched_end - open_ts
    life_h = life / 3600.0
    hours_to_end = max(0.0, (sched_end - t) / 3600.0)

    def dp(window: float) -> float:
        then = t - window
        if then < open_ts:
            return float('nan')
        q = _price_at(bars, upto, then)
        return p - q if math.isfinite(q) else float('nan')

    vol = _recent_volatility(bars, upto, t - 0.2 * life)
    since_change = _hours_since_change(bars, upto, t)
    if platform == 'kalshi' and upto > 0:
        volume_feats = _kalshi_book_features(bars, upto, t - 0.1 * life)
    else:
        volume_feats = (float('nan'),) * 4

    row = {
        'price': p,
        'logit_price': _logit(p),
        'extremity': abs(p - 0.5),
        'log_hours_to_end': math.log1p(hours_to_end),
        'log_life_hours': math.log1p(life_h),
        'elapsed_fraction': min(1.0, max(0.0, (t - open_ts) / life)),
        'dp_life05': dp(0.05 * life),
        'dp_life20': dp(0.20 * life),
        'dp_1h': dp(3600.0),
        'dp_24h': dp(86400.0),
        'vol_logit_life20': vol,
        'hours_since_change': since_change,
        'spread': volume_feats[0],
        'log_volume_to_date': volume_feats[1],
        'log_volume_life10': volume_feats[2],
        'active_share_life10': volume_feats[3],
        'is_kalshi': 1.0 if platform == 'kalshi' else 0.0,
        'staleness_hours': max(0.0, (t - last_ts) / 3600.0),
    }
    for c in CATEGORIES:
        row[f'cat_{c}'] = 1.0 if category == c else 0.0
    return row


def _recent_volatility(bars: Bars, upto: int, since: float) -> float:
    """Standard deviation of bar-to-bar changes in log-odds after ``since``."""
    rp = bars.price[:upto][bars.ts[:upto] > since]
    rp = rp[np.isfinite(rp)]
    if len(rp) < 3:
        return float('nan')
    q = np.clip(rp, PRICE_FLOOR, 1 - PRICE_FLOOR)
    return float(np.std(np.diff(np.log(q / (1 - q)))))


def _hours_since_change(bars: Bars, upto: int, t: float) -> float:
    known = bars.price[:upto]
    ok = np.isfinite(known)
    hist, ts_hist = known[ok], bars.ts[:upto][ok]
    if not len(hist):
        return float('nan')
    changed = np.flatnonzero(np.abs(np.diff(hist)) > 1e-9)
    ref = ts_hist[changed[-1] + 1] if len(changed) else ts_hist[0]
    return max(0.0, (t - float(ref)) / 3600.0)


def _kalshi_book_features(
    bars: Bars, upto: int, since: float
) -> tuple[float, float, float, float]:
    """Spread at the last bar, log volume to date, log recent volume, share of recent bars with trades."""
    b, a = bars.bid[upto - 1], bars.ask[upto - 1]
    bid = b if np.isfinite(b) and b > 0 else 0.0
    ask = a if np.isfinite(a) and 0 < a < 1 else 1.0
    vols = bars.volume[:upto]
    window = bars.ts[:upto] > since
    vol_to_date = float(np.nansum(vols))
    vol_recent = float(np.nansum(vols[window]))
    active = float(np.mean(np.nan_to_num(vols[window]) > 0)) if window.any() else 0.0
    return (
        float(ask - bid),
        math.log1p(max(vol_to_date, 0.0)),
        math.log1p(max(vol_recent, 0.0)),
        active,
    )
