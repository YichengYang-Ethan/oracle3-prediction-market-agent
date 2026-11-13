"""Collect resolved Kalshi and Polymarket contracts with their price histories.

The output is a raw cache that :mod:`oracle3.ml.dataset` turns into leakage-safe
prediction snapshots. Every request is a cached GET, so an interrupted run
resumes where it stopped.

Sampling is fixed before any price is looked at: events are drawn per series
evenly over the window, capped per category, and markets are drawn per event
with a seeded hash of the ticker. Lifetime volume, final prices and results are
not used to choose contracts.

    python -m oracle3.ml.collect kalshi --env-file PATH --out DIR
    python -m oracle3.ml.collect polymarket --out DIR
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import time
from collections import defaultdict
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from oracle3.ml.sources import (
    GammaReader,
    KalshiReader,
    PolyDataReader,
    SourceError,
    kalshi_signer_from_env,
)
from oracle3.ml.taxonomy import kalshi_category, polymarket_category

FRACTIONS = (0.25, 0.5, 0.75)  # snapshot points as fractions of the scheduled life
MINUTE_LIFE_H = (
    48.0  # contracts scheduled to live at most this long get one-minute bars
)
DAILY_LIFE_H = 24.0 * 120  # contracts scheduled to live longer than this get daily bars
MAX_BARS = 4900  # Kalshi caps one candlestick request at 5,000 bars
POLY_BUCKET_S = 10800  # Polymarket's finest permanent price-history grain
POLY_MIN_LIFE_S = (
    24 * 3600
)  # Polymarket contracts must be scheduled to live at least a day
# Auto-generated short-horizon series ("Up or Down", 5M, 15M, 1H, 4H) are thousands of events a week whose
# trading window is far shorter than the 3-hour price grain; leaving them out keeps the weekly listing complete.
POLY_EXCLUDED_TAGS = (102127, 102892, 102467, 102175, 102531)
WEEK_S = 7 * 86400
WEEK_GRID_ORIGIN = (
    2 * 86400
)  # 1970-01-03, a Saturday: listing weeks run Saturday to Saturday (UTC)


def iso_ts(value: Any) -> float | None:
    if not value:
        return None
    text = str(value).replace(' ', 'T')
    if text.endswith('+00'):
        text += ':00'
    try:
        dt = datetime.fromisoformat(text.replace('Z', '+00:00'))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


_MONTHS = (
    'JAN',
    'FEB',
    'MAR',
    'APR',
    'MAY',
    'JUN',
    'JUL',
    'AUG',
    'SEP',
    'OCT',
    'NOV',
    'DEC',
)
_TICKER_DATE = re.compile(r'(\d{2})(' + '|'.join(_MONTHS) + r')(\d{2})?')


def required_ts(value: Any) -> float:
    ts = iso_ts(value)
    if ts is None:
        raise ValueError(f'not an ISO-8601 time: {value!r}')
    return ts


def ticker_date(event_ticker: str) -> float | None:
    """Approximate date encoded in a Kalshi event ticker (26JUN13 -> 2026-06-13, 26JUL -> 2026-07-15).

    Used only to spread the sample over time when the event has no ``strike_date``.
    """
    suffix = event_ticker.split('-', 1)[1] if '-' in event_ticker else ''
    m = _TICKER_DATE.search(suffix)
    if not m:
        return None
    day = int(m.group(3)) if m.group(3) else 15
    try:
        dt = datetime(
            2000 + int(m.group(1)),
            _MONTHS.index(m.group(2)) + 1,
            min(max(day, 1), 28) if day > 28 else day,
            tzinfo=timezone.utc,
        )
    except ValueError:
        return None
    return dt.timestamp()


def event_time(event: dict) -> float | None:
    return iso_ts(event.get('strike_date')) or ticker_date(
        event.get('event_ticker') or ''
    )


def stable_hash(text: str, seed: int = 7) -> int:
    return int(hashlib.sha256(f'{seed}:{text}'.encode()).hexdigest()[:12], 16)


def spread_pick(items: list[Any], k: int, key: Callable[[Any], float]) -> list[Any]:
    """Up to ``k`` items spread evenly over the order given by ``key``."""
    items = sorted(items, key=key)
    if len(items) <= k:
        return items
    idx = (
        sorted({round(i * (len(items) - 1) / (k - 1)) for i in range(k)})
        if k > 1
        else [len(items) // 2]
    )
    return [items[i] for i in idx]


def snapshot_times(open_ts: float, sched_end: float, close_ts: float) -> list[float]:
    """Prediction times placed on the scheduled life, kept only while the market still traded."""
    if open_ts is None or sched_end is None or close_ts is None or sched_end <= open_ts:
        return []
    life = sched_end - open_ts
    return [open_ts + f * life for f in FRACTIONS if open_ts + f * life < close_ts]


def bar_period(open_ts: float, sched_end: float) -> int:
    life_h = (sched_end - open_ts) / 3600.0
    if life_h <= MINUTE_LIFE_H:
        return 1
    if life_h <= DAILY_LIFE_H:
        return 60
    return 1440


def read_json(path: Path) -> Any:
    if path.name.endswith('.gz'):
        with gzip.open(path, 'rt', encoding='utf-8') as fh:
            return json.load(fh)
    return json.loads(path.read_text())


class Cache:
    """Gzipped JSON files under one root; plain ``.json`` files written by older runs are still read."""

    def __init__(self, root: Path):
        self.root = root

    def _gz(self, parts: tuple[str, ...]) -> Path:
        p = self.root.joinpath(*parts)
        return p.with_name(p.name + '.gz')

    def has(self, *parts: str) -> bool:
        return self._gz(parts).exists() or self.root.joinpath(*parts).exists()

    def load(self, *parts: str) -> Any:
        for p in (self._gz(parts), self.root.joinpath(*parts)):
            if p.exists():
                return read_json(p)
        return None

    def save(self, obj: Any, *parts: str) -> None:
        gz = self._gz(parts)
        gz.parent.mkdir(parents=True, exist_ok=True)
        tmp = gz.with_name(gz.name + '.tmp')
        with gzip.open(tmp, 'wt', encoding='utf-8') as fh:
            json.dump(obj, fh, separators=(',', ':'))
        tmp.replace(gz)
        self.root.joinpath(*parts).unlink(missing_ok=True)  # superseded plain copy

    def files(self, *parts: str) -> list[Path]:
        folder = self.root.joinpath(*parts)
        return sorted([*folder.glob('*.json'), *folder.glob('*.json.gz')])


def window_tag(start: float, end: float, *caps: tuple[str, int]) -> str:
    """Name for a sample drawn with these settings, so samples drawn with other settings sit beside it."""

    def day(t: float) -> str:
        return datetime.fromtimestamp(t, timezone.utc).strftime('%Y%m%d')

    return '_'.join([f'{day(start)}-{day(end)}', *(f'{k}{v}' for k, v in caps)])


def run_pool(
    fn: Callable[[Any], Any],
    items: Iterable[Any],
    workers: int,
    label: str,
    every: int = 500,
) -> list[Any]:
    items = list(items)
    out: list[Any] = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, res in enumerate(pool.map(fn, items), 1):
            out.append(res)
            if i % every == 0 or i == len(items):
                print(f'[{label}] {i}/{len(items)} {time.time() - t0:.0f}s', flush=True)
    return out


# ── Kalshi ───────────────────────────────────────────────────────────────


def _kalshi_bar(c: dict[str, Any]) -> list[Any]:
    def close_of(name: str) -> float | None:
        block = c.get(name) or {}
        for key in ('close_dollars', 'close'):
            v = block.get(key)
            if v is not None and v != '':
                return float(v)
        return None

    def prev_price() -> float | None:
        block = c.get('price') or {}
        for key in ('previous_dollars', 'previous'):
            v = block.get(key)
            if v is not None and v != '':
                return float(v)
        return None

    vol = c.get('volume_fp', c.get('volume'))
    return [
        int(c['end_period_ts']),
        close_of('yes_bid'),
        close_of('yes_ask'),
        close_of('price'),
        prev_price(),
        float(vol) if vol not in (None, '') else 0.0,
    ]


class KalshiCollector:
    def __init__(
        self,
        out: Path,
        env_file: str | None,
        start: str,
        end: str,
        per_series: int,
        per_category: int,
        per_event: int,
        workers: int,
        rate: float | None,
        max_pages: int,
    ):
        self.cache = Cache(out / 'kalshi')
        self.reader = KalshiReader(kalshi_signer_from_env(env_file), rate)
        self.start, self.end = required_ts(start), required_ts(end)
        self.per_series, self.per_category, self.per_event = (
            per_series,
            per_category,
            per_event,
        )
        self.workers, self.max_pages = workers, max_pages
        print(
            f'[kalshi] signed={self.reader.signer is not None} rate={1 / self.reader.limiter.interval:.1f}/s',
            flush=True,
        )

    def series(self) -> list[dict]:
        cached = self.cache.load('series.json')
        if cached is None:
            cached = self.reader.get('/series').get('series') or []
            self.cache.save(cached, 'series.json')
        return cached

    def tag(self) -> str:
        return window_tag(
            self.start,
            self.end,
            ('s', self.per_series),
            ('c', self.per_category),
            ('e', self.per_event),
        )

    def events_of(self, series: dict) -> list[dict]:
        """Settled events of one series, listed newest first until the listing passes the window start.

        The cache notes whether the listing ran out (``complete``) and the oldest event it reached, so a
        later run with an earlier start lists the series again only when it has to.
        """
        name = series['ticker']
        cached = self.cache.load('events', f'{name}.json')
        if isinstance(cached, dict) and (
            cached['complete']
            or (cached['oldest'] is not None and cached['oldest'] < self.start)
        ):
            return cached['events']
        keep = (
            'event_ticker',
            'series_ticker',
            'title',
            'sub_title',
            'strike_date',
            'mutually_exclusive',
            'category',
        )
        events: list[dict] = []
        oldest: float | None = None
        pages, passed_start = 0, False
        try:
            for page in self.reader.paginate(
                '/events',
                'events',
                max_pages=self.max_pages,
                series_ticker=name,
                status='settled',
                limit=200,
            ):
                pages += 1
                events += [{k: e.get(k) for k in keep} for e in page]
                dates = [d for d in (event_time(e) for e in page) if d]
                if dates:
                    oldest = min([*dates, *([oldest] if oldest is not None else [])])
                if oldest is not None and oldest < self.start:
                    passed_start = True
                    break
        except SourceError as exc:
            print(f'[kalshi] events {name}: {exc}', flush=True)
            return events  # not cached, so a later run retries this series
        complete = not passed_start and pages < self.max_pages
        self.cache.save(
            {'complete': complete, 'oldest': oldest, 'events': events},
            'events',
            f'{name}.json',
        )
        return events

    def sample_events(self, series_list: list[dict]) -> list[dict]:
        name = f'sample_events_{self.tag()}.json'
        cached = self.cache.load(name)
        if cached is not None:
            return cached
        by_cat: dict[str, list[dict]] = defaultdict(list)
        for s in series_list:
            listing = self.cache.load('events', f'{s["ticker"]}.json') or {}
            events = listing.get('events') or []
            in_window = [
                e for e in events if self.start <= (event_time(e) or 0.0) < self.end
            ]
            picked = spread_pick(
                in_window, self.per_series, key=lambda e: event_time(e) or 0.0
            )
            cat = kalshi_category(s.get('category'), s.get('ticker'))
            for e in picked:
                e = dict(
                    e,
                    series_category=s.get('category'),
                    category=cat,
                    series_title=s.get('title'),
                )
                by_cat[cat].append(e)
        sample: list[dict] = []
        for cat, events in sorted(by_cat.items()):
            if len(events) > self.per_category:
                events = sorted(events, key=lambda e: stable_hash(e['event_ticker']))[
                    : self.per_category
                ]
            sample += events
            print(f'[kalshi] category {cat}: {len(events)} events', flush=True)
        self.cache.save(sample, name)
        return sample

    def markets_of(self, event: dict) -> list[dict]:
        name = event['event_ticker']
        cached = self.cache.load('markets', f'{name}.json')
        if cached is not None:
            return cached
        keep = (
            'ticker',
            'event_ticker',
            'title',
            'yes_sub_title',
            'open_time',
            'close_time',
            'expected_expiration_time',
            'settlement_ts',
            'result',
            'strike_type',
            'floor_strike',
            'cap_strike',
            'can_close_early',
            'market_type',
        )
        markets: list[dict] = []
        for path, extra in (
            ('/historical/markets', {}),
            ('/markets', {'status': 'settled'}),
        ):
            try:
                rows = (
                    self.reader.get(path, event_ticker=name, limit=1000, **extra).get(
                        'markets'
                    )
                    or []
                )
            except SourceError as exc:
                if exc.status != 404:
                    print(f'[kalshi] markets {name} via {path}: {exc}', flush=True)
                rows = []
            if rows:
                markets = [
                    dict(
                        {k: m.get(k) for k in keep},
                        archived=path.startswith('/historical'),
                    )
                    for m in rows
                ]
                break
        self.cache.save(markets, 'markets', f'{name}.json')
        return markets

    def candles_of(self, job: tuple[dict, dict]) -> int:
        market, event = job
        ticker = market['ticker']
        if self.cache.has('candles', f'{ticker}.json'):
            return 0
        open_ts = iso_ts(market.get('open_time'))
        sched_end = iso_ts(market.get('expected_expiration_time')) or iso_ts(
            market.get('close_time')
        )
        close_ts = iso_ts(market.get('close_time'))
        times = (
            snapshot_times(open_ts, sched_end, close_ts)
            if open_ts and sched_end and close_ts
            else []
        )
        record: dict[str, Any] = {
            'market': market,
            'event': event,
            'period': None,
            'bars': [],
            'snapshot_times': times,
        }
        if times and open_ts and sched_end:
            period = bar_period(open_ts, sched_end)
            record['period'] = period
            start = int(open_ts) - 60 * period
            stop = int(max(times)) + 60
            step = MAX_BARS * 60 * period
            bars: dict[int, list[Any]] = {}
            series = event.get('series_ticker') or ticker.split('-', 1)[0]
            for lo in range(start, stop, step):
                hi = min(stop, lo + step)
                paths = (
                    [f'/historical/markets/{ticker}/candlesticks']
                    if market.get('archived')
                    else [
                        f'/series/{series}/markets/{ticker}/candlesticks',
                        f'/historical/markets/{ticker}/candlesticks',
                    ]
                )
                for path in paths:
                    try:
                        rows = (
                            self.reader.get(
                                path, start_ts=lo, end_ts=hi, period_interval=period
                            ).get('candlesticks')
                            or []
                        )
                    except SourceError as exc:
                        if exc.status == 404:
                            continue
                        print(f'[kalshi] candles {ticker}: {exc}', flush=True)
                        rows = []
                    for c in rows:
                        bar = _kalshi_bar(c)
                        bars[bar[0]] = bar
                    break
            record['bars'] = [bars[k] for k in sorted(bars)]
        self.cache.save(record, 'candles', f'{ticker}.json')
        return 1

    def run(self) -> None:
        series_list = self.series()
        print(f'[kalshi] {len(series_list)} series', flush=True)
        run_pool(self.events_of, series_list, self.workers, 'kalshi events', every=1000)
        sample = self.sample_events(series_list)
        print(f'[kalshi] {len(sample)} sampled events', flush=True)
        run_pool(self.markets_of, sample, self.workers, 'kalshi markets')
        jobs: list[tuple[dict, dict]] = []
        for e in sample:
            markets = [
                m
                for m in (self.cache.load('markets', f'{e["event_ticker"]}.json') or [])
                if m.get('result') in ('yes', 'no')
            ]
            markets = sorted(markets, key=lambda m: stable_hash(m['ticker']))[
                : self.per_event
            ]
            jobs += [(m, e) for m in markets]
        print(f'[kalshi] {len(jobs)} markets to fetch bars for', flush=True)
        run_pool(self.candles_of, jobs, self.workers, 'kalshi bars')
        # The dataset is built from this selection only, never from whatever else sits in the cache.
        self.cache.save(
            {'sample': self.tag(), 'files': [m['ticker'] for m, _ in jobs]},
            'selection.json',
        )
        print(f'[kalshi] done; requests {self.reader.stats}', flush=True)


# ── Polymarket ───────────────────────────────────────────────────────────


def _json_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, str) and value.startswith('['):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return []
    return []


class PolymarketCollector:
    def __init__(
        self,
        out: Path,
        start: str,
        end: str,
        per_series: int,
        per_category: int,
        per_event: int,
        workers: int,
        max_pages: int,
    ):
        self.cache = Cache(out / 'polymarket')
        self.gamma = GammaReader()
        self.data = PolyDataReader()
        self.start, self.end = required_ts(start), required_ts(end)
        self.per_series, self.per_category, self.per_event = (
            per_series,
            per_category,
            per_event,
        )
        self.workers, self.max_pages = workers, max_pages

    def tag(self) -> str:
        return window_tag(
            self.start,
            self.end,
            ('s', self.per_series),
            ('c', self.per_category),
            ('e', self.per_event),
        )

    def week_windows(self) -> list[tuple[str, str]]:
        """Weekly listing windows on a fixed UTC grid, so runs with other start dates reuse cached weeks."""
        fmt = '%Y-%m-%dT%H:%M:%SZ'
        t = self.start - (self.start - WEEK_GRID_ORIGIN) % WEEK_S
        out = []
        while t < self.end:
            lo, hi = max(t, self.start), min(t + WEEK_S, self.end)
            out.append(
                (
                    datetime.fromtimestamp(lo, timezone.utc).strftime(fmt),
                    datetime.fromtimestamp(hi, timezone.utc).strftime(fmt),
                )
            )
            t += WEEK_S
        return out

    def events_in(self, window: tuple[str, str]) -> int:
        lo, hi = window
        name = f'{lo.replace(":", "")}_{hi.replace(":", "")}.json'
        if self.cache.has('events', name):
            return 0
        keep_e = (
            'id',
            'slug',
            'title',
            'startDate',
            'endDate',
            'closedTime',
            'createdAt',
            'seriesSlug',
            'negRisk',
        )
        keep_m = (
            'id',
            'conditionId',
            'question',
            'startDate',
            'createdAt',
            'endDate',
            'closedTime',
            'outcomes',
            'outcomePrices',
            'clobTokenIds',
            'umaResolutionStatus',
            'negRisk',
            'groupItemTitle',
        )
        events: list[dict] = []
        cursor = None
        for _ in range(self.max_pages):  # keyset paging: Gamma rejects deep offsets
            try:
                page = self.gamma.get(
                    '/events/keyset',
                    closed='true',
                    end_date_min=lo,
                    end_date_max=hi,
                    limit=100,
                    order='endDate',
                    ascending='true',
                    after_cursor=cursor,
                    exclude_tag_id=list(POLY_EXCLUDED_TAGS),
                )
            except SourceError as exc:
                print(f'[poly] events {lo}: {exc}', flush=True)
                return 0  # not cached, so a later run retries this week
            for e in page.get('events') or []:
                rec = {k: e.get(k) for k in keep_e}
                rec['tags'] = [
                    t.get('label') for t in (e.get('tags') or []) if t.get('label')
                ]
                series = e.get('series') or []
                rec['series'] = series[0].get('slug') if series else e.get('seriesSlug')
                rec['markets'] = [
                    {k: m.get(k) for k in keep_m} for m in (e.get('markets') or [])
                ]
                events.append(rec)
            cursor = page.get('next_cursor')
            if not cursor or not page.get('events'):
                break
        self.cache.save(events, 'events', name)
        return len(events)

    @staticmethod
    def resolved_markets(event: dict) -> list[dict]:
        """Binary markets with a clean 1/0 payout that were scheduled to trade for at least a day."""
        out = []
        for m in event.get('markets') or []:
            prices = [str(x) for x in _json_list(m.get('outcomePrices'))]
            tokens = _json_list(m.get('clobTokenIds'))
            open_ts = iso_ts(m.get('startDate')) or iso_ts(m.get('createdAt'))
            end_ts = iso_ts(m.get('endDate'))
            if (
                len(tokens) != 2
                or prices not in (['1', '0'], ['0', '1'])
                or not open_ts
                or not end_ts
            ):
                continue
            if end_ts - open_ts < POLY_MIN_LIFE_S:
                continue
            out.append(
                dict(m, yes_token=str(tokens[0]), label=1 if prices[0] == '1' else 0)
            )
        return out

    def events_by_series(self) -> dict[str, list[dict]]:
        """Events in the window with at least one usable market, keyed by series (or by event when standalone)."""
        events: dict[str, dict] = {}
        for p in self.cache.files('events'):
            for e in read_json(p):
                end = iso_ts(e.get('endDate'))
                if end is None or not self.start <= end < self.end:
                    continue
                usable = self.resolved_markets(e)
                if usable:  # keep only what sampling needs; full weekly listings run to gigabytes
                    events[str(e['id'])] = dict(e, markets=usable)
        by_series: dict[str, list[dict]] = defaultdict(list)
        for e in events.values():
            by_series[e.get('series') or f'event:{e["id"]}'].append(e)
        return by_series

    def sample_markets(self) -> list[dict]:
        name = f'sample_markets_{self.tag()}.json'
        cached = self.cache.load(name)
        if cached is not None:
            return cached
        by_series = self.events_by_series()
        by_cat: dict[str, list[dict]] = defaultdict(list)
        for evs in by_series.values():
            for e in spread_pick(
                evs, self.per_series, key=lambda x: iso_ts(x.get('endDate')) or 0.0
            ):
                by_cat[
                    polymarket_category(e.get('tags') or [], e.get('title') or '')
                ].append(e)
        sample: list[dict] = []
        keep = ('id', 'slug', 'title', 'tags', 'series', 'endDate', 'negRisk')
        for cat, evs in sorted(by_cat.items()):
            evs = sorted(evs, key=lambda e: stable_hash(str(e['id'])))[
                : self.per_category
            ]
            n_markets = 0
            for e in evs:
                for m in sorted(
                    self.resolved_markets(e), key=lambda m: stable_hash(str(m['id']))
                )[: self.per_event]:
                    sample.append(
                        {
                            'market': m,
                            'event': {k: e.get(k) for k in keep},
                            'category': cat,
                        }
                    )
                    n_markets += 1
            print(
                f'[poly] category {cat}: {len(evs)} events, {n_markets} markets',
                flush=True,
            )
        self.cache.save(sample, name)
        return sample

    def history_of(self, item: dict) -> int:
        m = item['market']
        token = m['yes_token']
        if self.cache.has('history', f'{token[-24:]}.json'):
            return 0
        open_ts = iso_ts(m.get('startDate')) or iso_ts(m.get('createdAt'))
        sched_end = iso_ts(m.get('endDate'))
        close_ts = iso_ts(m.get('closedTime')) or sched_end
        times = (
            snapshot_times(open_ts, sched_end, close_ts)
            if open_ts and sched_end and close_ts
            else []
        )
        record = {**item, 'snapshot_times': times, 'bucket': None, 'points': []}
        if times:
            # 3-hour and 12-hour series are the only permanent grains (finer ones expire after 1-3 months).
            bucket = POLY_BUCKET_S
            record['bucket'] = bucket
            window: dict[str, Any] = {'interval': 'max'}
            points: list[list[Any]] = []
            cursor = None
            for _ in range(20):
                try:
                    page = self.data.get(
                        '/v2/prices-history',
                        token_id=token,
                        bucket_seconds=bucket,
                        limit=10000,
                        cursor=cursor,
                        **({} if cursor else window),
                    )
                except SourceError as exc:
                    print(f'[poly] history {token[-12:]}: {exc}', flush=True)
                    return 0  # not cached, so a later run retries it
                points += [
                    [
                        int(p['timestamp']),
                        float(p['price']),
                        int(p.get('resolution_seconds') or 0),
                    ]
                    for p in page.get('data') or []
                ]
                cursor = (page.get('pagination') or {}).get('next_cursor')
                if not cursor:
                    break
            record['points'] = points
        self.cache.save(record, 'history', f'{token[-24:]}.json')
        return 1

    def run(self) -> None:
        windows = self.week_windows()
        counts = run_pool(
            self.events_in, windows, min(self.workers, 3), 'poly events', every=10
        )
        print(
            f'[poly] fetched {sum(counts)} new events over {len(windows)} weeks',
            flush=True,
        )
        sample = self.sample_markets()
        print(f'[poly] {len(sample)} sampled markets', flush=True)
        run_pool(self.history_of, sample, self.workers, 'poly history')
        # The dataset is built from this selection only, never from whatever else sits in the cache.
        self.cache.save(
            {
                'sample': self.tag(),
                'files': [item['market']['yes_token'][-24:] for item in sample],
            },
            'selection.json',
        )
        print(
            f'[poly] done; gamma {self.gamma.stats} data {self.data.stats}', flush=True
        )


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument('venue', choices=['kalshi', 'polymarket'])
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument(
        '--env-file',
        help='Kalshi key file (KALSHI_API_KEY_ID, KALSHI_PRIVATE_KEY_PATH); optional',
    )
    ap.add_argument(
        '--start', default='2024-01-01T00:00:00Z', help='earliest scheduled end (UTC)'
    )
    ap.add_argument(
        '--end',
        default='2026-11-05T00:00:00Z',
        help='latest scheduled end, exclusive (UTC)',
    )
    ap.add_argument('--per-series', type=int, default=40)
    ap.add_argument('--per-category', type=int, default=8000)
    ap.add_argument('--per-event', type=int, default=4)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument(
        '--rate', type=float, default=None, help='Kalshi requests per second'
    )
    ap.add_argument('--max-pages', type=int, default=120)
    a = ap.parse_args(argv)
    if a.venue == 'kalshi':
        KalshiCollector(
            a.out,
            a.env_file,
            a.start,
            a.end,
            a.per_series,
            a.per_category,
            a.per_event,
            a.workers,
            a.rate,
            a.max_pages,
        ).run()
    else:
        PolymarketCollector(
            a.out,
            a.start,
            a.end,
            a.per_series,
            a.per_category,
            a.per_event,
            a.workers,
            max(a.max_pages, 1),
        ).run()


if __name__ == '__main__':
    main()
