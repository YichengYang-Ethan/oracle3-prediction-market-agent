"""Tests for the machine-learning layer: leakage, splits, metrics, baselines and the MCP tool."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

pytest.importorskip('pandas')
pytest.importorskip('sklearn')

import httpx  # noqa: E402
import pandas as pd  # noqa: E402

from oracle3.ml.cv import (  # noqa: E402
    inner_split,
    inner_walk_forward,
    purged_walk_forward,
)
from oracle3.ml.dataset import polymarket_rows  # noqa: E402
from oracle3.ml.features import (  # noqa: E402
    FEATURES,
    Bars,
    displayed_prices,
    snapshot_features,
)
from oracle3.ml.metrics import brier, ece, log_loss  # noqa: E402
from oracle3.ml.models import WangBaseline  # noqa: E402

HOUR = 3600.0


def _bars(n: int, start: float = 0.0, step: float = HOUR, seed: int = 0) -> Bars:
    rng = np.random.default_rng(seed)
    ts = start + step * np.arange(1, n + 1)
    mid = np.clip(0.5 + np.cumsum(rng.normal(0, 0.02, n)), 0.02, 0.98)
    return Bars.from_rows(
        list(ts),
        list(mid - 0.01),
        list(mid + 0.01),
        list(mid),
        list(rng.integers(0, 50, n)),
    )


def test_features_ignore_everything_after_the_prediction_time():
    bars = _bars(200)
    t = 120 * HOUR + 1800
    full = snapshot_features(bars, 0.0, 200 * HOUR, t, 'kalshi', 'crypto')
    cut = int(np.searchsorted(bars.ts, t, side='right'))
    truncated = Bars(
        bars.ts[:cut],
        bars.bid[:cut],
        bars.ask[:cut],
        bars.last[:cut],
        bars.volume[:cut],
        bars.price[:cut],
    )
    # extreme values after t must not move any feature
    shocked = Bars(
        bars.ts.copy(),
        bars.bid.copy(),
        bars.ask.copy(),
        bars.last.copy(),
        bars.volume.copy(),
        bars.price.copy(),
    )
    shocked.price[cut:] = 0.999
    shocked.volume[cut:] = 1e9
    a = snapshot_features(truncated, 0.0, 200 * HOUR, t, 'kalshi', 'crypto')
    b = snapshot_features(shocked, 0.0, 200 * HOUR, t, 'kalshi', 'crypto')
    for f in FEATURES:
        x, y, z = full[f], a[f], b[f]
        assert (math.isnan(x) and math.isnan(y) and math.isnan(z)) or (
            x == pytest.approx(y) == pytest.approx(z)
        ), f


def test_a_bar_counts_only_once_it_has_ended():
    bars = Bars.from_prices([HOUR, 2 * HOUR], [0.30, 0.90])
    row = snapshot_features(
        bars, 0.0, 10 * HOUR, 2 * HOUR - 1, 'polymarket', 'politics'
    )
    assert row['price'] == pytest.approx(0.30)


def test_time_to_end_uses_the_scheduled_end_not_the_actual_close():
    bars = _bars(10)
    row = snapshot_features(bars, 0.0, 100 * HOUR, 5 * HOUR, 'kalshi', 'politics')
    assert row['log_hours_to_end'] == pytest.approx(math.log1p(95.0))
    assert row['elapsed_fraction'] == pytest.approx(0.05)


def test_displayed_price_rule():
    nan = np.nan
    out = displayed_prices(
        np.array([0.40, 0.10, nan, nan]),
        np.array([0.44, 0.60, 0.02, nan]),
        np.array([0.41, 0.55, nan, nan]),
    )
    assert out[0] == pytest.approx(0.42)  # tight spread: midpoint
    assert out[1] == pytest.approx(0.55)  # wide spread: last trade
    assert out[2] == pytest.approx(0.01)  # missing bid counts as 0
    assert math.isnan(out[3])  # no quote and no trade


def test_polymarket_settlement_tick_never_becomes_a_feature():
    # The resolved series ends with a 0/1 payout tick (resolution_seconds 0) that can fall before t
    # when trading went quiet; it must be dropped.
    record = {
        'market': {
            'id': 'm1',
            'startDate': '2026-01-01T00:00:00Z',
            'endDate': '2026-01-11T00:00:00Z',
            'closedTime': '2026-01-11T00:00:00Z',
            'label': 1,
            'question': 'q',
        },
        'event': {'id': 'e1', 'series': None},
        'category': 'politics',
        'bucket': 10800,
        'snapshot_times': [pd.Timestamp('2026-01-08T00:00:00Z').timestamp()],
        'points': [
            [int(pd.Timestamp('2026-01-02T00:00:00Z').timestamp()), 0.2, 10800],
            [int(pd.Timestamp('2026-01-02T03:00:00Z').timestamp()), 1.0, 0],
        ],
    }
    rows = polymarket_rows(record)
    assert len(rows) == 1 and rows[0]['price'] == pytest.approx(0.2)


def _synthetic_events(n_events: int = 300, seed: int = 1):
    rng = np.random.default_rng(seed)
    starts = np.sort(rng.uniform(0, 300 * 86400, n_events))
    t, resolve, groups = [], [], []
    for i, s in enumerate(starts):
        life = rng.uniform(HOUR, 20 * 86400)
        for k in range(int(rng.integers(1, 4))):
            t.append(s + k * life / 4)
            resolve.append(s + life)
            groups.append(f'e{i}')
    return np.array(t), np.array(resolve), np.array(groups)


def test_walk_forward_folds_are_grouped_purged_and_forward():
    t, resolve, groups = _synthetic_events()
    embargo = 86400.0
    folds = purged_walk_forward(
        t, resolve, groups, n_splits=4, embargo=embargo, min_train_groups=5
    )
    assert len(folds) >= 3
    for fold in folds:
        train_g, test_g = set(groups[fold.train]), set(groups[fold.test])
        assert not train_g & test_g
        assert resolve[fold.train].max() < fold.test_start - embargo
        assert t[fold.test].min() >= fold.test_start
        fit, val = inner_split(t, resolve, groups, fold.train, 0.2, embargo)
        assert not set(groups[fit]) & set(groups[val])
        assert resolve[fit].max() < t[val].min() - embargo
        pairs = inner_walk_forward(
            t, resolve, groups, fold.train, 3, embargo, min_fit_groups=5
        )
        assert pairs
        for fit, val in pairs:
            assert set(fit) <= set(fold.train) and set(val) <= set(fold.train)
            assert not set(groups[fit]) & set(groups[val])
            assert resolve[fit].max() < t[val].min() - embargo


def test_metrics_known_values():
    p = np.array([0.1, 0.9, 0.8, 0.3])
    y = np.array([0, 1, 0, 0])
    assert brier(p, y) == pytest.approx(np.mean((p - y) ** 2))
    assert log_loss(p, y) > 0
    # two bins: [0.1, 0.3] -> mean 0.2 vs 0; [0.8, 0.9] -> mean 0.85 vs 0.5
    assert ece(p, y, n_bins=2) == pytest.approx(0.5 * 0.2 + 0.5 * 0.35)


def test_wang_baseline_recovers_a_probit_shift():
    rng = np.random.default_rng(3)
    n, lam = 20000, 0.2
    true_p = rng.uniform(0.05, 0.95, n)
    from scipy import stats

    price = stats.norm.cdf(stats.norm.ppf(true_p) + lam)
    df = pd.DataFrame(
        {
            'price': price,
            'label': rng.binomial(1, true_p),
            'platform': 'kalshi',
            'log_life_hours': 1.0,
            'extremity': np.abs(price - 0.5),
            'log_volume_to_date': 1.0,
            'spread': 0.02,
        }
    )
    model = WangBaseline().fit(df)
    assert model.beta['kalshi'][0] == pytest.approx(lam, abs=0.04)
    assert brier(model.predict(df), df['label']) <= brier(df['price'], df['label'])


# ── MCP tool ─────────────────────────────────────────────────────────────


def _synthetic_snapshots(n: int = 600, seed: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        p = float(rng.uniform(0.05, 0.95))
        row = dict.fromkeys(FEATURES, 0.0)
        row.update(
            price=p,
            logit_price=math.log(p / (1 - p)),
            extremity=abs(p - 0.5),
            is_kalshi=1.0,
            cat_economics=1.0,
        )
        row.update(
            platform='kalshi',
            market_id=f'M{i}',
            event_id=f'E{i // 2}',
            label=int(rng.random() < p),
            t=1_700_000_000 + 3600 * i,
            resolve_ts=1_700_000_000 + 3600 * i + 1800,
        )
        rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture
def tiny_bundle(tmp_path, monkeypatch):
    from oracle3.ml import serve

    df = _synthetic_snapshots()
    summary = {
        'pooled': {
            'logistic': {'brier': 0.2},
            'market': {'brier': 0.21},
            'wang': {'brier': 0.205},
        },
        'vs_market': {'logistic': {'brier_reduction_pct': 1.0}},
    }
    serve.train_bundle(df, 'logistic', tmp_path / 'bundle', 3, 0.0, summary)
    monkeypatch.setenv('ORACLE3_ML_BUNDLE', str(tmp_path / 'bundle'))
    serve._cache.clear()
    return tmp_path / 'bundle'


def test_bundle_refuses_a_tampered_model_file(tiny_bundle):
    from oracle3.ml import serve

    with open(tiny_bundle / 'isotonic.json', 'a') as fh:
        fh.write(' ')
    serve._cache.clear()
    with pytest.raises(ValueError, match='does not match'):
        serve.load_bundle()


async def test_model_probability_tool_scores_a_live_kalshi_market(tiny_bundle):
    pytest.importorskip('mcp')
    from oracle3.mcp_server import venues
    from oracle3.mcp_server.server import mcp

    now = pd.Timestamp.now(tz='UTC')
    market = {
        'ticker': 'KXFED-27JAN-H0',
        'title': 'Fed holds?',
        'status': 'active',
        'open_time': (now - pd.Timedelta(days=10)).isoformat(),
        'expected_expiration_time': (now + pd.Timedelta(days=10)).isoformat(),
        'close_time': (now + pd.Timedelta(days=10)).isoformat(),
        'yes_bid_dollars': '0.6000',
        'yes_ask_dollars': '0.6200',
        'last_price_dollars': '0.6100',
        'volume_fp': '1500.00',
    }
    end = int(now.timestamp()) // 3600 * 3600
    candles = [
        {
            'end_period_ts': end - 3600 * k,
            'yes_bid': {'close': '0.59'},
            'yes_ask': {'close': '0.61'},
            'price': {'close': '0.60', 'previous': '0.60'},
            'volume': '10',
        }
        for k in range(48, 0, -1)
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.endswith('/candlesticks'):
            return httpx.Response(200, json={'candlesticks': candles})
        if path.endswith('/series/KXFED'):
            return httpx.Response(200, json={'series': {'category': 'Economics'}})
        if path.endswith('/markets/KXFED-27JAN-H0'):
            return httpx.Response(200, json={'market': market})
        return httpx.Response(404, json={})

    venues.set_transport(httpx.MockTransport(handler))
    try:
        result = await mcp.call_tool(
            'model_probability', {'venue': 'kalshi', 'market_id': 'KXFED-27JAN-H0'}
        )
    finally:
        venues.set_transport(None)
    content = getattr(result, 'structuredContent', None) or getattr(
        result, 'structured_content', None
    )
    if content is None:
        content = (
            result[1]
            if isinstance(result, tuple)
            else json.loads(result.content[0].text)
        )
    out = content.get('result', content)
    assert 0.0 < out['model_probability'] < 1.0
    assert out['market_price'] == pytest.approx(0.61)
    assert out['features']['cat_economics'] == 1.0
    assert out['model'] == 'logistic'


def test_sampling_helpers_place_events_and_snapshots_without_outcome_data():
    from oracle3.ml.collect import snapshot_times, spread_pick, ticker_date

    assert (
        pd.Timestamp(ticker_date('KXNBAGAME-26JUN13NYKSAS'), unit='s')
        .date()
        .isoformat()
        == '2026-06-13'
    )
    assert (
        pd.Timestamp(ticker_date('KXFEDDECISION-26JUL'), unit='s').date().isoformat()
        == '2026-07-15'
    )
    assert ticker_date('KXPRES-2028') is None
    # snapshots sit on the scheduled life and stop once the market actually closed
    assert snapshot_times(0.0, 100.0, 1000.0) == [25.0, 50.0, 75.0]
    assert snapshot_times(0.0, 100.0, 60.0) == [25.0, 50.0]
    assert snapshot_times(0.0, 100.0, 10.0) == []
    picked = spread_pick(list(range(100)), 5, key=float)
    assert picked[0] == 0 and picked[-1] == 99 and len(picked) == 5


def test_kalshi_rows_use_only_bars_before_each_snapshot():
    from oracle3.ml.dataset import kalshi_rows

    open_ts = pd.Timestamp('2026-01-01T00:00:00Z').timestamp()
    end_ts = open_ts + 100 * HOUR
    bars = [
        [int(open_ts + k * HOUR), 0.40, 0.42, 0.41, None, 5.0] for k in range(1, 100)
    ]
    bars[60:] = [
        [b[0], 0.95, 0.97, 0.96, None, 500.0] for b in bars[60:]
    ]  # the price jumps after hour 60
    record = {
        'market': {
            'ticker': 'KXT-1',
            'event_ticker': 'KXT',
            'result': 'yes',
            'open_time': '2026-01-01T00:00:00Z',
            'expected_expiration_time': pd.Timestamp(
                end_ts, unit='s', tz='UTC'
            ).isoformat(),
            'close_time': pd.Timestamp(end_ts, unit='s', tz='UTC').isoformat(),
            'settlement_ts': pd.Timestamp(
                end_ts + HOUR, unit='s', tz='UTC'
            ).isoformat(),
        },
        'event': {'category': 'economics', 'series_ticker': 'KXT'},
        'period': 60,
        'bars': bars,
        'snapshot_times': [
            open_ts + 25 * HOUR + 1,
            open_ts + 50 * HOUR + 1,
            open_ts + 75 * HOUR + 1,
        ],
    }
    rows = kalshi_rows(record)
    assert [r['price'] for r in rows] == [
        pytest.approx(0.41),
        pytest.approx(0.41),
        pytest.approx(0.96),
    ]
    assert rows[1]['log_volume_to_date'] == pytest.approx(math.log1p(5.0 * 50))


def test_cache_writes_gzip_and_still_reads_plain_files(tmp_path):
    from oracle3.ml.collect import Cache

    cache = Cache(tmp_path)
    (tmp_path / 'old').mkdir()
    (tmp_path / 'old' / 'a.json').write_text('[1, 2]')
    assert cache.load('old', 'a.json') == [1, 2]
    cache.save({'x': 1}, 'old', 'a.json')  # the gzipped copy replaces the plain one
    assert not (tmp_path / 'old' / 'a.json').exists()
    assert cache.load('old', 'a.json') == {'x': 1}
    (tmp_path / 'old' / 'b.json').write_text('{}')
    assert [p.name for p in cache.files('old')] == ['a.json.gz', 'b.json']
    assert cache.load('old', 'missing.json') is None
    assert cache.has('old', 'a.json') and cache.has('old', 'b.json')
    assert not cache.has('old', 'missing.json')


class _FakeKalshi:
    def __init__(self, pages: list[list[dict]]):
        self.pages, self.calls = pages, 0

    def paginate(self, path, key, max_pages=10_000, **params):
        for page in self.pages[:max_pages]:
            self.calls += 1
            yield page


def test_kalshi_listing_is_reused_only_when_it_covers_the_window(tmp_path):
    from oracle3.ml.collect import Cache, KalshiCollector, iso_ts

    pages = [  # newest first, dated by ticker
        [{'event_ticker': 'KXS-26JAN28'}, {'event_ticker': 'KXS-26JAN20'}],
        [{'event_ticker': 'KXS-26JAN12'}, {'event_ticker': 'KXS-26JAN04'}],
    ]
    col = object.__new__(KalshiCollector)
    col.cache, col.max_pages, col.reader = Cache(tmp_path), 10, _FakeKalshi(pages)

    col.start = iso_ts('2026-01-15T00:00:00Z')
    assert (
        len(col.events_of({'ticker': 'KXS'})) == 4
    )  # stops on the page that passes the start
    col.events_of({'ticker': 'KXS'})
    assert col.reader.calls == 2  # second call served from the cache

    col.start = iso_ts('2026-01-10T00:00:00Z')
    col.events_of({'ticker': 'KXS'})
    assert col.reader.calls == 2  # the cached listing already reaches back past Jan 10

    col.start = iso_ts('2026-01-01T00:00:00Z')
    col.events_of({'ticker': 'KXS'})
    assert col.reader.calls == 4  # an earlier start lists the series again, to its end
    col.start = iso_ts('2020-01-01T00:00:00Z')
    col.events_of({'ticker': 'KXS'})
    assert col.reader.calls == 4  # a complete listing serves any start

    col.max_pages, col.reader = 1, _FakeKalshi(pages)
    col.events_of({'ticker': 'KXT'})
    col.events_of({'ticker': 'KXT'})
    assert col.reader.calls == 2  # a listing cut by the page limit is never trusted


class _FlakyKalshi:
    """Fails the first candlestick request with a 429, then answers."""

    def __init__(self):
        self.calls = 0

    def get(self, path, **params):
        from oracle3.ml.sources import SourceError

        self.calls += 1
        if self.calls == 1:
            raise SourceError(429, path, 'too many requests')
        bar = {
            'end_period_ts': int(params['start_ts']) + 7200,
            'yes_bid': {'close_dollars': '0.40'},
            'yes_ask': {'close_dollars': '0.42'},
            'price': {'close_dollars': '0.41'},
            'volume_fp': '5',
        }
        return {'candlesticks': [bar]}


def test_kalshi_bars_are_not_cached_when_a_request_fails(tmp_path):
    from oracle3.ml.collect import Cache, KalshiCollector

    col = object.__new__(KalshiCollector)
    col.cache, col.reader = Cache(tmp_path), _FlakyKalshi()
    market = {
        'ticker': 'KXT-1',
        'archived': True,
        'open_time': '2026-01-01T00:00:00Z',
        'expected_expiration_time': '2026-01-05T00:00:00Z',
        'close_time': '2026-01-05T00:00:00Z',
    }
    job = (market, {'series_ticker': 'KXT'})
    assert col.candles_of(job) == -1  # rate-limited: nothing cached
    assert not col.cache.has('candles', 'KXT-1.json')
    assert col.candles_of(job) == 1  # the retry fetches and caches the bars
    assert col.cache.load('candles', 'KXT-1.json')['bars']
    assert col.candles_of(job) == 0


def test_isotonic_is_kept_only_when_it_helps_later_inner_blocks():
    from oracle3.ml.benchmark import isotonic_helps

    rng = np.random.default_rng(0)
    n = 3000
    q = rng.uniform(0.05, 0.95, 3 * n)
    y = (rng.random(3 * n) < q).astype(int)
    biased = np.clip(q + 0.2, 0.0, 1.0)  # every inner block over-predicts the same way
    keep, check = isotonic_helps(biased, y, [n, n, n])
    assert keep and check['inner_brier_isotonic'] < check['inner_brier_raw']
    # only the first block is biased; a map learned from it would distort the later ones
    first_only = np.concatenate([biased[:n], q[n:]])
    keep, check = isotonic_helps(first_only, y, [n, n, n])
    assert not keep and check['inner_brier_isotonic'] > check['inner_brier_raw']


@pytest.mark.parametrize(
    'name',
    [
        'logistic',
        'spline_logistic',
        'random_forest',
        'lightgbm',
        'lightgbm_monotone',
        'xgboost',
        'catboost',
        'mlp',
    ],
)
def test_every_learner_tunes_refits_and_predicts(name):
    from oracle3.ml.features import KALSHI_ONLY, NUMERIC_FEATURES
    from oracle3.ml.models import LEARNERS, fit_learner, predict

    if name in ('xgboost', 'catboost'):
        pytest.importorskip(name)
    learner = next(lr for lr in LEARNERS if lr.name == name)
    rng = np.random.default_rng(1)
    n = 600
    X = pd.DataFrame(rng.normal(size=(n, len(FEATURES))), columns=list(FEATURES))
    binary = [f for f in FEATURES if f not in NUMERIC_FEATURES]
    X[binary] = rng.integers(0, 2, (n, len(binary))).astype(float)
    X['price'] = rng.uniform(0.05, 0.95, n)
    X['logit_price'] = np.log(X['price'] / (1 - X['price']))
    X.loc[X['is_kalshi'] == 0, list(KALSHI_ONLY)] = np.nan  # Polymarket rows
    y = (rng.random(n) < X['price']).astype(int).to_numpy()
    est, used = fit_learner(
        learner, dict(learner.grid[0]), X[:400], y[:400], X[400:], y[400:]
    )
    if learner.early_stopping:
        assert used['n_estimators'] >= 1
    p = predict(fit_learner(learner, used, X, y)[0], X)
    assert p.shape == (n,) and np.all((p > 0) & (p < 1))


def test_polymarket_listing_weeks_sit_on_a_fixed_grid():
    from oracle3.ml.collect import PolymarketCollector, iso_ts

    def weeks(start: str) -> list[tuple[str, str]]:
        col = object.__new__(PolymarketCollector)
        col.start, col.end = iso_ts(start), iso_ts('2025-12-01T00:00:00Z')
        return col.week_windows()

    late, early = weeks('2025-11-01T00:00:00Z'), weeks('2025-10-15T00:00:00Z')
    assert late[0] == ('2025-11-01T00:00:00Z', '2025-11-08T00:00:00Z')
    assert early[0] == ('2025-10-15T00:00:00Z', '2025-10-18T00:00:00Z')
    assert late[-1] == ('2025-11-29T00:00:00Z', '2025-12-01T00:00:00Z')
    assert set(late) <= set(early)  # an earlier start reuses every cached later week


async def test_model_probability_tool_scores_a_live_polymarket_market(tiny_bundle):
    pytest.importorskip('mcp')
    from oracle3.mcp_server import venues
    from oracle3.mcp_server.server import mcp

    now = pd.Timestamp.now(tz='UTC')
    market = {
        'id': '555',
        'question': 'Will the Fed cut in December?',
        'active': True,
        'closed': False,
        'clobTokenIds': '["111", "222"]',
        'startDate': (now - pd.Timedelta(days=20)).isoformat(),
        'endDate': (now + pd.Timedelta(days=20)).isoformat(),
        'bestBid': 0.30,
        'bestAsk': 0.32,
        'lastTradePrice': 0.31,
        'tags': [{'label': 'Fed Rates'}, {'label': 'Economy'}],
        'events': [],
    }
    start = int(now.timestamp()) - 10 * 86400
    history = {
        'data': [
            {'timestamp': start + 10800 * k, 'price': 0.30, 'resolution_seconds': 10800}
            for k in range(70)
        ]
        + [
            {
                'timestamp': int(now.timestamp()) - 5,
                'price': 0.31,
                'resolution_seconds': 0,
            }
        ]
    }

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith('/markets/555'):
            assert request.url.params.get('include_tag') == 'true'
            return httpx.Response(200, json=market)
        if request.url.path.endswith('/v2/prices-history'):
            return httpx.Response(200, json=history)
        return httpx.Response(404, json={})

    venues.set_transport(httpx.MockTransport(handler))
    try:
        result = await mcp.call_tool(
            'model_probability', {'venue': 'polymarket', 'market_id': '555'}
        )
    finally:
        venues.set_transport(None)
    content = getattr(result, 'structuredContent', None) or getattr(
        result, 'structured_content', None
    )
    if content is None:
        content = (
            result[1]
            if isinstance(result, tuple)
            else json.loads(result.content[0].text)
        )
    out = content.get('result', content)
    assert out['category'] == 'economics'
    assert out['market_price'] == pytest.approx(0.31)
    assert out['features']['spread'] is None  # no spread history is used for Polymarket
