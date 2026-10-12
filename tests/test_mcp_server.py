"""Tests for the Oracle3 MCP server, with every venue request mocked."""

from __future__ import annotations

import json

import httpx
import pytest

pytest.importorskip('mcp')

from oracle3.mcp_server import venues  # noqa: E402
from oracle3.mcp_server.server import mcp  # noqa: E402

KALSHI_MARKET = {
    'ticker': 'KXFEDDECISION-28JAN-H0',
    'event_ticker': 'KXFEDDECISION-28JAN',
    'title': 'Will the Fed hold rates in January?',
    'yes_bid_dollars': '0.6000',
    'yes_ask_dollars': '0.6200',
    'no_bid_dollars': '0.3800',
    'no_ask_dollars': '0.4000',
    'volume_fp': '1599.47',
    'close_time': '2028-01-26T19:00:00Z',
    'status': 'active',
}
KALSHI_BOOK = {
    'orderbook_fp': {
        'yes_dollars': [['0.5800', '10.00'], ['0.6000', '25.00']],
        'no_dollars': [['0.3600', '40.00'], ['0.3800', '5.00']],
    }
}
POLY_MARKET = {
    'id': '2589812',
    'question': 'Will there be no change in Fed interest rates in January?',
    'outcomes': '["Yes", "No"]',
    'clobTokenIds': '["111", "222"]',
    'bestBid': 0.64,
    'bestAsk': 0.65,
    'volumeNum': 125000.0,
    'endDate': '2028-01-26T00:00:00Z',
    'feesEnabled': True,
    'feeType': 'economics_fees',
    'feeSchedule': {'exponent': 1, 'rate': 0.05, 'takerOnly': True, 'rebateRate': 0.25},
    'events': [{'title': 'Fed Decision in January?'}],
}
POLY_BOOKS = {
    '111': {
        'bids': [{'price': '0.63', 'size': '100'}, {'price': '0.64', 'size': '50'}],
        'asks': [{'price': '0.66', 'size': '80'}, {'price': '0.65', 'size': '30'}],
    },
    '222': {
        'bids': [{'price': '0.35', 'size': '30'}],
        'asks': [{'price': '0.36', 'size': '50'}, {'price': '0.37', 'size': '100'}],
    },
}


def kalshi_route(path: str) -> httpx.Response:
    if path.endswith('/orderbook'):
        return httpx.Response(200, json=KALSHI_BOOK)
    if path.endswith(f'/markets/{KALSHI_MARKET["ticker"]}'):
        return httpx.Response(200, json={'market': KALSHI_MARKET})
    if path.endswith('/markets'):
        return httpx.Response(200, json={'markets': [KALSHI_MARKET]})
    if path.endswith('/events/KXFEDDECISION-28JAN'):
        return httpx.Response(200, json={'event': {'series_ticker': 'KXFEDDECISION'}})
    if path.endswith('/events'):
        other = dict(KALSHI_MARKET, ticker='KXOTHER-1', title='Other')
        return httpx.Response(
            200,
            json={
                'events': [
                    {
                        'title': 'Fed decision in January',
                        'series_ticker': 'KXFEDDECISION',
                        'markets': [KALSHI_MARKET],
                    },
                    {
                        'title': 'Unrelated',
                        'series_ticker': 'KXOTHER',
                        'markets': [other],
                    },
                ],
                'cursor': '',
            },
        )
    if path.endswith('/series/KXFEDDECISION'):
        return httpx.Response(
            200,
            json={
                'series': {'fee_type': 'quadratic_with_maker_fees', 'fee_multiplier': 1}
            },
        )
    return httpx.Response(404, text=f'unmocked {path}')


def polymarket_route(host: str, path: str, params: httpx.QueryParams) -> httpx.Response:
    if host == 'clob.polymarket.com' and path == '/book':
        return httpx.Response(200, json=POLY_BOOKS[params['token_id']])
    if path == f'/markets/{POLY_MARKET["id"]}':
        return httpx.Response(200, json=POLY_MARKET)
    if path == '/markets':
        return httpx.Response(200, json=[POLY_MARKET])
    if path == '/public-search':
        return httpx.Response(
            200,
            json={
                'events': [
                    {
                        'id': 77,
                        'title': 'Fed Decision in January?',
                        'markets': [POLY_MARKET],
                    }
                ]
            },
        )
    return httpx.Response(404, text=f'unmocked {path}')


def handler(request: httpx.Request) -> httpx.Response:
    if request.url.host == 'api.elections.kalshi.com':
        return kalshi_route(request.url.path)
    return polymarket_route(request.url.host, request.url.path, request.url.params)


@pytest.fixture(autouse=True)
def mocked_venues(tmp_path, monkeypatch):
    monkeypatch.setenv('ORACLE3_MCP_LEDGER', str(tmp_path / 'ledger.json'))
    venues.set_transport(httpx.MockTransport(handler))
    yield
    venues.set_transport(None)


def _unwrap(result):
    """Normalize call_tool output across mcp 1.x (tuple) and 2.x (CallToolResult)."""
    if isinstance(result, tuple):
        content, structured = result
        is_error = False
    else:
        # mcp 1.x uses camelCase attributes; mcp 2.x uses snake_case with aliases.
        content = getattr(result, 'content', result)
        structured = getattr(result, 'structuredContent', None)
        if structured is None:
            structured = getattr(result, 'structured_content', None)
        is_error = bool(
            getattr(result, 'isError', None) or getattr(result, 'is_error', False)
        )
    if is_error:
        raise RuntimeError(content[0].text if content else 'tool error')
    data = structured if structured is not None else json.loads(content[0].text)
    if isinstance(data, dict) and set(data) == {'result'}:
        return data['result']
    return data


async def call(name, **arguments):
    return _unwrap(await mcp.call_tool(name, arguments))


async def test_server_reports_package_version():
    import oracle3

    server = getattr(mcp, '_lowlevel_server', None) or mcp._mcp_server
    assert server.version == oracle3.__version__


async def test_tool_surface_has_no_live_trading():
    names = {t.name for t in await mcp.list_tools()}
    assert names == {
        'search_markets',
        'get_market',
        'get_orderbook',
        'get_quote',
        'trading_fee',
        'check_constraint',
        'check_constraint_live',
        'list_relation_types',
        'fair_value',
        'list_relations',
        'paper_order',
        'paper_portfolio',
        'paper_reset',
    }
    read_only = {
        t.name
        for t in await mcp.list_tools()
        if t.annotations and t.annotations.model_dump(by_alias=True).get('readOnlyHint')
    }
    assert {'paper_order', 'paper_reset'}.isdisjoint(read_only)


async def test_kalshi_search_filters_events_by_query():
    out = await call('search_markets', venue='kalshi', query='fed')
    assert out['count'] == 1
    market = out['markets'][0]
    assert market['market_id'] == 'KXFEDDECISION-28JAN-H0'
    assert market['yes_bid'] == 0.60 and market['yes_ask'] == 0.62


async def test_kalshi_orderbook_derives_asks_from_opposite_bids():
    book = await call(
        'get_orderbook', venue='kalshi', market_id='KXFEDDECISION-28JAN-H0'
    )
    assert book['yes']['bids'][0] == [0.60, 25.0]
    assert book['yes']['asks'][0] == [0.62, 5.0]
    assert book['no']['asks'][0] == [0.40, 25.0]


async def test_kalshi_quote_carries_series_fee_schedule():
    quote = await call('get_quote', venue='kalshi', market_id='KXFEDDECISION-28JAN-H0')
    assert quote['yes_ask'] == 0.62 and quote['no_ask'] == 0.40
    assert quote['fee_schedule']['multiplier_M'] == 1.0
    assert quote['fee_schedule']['maker_k'] == 0.0175


async def test_polymarket_search_and_quote():
    out = await call('search_markets', venue='polymarket', query='fed')
    assert out['markets'][0]['fee_schedule']['taker_rate'] == 0.05
    quote = await call('get_quote', venue='polymarket', market_id='2589812')
    assert (quote['yes_bid'], quote['yes_ask'], quote['no_bid'], quote['no_ask']) == (
        0.64,
        0.65,
        0.35,
        0.36,
    )


async def test_live_same_event_check_across_venues():
    out = await call(
        'check_constraint_live',
        relation='same_event',
        markets=[
            {'venue': 'kalshi', 'market_id': 'KXFEDDECISION-28JAN-H0'},
            {'venue': 'polymarket', 'market_id': '2589812'},
        ],
    )
    # Kalshi YES 0.62 + Polymarket NO 0.36 = 0.98 < 1
    assert out['violated']
    assert out['best']['description'] == 'YES on A + NO on B'
    assert out['best']['gross_edge'] == pytest.approx(0.02)
    assert len(out['quotes']) == 2
    assert out['assessment'] == {
        'relation_verified': False,
        'execution_verified': False,
        'conditional_on_relation': True,
    }


async def test_offline_check_constraint():
    out = await call(
        'check_constraint',
        relation='implication',
        quotes=[
            {'market_id': 'A', 'yes_bid': 0.60},
            {'market_id': 'B', 'yes_ask': 0.55},
        ],
    )
    assert out['best']['net_edge'] == pytest.approx(0.0158)
    assert out['assessment'] == {
        'relation_verified': False,
        'execution_verified': False,
        'conditional_on_relation': True,
    }


@pytest.mark.parametrize('contracts', [float('nan'), float('inf'), -float('inf')])
async def test_constraint_rejects_nonfinite_size_when_all_prices_missing(contracts):
    with pytest.raises(Exception, match='contracts must be finite and positive'):
        await call(
            'check_constraint',
            relation='implication',
            quotes=[{'market_id': 'A'}, {'market_id': 'B'}],
            contracts=contracts,
        )


@pytest.mark.parametrize('field', ['yes_bid', 'yes_ask', 'no_bid', 'no_ask'])
@pytest.mark.parametrize('price', [float('nan'), float('inf'), -0.01, 1.01])
async def test_constraint_rejects_invalid_unused_quote_field(field, price):
    with pytest.raises(Exception, match=f'{field} must be finite and between 0 and 1'):
        await call(
            'check_constraint',
            relation='implication',
            quotes=[{'market_id': 'A'}, {'market_id': 'B', field: price}],
        )


async def test_constraint_rejects_repeated_outcome_in_group():
    with pytest.raises(Exception, match='duplicate contract: kalshi:A'):
        await call(
            'check_constraint',
            relation='event_sum',
            quotes=[{'market_id': 'A', 'yes_ask': 0.3}] * 2,
        )


async def test_constraint_allows_same_id_on_different_venues():
    out = await call(
        'check_constraint',
        relation='same_event',
        quotes=[
            {'market_id': 'A', 'venue': 'kalshi', 'yes_ask': 0.4},
            {'market_id': 'A', 'venue': 'polymarket', 'no_ask': 0.55},
        ],
    )
    assert out['profitable_after_fees']
    assert {leg['venue'] for leg in out['best']['legs']} == {'kalshi', 'polymarket'}


@pytest.mark.parametrize('market_id', ['', '  '])
async def test_constraint_rejects_blank_identity(market_id):
    with pytest.raises(Exception, match='market_id must be a nonblank string'):
        await call(
            'check_constraint',
            relation='implication',
            quotes=[{'market_id': market_id}, {'market_id': 'B'}],
        )


@pytest.mark.parametrize('invalid', ['size', 'duplicate', 'blank'])
async def test_live_constraint_rejects_invalid_input_before_http(invalid):
    requests = []

    def unexpected_request(request):
        requests.append(request)
        return httpx.Response(500, text='invalid input should not fetch quotes')

    venues.set_transport(httpx.MockTransport(unexpected_request))
    markets = [
        {'venue': 'kalshi', 'market_id': 'A'},
        {'venue': 'kalshi', 'market_id': 'B'},
    ]
    contracts = 1.0
    if invalid == 'size':
        contracts = float('nan')
        message = 'contracts must be finite and positive'
    elif invalid == 'duplicate':
        markets[1]['market_id'] = 'A'
        message = 'duplicate contract'
    else:
        markets[0]['market_id'] = ' '
        message = 'market_id must be a nonblank string'
    with pytest.raises(Exception, match=message):
        await call(
            'check_constraint_live',
            relation='same_event',
            markets=markets,
            contracts=contracts,
        )
    assert requests == []


async def test_trading_fee_and_fair_value():
    fee = await call('trading_fee', venue='kalshi', price=0.5, contracts=100)
    assert fee['fee'] == pytest.approx(1.75)
    fv = await call('fair_value', market_price=0.57)
    assert fv['implied_probability'] == pytest.approx(0.497, abs=1e-3)


@pytest.mark.parametrize('lam', [float('nan'), float('inf'), -float('inf')])
async def test_fair_value_rejects_nonfinite_lambda(lam):
    with pytest.raises(Exception, match='lam must be finite'):
        await call('fair_value', market_price=0.57, lam=lam)


async def test_paper_order_walks_the_book_and_charges_fees():
    out = await call(
        'paper_order',
        venue='polymarket',
        market_id='2589812',
        side='yes',
        contracts=50,
        limit_price=0.66,
    )
    assert out['status'] == 'filled'
    assert [lv['price'] for lv in out['levels']] == [0.65, 0.66]
    assert out['cost'] == pytest.approx(30 * 0.65 + 20 * 0.66)
    portfolio = await call('paper_portfolio')
    assert portfolio['positions'][0]['contracts'] == 50
    assert portfolio['cash'] == pytest.approx(10_000 - out['cost'] - out['fees'])


async def test_paper_order_respects_limit_and_reset_needs_confirmation():
    out = await call(
        'paper_order',
        venue='kalshi',
        market_id='KXFEDDECISION-28JAN-H0',
        side='yes',
        contracts=5,
        limit_price=0.50,
    )
    assert out['status'] == 'unfilled'
    assert (await call('paper_reset'))['status'] == 'not_reset'
    assert (await call('paper_reset', confirm=True))['cash'] == 10_000


@pytest.mark.parametrize(
    'field,value',
    [
        ('contracts', 'NaN'),
        ('contracts', 'Infinity'),
        ('contracts', '-Infinity'),
        ('contracts', 0),
        ('contracts', -1),
        ('limit_price', 'NaN'),
        ('limit_price', 'Infinity'),
        ('limit_price', 1),
        ('market_id', '  '),
        ('request_id', '  '),
    ],
)
@pytest.mark.asyncio
async def test_invalid_paper_dispatch_does_not_contact_venues_or_write(
    tmp_path, field, value
):
    requests = []

    def no_http(request):
        requests.append(request)
        return httpx.Response(500)

    venues.set_transport(httpx.MockTransport(no_http))
    arguments = {
        'venue': 'polymarket',
        'market_id': '2589812',
        'side': 'yes',
        'contracts': 1,
        'limit_price': 0.7,
        field: value,
    }
    with pytest.raises(Exception, match='finite|positive|between|nonblank'):
        await call('paper_order', **arguments)
    assert requests == []
    assert not (tmp_path / 'ledger.json').exists()


@pytest.mark.asyncio
async def test_paper_dispatch_retry_and_conflict_skip_http():
    arguments = {
        'venue': 'polymarket',
        'market_id': '2589812',
        'side': 'yes',
        'contracts': 1,
        'limit_price': 0.7,
        'request_id': 'retry-this-order',
    }
    first = await call('paper_order', **arguments)
    requests = []

    def no_http(request):
        requests.append(request)
        return httpx.Response(500)

    venues.set_transport(httpx.MockTransport(no_http))
    assert await call('paper_order', **arguments) == first
    with pytest.raises(Exception, match='different paper order'):
        await call('paper_order', **{**arguments, 'contracts': 2})
    assert requests == []
    assert (await call('paper_portfolio'))['fills'] == 1


@pytest.mark.parametrize(
    'venue,field,value',
    [
        ('kalshi', 'kalshi_multiplier', 'NaN'),
        ('kalshi', 'kalshi_multiplier', 'Infinity'),
        ('kalshi', 'kalshi_multiplier', -1),
        ('polymarket', 'polymarket_rate', 'NaN'),
        ('polymarket', 'polymarket_rate', 'Infinity'),
        ('polymarket', 'polymarket_rate', -0.1),
    ],
)
@pytest.mark.asyncio
async def test_fee_dispatch_rejects_invalid_schedule_inputs(venue, field, value):
    with pytest.raises(Exception, match='finite|nonnegative'):
        await call('trading_fee', venue=venue, price=0.5, contracts=1, **{field: value})


@pytest.mark.asyncio
async def test_corrupt_ledger_is_a_readable_tool_error_without_http(tmp_path):
    path = tmp_path / 'ledger.json'
    path.write_text('{"cash":NaN}')
    requests = []

    def no_http(request):
        requests.append(request)
        return httpx.Response(500)

    venues.set_transport(httpx.MockTransport(no_http))
    with pytest.raises(Exception, match='Invalid paper ledger.*preserved'):
        await call(
            'paper_order',
            venue='polymarket',
            market_id='2589812',
            side='yes',
            contracts=1,
            limit_price=0.7,
        )
    with pytest.raises(Exception, match='Invalid paper ledger.*preserved'):
        await call('paper_reset', confirm=True)
    assert path.read_text() == '{"cash":NaN}'
    assert requests == []


async def test_venue_errors_surface_as_tool_errors():
    # The venue's message must reach the agent under both SDK versions.
    with pytest.raises(Exception, match='HTTP 404'):
        await call('get_market', venue='kalshi', market_id='MISSING')


def test_registry_manifest_matches_package():
    from pathlib import Path

    import oracle3

    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / 'server.json').read_text())
    assert manifest['version'] == oracle3.__version__
    assert manifest['packages'][0]['version'] == oracle3.__version__
    assert manifest['packages'][0]['identifier'] == 'oracle3'
    assert len(manifest['description']) <= 100
    readme = (root / 'README.md').read_text()
    assert f'mcp-name: {manifest["name"]}' in readme


async def test_non_finite_size_is_a_readable_error():
    with pytest.raises(Exception, match='quantity|Invalid|NaN|nan'):
        await call('trading_fee', venue='kalshi', price=0.5, contracts=float('nan'))
