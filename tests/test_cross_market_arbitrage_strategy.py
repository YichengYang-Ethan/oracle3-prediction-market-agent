"""Tests for Feature 1: Cross-Market Arbitrage Strategy."""

from __future__ import annotations

import time
from decimal import Decimal
from unittest.mock import MagicMock

import pytest

from oracle3.strategy.contrib.cross_market_arbitrage_strategy import (
    ArbPosition,
    CrossMarketArbitrageStrategy,
    TickerGrouper,
    _normalize,
)
from oracle3.ticker.ticker import KalshiTicker, PolyMarketTicker, SolanaTicker


class TestNormalize:
    def test_lowercases_and_removes_punctuation(self):
        assert _normalize('Will BTC hit $100k?') == 'btc hit 100k'

    def test_removes_stopwords(self):
        assert _normalize('the price of ETH') == 'price eth'


class TestTickerGrouper:
    def test_groups_similar_tickers(self):
        grouper = TickerGrouper(min_similarity=0.5)
        t1 = PolyMarketTicker(
            symbol='BTC_100K_YES',
            name='Will BTC hit 100k by end of 2026',
            token_id='tok1',
            market_id='mkt1',
            event_id='evt1',
        )
        t2 = SolanaTicker(
            symbol='BTC_100K_DFLOW',
            name='Will Bitcoin hit 100k by 2026',
            market_ticker='mkt2',
            event_ticker='evt2',
        )

        groups = grouper.group({
            'polymarket': [t1],
            'dflow': [t2],
        })
        assert len(groups) >= 1
        assert 'polymarket' in groups[0].tickers
        assert 'dflow' in groups[0].tickers

    def test_does_not_group_dissimilar(self):
        grouper = TickerGrouper(min_similarity=0.9)
        t1 = PolyMarketTicker(
            symbol='BTC_YES', name='Bitcoin price',
            token_id='a', market_id='b', event_id='c',
        )
        t2 = SolanaTicker(
            symbol='ETH_YES', name='Ethereum merge date',
            market_ticker='d', event_ticker='e',
        )
        groups = grouper.group({'poly': [t1], 'dflow': [t2]})
        assert len(groups) == 0

    def test_needs_two_platforms(self):
        grouper = TickerGrouper()
        t1 = PolyMarketTicker(
            symbol='A', name='test', token_id='a', market_id='b', event_id='c',
        )
        assert grouper.group({'poly': [t1]}) == []


class TestCrossMarketArbitrageStrategy:
    def test_find_arbitrage_opportunities_no_prices(self):
        s = CrossMarketArbitrageStrategy(min_edge=0.03)
        assert s.find_arbitrage_opportunities() == []

    def test_find_opportunities_with_edge(self):
        s = CrossMarketArbitrageStrategy(min_edge=0.03, fee_per_side=0.0)
        t1 = PolyMarketTicker(
            symbol='EVT_POLY', name='Same Event',
            token_id='a', market_id='b', event_id='c',
        )
        t2 = SolanaTicker(
            symbol='EVT_DFLOW', name='Same Event',
            market_ticker='d', event_ticker='e',
        )
        s.register_price('polymarket', t1, Decimal('0.40'))
        s.register_price('dflow', t2, Decimal('0.50'))

        opps = s.find_arbitrage_opportunities()
        assert len(opps) >= 1
        assert opps[0]['gross_edge'] != 0

    def test_no_opportunity_when_edge_too_small(self):
        s = CrossMarketArbitrageStrategy(min_edge=0.20)
        t1 = PolyMarketTicker(
            symbol='EVT_P', name='Event X',
            token_id='a', market_id='b', event_id='c',
        )
        t2 = SolanaTicker(
            symbol='EVT_D', name='Event X',
            market_ticker='d', event_ticker='e',
        )
        s.register_price('polymarket', t1, Decimal('0.50'))
        s.register_price('dflow', t2, Decimal('0.51'))
        assert s.find_arbitrage_opportunities() == []

    def test_detect_platform(self):
        s = CrossMarketArbitrageStrategy()
        poly = PolyMarketTicker(
            symbol='A', name='', token_id='a', market_id='b', event_id='c',
        )
        sol = SolanaTicker(symbol='B', name='', market_ticker='d', event_ticker='e')
        assert s._detect_platform(poly) == 'polymarket'
        assert s._detect_platform(sol) == 'dflow'

    @pytest.mark.asyncio
    async def test_process_event_no_crash(self):
        from oracle3.data.market_data_manager import MarketDataManager
        from oracle3.events.events import PriceChangeEvent
        from oracle3.position.position_manager import PositionManager
        from oracle3.risk.risk_manager import NoRiskManager
        from oracle3.trader.paper_trader import PaperTrader

        s = CrossMarketArbitrageStrategy(min_edge=0.03)
        t = SolanaTicker(
            symbol='TEST', name='Test', market_ticker='m', event_ticker='e',
        )
        event = PriceChangeEvent(ticker=t, price=Decimal('0.50'))

        md = MarketDataManager()
        pm = PositionManager()
        trader = PaperTrader(
            market_data=md, risk_manager=NoRiskManager(),
            position_manager=pm, min_fill_rate=Decimal('1'), max_fill_rate=Decimal('1'),
            commission_rate=Decimal('0'),
        )
        s.bind_context(event, trader)
        await s.process_event(event, trader)


def _thin_fed_spread(strategy: CrossMarketArbitrageStrategy) -> tuple[str, str]:
    """Kalshi YES at 0.48 vs Polymarket YES at 0.50: 0.02 gross per contract."""
    name = 'Fed cuts rates in December'
    kalshi = KalshiTicker('KXFED-26DEC', name, market_ticker='KXFED-26DEC')
    poly = PolyMarketTicker('PM_FED_YES', name, market_id='pm-fed')
    strategy.register_price('kalshi', kalshi, Decimal('0.48'))
    strategy.register_price('polymarket', poly, Decimal('0.50'))
    return kalshi.symbol, poly.symbol


class TestCrossMarketArbitrageFees:
    def test_published_fees_reject_spread_below_fee_hurdle(self):
        # Kalshi's taker fee at 0.48 and Polymarket's default at 0.50 are each
        # about 0.0175 per contract, so 0.02 gross is a net loss. The former
        # flat 0.005 per side reported 0.01 net edge here.
        s = CrossMarketArbitrageStrategy(min_edge=0.005)
        _thin_fed_spread(s)
        assert s.find_arbitrage_opportunities() == []

    def test_market_fee_terms_override_the_schedules(self):
        s = CrossMarketArbitrageStrategy(
            min_edge=0.005,
            market_fee_terms={
                'KXFED-26DEC': {'fee_type': 'quadratic', 'fee_multiplier': 0},
                'pm-fed': {'rate': 0},
            },
        )
        _thin_fed_spread(s)
        [opp] = s.find_arbitrage_opportunities()
        assert opp['direction'] == 'buy_a_sell_b'
        assert opp['net_edge'] == pytest.approx(0.02)
        assert opp['fees'] == 0

    def test_explicit_flat_fee_per_side_still_applies(self):
        s = CrossMarketArbitrageStrategy(min_edge=0.005, fee_per_side=0.005)
        _thin_fed_spread(s)
        [opp] = s.find_arbitrage_opportunities()
        assert opp['net_edge'] == pytest.approx(0.01)

    def test_venue_without_shared_schedule_needs_explicit_fee(self):
        s = CrossMarketArbitrageStrategy(min_edge=0.03)
        s.register_price(
            'polymarket',
            PolyMarketTicker(symbol='P', name='Same Event', market_id='b'),
            Decimal('0.40'),
        )
        s.register_price(
            'dflow',
            SolanaTicker(symbol='D', name='Same Event', market_ticker='d'),
            Decimal('0.50'),
        )
        assert s.find_arbitrage_opportunities() == []

    # Gross edge of +/-0.02 minus 0.01748 (Kalshi) and 0.0175 (Polymarket).
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        'state,net', [('long_a_short_b', -0.01498), ('long_b_short_a', -0.05498)]
    )
    async def test_exit_check_uses_schedule_fees(self, state, net):
        s = CrossMarketArbitrageStrategy(exit_threshold=0.005)
        symbol_a, symbol_b = _thin_fed_spread(s)
        s._positions['fed'] = ArbPosition(
            state=state,
            entry_time=time.monotonic(),
            ticker_a_symbol=symbol_a,
            ticker_b_symbol=symbol_b,
            label='fed',
        )
        await s._check_exits(MagicMock())  # no holdings to sell
        [decision] = s.get_decisions()
        assert decision.action == 'ARB_EXIT_SPREAD_COLLAPSED'
        assert decision.signal_values['current_net_edge'] == pytest.approx(net)
