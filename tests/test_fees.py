"""Tests for oracle3.fees against the venues' published numbers."""

from __future__ import annotations

from decimal import Decimal

import pytest

from oracle3.fees import (
    CENT,
    KalshiSchedule,
    PolymarketSchedule,
    UnsupportedFeeSchedule,
    default_schedule,
)

# Kalshi fee schedule, "General Trading Fees Table" (effective July 7, 2026).
# The table rounds to the cent, so these use increment=CENT.
KALSHI_TABLE_100_CONTRACTS = [
    (0.01, '0.07'),
    (0.05, '0.34'),
    (0.10, '0.63'),
    (0.20, '1.12'),
    (0.30, '1.47'),
    (0.50, '1.75'),
    (0.70, '1.47'),
    (0.95, '0.34'),
    (0.99, '0.07'),
]


@pytest.mark.parametrize('price,expected', KALSHI_TABLE_100_CONTRACTS)
def test_kalshi_taker_matches_published_table_for_100_contracts(price, expected):
    assert KalshiSchedule(increment=CENT).fee(price, 100) == Decimal(expected)


@pytest.mark.parametrize(
    'price,expected', [(0.01, '0.01'), (0.20, '0.02'), (0.50, '0.02'), (0.99, '0.01')]
)
def test_kalshi_taker_matches_published_table_for_one_contract(price, expected):
    assert KalshiSchedule(increment=CENT).fee(price, 1) == Decimal(expected)


def test_kalshi_default_rounds_up_to_a_centicent():
    # 0.07 * 1 * 0.5 * 0.5 = 0.0175 exactly
    assert KalshiSchedule().fee(0.5, 1) == Decimal('0.0175')
    # 0.07 * 1 * 0.55 * 0.45 = 0.017325 -> 0.0174
    assert KalshiSchedule().fee(0.55, 1) == Decimal('0.0174')


def test_kalshi_multiplier_scales_and_zero_is_free():
    assert KalshiSchedule(multiplier=Decimal('0')).fee(0.5, 100) == 0
    assert KalshiSchedule(multiplier=Decimal('2')).fee(0.5, 100) == Decimal('3.5')


def test_kalshi_maker_fee_only_on_series_with_maker_fees():
    assert KalshiSchedule(maker_fees=False).fee(0.5, 100, maker=True) == 0
    # 0.0175 * 100 * 0.25 = 0.4375
    assert KalshiSchedule(maker_fees=True).fee(0.5, 100, maker=True) == Decimal(
        '0.4375'
    )


def test_kalshi_from_series_reads_fee_type_and_multiplier():
    fed = KalshiSchedule.from_series(
        {'fee_type': 'quadratic_with_maker_fees', 'fee_multiplier': 1}
    )
    assert fed.maker_fees is True and fed.multiplier == 1
    btc = KalshiSchedule.from_series({'fee_type': 'quadratic', 'fee_multiplier': 0})
    assert btc.fee(0.5, 100) == 0
    with pytest.raises(UnsupportedFeeSchedule):
        KalshiSchedule.from_series(
            {'fee_type': 'quadratic_with_combo_maker_fees', 'fee_multiplier': 1}
        )
    with pytest.raises(UnsupportedFeeSchedule):
        KalshiSchedule.from_series({'fee_type': 'flat', 'fee_multiplier': 1})


def test_polymarket_documented_example():
    # docs.polymarket.com: 100 Crypto shares at $0.30 or $0.70 pay $1.47.
    crypto = PolymarketSchedule.for_category('crypto')
    assert crypto.fee(0.30, 100) == Decimal('1.47000')
    assert crypto.fee(0.70, 100) == Decimal('1.47000')


def test_polymarket_makers_and_geopolitics_are_free():
    assert PolymarketSchedule.for_category('sports').fee(0.5, 100, maker=True) == 0
    assert PolymarketSchedule.for_category('geopolitics').fee(0.5, 100) == 0


def test_polymarket_rounds_to_five_decimals_with_minimum():
    assert PolymarketSchedule(rate=Decimal('0.04')).fee(0.001, 0.01) == Decimal(
        '0.00001'
    )


def test_polymarket_from_gamma_market():
    market = {
        'feesEnabled': True,
        'feeType': 'sports_fees_v3',
        'feeSchedule': {
            'exponent': 1,
            'rate': 0.05,
            'takerOnly': True,
            'rebateRate': 0.15,
        },
    }
    schedule = PolymarketSchedule.from_gamma_market(market)
    assert schedule.rate == Decimal('0.05')
    assert (
        PolymarketSchedule.from_gamma_market({'feesEnabled': False}).fee(0.5, 100) == 0
    )
    with pytest.raises(UnsupportedFeeSchedule):
        PolymarketSchedule.from_gamma_market(
            {'feeSchedule': {'exponent': 2, 'rate': 0.05}}
        )
    with pytest.raises(UnsupportedFeeSchedule):
        PolymarketSchedule.from_gamma_market(
            {'feeSchedule': {'rate': 0.05, 'takerOnly': False}}
        )


def test_unknown_market_schedule_defaults_are_conservative():
    assert default_schedule('polymarket').rate == Decimal('0.07')
    assert default_schedule('kalshi').multiplier == 1
    with pytest.raises(ValueError):
        default_schedule('nyse')


@pytest.mark.parametrize('bad', [0, 1, -0.1, 1.2])
def test_price_must_be_inside_the_unit_interval(bad):
    with pytest.raises(ValueError):
        KalshiSchedule().fee(bad, 1)
    with pytest.raises(ValueError):
        PolymarketSchedule().fee(bad, 1)


def test_fee_is_symmetric_in_price():
    for schedule in (KalshiSchedule(), PolymarketSchedule()):
        for p in (0.1, 0.25, 0.4):
            assert schedule.fee(p, 37) == schedule.fee(1 - p, 37)


@pytest.mark.parametrize('schedule', [{}, {'rate': None}, None, 'unknown'])
def test_present_but_unknown_polymarket_schedule_is_not_free(schedule):
    with pytest.raises(UnsupportedFeeSchedule):
        PolymarketSchedule.from_gamma_market(
            {'feesEnabled': True, 'feeSchedule': schedule}
        )


@pytest.mark.parametrize(
    'rate', [-0.01, float('nan'), float('inf'), '-Infinity', '', False]
)
def test_invalid_polymarket_rate_is_rejected_at_api_boundary(rate):
    with pytest.raises(UnsupportedFeeSchedule, match='rate'):
        PolymarketSchedule.from_gamma_market({'feeSchedule': {'rate': rate}})


def test_explicit_zero_and_explicit_disabled_fees_remain_free():
    zero = PolymarketSchedule.from_gamma_market({'feeSchedule': {'rate': 0}})
    disabled = PolymarketSchedule.from_gamma_market(
        {'feesEnabled': False, 'feeSchedule': {'rate': None}}
    )
    assert zero.fee(0.5, 100) == 0
    assert disabled.fee(0.5, 100) == 0
    assert disabled.describe()['source'] == 'feesEnabled=false'


def test_absent_schedule_uses_labeled_default_instead_of_zero():
    schedule = PolymarketSchedule.from_gamma_market({'feesEnabled': True})
    assert schedule.fee(0.5, 100) > 0
    assert 'default' in schedule.describe()['source']
    assert 'unverified' in default_schedule('kalshi').describe()['source']


@pytest.mark.parametrize('bad', [-1, 'NaN', 'Infinity', '-Infinity', None, ''])
def test_schedule_constructors_reject_invalid_rates_and_multipliers(bad):
    with pytest.raises(ValueError):
        PolymarketSchedule(rate=bad)
    with pytest.raises(ValueError):
        KalshiSchedule(multiplier=bad)
    with pytest.raises(UnsupportedFeeSchedule):
        KalshiSchedule.from_series({'fee_type': 'quadratic', 'fee_multiplier': bad})


@pytest.mark.parametrize('bad', [0, -1, 'NaN', 'Infinity', None])
def test_rounding_increment_must_be_finite_and_positive(bad):
    with pytest.raises(ValueError, match='increment'):
        KalshiSchedule(increment=bad)


def test_schedule_constructors_normalize_valid_numeric_terms():
    assert KalshiSchedule(multiplier=2.0, increment=0.01).fee(0.5, 100) == Decimal(
        '3.5'
    )
    assert PolymarketSchedule(rate=0.05).fee(0.5, 100) == Decimal('1.25000')


@pytest.mark.parametrize(
    'bad', [float('nan'), float('inf'), Decimal('-Infinity'), 'bad']
)
@pytest.mark.parametrize('maker', [False, True])
def test_price_and_quantity_must_be_finite_even_when_fees_are_zero(bad, maker):
    for schedule in (
        KalshiSchedule(multiplier=Decimal('0')),
        PolymarketSchedule(rate=Decimal('0')),
    ):
        with pytest.raises(ValueError, match='price'):
            schedule.fee(bad, 1, maker=maker)
        with pytest.raises(ValueError, match='quantity'):
            schedule.fee(0.5, bad, maker=maker)


@pytest.mark.parametrize('bad', [0, -1])
def test_quantity_must_be_positive_for_both_venues(bad):
    for schedule in (KalshiSchedule(), PolymarketSchedule()):
        with pytest.raises(ValueError, match='quantity'):
            schedule.fee(0.5, bad)


def test_unknown_rate_cannot_create_a_false_fee_free_opportunity():
    from oracle3.arbitrage import Quote, check_constraint

    raw = {'feesEnabled': True, 'feeSchedule': {'takerOnly': True}}
    with pytest.raises(UnsupportedFeeSchedule, match='rate'):
        PolymarketSchedule.from_gamma_market(raw)

    # The same two-cent gross gap is not positive after the explicit default.
    schedule = PolymarketSchedule.from_gamma_market({})
    check = check_constraint(
        'implication',
        [
            Quote('A', venue='polymarket', yes_bid=0.51, schedule=schedule),
            Quote('B', venue='polymarket', yes_ask=0.49, schedule=schedule),
        ],
    )
    assert check.violated
    assert not check.profitable_after_fees


def test_strategy_fee_helper_uses_ticker_venue_and_explicit_terms():
    from oracle3.strategy.fee_utils import estimate_fee, resolve_fee_schedule
    from oracle3.ticker.ticker import KalshiTicker, PolyMarketTicker

    kalshi = KalshiTicker(symbol='K-YES', market_ticker='K')
    poly = PolyMarketTicker(symbol='P-YES', market_id='P')
    assert estimate_fee(kalshi, Decimal('0.5'), Decimal('100')) == Decimal('1.75')
    assert estimate_fee(poly, Decimal('0.5'), Decimal('100')) == Decimal('1.75')
    assert 'default' in resolve_fee_schedule(poly).describe()['source']
    terms = {
        'kalshi:K': {'fee_type': 'quadratic', 'fee_multiplier': 2},
        'P': {'rate': 0.04},
    }
    assert estimate_fee(kalshi, 0.5, 100, market_fee_terms=terms) == Decimal('3.5')
    assert estimate_fee(poly, 0.5, 100, market_fee_terms=terms) == Decimal('1')


def test_strategy_fee_helper_prefers_venue_specific_keys_and_shared_schedules():
    from oracle3.strategy.fee_utils import estimate_fee
    from oracle3.ticker.ticker import PolyMarketTicker

    ticker = PolyMarketTicker(symbol='TOKEN', market_id='M')
    terms = {'polymarket:M': PolymarketSchedule(rate=Decimal('0.05')), 'M': {'rate': 0}}
    assert estimate_fee(ticker, 0.5, 100, market_fee_terms=terms) == Decimal('1.25')
    with pytest.raises(UnsupportedFeeSchedule, match='another venue'):
        estimate_fee(ticker, 0.5, 100, market_fee_terms={'M': KalshiSchedule()})
    with pytest.raises(UnsupportedFeeSchedule, match='missing'):
        estimate_fee(ticker, 0.5, 100, market_fee_terms={'M': {'feeSchedule': {}}})


def test_strategy_fee_helper_requires_explicit_override_for_unknown_venue():
    from oracle3.strategy.fee_utils import estimate_fee
    from oracle3.ticker.ticker import SolanaTicker

    ticker = SolanaTicker(symbol='TOKEN')
    with pytest.raises(UnsupportedFeeSchedule, match='ticker type'):
        estimate_fee(ticker, 0.5, 100)
    assert estimate_fee(ticker, 0.5, 100, legacy_fee_per_contract=0) == 0
    assert estimate_fee(ticker, 0.5, 100, legacy_fee_per_contract=0.005) == Decimal(
        '0.5'
    )
    for invalid in (-1, float('nan'), float('inf')):
        with pytest.raises(ValueError, match='legacy_fee_per_contract'):
            estimate_fee(ticker, 0.5, 100, legacy_fee_per_contract=invalid)
