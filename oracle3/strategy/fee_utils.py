"""Shared fee estimates for strategies; no network or verified-current defaults.

Configured terms override the published defaults. Callers must keep configured
terms current: this helper does not fetch or verify a venue's fee schedule.
"""

from __future__ import annotations

from collections.abc import Mapping
from decimal import Decimal
from typing import Any, TypeAlias

from oracle3.fees import (
    FeeSchedule,
    KalshiSchedule,
    PolymarketSchedule,
    UnsupportedFeeSchedule,
    _nonnegative,
    _price,
    _quantity,
    default_schedule,
)
from oracle3.ticker.ticker import KalshiTicker, PolyMarketTicker, Ticker

MarketFeeTerms: TypeAlias = Mapping[str, FeeSchedule | Mapping[str, Any]]


def _venue_and_market(ticker: Ticker) -> tuple[str, str]:
    if isinstance(ticker, KalshiTicker):
        return 'kalshi', ticker.market_ticker
    if isinstance(ticker, PolyMarketTicker):
        return 'polymarket', ticker.market_id
    raise UnsupportedFeeSchedule(
        f'no shared venue fee schedule for ticker type {type(ticker).__name__}'
    )


def _configured_schedule(venue: str, terms: Any) -> FeeSchedule:
    if isinstance(terms, (KalshiSchedule, PolymarketSchedule)):
        if terms.venue != venue:
            raise UnsupportedFeeSchedule(
                'configured fee schedule belongs to another venue'
            )
        return terms
    if not isinstance(terms, Mapping):
        raise UnsupportedFeeSchedule(
            'configured market fee terms must be a schedule or mapping'
        )
    if venue == 'kalshi':
        if not {'fee_type', 'fee_multiplier'}.intersection(terms):
            raise UnsupportedFeeSchedule(
                'configured Kalshi terms need series fee fields'
            )
        return KalshiSchedule.from_series(terms)
    if 'feeSchedule' in terms or 'feesEnabled' in terms:
        return PolymarketSchedule.from_gamma_market(terms)
    if 'rate' in terms:
        return PolymarketSchedule.from_gamma_market({'feeSchedule': terms})
    raise UnsupportedFeeSchedule('configured Polymarket terms need feeSchedule or rate')


def resolve_fee_schedule(
    ticker: Ticker,
    *,
    market_fee_terms: MarketFeeTerms | None = None,
) -> FeeSchedule:
    """Resolve a ticker's configured schedule, or an explicitly labeled default.

    Keys are checked in this order: ``venue:market_id``, ``venue:symbol``,
    ``market_id``, ``symbol``. Kalshi uses ``market_ticker`` for its market ID.
    Values are schedule objects or native Kalshi series / Polymarket Gamma
    mappings. A Polymarket ``{'rate': ...}`` schedule mapping is also accepted.
    Unknown venues and malformed explicitly configured terms fail closed.
    """
    venue, market_id = _venue_and_market(ticker)
    identifiers = list(
        dict.fromkeys(item for item in (market_id, ticker.symbol) if item)
    )
    keys = [f'{venue}:{item}' for item in identifiers] + identifiers
    terms: FeeSchedule | Mapping[str, Any] | None = None
    configured = False
    if market_fee_terms is not None:
        for key in keys:
            if key in market_fee_terms:
                terms = market_fee_terms[key]
                configured = True
                break
    if not configured:
        return default_schedule(venue)
    return _configured_schedule(venue, terms)


def estimate_fee(
    ticker: Ticker,
    price: Decimal | float,
    contracts: Decimal | float,
    *,
    market_fee_terms: MarketFeeTerms | None = None,
    legacy_fee_per_contract: Decimal | float | None = None,
    maker: bool = False,
) -> Decimal:
    """Estimate the total fee at the proposed fill price and share quantity.

    ``legacy_fee_per_contract`` is an explicit flat dollars-per-share override,
    not a percentage and not a venue schedule. It takes precedence over market
    terms only when provided; zero must be intentional. Defaults are estimates,
    not verified current market terms. Unsupported venues require an explicit
    legacy override rather than inheriting another venue's fee schedule.
    """
    parsed_price = _price(price)
    quantity = _quantity(contracts)
    if legacy_fee_per_contract is not None:
        return (
            _nonnegative(legacy_fee_per_contract, 'legacy_fee_per_contract') * quantity
        )
    return resolve_fee_schedule(ticker, market_fee_terms=market_fee_terms).fee(
        parsed_price, quantity, maker=maker
    )
