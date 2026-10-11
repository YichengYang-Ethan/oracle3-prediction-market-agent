"""Trade a declared, mutually exclusive and exhaustive event partition.

Observing two or more markets with the same event ID does not prove that every
outcome has arrived, or that the outcomes form a partition. Trading therefore
requires an explicit complete market list, a YES-token mapping, and both
partition assertions. Callers must verify these assertions against settlement
rules; this strategy does not infer them from titles or event membership.

For equal quantities of every declared outcome, buying all YES pays one dollar
per basket; buying all NO pays N - 1 dollars. YES purchases use asks. NO
purchases prefer their own ask book; only an absent NO book permits the
complement quote 1 - YES bid. Last-trade prices and YES asks are never used as
NO purchase quotes. Market-data books may themselves be simulated by a paper
feed; these checks do not certify live execution or atomic multi-leg fills.

Example strategy kwargs (replace IDs with the verified event's identifiers)::

    {"event_id": "event", "outcome_market_ids": ["m1", "m2"],
     "yes_token_ids": {"m1": "yes1", "m2": "yes2"},
     "outcomes_are_exhaustive": true,
     "outcomes_are_mutually_exclusive": true}
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Mapping, Sequence
from decimal import Decimal

from oracle3.events.events import Event, OrderBookEvent, PriceChangeEvent
from oracle3.fees import PolymarketSchedule
from oracle3.order.order_book import Level
from oracle3.strategy.quant_strategy import QuantStrategy
from oracle3.ticker.ticker import PolyMarketTicker
from oracle3.trader.trader import Trader
from oracle3.trader.types import OrderStatus, TradeSide

logger = logging.getLogger(__name__)


class EventSumArbStrategy(QuantStrategy):
    """Event-partition baskets, disabled until the partition is declared.

    ``outcome_market_ids`` is the complete list of binary outcome markets;
    ``yes_token_ids`` maps each market to its YES token, so NO-first market
    data cannot silently reverse a leg. Both partition flags must be true.
    An unexpected market in the event invalidates the declaration for this
    strategy instance; refresh the verified configuration before restarting.

    ``trade_size`` is the number of shares per leg, not dollars. All legs use
    that same quantity. Only quotes with enough displayed size at the selected
    price qualify; this still cannot guarantee fills across separate orders.
    ``min_markets`` remains a lower bound, never a completeness check.

    Fees use :class:`oracle3.fees.PolymarketSchedule`, with optional per-market
    ``market_fee_rates`` from the venue and its conservative default otherwise.
    ``fee_rate`` is retained as an explicit legacy *per-share flat fee*
    override. Its default is now None, so omitting it uses published schedules.
    """

    name = 'event_sum_arb'
    version = '1.0.0'
    author = 'oracle3'

    def __init__(
        self,
        event_id: str = '',
        min_edge: float = 0.02,
        trade_size: float = 10.0,
        cooldown_seconds: float = 120.0,
        min_markets: int = 2,
        fee_rate: float | None = None,
        outcome_market_ids: Sequence[str] | None = None,
        yes_token_ids: Mapping[str, str] | None = None,
        outcomes_are_exhaustive: bool = False,
        outcomes_are_mutually_exclusive: bool = False,
        market_fee_rates: Mapping[str, float] | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(event_id, str) or event_id != event_id.strip():
            raise ValueError('event_id must not contain surrounding whitespace')
        if not math.isfinite(cooldown_seconds) or cooldown_seconds < 0:
            raise ValueError('cooldown_seconds must be finite and nonnegative')
        if isinstance(outcome_market_ids, (str, bytes)):
            raise ValueError('outcome_market_ids must be a sequence of market IDs')
        ids = tuple(outcome_market_ids or ())
        if any(
            not isinstance(mid, str) or not mid or mid != mid.strip() for mid in ids
        ):
            raise ValueError(
                'outcome_market_ids must contain nonempty, trimmed strings'
            )
        if len(ids) != len(set(ids)):
            raise ValueError('outcome_market_ids must not contain duplicates')
        token_ids = dict(yes_token_ids or {})
        if token_ids and set(token_ids) != set(ids):
            raise ValueError('yes_token_ids must cover exactly outcome_market_ids')
        if any(
            not isinstance(tid, str) or not tid or tid != tid.strip()
            for tid in token_ids.values()
        ):
            raise ValueError('yes_token_ids must contain nonempty, trimmed token IDs')
        if len(set(token_ids.values())) != len(token_ids):
            raise ValueError('each outcome must have a distinct YES token')
        if not isinstance(outcomes_are_exhaustive, bool) or not isinstance(
            outcomes_are_mutually_exclusive, bool
        ):
            raise ValueError('partition assertions must be booleans')
        partition_declared = outcomes_are_exhaustive and outcomes_are_mutually_exclusive
        if partition_declared and (not event_id or not ids or not token_ids):
            raise ValueError(
                'a declared partition requires event, market and YES token IDs'
            )
        if (
            isinstance(min_markets, bool)
            or not isinstance(min_markets, int)
            or min_markets < 2
        ):
            raise ValueError('min_markets must be an integer of at least two')
        self.event_id = event_id
        self.min_edge = Decimal(str(min_edge))
        self.trade_size = Decimal(str(trade_size))
        self.cooldown_seconds = cooldown_seconds
        self.min_markets = min_markets
        self.fee_rate = None if fee_rate is None else Decimal(str(fee_rate))
        for name, value in (
            ('min_edge', self.min_edge),
            ('trade_size', self.trade_size),
        ):
            if not value.is_finite() or value <= 0:
                raise ValueError(f'{name} must be finite and positive')
        if self.fee_rate is not None and (
            not self.fee_rate.is_finite() or self.fee_rate < 0
        ):
            raise ValueError('fee_rate must be finite and nonnegative')
        self.outcome_market_ids = ids
        self.yes_token_ids = token_ids
        self.outcomes_are_exhaustive = outcomes_are_exhaustive
        self.outcomes_are_mutually_exclusive = outcomes_are_mutually_exclusive
        self._partition_declared = partition_declared
        self._invalid_partition = False
        self._fee_schedules: dict[str, PolymarketSchedule] = {}
        for mid, raw_rate in (market_fee_rates or {}).items():
            rate = Decimal(str(raw_rate))
            if mid not in ids or not rate.is_finite() or rate < 0:
                raise ValueError(
                    'market_fee_rates require declared markets and valid rates'
                )
            self._fee_schedules[mid] = PolymarketSchedule(
                rate=rate, source='strategy config'
            )
        self._tickers: dict[str, PolyMarketTicker] = {}
        self._last_arb_time = 0.0

    async def process_event(self, event: Event, trader: Trader) -> None:
        if self.is_paused() or not isinstance(
            event, (OrderBookEvent, PriceChangeEvent)
        ):
            return
        ticker = event.ticker
        if not isinstance(ticker, PolyMarketTicker) or ticker.event_id != self.event_id:
            return
        mid = ticker.market_id
        if not mid:
            return
        if self.outcome_market_ids and mid not in self.outcome_market_ids:
            self._invalid_partition = True
            return
        expected_yes = self.yes_token_ids.get(mid)
        if expected_yes is not None and ticker.token_id != expected_yes:
            # A NO update can change the executable basket, but not YES identity.
            await self._check_arb(trader)
            return
        if ticker.no_token_id and ticker.no_token_id in self.yes_token_ids.values():
            self._invalid_partition = True
            return
        self._tickers[mid] = ticker
        await self._check_arb(trader)

    def _valid_quote(self, level: Level | None) -> bool:
        return (
            level is not None
            and level.price.is_finite()
            and level.size.is_finite()
            and Decimal('0') < level.price < Decimal('1')
            and level.size >= self.trade_size
        )

    def _no_ask(self, ticker: PolyMarketTicker, trader: Trader) -> Decimal | None:
        no_ticker = ticker.get_no_ticker()
        if no_ticker is None:
            return None
        # A known but empty/insufficient NO book is authoritative: do not invent
        # executable NO liquidity from its complementary YES book.
        if no_ticker in trader.market_data.order_books:
            ask = trader.market_data.get_best_ask(no_ticker)
            return ask.price if self._valid_quote(ask) and ask is not None else None
        bid = trader.market_data.get_best_bid(ticker)
        if self._valid_quote(bid) and bid is not None:
            return Decimal('1') - bid.price
        return None

    def _basket_fee(self, prices: Mapping[str, Decimal]) -> Decimal:
        if self.fee_rate is not None:
            return self.fee_rate * len(prices)
        return sum(
            (
                self._fee_schedules.get(mid, PolymarketSchedule()).fee(
                    price, self.trade_size
                )
                / self.trade_size
                for mid, price in prices.items()
            ),
            Decimal('0'),
        )

    async def _check_arb(self, trader: Trader) -> None:
        if not self._partition_declared or self._invalid_partition:
            return
        market_ids = self.outcome_market_ids
        if len(market_ids) < self.min_markets or any(
            mid not in self._tickers for mid in market_ids
        ):
            return
        yes_prices: dict[str, Decimal] = {}
        no_prices: dict[str, Decimal] = {}
        for mid in market_ids:
            ticker = self._tickers[mid]
            ask = trader.market_data.get_best_ask(ticker)
            if self._valid_quote(ask) and ask is not None:
                yes_prices[mid] = ask.price
            no_ask = self._no_ask(ticker, trader)
            if no_ask is not None:
                no_prices[mid] = no_ask
        n = len(market_ids)
        candidates: dict[str, tuple[Decimal, dict[str, Decimal]]] = {}
        if len(yes_prices) == n:
            edge = (
                Decimal('1') - sum(yes_prices.values()) - self._basket_fee(yes_prices)
            )
            candidates['BUY_YES'] = (edge, yes_prices)
        if len(no_prices) == n:
            edge = (
                Decimal(n - 1) - sum(no_prices.values()) - self._basket_fee(no_prices)
            )
            candidates['BUY_NO'] = (edge, no_prices)
        if not candidates:
            return
        action = max(candidates, key=lambda side: candidates[side][0])
        best_edge, prices = candidates[action]
        signal: dict[str, float] = {
            'n_markets': float(n),
            'best_edge': float(best_edge),
        }
        if len(yes_prices) == n:
            signal['sum_yes'] = float(sum(yes_prices.values()))
        for side, (edge, _) in candidates.items():
            signal[f'edge_{side.lower()}'] = float(edge)
        if best_edge < self.min_edge:
            self.record_decision(
                ticker_name=f'event:{self.event_id[:16]}',
                action='HOLD',
                executed=False,
                reasoning=f'complete partition: best edge {best_edge} < {self.min_edge}',
                signal_values=signal,
            )
            return
        now = time.monotonic()
        if now - self._last_arb_time < self.cooldown_seconds:
            return
        self._last_arb_time = now
        executed_legs = 0
        failed_legs = 0
        for mid, leg_price in prices.items():
            ticker = self._tickers[mid]
            trade_ticker = ticker if action == 'BUY_YES' else ticker.get_no_ticker()
            if trade_ticker is None:  # Already excluded during basket construction.
                failed_legs += 1
                break
            try:
                result = await trader.place_order(
                    side=TradeSide.BUY,
                    ticker=trade_ticker,
                    limit_price=leg_price,
                    quantity=self.trade_size,
                )
                order = result.order
                if (
                    order is not None
                    and order.ticker == trade_ticker
                    and order.side == TradeSide.BUY
                    and order.filled_quantity.is_finite()
                    and order.filled_quantity > 0
                ):
                    executed_legs += 1
                if (
                    result.failure_reason is not None
                    or order is None
                    or order.status != OrderStatus.FILLED
                    or order.ticker != trade_ticker
                    or order.side != TradeSide.BUY
                    or order.filled_quantity != self.trade_size
                    or order.remaining != Decimal('0')
                ):
                    failed_legs += 1
                    # A partial/failed leg leaves exposure; do not add more legs.
                    break
            except Exception:
                logger.exception('EventSumArb: leg failed for market %s', mid[:16])
                failed_legs += 1
                break
        self.record_decision(
            ticker_name=f'event:{self.event_id[:16]}',
            action=action,
            executed=executed_legs > 0,
            reasoning=(
                f'complete partition n={n} edge={best_edge} '
                f'filled legs={executed_legs}/{n} incomplete legs={failed_legs}; '
                'separate orders do not guarantee an atomic basket'
            ),
            signal_values={
                **signal,
                'executed_legs': executed_legs,
                'failed_legs': failed_legs,
            },
        )
