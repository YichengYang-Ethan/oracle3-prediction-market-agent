"""Shared execution state for the two-market strategies.

Ownership is local to this strategy instance. It is not a durable account ledger:
unknown orders and negative residuals stop automation until externally reconciled.
"""

from __future__ import annotations

import math
import time
from collections.abc import Mapping
from decimal import Decimal
from typing import Any

from oracle3.events.events import Event, PriceChangeEvent
from oracle3.strategy.fee_utils import estimate_fee
from oracle3.strategy.quant_strategy import QuantStrategy
from oracle3.ticker.ticker import KalshiTicker, PolyMarketTicker, Ticker
from oracle3.trader.spread_executor import SpreadExecutor, SpreadLeg, SpreadOrderResult
from oracle3.trader.trader import Trader
from oracle3.trader.types import TradeSide


class BasketExecutionStrategy(QuantStrategy):
    """Preflight complete baskets and retain only this instance's known exposure."""

    def __init__(
        self,
        market_id_a: str,
        market_id_b: str,
        trade_size: float,
        min_edge: float,
        cooldown_seconds: float,
        fee_rate: float | None,
        market_fee_terms: Mapping[str, Any] | None,
        yes_token_ids: Mapping[str, str] | None,
    ) -> None:
        super().__init__()
        for name, value in [('market_id_a', market_id_a), ('market_id_b', market_id_b)]:
            if not isinstance(value, str) or value.strip() != value:
                raise ValueError(f'{name} must be a trimmed identifier')
        if market_id_a and market_id_a == market_id_b:
            raise ValueError('markets A and B must be distinct')
        self._id_a, self._id_b = market_id_a, market_id_b
        self.trade_size = self._number(trade_size, 'trade_size', positive=True)
        self.min_edge = self._number(min_edge, 'min_edge')
        if not math.isfinite(cooldown_seconds) or cooldown_seconds < 0:
            raise ValueError('cooldown_seconds must be finite and nonnegative')
        self.cooldown_seconds = cooldown_seconds
        self.fee_rate = None if fee_rate is None else self._number(fee_rate, 'fee_rate')
        self.market_fee_terms = dict(market_fee_terms or {})
        self.yes_token_ids = dict(yes_token_ids or {})
        if any(
            key not in {market_id_a, market_id_b}
            or not isinstance(value, str)
            or not value
            or value.strip() != value
            for key, value in self.yes_token_ids.items()
        ) or len(set(self.yes_token_ids.values())) != len(self.yes_token_ids):
            raise ValueError(
                'yes_token_ids must map the configured markets to distinct tokens'
            )
        self._price_a: Decimal | None = None
        self._price_b: Decimal | None = None
        self._position_state = 'flat'
        self._last_entry_time = float('-inf')
        self._owned_positions: dict[Ticker, Decimal] = {}
        self.requires_reconciliation = False
        self.last_execution_result: SpreadOrderResult | None = None
        self._basket_busy = False

    @staticmethod
    def _number(
        value: float | Decimal, name: str, *, positive: bool = False
    ) -> Decimal:
        result = Decimal(str(value))
        if not result.is_finite() or (result <= 0 if positive else result < 0):
            raise ValueError(
                f'{name} must be finite and {"positive" if positive else "nonnegative"}'
            )
        return result

    @staticmethod
    def _matches(ticker_id: str, market_id: str) -> bool:
        return bool(market_id) and ticker_id == market_id

    @staticmethod
    def _extract_ticker_id(ticker: Ticker) -> str:
        return (
            getattr(ticker, 'market_id', '')
            or getattr(ticker, 'market_ticker', '')
            or getattr(ticker, 'token_id', '')
            or ticker.symbol
        )

    def _is_yes(self, ticker: Ticker) -> bool:
        if isinstance(ticker, KalshiTicker):
            return not ticker.is_no_side
        if not isinstance(ticker, PolyMarketTicker):
            return False
        expected = self.yes_token_ids.get(self._extract_ticker_id(ticker))
        if expected:
            return ticker.token_id == expected
        if ticker.symbol.endswith('_NO') or ticker.name.startswith('NO '):
            return False
        # Polymarket's opaque token IDs do not encode outcome polarity. Require
        # configuration unless a local fixture explicitly labels its outcome.
        return (
            ticker.symbol.endswith('_YES')
            or ticker.name.startswith('YES ')
            or ticker.no_token_id.endswith('_NO')
        )

    def _find_ticker(
        self, trader: Trader, market_id: str, *, yes: bool = True
    ) -> Ticker | None:
        candidates = [
            ticker
            for ticker in trader.market_data.order_books
            if self._matches(self._extract_ticker_id(ticker), market_id)
            and self._is_yes(ticker)
        ]
        if len(candidates) != 1:
            return None
        ticker = candidates[0]
        if yes:
            return ticker
        complement = getattr(ticker, 'get_no_ticker', lambda: None)()
        return complement if complement in trader.market_data.order_books else None

    def _accept_event(self, event: Event) -> bool:
        if (
            self.is_paused()
            or self._basket_busy
            or not isinstance(event, PriceChangeEvent)
        ):
            return False
        if not isinstance(event.price, Decimal) or not event.price.is_finite():
            return False
        if not 0 <= event.price <= 1 or not self._is_yes(event.ticker):
            return False
        tid = self._extract_ticker_id(event.ticker)
        if self._matches(tid, self._id_a):
            self._price_a = event.price
        elif self._matches(tid, self._id_b):
            self._price_b = event.price
        else:
            return False
        return self._price_a is not None and self._price_b is not None

    def _hold(self, reasoning: str, **signals: float) -> None:
        self.record_decision(
            ticker_name=f'{self.name}({self._id_a}|{self._id_b})',
            action='HOLD',
            executed=False,
            reasoning=reasoning,
            signal_values=signals,
        )

    def _buy_legs(
        self, trader: Trader, specs: list[tuple[str, bool, Decimal]]
    ) -> list[SpreadLeg] | None:
        legs: list[SpreadLeg] = []
        for market_id, yes, weight in specs:
            if weight == 0:
                continue
            ticker = self._find_ticker(trader, market_id, yes=yes)
            if ticker is None:
                self._hold('Missing or ambiguous outcome identity/order book')
                return None
            ask = trader.market_data.get_best_ask(ticker)
            quantity = self.trade_size * weight
            if (
                ask is None
                or not ask.price.is_finite()
                or not ask.size.is_finite()
                or not 0 < ask.price < 1
                or ask.size < quantity
            ):
                self._hold('Missing executable ask or insufficient visible size')
                return None
            legs.append(SpreadLeg(TradeSide.BUY, ticker, ask.price, quantity))
        return legs or None

    async def _enter_basket(
        self,
        trader: Trader,
        specs: list[tuple[str, bool, Decimal]],
        reference_value: Decimal,
        state: str,
        action: str,
        *,
        model_dependent: bool = False,
    ) -> None:
        if (
            self._position_state != 'flat'
            or self._owned_positions
            or self.requires_reconciliation
        ):
            return
        if time.monotonic() - self._last_entry_time < self.cooldown_seconds:
            return
        legs = self._buy_legs(trader, specs)
        if not legs:
            return
        try:
            fees = sum(
                (
                    estimate_fee(
                        leg.ticker,
                        leg.price,
                        leg.quantity,
                        market_fee_terms=self.market_fee_terms,
                        legacy_fee_per_contract=self.fee_rate,
                    )
                    for leg in legs
                ),
                Decimal('0'),
            )
        except ValueError as exc:
            self._hold(f'Cannot establish fees: {exc}')
            return
        cost = sum((leg.price * leg.quantity for leg in legs), Decimal('0'))
        gross_edge = reference_value - cost / self.trade_size
        net_edge = gross_edge - fees / self.trade_size
        if net_edge <= self.min_edge:
            self._hold(
                'Executable edge after entry fees is below threshold',
                gross_edge=float(gross_edge),
                net_edge=float(net_edge),
            )
            return
        self._last_entry_time = time.monotonic()
        self._basket_busy = True
        try:
            result = await SpreadExecutor(trader).execute_spread(legs)
        except BaseException:
            # Cancellation may interrupt an order after venue acceptance.
            self.requires_reconciliation = True
            self._position_state = 'reconciliation_required'
            raise
        finally:
            self._basket_busy = False
        self.last_execution_result = result
        if result.success:
            self._owned_positions = {leg.ticker: leg.quantity for leg in legs}
            self._position_state = state
        else:
            self._owned_positions = {
                item.ticker: item.quantity for item in result.residual_exposures
            }
            self.requires_reconciliation = result.execution_uncertain or any(
                quantity < 0 for quantity in self._owned_positions.values()
            )
            self._position_state = (
                'reconciliation_required'
                if self.requires_reconciliation
                else 'recovery_required'
                if self._owned_positions
                else 'flat'
            )
        self.record_decision(
            ticker_name=f'{self.name}({self._id_a}|{self._id_b})',
            action=action,
            executed=result.success,
            reasoning=(
                'Conditional on caller-supplied relationship; '
                + (
                    'model convergence is not a guaranteed payout; '
                    if model_dependent
                    else ''
                )
                + ('all legs filled' if result.success else result.failure_reason)
            ),
            signal_values={
                'gross_edge': float(gross_edge),
                'net_edge': float(net_edge),
                'entry_fees': float(fees),
                'relation_verified': 0.0,
                'model_dependent': float(model_dependent),
                'executed_legs': float(sum(r.executed for r in result.leg_results)),
                'all_filled': float(result.all_filled),
                'requires_reconciliation': float(self.requires_reconciliation),
            },
        )

    async def _close_owned(self, trader: Trader, *, action: str = 'EXIT') -> None:
        if self.requires_reconciliation or self._basket_busy:
            self._hold('Reconcile outstanding orders before further execution')
            return
        if not self._owned_positions:
            self._position_state = 'flat'
            return
        legs: list[SpreadLeg] = []
        for ticker, quantity in self._owned_positions.items():
            position = trader.position_manager.positions.get(ticker.symbol)
            if (
                position is None
                or position.ticker != ticker
                or position.quantity < quantity
            ):
                self.requires_reconciliation = True
                self._position_state = 'reconciliation_required'
                self._hold(
                    'Strategy exposure does not match available account holdings'
                )
                return
            bid = trader.market_data.get_best_bid(ticker)
            if (
                bid is None
                or not bid.price.is_finite()
                or not bid.size.is_finite()
                or not 0 < bid.price < 1
                or bid.size < quantity
            ):
                self._hold('Owned exposure remains: missing executable exit liquidity')
                return
            legs.append(SpreadLeg(TradeSide.SELL, ticker, bid.price, quantity))
        self._basket_busy = True
        try:
            # Closing one leg must not buy it back when another exit fails.
            result = await SpreadExecutor(trader).execute_spread(
                legs, unwind_on_partial=False
            )
        except BaseException:
            self.requires_reconciliation = True
            self._position_state = 'reconciliation_required'
            raise
        finally:
            self._basket_busy = False
        self.last_execution_result = result
        if result.success:
            self._owned_positions.clear()
        else:
            for item in result.residual_exposures:
                self._owned_positions[item.ticker] += item.quantity
            self._owned_positions = {
                t: q for t, q in self._owned_positions.items() if q != 0
            }
        self.requires_reconciliation = result.execution_uncertain or any(
            q < 0 for q in self._owned_positions.values()
        )
        self._position_state = (
            'reconciliation_required'
            if self.requires_reconciliation
            else 'recovery_required'
            if self._owned_positions
            else 'flat'
        )
        self.record_decision(
            ticker_name=f'{self.name}({self._id_a}|{self._id_b})',
            action=action,
            executed=result.success,
            reasoning='Owned exposure closed'
            if result.success
            else result.failure_reason,
            signal_values={
                'all_filled': float(result.all_filled),
                'requires_reconciliation': float(self.requires_reconciliation),
                'remaining_owned_quantity': float(sum(self._owned_positions.values())),
            },
        )

    async def _recover_if_needed(self, trader: Trader) -> bool:
        if self.requires_reconciliation:
            self._hold('Reconcile outstanding orders before further execution')
            return True
        if self._position_state == 'recovery_required':
            await self._close_owned(trader, action='RECOVER')
            return True
        return False

    def execution_state(self) -> dict[str, Any]:
        """Inspectable state; callers must persist/reconcile it before restarting."""
        return {
            'position_state': self._position_state,
            'requires_reconciliation': self.requires_reconciliation,
            'owned_positions': {
                ticker.symbol: str(q) for ticker, q in self._owned_positions.items()
            },
            'last_execution': self.last_execution_result.to_payload()
            if self.last_execution_result
            else None,
        }
