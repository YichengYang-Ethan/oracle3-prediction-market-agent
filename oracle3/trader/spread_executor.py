"""Sequential spread execution with verified, best-effort recovery.

This is not atomic execution. A trader may leave orders outstanding or raise
without reporting whether an order filled. Known fills are unwound in LIFO order
only after terminal failures; uncertain venue state must be reconciled first.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from oracle3.ticker.ticker import Ticker
from oracle3.trader.types import (
    Order,
    OrderFailureReason,
    OrderStatus,
    PlaceOrderResult,
    TradeSide,
)

if TYPE_CHECKING:
    from oracle3.trader.trader import Trader

logger = logging.getLogger(__name__)


@dataclass
class SpreadLeg:
    """One order in a spread, with quantities expressed in contracts."""

    side: TradeSide
    ticker: Ticker
    price: Decimal
    quantity: Decimal

    def to_payload(self) -> dict[str, Any]:
        return {
            'side': self.side.value,
            'ticker': {'symbol': self.ticker.symbol, 'name': self.ticker.name},
            'price': str(self.price),
            'quantity': str(self.quantity),
        }


@dataclass
class ResidualExposure:
    """Known position change from an unsuccessful spread; positive means long.

    Quantities are relative to positions before the attempt. They exclude any
    unknown fills, so an empty list does not prove flat when execution is uncertain.
    """

    ticker: Ticker
    quantity: Decimal

    def to_payload(self) -> dict[str, Any]:
        return {
            'ticker': {'symbol': self.ticker.symbol, 'name': self.ticker.name},
            'quantity': str(self.quantity),
        }


@dataclass
class UnwindAttempt:
    """Recovery of one original leg (``leg_index`` is one-based)."""

    leg_index: int
    leg: SpreadLeg
    result: PlaceOrderResult | None = None
    failure_reason: str = ''
    execution_uncertain: bool = False

    def to_payload(self) -> dict[str, Any]:
        return {
            'leg_index': self.leg_index,
            'leg': self.leg.to_payload(),
            'result': self.result.to_payload() if self.result is not None else None,
            'failure_reason': self.failure_reason,
            'execution_uncertain': self.execution_uncertain,
        }


@dataclass
class SpreadOrderResult:
    """Observed execution, never a guarantee of atomicity or venue cancellation.

    ``success`` requires every requested quantity to be confirmed fully filled.
    ``hedged`` means recovery neutralized all known fills and no order is uncertain.
    It describes this attempt's position change, not the account's total position.
    Pending/partial orders and exceptions require reconciliation before
    any automatic reversal is attempted. No cancellation or retry is implied by this result.
    """

    success: bool
    leg_results: list[PlaceOrderResult] = field(default_factory=list)
    hedged: bool = False
    unwind_details: str = ''
    failure_reason: str = ''
    requested_legs: list[SpreadLeg] = field(default_factory=list)
    unwind_results: list[UnwindAttempt] = field(default_factory=list)
    residual_exposures: list[ResidualExposure] = field(default_factory=list)
    execution_uncertain: bool = False
    uncertain_leg_indices: list[int] = field(default_factory=list)

    @property
    def all_filled(self) -> bool:
        """Require nonempty, complete, exact confirmations for all requested legs."""
        return (
            bool(self.requested_legs)
            and len(self.leg_results) == len(self.requested_legs)
            and all(
                not _inspect_result(result, leg)[1]
                for leg, result in zip(self.requested_legs, self.leg_results)
            )
        )

    @property
    def requires_reconciliation(self) -> bool:
        return self.execution_uncertain or bool(self.residual_exposures)

    def to_payload(self) -> dict[str, Any]:
        """JSON-safe state for agents; decimal quantities retain their precision."""
        return {
            'success': self.success,
            'all_filled': self.all_filled,
            'hedged': self.hedged,
            'failure_reason': self.failure_reason,
            'unwind_details': self.unwind_details,
            'execution_uncertain': self.execution_uncertain,
            'requires_reconciliation': self.requires_reconciliation,
            'uncertain_leg_indices': self.uncertain_leg_indices,
            'requested_legs': [leg.to_payload() for leg in self.requested_legs],
            'leg_results': [result.to_payload() for result in self.leg_results],
            'unwind_results': [attempt.to_payload() for attempt in self.unwind_results],
            'residual_exposures': [
                item.to_payload() for item in self.residual_exposures
            ],
        }


def _valid_decimal(value: object, *, positive: bool = False) -> bool:
    return (
        isinstance(value, Decimal)
        and value.is_finite()
        and (value > 0 if positive else value >= 0)
    )


def _inspect_result(
    result: PlaceOrderResult, leg: SpreadLeg
) -> tuple[Decimal, str, bool]:
    """Return known fill, failure description and uncertainty for one response."""
    order = result.order
    if order is None:
        reason = result.failure_reason
        known_rejection = reason is not None and not result.requires_reconciliation
        return (
            Decimal('0'),
            reason.value if reason else 'no order returned',
            not known_rejection,
        )
    if (
        not isinstance(order, Order)
        or order.ticker != leg.ticker
        or order.side != leg.side
        or not _valid_decimal(order.filled_quantity)
    ):
        return Decimal('0'), 'invalid or mismatched order response', True
    if order.execution_uncertain or not order.accounting_confirmed:
        return (
            Decimal('0'),
            order.reconciliation_reason or 'fill accounting is unconfirmed',
            True,
        )
    filled = order.filled_quantity
    if not _valid_decimal(order.remaining) or not isinstance(order.status, OrderStatus):
        return filled, 'invalid order state', True
    uncertain = order.status in {
        OrderStatus.PLACED,
        OrderStatus.PARTIALLY_FILLED,
        OrderStatus.UNKNOWN,
    } or (order.status == OrderStatus.FILLED and order.remaining != 0)
    if order.status == OrderStatus.REJECTED and filled > 0:
        uncertain = True
    if result.requires_reconciliation:
        uncertain = True
    if (
        order.status == OrderStatus.FILLED
        and filled == leg.quantity
        and order.remaining == 0
        and result.failure_reason is None
        and not uncertain
    ):
        return filled, '', False
    failure = (
        result.failure_reason.value
        if result.failure_reason is not None
        else f'{order.status.value}: filled {filled} of {leg.quantity}'
        + ('; reconciliation required' if uncertain else '')
    )
    return filled, failure, uncertain


class SpreadExecutor:
    """Submit legs sequentially; stop at the first incomplete or uncertain order.

    On a terminal failure, optionally reverse confirmed fills, including the
    current terminal partial leg. An open order, exception or unconfirmed account
    delta stops execution and recovery until the caller reconciles venue state.
    """

    def __init__(self, trader: Trader) -> None:
        self._trader = trader

    async def execute_spread(
        self, legs: list[SpreadLeg], *, unwind_on_partial: bool = True
    ) -> SpreadOrderResult:
        if not legs:
            return SpreadOrderResult(success=False, failure_reason='no legs provided')
        for i, leg in enumerate(legs, start=1):
            if (
                not isinstance(leg, SpreadLeg)
                or not isinstance(leg.side, TradeSide)
                or not isinstance(leg.ticker, Ticker)
                or not _valid_decimal(leg.price, positive=True)
                or not _valid_decimal(leg.quantity, positive=True)
            ):
                return SpreadOrderResult(
                    success=False, failure_reason=f'leg {i} has invalid order inputs'
                )
        # Snapshot requests so a caller cannot mutate them across awaits.
        requested = [
            SpreadLeg(leg.side, leg.ticker, leg.price, leg.quantity) for leg in legs
        ]
        outcome = SpreadOrderResult(success=False, requested_legs=requested)
        known: dict[Ticker, Decimal] = {}
        filled_legs: list[tuple[int, SpreadLeg, Decimal]] = []
        for index, leg in enumerate(requested, start=1):
            result, failure, uncertain, filled = await self._submit(leg)
            outcome.leg_results.append(result)
            if filled > 0:
                self._add_exposure(known, leg, filled)
                filled_legs.append((index, leg, filled))
            if uncertain:
                outcome.execution_uncertain = True
                outcome.uncertain_leg_indices.append(index)
            if failure:
                outcome.failure_reason = f'leg {index} failed: {failure}'
                if (
                    unwind_on_partial
                    and filled_legs
                    and not outcome.execution_uncertain
                ):
                    await self._unwind(filled_legs, known, outcome)
                outcome.residual_exposures = [
                    ResidualExposure(ticker, quantity)
                    for ticker, quantity in known.items()
                    if quantity != 0
                ]
                outcome.hedged = (
                    bool(outcome.unwind_results)
                    and not outcome.residual_exposures
                    and not outcome.execution_uncertain
                )
                return outcome
        outcome.success = True
        return outcome

    async def _submit(
        self, leg: SpreadLeg
    ) -> tuple[PlaceOrderResult, str, bool, Decimal]:
        try:
            result = await self._trader.place_order(
                side=leg.side,
                ticker=leg.ticker,
                limit_price=leg.price,
                quantity=leg.quantity,
            )
        except Exception as exc:
            # A timeout/transport exception cannot establish whether a venue filled.
            logger.warning(
                'Order outcome uncertain for %s (%s)',
                leg.ticker.symbol,
                type(exc).__name__,
            )
            return (
                PlaceOrderResult(failure_reason=OrderFailureReason.UNKNOWN),
                f'order raised {type(exc).__name__}; reconcile venue state',
                True,
                Decimal('0'),
            )
        if not isinstance(result, PlaceOrderResult):
            return (
                PlaceOrderResult(failure_reason=OrderFailureReason.UNKNOWN),
                'invalid trader response; reconcile venue state',
                True,
                Decimal('0'),
            )
        try:
            filled, failure, uncertain = _inspect_result(result, leg)
            # A malformed adapter response must not break the agent-facing payload.
            result.to_payload()
        except Exception:
            return (
                PlaceOrderResult(failure_reason=OrderFailureReason.UNKNOWN),
                'malformed order response; reconcile venue state',
                True,
                Decimal('0'),
            )
        return result, failure, uncertain, filled

    @staticmethod
    def _add_exposure(
        known: dict[Ticker, Decimal], leg: SpreadLeg, filled: Decimal
    ) -> None:
        signed = filled if leg.side == TradeSide.BUY else -filled
        known[leg.ticker] = known.get(leg.ticker, Decimal('0')) + signed

    async def _unwind(
        self,
        filled_legs: list[tuple[int, SpreadLeg, Decimal]],
        known: dict[Ticker, Decimal],
        outcome: SpreadOrderResult,
    ) -> None:
        details: list[str] = []
        for index, original, filled in reversed(filled_legs):
            side = TradeSide.SELL if original.side == TradeSide.BUY else TradeSide.BUY
            leg = SpreadLeg(side, original.ticker, original.price, filled)
            attempt = UnwindAttempt(index, leg)
            outcome.unwind_results.append(attempt)
            try:
                md = self._trader.market_data
                quote = (
                    md.get_best_bid(leg.ticker)
                    if side == TradeSide.SELL
                    else md.get_best_ask(leg.ticker)
                )
                if quote is None or not _valid_decimal(quote.price, positive=True):
                    attempt.failure_reason = 'no valid executable quote for unwind'
                else:
                    leg.price = quote.price
            except Exception as exc:
                # Quote access failed before an order was submitted.
                attempt.failure_reason = (
                    f'unwind preparation raised {type(exc).__name__}'
                )
            if not attempt.failure_reason:
                result, failure, uncertain, reversed_fill = await self._submit(leg)
                attempt.result = result
                attempt.failure_reason = failure
                attempt.execution_uncertain = uncertain
                outcome.execution_uncertain |= uncertain
                self._add_exposure(known, leg, reversed_fill)
            label = 'FAILED' if attempt.failure_reason else 'FILLED'
            details.append(
                f'{label} unwind {side.value} {leg.ticker.symbol} qty={filled}: {attempt.failure_reason or "confirmed"}'
            )
            if attempt.execution_uncertain:
                details.append(
                    'Recovery stopped: reconcile uncertain unwind before further orders'
                )
                break
        outcome.unwind_details = '; '.join(details)

    async def execute_pair_spread(
        self,
        buy_ticker: Ticker,
        sell_ticker: Ticker,
        quantity: Decimal,
        *,
        unwind_on_partial: bool = True,
    ) -> SpreadOrderResult:
        """Read best ask/bid and attempt a buy/sell pair with the same contract size."""
        md = self._trader.market_data
        ask = md.get_best_ask(buy_ticker)
        bid = md.get_best_bid(sell_ticker)
        if ask is None:
            return SpreadOrderResult(
                success=False,
                failure_reason=f'no ask available for {buy_ticker.symbol}',
            )
        if bid is None:
            return SpreadOrderResult(
                success=False,
                failure_reason=f'no bid available for {sell_ticker.symbol}',
            )
        return await self.execute_spread(
            [
                SpreadLeg(TradeSide.BUY, buy_ticker, ask.price, quantity),
                SpreadLeg(TradeSide.SELL, sell_ticker, bid.price, quantity),
            ],
            unwind_on_partial=unwind_on_partial,
        )
