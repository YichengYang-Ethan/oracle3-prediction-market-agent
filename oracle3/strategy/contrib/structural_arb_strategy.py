"""Convergence trading under the assumed relation P(A) = slope*P(B) + intercept.

A fitted or supplied probability relationship is not a verified payoff identity.
The basket may lose money when the relation changes or does not hold at settlement.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from decimal import Decimal
from typing import Any

from oracle3.events.events import Event
from oracle3.strategy.contrib.basket_execution import BasketExecutionStrategy
from oracle3.trader.trader import Trader


class StructuralArbStrategy(BasketExecutionStrategy):
    """Trade a weighted residual; no guaranteed convergence or settlement payout.

    ``trade_size`` is A contracts and B uses ``abs(slope)`` times that quantity.
    ``min_edge`` applies to modeled residual after entry fees per A contract;
    future exit fees/slippage are additional. Shared venue schedules are default;
    ``fee_rate`` is an explicit legacy flat dollar fee per contract.
    ``market_fee_terms`` overrides market schedules. ``yes_token_ids`` identifies
    YES outcomes for opaque Polymarket token IDs. Unknown orders stop automation.
    """

    name = 'structural_arb'
    version = '1.0.0'
    author = 'oracle3'

    def __init__(
        self,
        market_id_a: str = '',
        market_id_b: str = '',
        trade_size: float = 10.0,
        slope: float = 1.0,
        intercept: float = 0.0,
        min_edge: float = 0.02,
        exit_fraction: float = 0.5,
        cooldown_seconds: float = 120.0,
        fee_rate: float | None = None,
        market_fee_terms: Mapping[str, Any] | None = None,
        yes_token_ids: Mapping[str, str] | None = None,
    ) -> None:
        super().__init__(
            market_id_a,
            market_id_b,
            trade_size,
            min_edge,
            cooldown_seconds,
            fee_rate,
            market_fee_terms,
            yes_token_ids,
        )
        if not all(math.isfinite(v) for v in (slope, intercept, exit_fraction)):
            raise ValueError('slope, intercept and exit_fraction must be finite')
        if not 0 <= exit_fraction <= 1:
            raise ValueError('exit_fraction must be between zero and one')
        self.slope, self.intercept, self.exit_fraction = slope, intercept, exit_fraction

    def _expected_a(self, price_b: float) -> float:
        return self.slope * price_b + self.intercept

    async def process_event(self, event: Event, trader: Trader) -> None:
        if not self._accept_event(event) or await self._recover_if_needed(trader):
            return
        assert self._price_a is not None and self._price_b is not None
        pa = float(self._price_a)
        expected = self._expected_a(float(self._price_b))
        residual = pa - expected
        if self._position_state == 'flat':
            if residual > float(self.min_edge):
                await self._enter_short_a(trader, pa, expected, residual)
            elif residual < -float(self.min_edge):
                await self._enter_long_a(trader, pa, expected, residual)
            else:
                self._hold(
                    'Residual is within the assumed convergence band',
                    residual=residual,
                    expected=expected,
                )
        elif abs(residual) <= float(self.min_edge) * self.exit_fraction:
            await self._exit(trader, residual)

    async def _enter_short_a(
        self, trader: Trader, pa: float, expected: float, residual: float
    ) -> None:
        slope, intercept = Decimal(str(self.slope)), Decimal(str(self.intercept))
        reference = Decimal('1') - intercept + max(-slope, Decimal('0'))
        await self._enter_basket(
            trader,
            [(self._id_a, False, Decimal('1')), (self._id_b, slope >= 0, abs(slope))],
            reference,
            'short_a_long_b' if slope >= 0 else 'short_both',
            'ENTER_SHORT_A',
            model_dependent=True,
        )

    async def _enter_long_a(
        self, trader: Trader, pa: float, expected: float, residual: float
    ) -> None:
        slope, intercept = Decimal(str(self.slope)), Decimal(str(self.intercept))
        reference = intercept + max(slope, Decimal('0'))
        await self._enter_basket(
            trader,
            [(self._id_a, True, Decimal('1')), (self._id_b, slope < 0, abs(slope))],
            reference,
            'long_a_short_b' if slope >= 0 else 'long_both',
            'ENTER_LONG_A',
            model_dependent=True,
        )

    async def _exit(self, trader: Trader, residual: float) -> None:
        await self._close_owned(trader)
