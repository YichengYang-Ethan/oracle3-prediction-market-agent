"""Convergence trading under caller-supplied conditional probability bounds.

The bounds lower*P(B) <= P(A) <= 1-(1-upper)*P(B) concern expectations. They do
not establish a guaranteed state-by-state payoff or automatically verify a
relationship between the contracts' settlement rules.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from decimal import Decimal
from typing import Any

from oracle3.events.events import Event
from oracle3.strategy.contrib.basket_execution import BasketExecutionStrategy
from oracle3.trader.trader import Trader


class ConditionalArbStrategy(BasketExecutionStrategy):
    """Trade a weighted basket reflecting the supplied conditional bound.

    Lower violation: buy one YES(A) and ``cond_lower`` NO(B).
    Upper violation: buy one NO(A) and ``1-cond_upper`` NO(B).
    ``trade_size`` is A contracts; B contracts use the indicated multiplier.
    ``min_edge`` is modeled edge per A contract after entry fees, excluding future
    exit costs. This is a convergence hypothesis, not a risk-free arbitrage.
    Shared fee schedules are the default; ``fee_rate`` explicitly overrides them
    with a flat dollar fee per contract. ``market_fee_terms`` supplies market terms;
    ``yes_token_ids`` identifies YES outcomes for opaque Polymarket token IDs.
    """

    name = 'conditional_arb'
    version = '1.0.0'
    author = 'oracle3'

    def __init__(
        self,
        market_id_a: str = '',
        market_id_b: str = '',
        trade_size: float = 10.0,
        cond_lower: float = 0.0,
        cond_upper: float = 1.0,
        min_edge: float = 0.02,
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
        if (
            not all(math.isfinite(v) for v in (cond_lower, cond_upper))
            or not 0 <= cond_lower <= cond_upper <= 1
        ):
            raise ValueError('conditional bounds must satisfy 0 <= lower <= upper <= 1')
        self.cond_lower, self.cond_upper = cond_lower, cond_upper

    def _compute_bounds(self, price_b: float) -> tuple[float, float]:
        return self.cond_lower * price_b, 1 - (1 - self.cond_upper) * price_b

    async def process_event(self, event: Event, trader: Trader) -> None:
        if not self._accept_event(event) or await self._recover_if_needed(trader):
            return
        assert self._price_a is not None and self._price_b is not None
        pa = float(self._price_a)
        lower, upper = self._compute_bounds(float(self._price_b))
        if self._position_state == 'flat':
            if pa > upper + float(self.min_edge):
                await self._enter_short_a(trader, pa, lower, upper)
            elif pa < lower - float(self.min_edge):
                await self._enter_long_a(trader, pa, lower, upper)
            else:
                self._hold(
                    'Price is within the assumed conditional band',
                    lower=lower,
                    upper=upper,
                    price_a=pa,
                )
        elif lower <= pa <= upper:
            await self._exit(trader, pa, lower, upper)

    async def _enter_short_a(
        self, trader: Trader, pa: float, lower: float, upper: float
    ) -> None:
        weight = Decimal('1') - Decimal(str(self.cond_upper))
        await self._enter_basket(
            trader,
            [(self._id_a, False, Decimal('1')), (self._id_b, False, weight)],
            weight,
            'short_both',
            'ENTER_SHORT_A',
            model_dependent=True,
        )

    async def _enter_long_a(
        self, trader: Trader, pa: float, lower: float, upper: float
    ) -> None:
        weight = Decimal(str(self.cond_lower))
        await self._enter_basket(
            trader,
            [(self._id_a, True, Decimal('1')), (self._id_b, False, weight)],
            weight,
            'long_a_short_b',
            'ENTER_LONG_A',
            model_dependent=True,
        )

    async def _exit(
        self, trader: Trader, pa: float, lower: float, upper: float
    ) -> None:
        await self._close_owned(trader)
