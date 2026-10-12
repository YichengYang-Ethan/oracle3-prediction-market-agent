"""Trade executable baskets conditional on an asserted implication A => B.

For example, winning an election implies participating in that election, so
P(win) <= P(participate). NO(A) + YES(B) pays at least one only when the asserted
relationship and the two contracts' settlement rules are valid.
"""

from __future__ import annotations

from collections.abc import Mapping
from decimal import Decimal
from typing import Any

from oracle3.events.events import Event
from oracle3.strategy.contrib.basket_execution import BasketExecutionStrategy
from oracle3.trader.trader import Trader


class ImplicationArbStrategy(BasketExecutionStrategy):
    """Buy NO(A) and YES(B) when executable cost plus entry fees is below one.

    ``trade_size`` is contracts per leg, not dollars. ``min_edge`` is the minimum
    edge per basket after entry fees; an early exit can incur further fees/slippage.
    ``fee_rate`` is an optional legacy flat dollar fee per contract. By default,
    shared venue schedules apply; ``market_fee_terms`` overrides per-market terms.
    ``yes_token_ids`` maps market IDs to YES token IDs for opaque Polymarket tokens.
    Only this instance's confirmed exposure is closed. Unknown fills halt execution.
    """

    name = 'implication_arb'
    version = '1.0.0'
    author = 'oracle3'

    def __init__(
        self,
        market_id_a: str = '',
        market_id_b: str = '',
        trade_size: float = 10.0,
        min_edge: float = 0.01,
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

    async def process_event(self, event: Event, trader: Trader) -> None:
        if not self._accept_event(event) or await self._recover_if_needed(trader):
            return
        assert self._price_a is not None and self._price_b is not None
        violation = self._price_a - self._price_b
        if self._position_state == 'flat':
            if violation > self.min_edge:
                await self._enter(trader, violation)
            else:
                self._hold('No implication violation', violation=float(violation))
        elif violation <= 0:
            await self._exit(trader, violation)

    async def _enter(self, trader: Trader, violation: Decimal) -> None:
        await self._enter_basket(
            trader,
            [(self._id_a, False, Decimal('1')), (self._id_b, True, Decimal('1'))],
            Decimal('1'),
            'short_a_long_b',
            'ENTER_ARB',
        )

    async def _exit(self, trader: Trader, violation: Decimal) -> None:
        await self._close_owned(trader, action='EXIT_ARB')
