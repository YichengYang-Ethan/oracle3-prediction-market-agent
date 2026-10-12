"""Trade executable NO baskets conditional on asserted mutual exclusivity.

If A and B cannot both resolve YES, NO(A) + NO(B) pays at least one. This strategy
requires the caller to establish that relation and compatible settlement rules.
"""

from __future__ import annotations

from collections.abc import Mapping
from decimal import Decimal
from typing import Any

from oracle3.events.events import Event
from oracle3.strategy.contrib.basket_execution import BasketExecutionStrategy
from oracle3.trader.trader import Trader


class ExclusivityArbStrategy(BasketExecutionStrategy):
    """Buy both NO tokens at an entry-fee-adjusted discount to one.

    ``trade_size`` is contracts per leg. ``min_edge`` applies after entry fees;
    fees/slippage on early exits are additional. Shared venue schedules replace
    the former flat default; ``fee_rate`` retains an explicit flat-dollar override.
    ``market_fee_terms`` accepts per-market schedule overrides. Supply
    ``yes_token_ids`` for opaque Polymarket tokens. Unknown fills stop automation;
    exits touch only the quantities acquired by this strategy instance.
    """

    name = 'exclusivity_arb'
    version = '1.0.0'
    author = 'oracle3'

    def __init__(
        self,
        market_id_a: str = '',
        market_id_b: str = '',
        trade_size: float = 10.0,
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

    async def process_event(self, event: Event, trader: Trader) -> None:
        if not self._accept_event(event) or await self._recover_if_needed(trader):
            return
        assert self._price_a is not None and self._price_b is not None
        violation = self._price_a + self._price_b - 1
        if self._position_state == 'flat':
            if violation > self.min_edge:
                await self._enter(trader, violation)
            else:
                self._hold('No exclusivity violation', violation=float(violation))
        elif violation <= 0:
            await self._exit(trader, violation)

    async def _enter(self, trader: Trader, violation: Decimal) -> None:
        await self._enter_basket(
            trader,
            [(self._id_a, False, Decimal('1')), (self._id_b, False, Decimal('1'))],
            Decimal('1'),
            'short_both',
            'ENTER_ARB',
        )

    async def _exit(self, trader: Trader, violation: Decimal) -> None:
        await self._close_owned(trader, action='EXIT_ARB')
