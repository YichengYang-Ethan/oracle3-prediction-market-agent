from __future__ import annotations

import json
import math
from decimal import Decimal

from oracle3.events.events import Event, PriceChangeEvent
from oracle3.strategy.agent_strategy import AgentStrategy, TradeRequest
from oracle3.trader.trader import Trader


class OpenAIRelatedMarketAgentStrategy(AgentStrategy):
    """OpenAI Agents SDK strategy that analyzes cross-market relationships."""

    def __init__(
        self,
        trade_size: Decimal = Decimal('10'),
        agent_model: str = 'gpt-4.1-mini',
        agent_max_turns: int = 8,
        decision_cooldown_events: int = 60,
    ) -> None:
        super().__init__()
        self.trade_size = Decimal(str(trade_size))
        self.agent_model = agent_model
        self.agent_max_turns = agent_max_turns
        self.decision_cooldown_events = max(1, int(decision_cooldown_events))
        self._last_analysis_sequence_by_symbol: dict[str, int] = {}

    def build_agent_instructions(self, context=None) -> str:
        base = super().build_agent_instructions(context)
        return '\n'.join(
            [
                base,
                'You are searching for lead-lag or dependency relationships across markets.',
                'Use the available tools before deciding.',
                'place_trade queues a request, not a fill. Queued requests take precedence '
                'over the final JSON action; do not describe them as executed.',
                'Return strict JSON with keys: action, side, target_symbol, reasoning, confidence.',
                'Allowed action values: buy, sell, hold.',
                'Allowed side values: yes, no.',
                'Only choose target_symbol from the base market symbols returned by list_available_tickers.',
                'Prefer HOLD when evidence is weak.',
            ]
        )

    def build_task_input(self, context=None) -> str:
        ctx = context or self.require_context()
        ticker = ctx.ticker
        ticker_symbol = getattr(ticker, 'symbol', 'none')
        return '\n'.join(
            [
                f'Current event ticker: {ticker_symbol}',
                'Find whether another visible market is leading or confirming this market.',
                'If there is a tradable lag opportunity, propose action buy or sell with side yes or no.',
                'Otherwise return hold.',
            ]
        )

    async def process_event(self, event: Event, trader: Trader) -> None:
        if self.is_paused():
            return
        if not isinstance(event, PriceChangeEvent):
            return
        if event.ticker.symbol.endswith('_NO'):
            return

        context = self.require_context()
        if len(context.available_tickers(include_complements=False)) < 2:
            return
        if not self._should_analyze(context, event.ticker.symbol):
            return

        try:
            run_result = await self.run_openai_agent(context=context)
        except Exception as exc:
            discarded = self.take_pending_trades()
            self.record_decision(
                ticker_name=event.ticker.name or event.ticker.symbol,
                action='HOLD',
                executed=False,
                reasoning=(
                    f'agent analysis failed: {exc}; '
                    f'{len(discarded)} unsubmitted requests discarded'
                ),
                signal_values={},
            )
            return

        final_output = self.get_run_final_output(run_result)
        payload = self._parse_output(final_output)
        pending = bool(getattr(self, '_pending_trades', []))
        if payload is None and not pending:
            self.record_decision(
                ticker_name=event.ticker.name or event.ticker.symbol,
                action='HOLD',
                executed=False,
                reasoning=f'invalid agent output: {final_output[:160]}',
                signal_values={},
            )
            return

        payload = payload or {}

        action = str(payload.get('action', 'hold')).lower()
        side = str(payload.get('side', 'yes')).lower()
        target_symbol = str(payload.get('target_symbol') or event.ticker.symbol)
        try:
            confidence = float(payload.get('confidence', 0.0) or 0.0)
        except (TypeError, ValueError):
            confidence = 0.0
        if not math.isfinite(confidence):
            confidence = 0.0
        reasoning = str(payload.get('reasoning', ''))

        if action not in {'buy', 'sell', 'hold'}:
            action = 'hold'
        if side not in {'yes', 'no'}:
            side = 'yes'

        if action == 'hold' and not pending:
            self.record_decision(
                ticker_name=event.ticker.name or event.ticker.symbol,
                action='HOLD',
                executed=False,
                reasoning=reasoning or final_output[:200],
                confidence=confidence,
                signal_values={},
            )
            return

        target_ticker = context.resolve_trade_ticker(target_symbol, side)
        if target_ticker is None and not pending:
            self.record_decision(
                ticker_name=event.ticker.name or event.ticker.symbol,
                action='HOLD',
                executed=False,
                reasoning=f'agent returned unknown trade contract: {target_symbol}/{side}',
                confidence=confidence,
                signal_values={},
            )
            return

        # Both representations use one submission path. Tool requests take
        # precedence, even when the final JSON repeats a buy/sell instruction.
        if not pending and target_ticker is not None:
            quote = (
                trader.market_data.get_best_ask(target_ticker)
                if action == 'buy'
                else trader.market_data.get_best_bid(target_ticker)
            )
            position = trader.position_manager.get_position(target_ticker)
            quantity = (
                self.trade_size
                if action == 'buy'
                else (position.quantity if position is not None else Decimal('0'))
            )
            if quote is None or quantity <= 0:
                self.record_decision(
                    ticker_name=target_ticker.name or target_ticker.symbol,
                    action='HOLD',
                    executed=False,
                    reasoning='No executable quote or position for the requested action',
                    confidence=confidence,
                    signal_values={},
                )
                return
            self.queue_trade(
                TradeRequest(
                    target_ticker.symbol, action, float(quantity), float(quote.price)
                ),
                context=context,
            )

        outcomes = await self.execute_pending_trades(context=context)
        for outcome in outcomes:
            execution = outcome.execution
            actual_ticker = context.resolve_ticker(outcome.request.symbol)
            is_no = actual_ticker is not None and any(
                getattr(ticker, 'get_no_ticker', lambda: None)() == actual_ticker
                for ticker in context.available_tickers(include_complements=False)
            )
            self.record_decision(
                ticker_name=outcome.request.symbol,
                action=f'{outcome.request.side.upper()}_{"NO" if is_no else "YES"}',
                executed=outcome.executed,
                reasoning=(
                    f'{reasoning or final_output[:200]}; '
                    f'execution: {execution.failure_reason or "confirmed full fill"}'
                ),
                confidence=confidence,
                signal_values={
                    'fully_filled': float(execution.success),
                    'requires_reconciliation': float(execution.requires_reconciliation),
                },
            )

    @staticmethod
    def _parse_output(output: str) -> dict[str, object] | None:
        try:
            payload = json.loads(output)
        except json.JSONDecodeError:
            return None
        return payload if isinstance(payload, dict) else None

    def _should_analyze(self, context, symbol: str) -> bool:
        latest = context.market_history(limit=1)
        sequence = latest[0].sequence if latest else 0
        last_sequence = self._last_analysis_sequence_by_symbol.get(symbol, 0)
        if sequence - last_sequence < self.decision_cooldown_events:
            return False
        self._last_analysis_sequence_by_symbol[symbol] = sequence
        return True
