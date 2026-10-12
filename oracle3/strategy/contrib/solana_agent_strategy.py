"""Solana/DFlow-optimized agent strategy for Oracle3."""

from __future__ import annotations

import logging
import re
from decimal import Decimal
from typing import Any

from oracle3.events.events import Event, NewsEvent, OrderBookEvent, PriceChangeEvent
from oracle3.strategy.agent_strategy import AgentStrategy, TradeRequest
from oracle3.trader.trader import Trader
from oracle3.trader.types import TradeSide

logger = logging.getLogger(__name__)


class SolanaAgentStrategy(AgentStrategy):
    """LLM-driven strategy optimized for DFlow prediction markets on Solana.

    Uses the OpenAI Agents SDK to analyze Solana-tokenized prediction markets,
    incorporating on-chain liquidity signals and DFlow-specific market structure.

    The agent has a ``place_trade`` tool it can call during analysis.  Queued
    trades are executed after the agent run finishes.  For non-news events
    (order-book and price changes) a lightweight heuristic fallback is used
    instead of calling the LLM.
    """

    name = 'SolanaAgentStrategy'
    version = '1.1.0'
    author = 'Yicheng Yang'

    agent_name = 'oracle3-solana-agent'
    agent_model = 'deepseek-chat'
    agent_max_turns = 8

    def __init__(
        self,
        trade_size: float = 10.0,
        confidence_threshold: float = 0.6,
        max_position_pct: float = 0.15,
    ) -> None:
        super().__init__()
        self.trade_size = Decimal(str(trade_size))
        self.confidence_threshold = confidence_threshold
        self.max_position_pct = max_position_pct

    # ------------------------------------------------------------------
    # Prompt
    # ------------------------------------------------------------------

    def get_prompt_guide(self) -> str:
        return (
            'You are an AI trading agent running inside Oracle3, specialized in '
            'Solana-based prediction markets via DFlow. '
            'DFlow tokenizes Kalshi prediction markets on Solana mainnet-beta, '
            'allowing on-chain settlement with SPL tokens. '
            'Use the bound StrategyContext to inspect market data: '
            'context.ticker_history() for price history, '
            'context.order_books() for current bid/ask, '
            'context.positions() for portfolio exposure, '
            'context.recent_news() for market-moving news. '
            'Consider the following when making decisions: '
            '1. DFlow markets mirror Kalshi — watch for cross-exchange pricing gaps. '
            '2. Solana settlement is near-instant — no need to worry about settlement risk. '
            '3. On-chain liquidity may be thinner — prefer smaller position sizes. '
            '4. Focus on high-conviction trades with clear catalysts. '
            'When you decide to trade, call the place_trade tool with the symbol, '
            'side (buy/sell), quantity, and limit_price. '
            'If your confidence is below the threshold or the signal is unclear, '
            'do NOT place a trade — simply state your analysis and recommendation. '
            'At the END of your response, always include a summary line in this format: '
            'SIGNAL: <BUY|SELL|HOLD> <TICKER> confidence=<0.0-1.0> '
            'Do not use future information. Only act on visible context data.'
        )

    # ------------------------------------------------------------------
    # Event routing
    # ------------------------------------------------------------------

    async def process_event(self, event: Event, trader: Trader) -> None:
        if self.is_paused():
            return

        context = self.require_context()

        if isinstance(event, NewsEvent):
            await self._handle_news_event(event, trader, context)
        elif isinstance(event, OrderBookEvent):
            await self._handle_order_book_event(event, trader, context)
        elif isinstance(event, PriceChangeEvent):
            await self._handle_price_change_event(event, trader, context)

    # ------------------------------------------------------------------
    # NewsEvent — full LLM agent path
    # ------------------------------------------------------------------

    async def _handle_news_event(
        self, event: NewsEvent, trader: Trader, context: Any
    ) -> None:
        ticker_name = str(event.ticker.symbol) if event.ticker else 'unknown'

        if self.sdk_available():
            try:
                result = await self.run_openai_agent(context=context)
                output = self.get_run_final_output(result)

                # Execute any trades the agent queued via place_trade tool
                pending = self.take_pending_trades()
                if pending:
                    await self._execute_pending_trades(pending, trader)

                # Parse the agent's text output for a signal summary
                signal = self._parse_signal(output)
                action = signal.get('action', 'HOLD')
                confidence = signal.get('confidence', 0.0)
                signal_ticker = signal.get('ticker', ticker_name)

                # If the agent did NOT use the place_trade tool but expressed
                # high confidence in its text output, place the trade now.
                if not pending and action in ('BUY', 'SELL'):
                    if confidence >= self.confidence_threshold:
                        await self._execute_signal_trade(
                            action=action,
                            ticker_name=signal_ticker,
                            trader=trader,
                            confidence=confidence,
                            reasoning=output[:300],
                        )
                        self.record_decision(
                            ticker_name=signal_ticker,
                            action=action,
                            executed=True,
                            confidence=confidence,
                            reasoning=output[:500],
                            signal_values={'source': 'openai_agent_signal'},
                        )
                        return

                self.record_decision(
                    ticker_name=signal_ticker,
                    action=action,
                    executed=bool(pending),
                    confidence=confidence,
                    reasoning=output[:500],
                    signal_values={'source': 'openai_agent'},
                )
            except Exception as e:
                logger.exception('Agent error for %s', ticker_name)
                self.record_decision(
                    ticker_name=ticker_name,
                    action='ERROR',
                    executed=False,
                    reasoning=f'Agent error: {e}',
                    signal_values={},
                )
        else:
            prompt = self.build_prompt_context(context)
            self.record_decision(
                ticker_name=ticker_name,
                action='HOLD',
                executed=False,
                reasoning=f'SDK not available. Context: {prompt[:200]}',
                signal_values={},
            )

    # ------------------------------------------------------------------
    # OrderBookEvent — spread-based heuristic (no LLM)
    # ------------------------------------------------------------------

    async def _handle_order_book_event(
        self, event: OrderBookEvent, trader: Trader, context: Any
    ) -> None:
        """Light heuristic: buy when the bid/ask spread is tight and
        the order book is heavily bid-skewed (more buying pressure)."""
        ticker = event.ticker
        symbol = ticker.symbol

        best_bid = trader.market_data.get_best_bid(ticker)
        best_ask = trader.market_data.get_best_ask(ticker)
        if best_bid is None or best_ask is None:
            return

        bid_price = float(best_bid.price)
        ask_price = float(best_ask.price)
        if ask_price <= 0:
            return

        spread = (ask_price - bid_price) / ask_price
        bid_size = float(best_bid.size) if best_bid.size else 0.0
        ask_size = float(best_ask.size) if best_ask.size else 0.0
        total_size = bid_size + ask_size

        if total_size <= 0:
            return

        imbalance = (bid_size - ask_size) / total_size  # -1 to +1

        # Tight spread + heavy bid imbalance -> BUY signal
        # Wide spread or heavy ask imbalance -> no action
        confidence = 0.0
        action = 'HOLD'
        if spread < 0.03 and imbalance > 0.3:
            confidence = min(0.5 + imbalance * 0.5, 1.0)
            action = 'BUY'
        elif spread < 0.03 and imbalance < -0.3:
            confidence = min(0.5 + abs(imbalance) * 0.5, 1.0)
            action = 'SELL'

        if action != 'HOLD' and confidence >= self.confidence_threshold:
            side = TradeSide.BUY if action == 'BUY' else TradeSide.SELL
            price = (
                Decimal(str(ask_price)) if action == 'BUY' else Decimal(str(bid_price))
            )
            quantity = self._cap_quantity(self.trade_size, ticker, trader)
            if quantity > Decimal('0'):
                result = await trader.place_order(side, ticker, price, quantity)
                executed = result.accepted
                self.record_decision(
                    ticker_name=symbol,
                    action=action,
                    executed=executed,
                    confidence=confidence,
                    reasoning=f'Spread={spread:.4f} imbalance={imbalance:.2f}',
                    signal_values={
                        'source': 'order_book_heuristic',
                        'spread': spread,
                        'imbalance': imbalance,
                    },
                )
                return

        self.record_decision(
            ticker_name=symbol,
            action='HOLD',
            executed=False,
            confidence=confidence,
            reasoning=f'OB spread={spread:.4f} imbalance={imbalance:.2f}',
            signal_values={
                'source': 'order_book_heuristic',
                'spread': spread,
                'imbalance': imbalance,
            },
        )

    # ------------------------------------------------------------------
    # PriceChangeEvent — momentum heuristic (no LLM)
    # ------------------------------------------------------------------

    async def _handle_price_change_event(
        self, event: PriceChangeEvent, trader: Trader, context: Any
    ) -> None:
        """Simple momentum: compare current price to short-term average.
        Buy when price is above average (uptrend), sell when below."""
        ticker = event.ticker
        symbol = ticker.symbol
        current_price = float(event.price)

        prices = context.price_history(limit=20)
        if len(prices) < 5:
            return

        avg_price = sum(float(p) for p in prices) / len(prices)
        if avg_price <= 0:
            return

        momentum = (current_price - avg_price) / avg_price

        action = 'HOLD'
        confidence = 0.0
        if momentum > 0.02:
            action = 'BUY'
            confidence = min(0.5 + momentum * 5, 1.0)
        elif momentum < -0.02:
            action = 'SELL'
            confidence = min(0.5 + abs(momentum) * 5, 1.0)

        if action != 'HOLD' and confidence >= self.confidence_threshold:
            side = TradeSide.BUY if action == 'BUY' else TradeSide.SELL
            best_ask = trader.market_data.get_best_ask(ticker)
            best_bid = trader.market_data.get_best_bid(ticker)

            if action == 'BUY' and best_ask is not None:
                price = best_ask.price
            elif action == 'SELL' and best_bid is not None:
                price = best_bid.price
            else:
                price = Decimal(str(current_price))

            quantity = self._cap_quantity(self.trade_size, ticker, trader)
            if quantity > Decimal('0'):
                result = await trader.place_order(side, ticker, price, quantity)
                executed = result.accepted
                self.record_decision(
                    ticker_name=symbol,
                    action=action,
                    executed=executed,
                    confidence=confidence,
                    reasoning=f'Momentum={momentum:.4f} avg={avg_price:.4f}',
                    signal_values={
                        'source': 'price_momentum',
                        'momentum': momentum,
                        'avg_price': avg_price,
                    },
                )
                return

        self.record_decision(
            ticker_name=symbol,
            action='HOLD',
            executed=False,
            confidence=confidence,
            reasoning=f'Momentum={momentum:.4f} avg={avg_price:.4f}',
            signal_values={
                'source': 'price_momentum',
                'momentum': momentum,
                'avg_price': avg_price,
            },
        )

    # ------------------------------------------------------------------
    # Trade execution helpers
    # ------------------------------------------------------------------

    async def _execute_pending_trades(
        self, trades: list[TradeRequest], trader: Trader
    ) -> None:
        """Execute trade requests queued by the agent's place_trade tool."""
        for req in trades:
            ticker = self.require_context().resolve_ticker(req.symbol)
            if ticker is None:
                logger.warning('Cannot resolve ticker %s for queued trade', req.symbol)
                continue

            side = TradeSide.BUY if req.side == 'buy' else TradeSide.SELL
            quantity = self._cap_quantity(Decimal(str(req.quantity)), ticker, trader)
            if quantity <= Decimal('0'):
                logger.info(
                    'Skipping queued trade for %s: capped quantity is 0', req.symbol
                )
                continue

            price = Decimal(str(req.limit_price))
            logger.info(
                'Executing queued trade: %s %s of %s @ %s',
                req.side,
                quantity,
                req.symbol,
                price,
            )
            result = await trader.place_order(side, ticker, price, quantity)
            if result.failure_reason is not None:
                logger.warning(
                    'Queued %s order failed for %s: %s',
                    req.side,
                    req.symbol,
                    result.failure_reason,
                )
            elif result.order is not None:
                logger.info(
                    'Queued %s order placed for %s: status=%s filled=%s',
                    req.side,
                    req.symbol,
                    result.order.status,
                    result.order.filled_quantity,
                )

    async def _execute_signal_trade(
        self,
        *,
        action: str,
        ticker_name: str,
        trader: Trader,
        confidence: float,
        reasoning: str,
    ) -> None:
        """Place a trade based on a parsed text signal from the agent."""
        ctx = self.require_context()
        ticker = ctx.resolve_ticker(ticker_name)
        if ticker is None:
            # Fall back to the event ticker
            ticker = ctx.ticker
        if ticker is None:
            logger.warning('Cannot resolve ticker %s for signal trade', ticker_name)
            return

        side = TradeSide.BUY if action == 'BUY' else TradeSide.SELL

        if side == TradeSide.BUY:
            best = trader.market_data.get_best_ask(ticker)
        else:
            best = trader.market_data.get_best_bid(ticker)

        if best is None:
            logger.warning(
                'No %s available for %s',
                'ask' if side == TradeSide.BUY else 'bid',
                ticker_name,
            )
            return

        price = best.price
        quantity = self._cap_quantity(self.trade_size, ticker, trader)
        if quantity <= Decimal('0'):
            logger.info('Capped quantity is 0 for %s, skipping', ticker_name)
            return

        logger.info(
            'Placing signal trade: %s %s of %s @ %s (confidence=%.2f)',
            action,
            quantity,
            ticker_name,
            price,
            confidence,
        )
        result = await trader.place_order(side, ticker, price, quantity)
        if result.failure_reason is not None:
            logger.warning(
                'Signal trade failed for %s: %s', ticker_name, result.failure_reason
            )

    # ------------------------------------------------------------------
    # Position sizing
    # ------------------------------------------------------------------

    def _cap_quantity(self, desired: Decimal, ticker: Any, trader: Trader) -> Decimal:
        """Cap *desired* quantity so the resulting position does not exceed
        ``max_position_pct`` of the total portfolio value."""
        portfolio_values = trader.position_manager.get_portfolio_value(
            trader.market_data
        )
        total_value = sum(portfolio_values.values(), Decimal('0'))
        if total_value <= Decimal('0'):
            return desired

        max_value = total_value * Decimal(str(self.max_position_pct))

        # Current position value for this ticker
        position = trader.position_manager.get_position(ticker)
        current_qty = position.quantity if position else Decimal('0')

        best_ask = trader.market_data.get_best_ask(ticker)
        price_est = best_ask.price if best_ask else Decimal('1')
        if price_est <= Decimal('0'):
            price_est = Decimal('1')

        current_value = current_qty * price_est
        remaining_value = max_value - current_value
        if remaining_value <= Decimal('0'):
            return Decimal('0')

        max_qty = (remaining_value / price_est).quantize(Decimal('1'))
        return min(desired, max_qty)

    # ------------------------------------------------------------------
    # Signal parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_signal(text: str) -> dict[str, Any]:
        """Extract a trading signal from the agent's free-form text output.

        Looks for a line like:
            SIGNAL: BUY AAPL confidence=0.85
        Returns ``{'action': 'BUY', 'ticker': 'AAPL', 'confidence': 0.85}``.
        Falls back to ``{'action': 'HOLD', ...}`` if no match.
        """
        pattern = r'SIGNAL:\s*(BUY|SELL|HOLD)\s+(\S+)\s+confidence\s*=\s*([\d.]+)'
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return {
                'action': match.group(1).upper(),
                'ticker': match.group(2),
                'confidence': float(match.group(3)),
            }
        return {'action': 'HOLD', 'ticker': 'unknown', 'confidence': 0.0}
