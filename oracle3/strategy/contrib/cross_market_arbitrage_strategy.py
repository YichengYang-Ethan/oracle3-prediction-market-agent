"""Cross-market arbitrage strategy across DFlow, Polymarket, and Kalshi.

Detects same-event price discrepancies across prediction market platforms
and trades when the spread exceeds a configurable threshold. Uses a
position state machine to manage entries and exits, and prices each leg
with its venue's published fee schedule from oracle3.fees.

Agent tool: find_arbitrage_opportunities() -> list[dict]

v2 improvements over v1:
- Per-leg venue fee schedules from oracle3.fees (net_edge = gross - leg fees)
- Position state machine (flat -> long_a_short_b / long_b_short_a -> flat)
- Exit logic when spread collapses (not just entry)
- Best bid/ask from order books instead of just last price
- Improved logging and decision recording with full signal payloads
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from decimal import Decimal
from difflib import SequenceMatcher
from typing import Any, ClassVar

from oracle3.events.events import Event, OrderBookEvent, PriceChangeEvent
from oracle3.strategy.fee_utils import estimate_fee
from oracle3.strategy.quant_strategy import QuantStrategy
from oracle3.ticker.ticker import Ticker
from oracle3.trader.trader import Trader
from oracle3.trader.types import TradeSide

logger = logging.getLogger(__name__)

_STOPWORDS = frozenset(
    {'will', 'the', 'a', 'an', 'of', 'in', 'on', 'by', 'to', 'for', 'be', 'is', 'at'}
)


def _normalize(text: str) -> str:
    """Lower, strip punctuation, remove stopwords."""
    text = re.sub(r'[^a-z0-9\s]', ' ', text.lower())
    tokens = [t for t in text.split() if t not in _STOPWORDS]
    return ' '.join(tokens)


# ---------------------------------------------------------------------------
# Ticker grouping across platforms
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TickerGroup:
    """A group of tickers from different platforms referring to the same event."""

    label: str
    tickers: dict[str, Ticker]  # platform_name -> Ticker
    similarity: float


class TickerGrouper:
    """Group tickers across DFlow/Polymarket/Kalshi by event name similarity."""

    def __init__(self, min_similarity: float = 0.60) -> None:
        self.min_similarity = min_similarity
        self._groups: list[TickerGroup] = []

    def group(self, platform_tickers: dict[str, list[Ticker]]) -> list[TickerGroup]:
        """Group tickers from multiple platforms by name similarity.

        Args:
            platform_tickers: mapping of platform_name -> list of tickers
        """
        platforms = list(platform_tickers.keys())
        if len(platforms) < 2:
            return []

        # Use the first platform as the base for matching
        base_platform = platforms[0]
        other_platforms = platforms[1:]
        base_tickers = platform_tickers[base_platform]

        groups: list[TickerGroup] = []
        for bt in base_tickers:
            bt_name = getattr(bt, 'name', '') or bt.symbol
            if not bt_name:
                continue
            bt_norm = _normalize(bt_name)

            group_tickers: dict[str, Ticker] = {base_platform: bt}
            min_sim = 1.0

            for other in other_platforms:
                best_score = 0.0
                best_ticker: Ticker | None = None
                for ot in platform_tickers[other]:
                    ot_name = getattr(ot, 'name', '') or ot.symbol
                    if not ot_name:
                        continue
                    score = SequenceMatcher(None, bt_norm, _normalize(ot_name)).ratio()
                    if score > best_score:
                        best_score = score
                        best_ticker = ot
                if best_ticker is not None and best_score >= self.min_similarity:
                    group_tickers[other] = best_ticker
                    min_sim = min(min_sim, best_score)

            if len(group_tickers) >= 2:
                groups.append(
                    TickerGroup(
                        label=bt_name[:60],
                        tickers=group_tickers,
                        similarity=round(min_sim, 3),
                    )
                )

        self._groups = groups
        return groups

    @property
    def groups(self) -> list[TickerGroup]:
        return list(self._groups)


# ---------------------------------------------------------------------------
# ArbitrageOpportunity
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ArbitrageOpportunity:
    """A detected arbitrage opportunity between two markets."""

    market_a: str  # platform:symbol
    market_b: str  # platform:symbol
    price_a: float  # best price for market A (bid or ask depending on direction)
    price_b: float  # best price for market B (bid or ask depending on direction)
    bid_a: float  # best bid for A
    ask_a: float  # best ask for A
    bid_b: float  # best bid for B
    ask_b: float  # best ask for B
    gross_edge: float  # raw spread before fees
    net_edge: float  # edge after per-side fees
    expected_profit: float  # net_edge * trade_size
    fees: float  # total fee cost for the round-trip
    direction: str  # 'buy_a_sell_b' or 'buy_b_sell_a'
    label: str = ''


# ---------------------------------------------------------------------------
# Position state for a single arb pair
# ---------------------------------------------------------------------------


@dataclass
class ArbPosition:
    """Tracks an active arbitrage position on a specific market pair.

    States: 'flat' -> 'long_a_short_b' | 'long_b_short_a' -> 'flat'

    long_a_short_b: bought YES on A, bought NO on B (A was cheaper)
    long_b_short_a: bought YES on B, bought NO on A (B was cheaper)
    """

    state: str = 'flat'  # flat | long_a_short_b | long_b_short_a
    entry_price_a: float = 0.0
    entry_price_b: float = 0.0
    entry_gross_edge: float = 0.0
    entry_net_edge: float = 0.0
    entry_time: float = 0.0
    ticker_a_symbol: str = ''
    ticker_b_symbol: str = ''
    label: str = ''


# ---------------------------------------------------------------------------
# CrossMarketArbitrageStrategy (v2)
# ---------------------------------------------------------------------------


class CrossMarketArbitrageStrategy(QuantStrategy):
    """Detect and trade cross-market arbitrage across DFlow/Polymarket/Kalshi.

    Monitors price events across platforms, matches tickers by event name
    similarity, and places arb trades when the spread exceeds ``min_edge``.

    v2 adds:
    - Per-leg venue fee schedules from oracle3.fees
    - Position state machine with proper entry and exit
    - Exit logic when spread collapses below ``exit_threshold``
    - Best bid/ask order book prices for accurate edge calculation
    - Comprehensive decision logging with full signal payloads
    """

    name: ClassVar[str] = 'cross_market_arbitrage'
    version: ClassVar[str] = '2.0.0'
    author: ClassVar[str] = 'oracle3'

    def __init__(
        self,
        min_edge: float = 0.03,
        exit_threshold: float = 0.005,
        trade_size: float = 10.0,
        cooldown_seconds: float = 60.0,
        fee_per_side: float | None = None,
        min_similarity: float = 0.60,
        max_hold_seconds: float = 3600.0,
        market_fee_terms: Mapping[str, Any] | None = None,
    ) -> None:
        """
        Parameters
        ----------
        min_edge:
            Minimum net edge (after fees) to trigger entry.
        exit_threshold:
            Exit when net edge collapses below this level (spread gone).
        trade_size:
            Dollar amount per trade leg.
        cooldown_seconds:
            Minimum seconds between arb entries on the same pair.
        fee_per_side:
            Optional flat dollar fee per side per contract that overrides the
            venue schedules. By default each leg pays its venue's taker fee
            (oracle3.fees); pairs on a venue without one (DFlow) are skipped.
        min_similarity:
            Minimum name similarity score (0-1) for cross-platform matching.
        max_hold_seconds:
            Maximum time to hold an arb position before forced exit.
        market_fee_terms:
            Per-market schedule overrides, keyed as in ``resolve_fee_schedule``.
        """
        super().__init__()
        self.min_edge = Decimal(str(min_edge))
        self.exit_threshold = Decimal(str(exit_threshold))
        self.trade_size = Decimal(str(trade_size))
        self.cooldown_seconds = cooldown_seconds
        self.fee_per_side = None if fee_per_side is None else Decimal(str(fee_per_side))
        self.market_fee_terms = dict(market_fee_terms or {})
        self._max_hold_seconds = max_hold_seconds

        self._grouper = TickerGrouper(min_similarity=min_similarity)

        # symbol -> latest YES price (from last event)
        self._prices: dict[str, Decimal] = {}
        # symbol -> best bid price
        self._best_bids: dict[str, Decimal] = {}
        # symbol -> best ask price
        self._best_asks: dict[str, Decimal] = {}
        # symbol -> platform name
        self._ticker_platform: dict[str, str] = {}
        # symbol -> ticker object
        self._ticker_map: dict[str, Ticker] = {}
        # label -> last entry time (for cooldown)
        self._last_entry_time: dict[str, float] = {}
        # label -> active ArbPosition
        self._positions: dict[str, ArbPosition] = {}
        # Detected opportunities (for agent tool)
        self._opportunities: list[ArbitrageOpportunity] = []
        # Flash loan integration hook (set by FlashLoanArbitrage)
        self.flash_loan_handler: Any = None

    # ------------------------------------------------------------------
    # Price registration (with order book awareness)
    # ------------------------------------------------------------------

    def register_price(
        self,
        platform: str,
        ticker: Ticker,
        price: Decimal,
        trader: Trader | None = None,
    ) -> None:
        """Register a price update from a platform.

        Also refreshes best bid/ask from the trader's MarketDataManager
        if available.
        """
        self._prices[ticker.symbol] = price
        self._ticker_platform[ticker.symbol] = platform
        self._ticker_map[ticker.symbol] = ticker

        # Update order book prices from MarketDataManager
        if trader is not None:
            bid_level = trader.market_data.get_best_bid(ticker)
            ask_level = trader.market_data.get_best_ask(ticker)
            if bid_level is not None:
                self._best_bids[ticker.symbol] = bid_level.price
            if ask_level is not None:
                self._best_asks[ticker.symbol] = ask_level.price

    def _get_effective_buy_price(self, symbol: str) -> Decimal | None:
        """Get the price we'd pay to buy YES (best ask, or last price)."""
        return self._best_asks.get(symbol) or self._prices.get(symbol)

    def _get_effective_sell_price(self, symbol: str) -> Decimal | None:
        """Get the price we'd receive selling YES (best bid, or last price)."""
        return self._best_bids.get(symbol) or self._prices.get(symbol)

    # ------------------------------------------------------------------
    # Fee-aware edge calculation
    # ------------------------------------------------------------------

    def _pair_fee(
        self, ticker_a: Ticker, price_a: Decimal, ticker_b: Ticker, price_b: Decimal
    ) -> Decimal:
        """Fee per contract for filling ``trade_size`` on both legs at YES prices.

        A NO fill at ``1 - p`` pays the YES fee at ``p`` under oracle3.fees.
        Raises ``ValueError`` when a leg's fee cannot be established.
        """
        fees = [
            estimate_fee(
                ticker,
                price,
                self.trade_size,
                market_fee_terms=self.market_fee_terms,
                legacy_fee_per_contract=self.fee_per_side,
            )
            for ticker, price in ((ticker_a, price_a), (ticker_b, price_b))
        ]
        return sum(fees, Decimal('0')) / self.trade_size

    def _compute_edge(
        self, buy: Ticker, buy_price: Decimal, sell: Ticker, sell_price: Decimal
    ) -> tuple[Decimal, Decimal, Decimal]:
        """Compute gross edge, fee cost, and net edge for a pair.

        The arb profit comes from: buy YES cheap on one platform,
        buy NO cheap on the other (equivalent to selling YES).

        For buy_a + sell_b (A cheaper):
            gross_edge = price_b - price_a
            (we buy A YES at ask_a, and buy B NO at (1 - bid_b))
            net_edge = gross_edge - venue fees of both legs (per contract)

        Returns (gross_edge, fees, net_edge).
        """
        gross_edge = sell_price - buy_price
        fees = self._pair_fee(buy, buy_price, sell, sell_price)
        net_edge = gross_edge - fees
        return gross_edge, fees, net_edge

    # ------------------------------------------------------------------
    # Opportunity scanning
    # ------------------------------------------------------------------

    def find_arbitrage_opportunities(self) -> list[dict[str, Any]]:
        """Agent tool: scan current prices for arbitrage opportunities.

        Uses best bid/ask from order books when available for more
        accurate edge calculation. Falls back to last traded price.

        Returns:
            List of opportunity dicts with full signal payload.
        """
        opportunities: list[ArbitrageOpportunity] = []

        # Group symbols by normalized name
        platform_symbols: dict[str, list[str]] = {}
        for symbol, platform in self._ticker_platform.items():
            platform_symbols.setdefault(platform, []).append(symbol)

        # Compare prices across platforms for same-event tickers
        platform_tickers: dict[str, list[Ticker]] = {}
        for platform, symbols in platform_symbols.items():
            platform_tickers[platform] = [
                self._ticker_map[s] for s in symbols if s in self._ticker_map
            ]

        groups = self._grouper.group(platform_tickers)

        for group in groups:
            platforms = list(group.tickers.keys())
            for i in range(len(platforms)):
                for j in range(i + 1, len(platforms)):
                    pa, pb = platforms[i], platforms[j]
                    ta, tb = group.tickers[pa], group.tickers[pb]

                    # Get order-book-aware prices
                    bid_a = self._get_effective_sell_price(ta.symbol)
                    ask_a = self._get_effective_buy_price(ta.symbol)
                    bid_b = self._get_effective_sell_price(tb.symbol)
                    ask_b = self._get_effective_buy_price(tb.symbol)

                    if bid_a is None or ask_a is None or bid_b is None or ask_b is None:
                        continue

                    try:
                        # Direction 1: A cheaper -> buy A YES (at ask_a), sell B YES (at bid_b)
                        # Equivalent to: buy A YES + buy B NO
                        gross_1, fees_1, net_1 = self._compute_edge(
                            ta, ask_a, tb, bid_b
                        )

                        # Direction 2: B cheaper -> buy B YES (at ask_b), sell A YES (at bid_a)
                        gross_2, fees_2, net_2 = self._compute_edge(
                            tb, ask_b, ta, bid_a
                        )
                    except ValueError as exc:  # a leg's fee cannot be established
                        logger.debug('Skipping %s: %s', group.label, exc)
                        continue

                    # Pick the better direction
                    if net_1 >= net_2 and net_1 > 0:
                        profit = float(self.trade_size * net_1)
                        if net_1 >= self.min_edge:
                            opp = ArbitrageOpportunity(
                                market_a=f'{pa}:{ta.symbol}',
                                market_b=f'{pb}:{tb.symbol}',
                                price_a=float(ask_a),
                                price_b=float(bid_b),
                                bid_a=float(bid_a),
                                ask_a=float(ask_a),
                                bid_b=float(bid_b),
                                ask_b=float(ask_b),
                                gross_edge=float(gross_1),
                                net_edge=float(net_1),
                                expected_profit=round(profit, 4),
                                fees=round(float(fees_1 * self.trade_size), 4),
                                direction='buy_a_sell_b',
                                label=group.label,
                            )
                            opportunities.append(opp)
                    elif net_2 > 0 and net_2 >= self.min_edge:
                        profit = float(self.trade_size * net_2)
                        opp = ArbitrageOpportunity(
                            market_a=f'{pa}:{ta.symbol}',
                            market_b=f'{pb}:{tb.symbol}',
                            price_a=float(bid_a),
                            price_b=float(ask_b),
                            bid_a=float(bid_a),
                            ask_a=float(ask_a),
                            bid_b=float(bid_b),
                            ask_b=float(ask_b),
                            gross_edge=float(gross_2),
                            net_edge=float(net_2),
                            expected_profit=round(profit, 4),
                            fees=round(float(fees_2 * self.trade_size), 4),
                            direction='buy_b_sell_a',
                            label=group.label,
                        )
                        opportunities.append(opp)

        self._opportunities = opportunities
        return [asdict(o) for o in opportunities]

    # ------------------------------------------------------------------
    # Main event handler
    # ------------------------------------------------------------------

    async def process_event(self, event: Event, trader: Trader) -> None:  # noqa: C901
        if self.is_paused():
            return

        ticker: Ticker | None = None
        price: Decimal | None = None

        if isinstance(event, PriceChangeEvent):
            ticker = event.ticker
            price = event.price
        elif isinstance(event, OrderBookEvent):
            ticker = event.ticker
            price = event.price if event.price > 0 else None
        else:
            return

        if ticker is None or price is None:
            return

        # Detect platform from ticker type
        platform = self._detect_platform(ticker)
        self.register_price(platform, ticker, price, trader=trader)

        # Check exit conditions for active positions
        await self._check_exits(trader)

        # Scan for new arb opportunities
        opps = self.find_arbitrage_opportunities()
        if not opps:
            return

        now = time.monotonic()
        for opp_dict in opps:
            label = opp_dict.get('label', '')
            opp_dict.get('direction', '')

            # Skip if we already have a position on this pair
            if label in self._positions and self._positions[label].state != 'flat':
                continue

            # Cooldown guard
            if now - self._last_entry_time.get(label, 0) < self.cooldown_seconds:
                self.record_decision(
                    ticker_name=label[:40],
                    action='HOLD',
                    executed=False,
                    reasoning=(
                        f'cooldown: {now - self._last_entry_time.get(label, 0):.1f}s '
                        f'< {self.cooldown_seconds:.1f}s'
                    ),
                    signal_values={
                        'net_edge': opp_dict['net_edge'],
                        'gross_edge': opp_dict['gross_edge'],
                    },
                )
                continue

            # Execute the arb entry
            await self._enter_arb(trader, opp_dict, now)

    # ------------------------------------------------------------------
    # Entry logic
    # ------------------------------------------------------------------

    async def _enter_arb(
        self, trader: Trader, opp: dict[str, Any], now: float
    ) -> None:
        """Enter an arbitrage position: buy YES cheap + buy NO on the other."""
        label = opp['label']
        direction = opp['direction']
        market_a = opp['market_a']
        market_b = opp['market_b']
        symbol_a = market_a.split(':', 1)[-1]
        symbol_b = market_b.split(':', 1)[-1]

        ticker_a = self._ticker_map.get(symbol_a)
        ticker_b = self._ticker_map.get(symbol_b)
        if ticker_a is None or ticker_b is None:
            return

        leg1_ok = False
        leg2_ok = False

        if direction == 'buy_a_sell_b':
            # Leg 1: Buy A YES at best ask
            buy_price_a = self._get_effective_buy_price(symbol_a)
            if buy_price_a is not None:
                leg1_ok = await self._place_arb_leg(
                    trader, ticker_a, TradeSide.BUY, buy_price_a, 'buy_yes_A'
                )

            # Leg 2: Buy B NO (equivalent to selling B YES)
            if leg1_ok:
                ticker_b_no = getattr(ticker_b, 'get_no_ticker', lambda: None)()
                if ticker_b_no is not None:
                    sell_price_b = self._get_effective_sell_price(symbol_b)
                    if sell_price_b is not None:
                        no_price = Decimal('1') - sell_price_b
                        leg2_ok = await self._place_arb_leg(
                            trader, ticker_b_no, TradeSide.BUY, no_price, 'buy_no_B'
                        )

            new_state = 'long_a_short_b'
            entry_price_a = float(buy_price_a or 0)
            entry_price_b = float(self._get_effective_sell_price(symbol_b) or 0)

        else:  # buy_b_sell_a
            # Leg 1: Buy B YES at best ask
            buy_price_b = self._get_effective_buy_price(symbol_b)
            if buy_price_b is not None:
                leg1_ok = await self._place_arb_leg(
                    trader, ticker_b, TradeSide.BUY, buy_price_b, 'buy_yes_B'
                )

            # Leg 2: Buy A NO (equivalent to selling A YES)
            if leg1_ok:
                ticker_a_no = getattr(ticker_a, 'get_no_ticker', lambda: None)()
                if ticker_a_no is not None:
                    sell_price_a = self._get_effective_sell_price(symbol_a)
                    if sell_price_a is not None:
                        no_price = Decimal('1') - sell_price_a
                        leg2_ok = await self._place_arb_leg(
                            trader, ticker_a_no, TradeSide.BUY, no_price, 'buy_no_A'
                        )

            new_state = 'long_b_short_a'
            entry_price_a = float(self._get_effective_sell_price(symbol_a) or 0)
            entry_price_b = float(buy_price_b or 0)

        executed = leg1_ok and leg2_ok

        # Record position state
        if executed:
            self._positions[label] = ArbPosition(
                state=new_state,
                entry_price_a=entry_price_a,
                entry_price_b=entry_price_b,
                entry_gross_edge=opp['gross_edge'],
                entry_net_edge=opp['net_edge'],
                entry_time=now,
                ticker_a_symbol=symbol_a,
                ticker_b_symbol=symbol_b,
                label=label,
            )
            self._last_entry_time[label] = now

        self.record_decision(
            ticker_name=label[:40],
            action=f'ARB_ENTRY_{direction.upper()}',
            executed=executed,
            reasoning=(
                f'Arb {direction}: {market_a} vs {market_b}, '
                f'gross_edge={opp["gross_edge"]:.4f}, '
                f'net_edge={opp["net_edge"]:.4f}, '
                f'profit={opp["expected_profit"]:.4f}, '
                f'leg1={"OK" if leg1_ok else "FAIL"}, '
                f'leg2={"OK" if leg2_ok else "FAIL"}'
            ),
            signal_values={
                'gross_edge': opp['gross_edge'],
                'net_edge': opp['net_edge'],
                'expected_profit': opp['expected_profit'],
                'fees': opp['fees'],
                'bid_a': opp['bid_a'],
                'ask_a': opp['ask_a'],
                'bid_b': opp['bid_b'],
                'ask_b': opp['ask_b'],
                'direction': 1.0 if direction == 'buy_a_sell_b' else -1.0,
            },
        )

        if executed:
            logger.info(
                'ARB ENTRY %s: %s vs %s, net_edge=%.4f, profit=%.4f',
                direction, market_a, market_b,
                opp['net_edge'], opp['expected_profit'],
            )
        elif leg1_ok and not leg2_ok:
            logger.warning(
                'ARB PARTIAL: leg1 filled but leg2 failed for %s — '
                'position may be unhedged',
                label,
            )

    # ------------------------------------------------------------------
    # Exit logic
    # ------------------------------------------------------------------

    async def _check_exits(self, trader: Trader) -> None:
        """Check all active arb positions for exit conditions.

        Exit when:
        1. Spread has collapsed: current net edge < exit_threshold
        2. Spread has reversed: we can profit by unwinding
        3. Max hold time exceeded: forced exit (timeout protection)
        """
        now = time.monotonic()
        labels_to_close: list[str] = []

        for label, pos in self._positions.items():
            if pos.state == 'flat':
                continue

            symbol_a = pos.ticker_a_symbol
            symbol_b = pos.ticker_b_symbol

            # Get current prices
            bid_a = self._get_effective_sell_price(symbol_a)
            ask_a = self._get_effective_buy_price(symbol_a)
            bid_b = self._get_effective_sell_price(symbol_b)
            ask_b = self._get_effective_buy_price(symbol_b)

            if bid_a is None or ask_a is None or bid_b is None or ask_b is None:
                continue

            ticker_a = self._ticker_map[symbol_a]
            ticker_b = self._ticker_map[symbol_b]

            # Compute current edge in same direction as entry
            try:
                if pos.state == 'long_a_short_b':
                    # We bought A YES and B NO.
                    # To exit: sell A YES (at bid_a) and sell B NO (at 1-ask_b).
                    # Current edge for same-direction re-entry:
                    edge = self._compute_edge(ticker_a, ask_a, ticker_b, bid_b)
                    exit_fees = self._pair_fee(ticker_a, bid_a, ticker_b, ask_b)
                else:  # long_b_short_a
                    edge = self._compute_edge(ticker_b, ask_b, ticker_a, bid_a)
                    exit_fees = self._pair_fee(ticker_a, ask_a, ticker_b, bid_b)
            except ValueError as exc:  # a leg's fee cannot be established
                logger.warning('Skipping exit check for %s: %s', label, exc)
                continue
            current_gross, _, current_net = edge

            # Determine exit reason
            exit_reason: str | None = None

            # 1. Spread collapsed — the arb is gone or reversed
            if current_net < self.exit_threshold:
                exit_reason = 'spread_collapsed'

            # 2. Max hold time exceeded
            hold_time = now - pos.entry_time
            if hold_time >= self._max_hold_seconds:
                exit_reason = 'timeout'

            if exit_reason is None:
                continue

            # Execute exit: sell our positions
            exit_ok = await self._exit_position(trader, pos)
            labels_to_close.append(label)

            # Estimate PnL
            if pos.state == 'long_a_short_b':
                # Bought A YES at entry_price_a, selling at bid_a
                pnl_a = float(bid_a) - pos.entry_price_a
                # Bought B NO at (1-entry_price_b), selling at (1-ask_b)
                pnl_b = (1.0 - float(ask_b)) - (1.0 - pos.entry_price_b)
                pnl_gross = pnl_a + pnl_b
            else:
                pnl_b = float(bid_b) - pos.entry_price_b
                pnl_a = (1.0 - float(ask_a)) - (1.0 - pos.entry_price_a)
                pnl_gross = pnl_a + pnl_b

            entry_fees = pos.entry_gross_edge - pos.entry_net_edge
            pnl_net = pnl_gross - entry_fees - float(exit_fees)

            self.record_decision(
                ticker_name=label[:40],
                action=f'ARB_EXIT_{exit_reason.upper()}',
                executed=exit_ok,
                reasoning=(
                    f'Exit({exit_reason}): entry_edge={pos.entry_net_edge:.4f}, '
                    f'current_net={float(current_net):.4f}, '
                    f'hold={hold_time:.0f}s, '
                    f'pnl_gross={pnl_gross:.4f}, pnl_net={pnl_net:.4f}'
                ),
                signal_values={
                    'entry_gross_edge': pos.entry_gross_edge,
                    'entry_net_edge': pos.entry_net_edge,
                    'current_gross_edge': float(current_gross),
                    'current_net_edge': float(current_net),
                    'hold_seconds': hold_time,
                    'pnl_gross': pnl_gross,
                    'pnl_net': pnl_net,
                    'bid_a': float(bid_a),
                    'ask_a': float(ask_a),
                    'bid_b': float(bid_b),
                    'ask_b': float(ask_b),
                },
            )

            logger.info(
                'ARB EXIT %s (%s): entry_edge=%.4f current=%.4f '
                'pnl_net=%.4f hold=%.0fs',
                label[:30], exit_reason, pos.entry_net_edge,
                float(current_net), pnl_net, hold_time,
            )

        # Reset closed positions to flat
        for label in labels_to_close:
            if label in self._positions:
                self._positions[label].state = 'flat'

    async def _exit_position(self, trader: Trader, pos: ArbPosition) -> bool:
        """Unwind an arb position by selling all related holdings."""
        any_sold = False
        for position in trader.position_manager.positions.values():
            if position.quantity <= 0:
                continue
            symbol = position.ticker.symbol
            # Check if this position belongs to our arb pair
            if symbol not in (
                pos.ticker_a_symbol, pos.ticker_b_symbol,
                f'{pos.ticker_a_symbol}_NO', f'{pos.ticker_b_symbol}_NO',
            ):
                # Also check via substring for NO tickers with different naming
                ticker_a_base = pos.ticker_a_symbol.replace('_NO', '')
                ticker_b_base = pos.ticker_b_symbol.replace('_NO', '')
                if ticker_a_base not in symbol and ticker_b_base not in symbol:
                    continue

            best_bid = trader.market_data.get_best_bid(position.ticker)
            if best_bid is not None:
                ok = await self._place_arb_leg(
                    trader, position.ticker, TradeSide.SELL,
                    best_bid.price, f'exit_{symbol[:20]}',
                )
                if ok:
                    any_sold = True
        return any_sold

    # ------------------------------------------------------------------
    # Order placement
    # ------------------------------------------------------------------

    async def _place_arb_leg(
        self,
        trader: Trader,
        ticker: Ticker,
        side: TradeSide,
        price: Decimal,
        leg_label: str = '',
    ) -> bool:
        """Place one leg of an arbitrage trade. Returns True if accepted."""
        try:
            result = await trader.place_order(
                side=side, ticker=ticker,
                limit_price=price, quantity=self.trade_size,
            )
            if result.failure_reason:
                logger.warning(
                    'Arb leg failed [%s]: %s %s @ %s - %s',
                    leg_label, side.value, ticker.symbol[:20],
                    price, result.failure_reason,
                )
                return False
            logger.info(
                'Arb leg placed [%s]: %s %s @ %s',
                leg_label, side.value, ticker.symbol[:20], price,
            )
            return True
        except Exception:
            logger.exception(
                'Error placing arb leg [%s]: %s %s',
                leg_label, side.value, ticker.symbol[:20],
            )
            return False

    # ------------------------------------------------------------------
    # Platform detection
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_platform(ticker: Ticker) -> str:
        """Detect platform from ticker type."""
        from oracle3.ticker.ticker import KalshiTicker, PolyMarketTicker, SolanaTicker

        if isinstance(ticker, SolanaTicker):
            return 'dflow'
        if isinstance(ticker, PolyMarketTicker):
            return 'polymarket'
        if isinstance(ticker, KalshiTicker):
            return 'kalshi'
        return 'unknown'

    # ------------------------------------------------------------------
    # Properties for external inspection
    # ------------------------------------------------------------------

    @property
    def opportunities(self) -> list[ArbitrageOpportunity]:
        return list(self._opportunities)

    @property
    def active_positions(self) -> dict[str, ArbPosition]:
        """Return all non-flat arb positions."""
        return {
            label: pos
            for label, pos in self._positions.items()
            if pos.state != 'flat'
        }

    @property
    def position_count(self) -> int:
        """Number of active (non-flat) arb positions."""
        return sum(1 for pos in self._positions.values() if pos.state != 'flat')
