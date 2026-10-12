import logging
from abc import ABC, abstractmethod
from decimal import Decimal

from oracle3.data.market_data_manager import MarketDataManager
from oracle3.position.position_manager import PositionManager
from oracle3.ticker.ticker import CashTicker, Ticker
from oracle3.trader.types import TradeSide

logger = logging.getLogger(__name__)


class RiskManager(ABC):
    @abstractmethod
    async def check_trade(
        self, ticker: Ticker, side: TradeSide, quantity: Decimal, price: Decimal
    ) -> bool:
        """Check if a trade meets risk management criteria."""
        pass


class NoRiskManager(RiskManager):
    """A risk manager that allows all trades (no risk checks)."""

    async def check_trade(
        self, ticker: Ticker, side: TradeSide, quantity: Decimal, price: Decimal
    ) -> bool:
        return True


class StandardRiskManager(RiskManager):
    """
    A comprehensive risk manager with configurable limits.

    Checks:
    - Maximum single trade size
    - Maximum position size per ticker
    - Maximum total portfolio exposure
    - Maximum drawdown limit
    - Daily loss limit
    """

    def __init__(
        self,
        position_manager: PositionManager,
        market_data: MarketDataManager,
        max_single_trade_size: Decimal = Decimal('1000'),
        max_position_size: Decimal = Decimal('5000'),
        max_total_exposure: Decimal = Decimal('50000'),
        max_drawdown_pct: Decimal = Decimal('0.20'),
        daily_loss_limit: Decimal | None = None,
        max_positions: int = 10,
        initial_capital: Decimal | None = None,
    ):
        """
        Initialize the risk manager.

        Args:
            position_manager: The position manager to check positions
            market_data: Market data manager for price lookups
            max_single_trade_size: Maximum value for a single trade
            max_position_size: Maximum position value per ticker
            max_total_exposure: Maximum total portfolio exposure
            max_drawdown_pct: Maximum drawdown as a percentage (0.20 = 20%)
            daily_loss_limit: Maximum daily loss allowed (optional)
            max_positions: Maximum number of open positions allowed
            initial_capital: Starting capital for drawdown calculation
        """
        self.position_manager = position_manager
        self.market_data = market_data
        self.max_single_trade_size = max_single_trade_size
        self.max_position_size = max_position_size
        self.max_total_exposure = max_total_exposure
        self.max_drawdown_pct = max_drawdown_pct
        self.daily_loss_limit = daily_loss_limit
        self.max_positions = max_positions
        self.initial_capital = initial_capital

        # Track peak portfolio value for drawdown calculation
        self._peak_portfolio_value: Decimal | None = None
        self._daily_starting_value: Decimal | None = None
        self._daily_pnl = Decimal('0')

    def _get_portfolio_value(self) -> Decimal:
        """Get the current total portfolio value."""
        portfolio_values = self.position_manager.get_portfolio_value(self.market_data)
        return sum(portfolio_values.values(), Decimal('0'))

    def _check_trade_size(self, quantity: Decimal, price: Decimal) -> bool:
        """Check if the trade size is within limits."""
        trade_value = quantity * price
        return trade_value <= self.max_single_trade_size

    def _check_position_limit(
        self, ticker: Ticker, side: TradeSide, quantity: Decimal, price: Decimal
    ) -> bool:
        """Check if the resulting position would exceed limits."""
        current_position = self.position_manager.get_position(ticker)
        current_quantity = (
            current_position.quantity if current_position else Decimal('0')
        )

        if side == TradeSide.BUY:
            new_quantity = current_quantity + quantity
        else:
            new_quantity = current_quantity - quantity

        # For sells, we don't need to check position limit (reducing position)
        if side == TradeSide.SELL:
            return True

        new_position_value = new_quantity * price
        return new_position_value <= self.max_position_size

    def _check_total_exposure(
        self, side: TradeSide, quantity: Decimal, price: Decimal
    ) -> bool:
        """Check if the trade would exceed total exposure limits."""
        if side == TradeSide.SELL:
            # Selling reduces exposure
            return True

        trade_value = quantity * price
        current_exposure = self._get_portfolio_value()

        # Subtract cash from exposure calculation (cash is not market exposure)
        cash_positions = self.position_manager.get_cash_positions()
        cash_value = sum(pos.quantity for pos in cash_positions)
        market_exposure = current_exposure - cash_value

        new_exposure = market_exposure + trade_value
        return new_exposure <= self.max_total_exposure

    def _check_drawdown(self) -> bool:
        """Check if we're within drawdown limits."""
        current_value = self._get_portfolio_value()

        # Initialize peak if not set
        if self._peak_portfolio_value is None:
            self._peak_portfolio_value = (
                self.initial_capital if self.initial_capital else current_value
            )

        # Update peak if current value is higher
        if current_value > self._peak_portfolio_value:
            self._peak_portfolio_value = current_value

        if self._peak_portfolio_value <= 0:
            return True

        drawdown = (
            self._peak_portfolio_value - current_value
        ) / self._peak_portfolio_value
        return drawdown < self.max_drawdown_pct

    def _check_daily_loss(self) -> bool:
        """Check if we're within daily loss limits."""
        if self.daily_loss_limit is None:
            return True

        current_value = self._get_portfolio_value()

        if self._daily_starting_value is None:
            self._daily_starting_value = current_value

        daily_pnl = current_value - self._daily_starting_value
        return daily_pnl > -self.daily_loss_limit

    def _check_max_positions(self, ticker: Ticker, side: TradeSide) -> bool:
        """Check if we would exceed maximum number of positions."""
        if side == TradeSide.SELL:
            return True

        current_positions = self.position_manager.get_non_cash_positions()
        open_positions = [p for p in current_positions if p.quantity > 0]

        # Check if we already have a position in this ticker
        existing_position = self.position_manager.get_position(ticker)
        if existing_position and existing_position.quantity > 0:
            return True  # Adding to existing position is fine

        return len(open_positions) < self.max_positions

    async def check_trade(
        self, ticker: Ticker, side: TradeSide, quantity: Decimal, price: Decimal
    ) -> bool:
        """
        Check if a trade meets all risk management criteria.

        Args:
            ticker: The ticker being traded
            side: Buy or sell
            quantity: Quantity to trade
            price: Price per unit

        Returns:
            True if the trade is allowed, False otherwise
        """
        if (
            not isinstance(side, TradeSide)
            or not quantity.is_finite()
            or not price.is_finite()
            or quantity <= 0
            or price < 0
        ):
            return False
        # Incomplete valuation must never create apparent buying capacity.
        if side == TradeSide.BUY and self.position_manager.get_missing_marks(
            self.market_data
        ):
            logger.warning('Risk check failed: held inventory has missing market marks')
            return False

        # Skip risk checks for cash tickers
        if isinstance(ticker, CashTicker):
            return True

        # Check trade size limit
        if not self._check_trade_size(quantity, price):
            logger.warning(
                'Risk check failed: Trade size %s exceeds limit %s',
                quantity * price,
                self.max_single_trade_size,
            )
            return False

        # Check position limit
        if not self._check_position_limit(ticker, side, quantity, price):
            logger.warning(
                'Risk check failed: Position would exceed limit %s',
                self.max_position_size,
            )
            return False

        # Check total exposure
        if not self._check_total_exposure(side, quantity, price):
            logger.warning(
                'Risk check failed: Total exposure would exceed limit %s',
                self.max_total_exposure,
            )
            return False

        # Check drawdown
        if not self._check_drawdown():
            logger.warning(
                'Risk check failed: Drawdown exceeds limit %s%%',
                self.max_drawdown_pct * 100,
            )
            return False

        # Check daily loss
        if not self._check_daily_loss():
            logger.warning(
                'Risk check failed: Daily loss exceeds limit %s',
                self.daily_loss_limit,
            )
            return False

        # Check max positions
        if not self._check_max_positions(ticker, side):
            logger.warning(
                'Risk check failed: Would exceed max positions %s', self.max_positions
            )
            return False

        return True

    def check_portfolio_health(self) -> tuple[bool, str]:
        """Post-trade/runtime health check for hard risk breaches.

        Returns:
            (ok, reason) where reason is non-empty when ok is False.
        """
        if not self._check_drawdown():
            return False, 'drawdown_limit_breached'
        if not self._check_daily_loss():
            return False, 'daily_loss_limit_breached'
        current_exposure = self._get_portfolio_value()
        cash_positions = self.position_manager.get_cash_positions()
        cash_value = sum(pos.quantity for pos in cash_positions)
        market_exposure = current_exposure - cash_value
        if market_exposure > self.max_total_exposure:
            return False, 'total_exposure_limit_breached'
        return True, ''

    def reset_daily_tracking(self) -> None:
        """Reset daily tracking (call at start of each trading day)."""
        self._daily_starting_value = self._get_portfolio_value()
        self._daily_pnl = Decimal('0')

    def update_peak(self) -> None:
        """Update the peak portfolio value."""
        current_value = self._get_portfolio_value()
        if (
            self._peak_portfolio_value is None
            or current_value > self._peak_portfolio_value
        ):
            self._peak_portfolio_value = current_value

    def get_current_drawdown(self) -> Decimal:
        """Get the current drawdown percentage."""
        current_value = self._get_portfolio_value()

        if self._peak_portfolio_value is None or self._peak_portfolio_value <= 0:
            return Decimal('0')

        return (self._peak_portfolio_value - current_value) / self._peak_portfolio_value

    def get_remaining_exposure(self) -> Decimal:
        """Get the remaining exposure capacity."""
        current_exposure = self._get_portfolio_value()
        cash_positions = self.position_manager.get_cash_positions()
        cash_value = sum(pos.quantity for pos in cash_positions)
        market_exposure = current_exposure - cash_value

        return self.max_total_exposure - market_exposure


class ConservativeRiskManager(StandardRiskManager):
    """A conservative risk manager with tighter limits."""

    def __init__(
        self,
        position_manager: PositionManager,
        market_data: MarketDataManager,
        initial_capital: Decimal | None = None,
    ):
        super().__init__(
            position_manager=position_manager,
            market_data=market_data,
            max_single_trade_size=Decimal('500'),
            max_position_size=Decimal('2000'),
            max_total_exposure=Decimal('10000'),
            max_drawdown_pct=Decimal('0.10'),
            daily_loss_limit=Decimal('500'),
            max_positions=5,
            initial_capital=initial_capital,
        )


class AggressiveRiskManager(StandardRiskManager):
    """An aggressive risk manager with looser limits."""

    def __init__(
        self,
        position_manager: PositionManager,
        market_data: MarketDataManager,
        initial_capital: Decimal | None = None,
    ):
        super().__init__(
            position_manager=position_manager,
            market_data=market_data,
            max_single_trade_size=Decimal('5000'),
            max_position_size=Decimal('20000'),
            max_total_exposure=Decimal('100000'),
            max_drawdown_pct=Decimal('0.30'),
            daily_loss_limit=None,  # No daily limit
            max_positions=20,
            initial_capital=initial_capital,
        )
