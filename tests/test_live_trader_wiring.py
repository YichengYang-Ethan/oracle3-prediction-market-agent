from __future__ import annotations

import sys
import types
from decimal import Decimal

import pytest

from oracle3.live import live_trader
from oracle3.position.position_manager import Position
from oracle3.risk.risk_manager import NoRiskManager
from oracle3.ticker.ticker import CashTicker, PolyMarketTicker


@pytest.mark.asyncio
async def test_run_live_trading_passes_continuous_and_drawdown(monkeypatch):
    captured: dict = {}

    class FakeEngine:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        async def start(self):
            captured['started'] = True

        async def stop(self):
            captured['stopped'] = True

    monkeypatch.setattr(live_trader, 'TradingEngine', FakeEngine)

    await live_trader.run_live_trading(
        data_source=object(),
        strategy=object(),
        trader=object(),
        continuous=False,
        drawdown_alert_pct=Decimal('0.12'),
    )

    assert captured['continuous'] is False
    assert captured['drawdown_alert_pct'] == Decimal('0.12')
    assert captured['started'] is True


@pytest.mark.asyncio
async def test_run_live_polymarket_trading_restores_missing_cash(monkeypatch):
    captured: dict = {}

    class FakePolymarketTrader:
        def __init__(self, **kwargs):
            self.position_manager = kwargs['position_manager']
            self.clob_client = self
            captured['trader'] = self

        def get_balance_allowance(self, params):
            return {'balance': '2500000'}  # 2.5 USDC (6 decimals)

    async def fake_run_live_trading(*args, **kwargs):
        captured['run_called'] = True

    # Provide a minimal fake py_clob_client.clob_types module for function-local import.
    pkg = types.ModuleType('py_clob_client')
    sub = types.ModuleType('py_clob_client.clob_types')

    class AssetType:
        COLLATERAL = 'collateral'

    class BalanceAllowanceParams:
        def __init__(self, asset_type):
            self.asset_type = asset_type

    sub.AssetType = AssetType
    sub.BalanceAllowanceParams = BalanceAllowanceParams
    monkeypatch.setitem(sys.modules, 'py_clob_client', pkg)
    monkeypatch.setitem(sys.modules, 'py_clob_client.clob_types', sub)

    monkeypatch.setattr(live_trader, 'PolymarketTrader', FakePolymarketTrader)
    monkeypatch.setattr(live_trader, 'run_live_trading', fake_run_live_trading)

    class FakeStore:
        def load_positions(self):
            # Deliberately no cash position.
            return [
                Position(
                    ticker=PolyMarketTicker(
                        symbol='T1', name='Market 1', token_id='tok1'
                    ),
                    quantity=Decimal('3'),
                    average_cost=Decimal('0.5'),
                    realized_pnl=Decimal('0'),
                )
            ]

    await live_trader.run_live_polymarket_trading(
        data_source=object(),
        strategy=object(),
        wallet_private_key='dummy',
        funder='0x0000000000000000000000000000000000000001',
        signature_type=0,
        risk_manager=NoRiskManager(),
        state_store=FakeStore(),
    )

    cash = captured['trader'].position_manager.get_position(CashTicker.POLYMARKET_USDC)
    assert cash is not None
    assert cash.quantity == Decimal('2.5')
    assert captured['run_called'] is True
