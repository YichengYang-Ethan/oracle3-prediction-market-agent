"""Edge-case tests for oracle3.pricing.greeks.kelly_fraction.

Covers boundary inputs that drive bet sizing:
- p_market at exactly 0 and 1 (clamped to the open interval)
- Non-finite p_market / lambda (must never reach position sizing)
- Extreme |lambda| (output stays capped at [-0.25, 0.25])
- Zero fee_rate (a real edge must produce a strictly non-zero bet)
- Negative fee_rate (rejected with ValueError)
"""

from __future__ import annotations

import math

import pytest

from oracle3.pricing.greeks import kelly_fraction


class TestKellyFractionEdgeCases:
    """Test kelly_fraction at boundary and degenerate inputs."""

    def test_kelly_at_p_zero(self):
        # p_market=0 clamps into the open interval; output stays finite
        # and within the safety cap (not NaN, not inf).
        k = kelly_fraction(0.0, 0.15)
        assert math.isfinite(k)
        assert -0.25 <= k <= 0.25

    def test_kelly_at_p_one(self):
        # p_market=1 clamps into the open interval; output stays finite
        # and within the safety cap.
        k = kelly_fraction(1.0, 0.15)
        assert math.isfinite(k)
        assert -0.25 <= k <= 0.25

    def test_kelly_nan_input(self):
        # Documented choice: a non-finite p_market returns 0.0 (bet
        # nothing) rather than NaN or an exception, so a bad quote can
        # never inject NaN into position sizing.
        k = kelly_fraction(float('nan'), 0.15)
        assert k == 0.0

    def test_kelly_nan_lambda(self):
        # A non-finite lambda must also bet nothing. Before this was
        # guarded, NaN lambda clamped into a spurious +0.25 max bet.
        k = kelly_fraction(0.5, float('nan'))
        assert k == 0.0

    def test_kelly_extreme_lambda_positive(self):
        # Strong longshot bias at p=0.5: deep NO bet, capped at -0.25.
        k = kelly_fraction(0.5, 5.0)
        assert math.isfinite(k)
        assert -0.25 <= k <= 0.25

    def test_kelly_extreme_lambda_negative(self):
        # Strong reverse bias at p=0.5: deep YES bet, capped at 0.25.
        k = kelly_fraction(0.5, -5.0)
        assert math.isfinite(k)
        assert -0.25 <= k <= 0.25

    def test_kelly_zero_fee(self):
        # With fee_rate=0.0 a real edge must produce a strictly non-zero
        # bet. Per the Wang transform documented on kelly_fraction
        # (p* = Phi(Phi^-1(p_market) - lam)), positive lambda makes
        # p* < p_market, so the optimal bet is NO and the fraction is
        # strictly negative; negative lambda flips it to a strictly
        # positive YES bet.
        k_no = kelly_fraction(0.4, 0.2, fee_rate=0.0)
        assert math.isfinite(k_no)
        assert k_no < 0
        k_yes = kelly_fraction(0.4, -0.2, fee_rate=0.0)
        assert math.isfinite(k_yes)
        assert k_yes > 0

    def test_kelly_negative_fee_rejected(self):
        with pytest.raises(ValueError):
            kelly_fraction(0.4, 0.2, fee_rate=-0.01)
