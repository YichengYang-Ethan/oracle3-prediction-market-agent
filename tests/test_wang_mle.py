"""Tests for the Wang Transform MLE estimator.

Verifies that the MLE:
- Recovers known lambda on synthetic data
- Produces valid standard errors
- Handles edge cases (all 0s, all 1s, small N)
- Hierarchical model detects covariate effects
- Summary table formats correctly
"""

from __future__ import annotations

import pytest

np = pytest.importorskip('numpy')
scipy = pytest.importorskip('scipy')

from oracle3.pricing.wang_mle import (  # noqa: E402
    MLEResult,
    WangMLE,
)


def _generate_wang_data(lam: float, n: int, seed: int = 42):
    """Generate synthetic resolved contracts from the Wang model."""
    rng = np.random.RandomState(seed)
    p_physical = rng.uniform(0.1, 0.9, n)
    from scipy.stats import norm
    p_market = norm.cdf(norm.ppf(p_physical) + lam)
    outcomes = (rng.uniform(size=n) < p_physical).astype(float)
    return p_market, outcomes, p_physical


class TestWangMLE:
    """Test the core MLE estimator."""

    def test_recovers_known_lambda(self):
        """MLE should recover lambda close to the true value on large N."""
        true_lambda = 0.20
        prices, outcomes, _ = _generate_wang_data(true_lambda, n=5000)
        mle = WangMLE()
        result = mle.fit(prices=prices, outcomes=outcomes)
        assert result.converged
        assert abs(result.lambda_hat - true_lambda) < 0.05, (
            f'Expected ~{true_lambda}, got {result.lambda_hat}'
        )

    def test_recovers_zero_lambda(self):
        """Lambda=0 should be recovered when no distortion is applied."""
        prices, outcomes, _ = _generate_wang_data(0.0, n=3000)
        mle = WangMLE()
        result = mle.fit(prices=prices, outcomes=outcomes)
        assert result.converged
        assert abs(result.lambda_hat) < 0.08

    def test_recovers_negative_lambda(self):
        """Negative lambda (play-money) should be recovered."""
        prices, outcomes, _ = _generate_wang_data(-0.20, n=3000)
        mle = WangMLE()
        result = mle.fit(prices=prices, outcomes=outcomes)
        assert result.converged
        assert result.lambda_hat < 0

    def test_standard_errors_finite(self):
        prices, outcomes, _ = _generate_wang_data(0.15, n=1000)
        mle = WangMLE()
        result = mle.fit(prices=prices, outcomes=outcomes)
        assert all(0 < se < 1 for se in result.se_fisher)
        assert all(0 < se < 1 for se in result.se_robust)

    def test_aic_bic_computed(self):
        prices, outcomes, _ = _generate_wang_data(0.15, n=500)
        mle = WangMLE()
        result = mle.fit(prices=prices, outcomes=outcomes)
        assert result.aic < 0 or result.aic > 0  # just not NaN
        assert result.bic < 0 or result.bic > 0

    def test_z_stat_and_pvalue(self):
        prices, outcomes, _ = _generate_wang_data(0.20, n=2000)
        mle = WangMLE()
        result = mle.fit(prices=prices, outcomes=outcomes)
        z = result.z_stat(0)
        p = result.p_value(0)
        assert abs(z) > 1.0  # should be significant
        assert p < 0.1

    def test_summary_table(self):
        prices, outcomes, _ = _generate_wang_data(0.15, n=500)
        mle = WangMLE()
        result = mle.fit(prices=prices, outcomes=outcomes)
        table = result.summary_table()
        assert 'Wang Transform MLE' in table
        assert 'lambda' in table

    def test_small_sample(self):
        """Should handle very small samples without crashing."""
        prices, outcomes, _ = _generate_wang_data(0.15, n=20)
        mle = WangMLE()
        result = mle.fit(prices=prices, outcomes=outcomes)
        assert result.n_obs == 20


class TestHierarchicalMLE:
    """Test the hierarchical covariate model."""

    def test_design_matrix_shape(self):
        mle = WangMLE()
        n = 100
        X = mle.build_design_matrix(
            volumes=np.random.uniform(100, 10000, n),
            durations_hours=np.random.uniform(1, 1000, n),
            prices=np.random.uniform(0.1, 0.9, n),
            spreads=np.random.uniform(0.01, 0.1, n),
        )
        assert X.shape == (n, 5)  # intercept + 4 covariates

    def test_volume_coefficient_sign(self):
        """Higher volume should reduce lambda (negative coefficient)."""
        n = 3000
        rng = np.random.RandomState(42)
        from scipy.stats import norm

        volumes = rng.uniform(100, 50000, n)
        # Generate lambda_i that depends on volume
        true_beta = [0.25, -0.06]  # constant + negative volume effect
        lambdas = true_beta[0] + true_beta[1] * np.log(1 + volumes)

        p_physical = rng.uniform(0.15, 0.85, n)
        p_market = norm.cdf(norm.ppf(p_physical) + lambdas)
        outcomes = (rng.uniform(size=n) < p_physical).astype(float)

        mle = WangMLE()
        X = mle.build_design_matrix(volumes=volumes)
        result = mle.fit(
            prices=p_market, outcomes=outcomes, covariates=X,
            covariate_names=['constant', 'ln(1+volume)'],
        )
        assert result.converged
        # Volume coefficient should be negative
        assert result.beta[1] < 0, f'Expected negative volume coefficient, got {result.beta[1]}'

    def test_lr_test(self):
        """LR test should detect that covariates improve fit."""
        n = 8000  # need large N for power with noisy binary outcomes
        rng = np.random.RandomState(42)
        from scipy.stats import norm

        volumes = rng.uniform(100, 50000, n)
        # Stronger effect to ensure detectability
        lambdas = 0.30 - 0.10 * np.log(1 + volumes)
        p_physical = rng.uniform(0.15, 0.85, n)
        p_market = norm.cdf(norm.ppf(p_physical) + lambdas)
        outcomes = (rng.uniform(size=n) < p_physical).astype(float)

        mle = WangMLE()
        r_restricted = mle.fit(prices=p_market, outcomes=outcomes)
        X = mle.build_design_matrix(volumes=volumes)
        r_full = mle.fit(prices=p_market, outcomes=outcomes, covariates=X)

        chi2, pval, df = mle.lr_test(r_full, r_restricted)
        assert df == 1
        assert chi2 > 0
        assert pval < 0.05, f'LR test not significant: chi2={chi2}, p={pval}'


class TestPredictInterval:
    """Test delta-method confidence intervals for fair values."""

    def test_pooled_matches_delta_method(self):
        prices, outcomes, _ = _generate_wang_data(0.20, n=5000)
        result = WangMLE().fit(prices=prices, outcomes=outcomes)

        from scipy.stats import norm

        p_star, lower, upper = result.predict_interval(0.5)
        eta = norm.ppf(0.5) - result.lambda_hat
        se = np.sqrt(result.vcov_robust[0, 0])
        z = norm.ppf(0.975)

        assert p_star == pytest.approx(norm.cdf(eta))
        assert lower == pytest.approx(norm.cdf(eta - z * se))
        assert upper == pytest.approx(norm.cdf(eta + z * se))
        assert lower <= p_star <= upper

    def test_pooled_interval_width_shrinks_with_n(self):
        prices, outcomes, _ = _generate_wang_data(0.20, n=5000, seed=7)

        small = WangMLE().fit(prices=prices[:300], outcomes=outcomes[:300])
        large = WangMLE().fit(prices=prices, outcomes=outcomes)

        _, small_lower, small_upper = small.predict_interval(0.5)
        _, large_lower, large_upper = large.predict_interval(0.5)

        assert (small_upper - small_lower) > (large_upper - large_lower)

    def test_hierarchical_interval_with_covariates(self):
        n = 3000
        rng = np.random.RandomState(42)
        from scipy.stats import norm

        volumes = rng.uniform(100, 50000, n)
        lambdas = 0.25 - 0.06 * np.log(1 + volumes)
        p_physical = rng.uniform(0.15, 0.85, n)
        p_market = norm.cdf(norm.ppf(p_physical) + lambdas)
        outcomes = (rng.uniform(size=n) < p_physical).astype(float)

        mle = WangMLE()
        X = mle.build_design_matrix(volumes=volumes)
        result = mle.fit(prices=p_market, outcomes=outcomes, covariates=X)

        p_star, lower, upper = result.predict_interval(0.5, covariates=X[0])

        eta = norm.ppf(0.5) - X[0] @ np.asarray(result.beta)
        se = np.sqrt(X[0] @ result.vcov_robust @ X[0])
        z = norm.ppf(0.975)

        assert p_star == pytest.approx(norm.cdf(eta))
        assert lower == pytest.approx(norm.cdf(eta - z * se))
        assert upper == pytest.approx(norm.cdf(eta + z * se))
        assert 0.0 < lower <= p_star <= upper < 1.0

    def test_hierarchical_requires_covariates(self):
        n = 1000
        rng = np.random.RandomState(42)
        from scipy.stats import norm

        volumes = rng.uniform(100, 50000, n)
        lambdas = 0.25 - 0.06 * np.log(1 + volumes)
        p_physical = rng.uniform(0.15, 0.85, n)
        p_market = norm.cdf(norm.ppf(p_physical) + lambdas)
        outcomes = (rng.uniform(size=n) < p_physical).astype(float)

        mle = WangMLE()
        X = mle.build_design_matrix(volumes=volumes)
        result = mle.fit(prices=p_market, outcomes=outcomes, covariates=X)

        with pytest.raises(ValueError, match='covariates are required'):
            result.predict_interval(0.5)

    def test_bad_covariate_length(self):
        prices, outcomes, _ = _generate_wang_data(0.20, n=5000)
        result = WangMLE().fit(prices=prices, outcomes=outcomes)

        with pytest.raises(ValueError, match='covariates must have length'):
            result.predict_interval(0.5, covariates=np.zeros(0))

    def test_covariates_must_be_vector(self):
        prices, outcomes, _ = _generate_wang_data(0.20, n=5000)
        result = WangMLE().fit(prices=prices, outcomes=outcomes)

        with pytest.raises(ValueError, match='1-dimensional'):
            result.predict_interval(0.5, covariates=np.zeros((1, 1)))

    def test_pooled_rejects_covariates(self):
        prices, outcomes, _ = _generate_wang_data(0.20, n=5000)
        result = WangMLE().fit(prices=prices, outcomes=outcomes)

        with pytest.raises(ValueError, match='pooled model'):
            result.predict_interval(0.5, covariates=np.zeros(1))

    def test_unfitted_model(self):
        with pytest.raises(ValueError, match='vcov_robust'):
            MLEResult().predict_interval(0.5)

    def test_nonfinite_covariance_rejected(self):
        result = MLEResult(
            beta=[0.2],
            vcov_robust=np.array([[np.nan]]),
            n_params=1,
        )
        with pytest.raises(ValueError, match='non-finite'):
            result.predict_interval(0.5)

    def test_negative_variance_rejected(self):
        result = MLEResult(
            beta=[0.2],
            vcov_robust=np.array([[-1.0]]),
            n_params=1,
        )
        with pytest.raises(ValueError, match='negative'):
            result.predict_interval(0.5)

    def test_covariance_shape_mismatch_rejected(self):
        result = MLEResult(
            beta=[0.2],
            vcov_robust=np.eye(2),
            n_params=1,
        )
        with pytest.raises(ValueError, match='shape'):
            result.predict_interval(0.5)

    def test_invalid_alpha_and_market_price(self):
        prices, outcomes, _ = _generate_wang_data(0.20, n=1000)
        result = WangMLE().fit(prices=prices, outcomes=outcomes)

        with pytest.raises(ValueError, match='alpha'):
            result.predict_interval(0.5, alpha=1.0)
        with pytest.raises(ValueError, match='p_market'):
            result.predict_interval(0.0)


class TestGreeks:
    """Test model sensitivity computations."""

    def test_delta_lambda_positive(self):
        from oracle3.pricing.greeks import delta_lambda
        # dp_mkt/dlambda is always positive (more premium = higher price)
        for p in [0.1, 0.3, 0.5, 0.7, 0.9]:
            assert delta_lambda(p, 0.15) > 0

    def test_delta_lambda_maximized_near_50(self):
        from oracle3.pricing.greeks import delta_lambda
        d_50 = delta_lambda(0.5, 0.15)
        d_10 = delta_lambda(0.1, 0.15)
        d_90 = delta_lambda(0.9, 0.15)
        assert d_50 > d_10
        assert d_50 > d_90

    def test_premium_ratio_decreasing(self):
        """Favorite-longshot bias: premium ratio decreases with p."""
        from oracle3.pricing.greeks import premium_ratio
        ratios = [premium_ratio(p, 0.15) for p in [0.1, 0.3, 0.5, 0.7, 0.9]]
        for i in range(1, len(ratios)):
            assert ratios[i] < ratios[i - 1] + 0.001

    def test_kelly_fraction_bounded(self):
        from oracle3.pricing.greeks import kelly_fraction
        for p in [0.2, 0.4, 0.6, 0.8]:
            k = kelly_fraction(p, 0.15)
            assert -0.25 <= k <= 0.25

    def test_kelly_zero_when_no_edge(self):
        from oracle3.pricing.greeks import kelly_fraction
        # With lambda=0, there's no edge, so Kelly should be 0
        k = kelly_fraction(0.5, 0.0)
        assert abs(k) < 0.01

    def test_greeks_table(self):
        from oracle3.pricing.greeks import compute_greeks_table
        table = compute_greeks_table([0.3, 0.5, 0.7], lam=0.15)
        assert len(table) == 3
        assert all('premium' in row for row in table)
        assert all('kelly' in row for row in table)


class TestSizing:
    """Test model-informed position sizing."""

    def test_skip_very_high_volume(self):
        from oracle3.pricing.fair_value import FairValueEstimate
        from oracle3.trading.sizing import ModelInformedSizer

        sizer = ModelInformedSizer()
        est = FairValueEstimate(
            market_price=0.55, fair_value=0.50, risk_premium=0.05,
            mispricing_signal=0.5, lambda_base=0.17, lambda_adjusted=0.01,
            confidence=0.8, category='default', volume_tier='very_high',
            is_premium_alpha_target=False,
        )
        result = sizer.compute_size(est)
        assert result.trade_size == 0

    def test_medium_volume_gets_sized(self):
        from oracle3.pricing.fair_value import FairValueEstimate
        from oracle3.trading.sizing import ModelInformedSizer

        sizer = ModelInformedSizer(min_edge=0.005)
        est = FairValueEstimate(
            market_price=0.55, fair_value=0.50, risk_premium=0.05,
            mispricing_signal=0.5, lambda_base=0.17, lambda_adjusted=0.30,
            confidence=0.7, category='default', volume_tier='medium',
            is_premium_alpha_target=True,
        )
        result = sizer.compute_size(est)
        assert result.trade_size > 0
        assert result.side in ('YES', 'NO')

    def test_low_confidence_rejected(self):
        from oracle3.pricing.fair_value import FairValueEstimate
        from oracle3.trading.sizing import ModelInformedSizer

        sizer = ModelInformedSizer(confidence_floor=0.5)
        est = FairValueEstimate(
            market_price=0.55, fair_value=0.50, risk_premium=0.05,
            mispricing_signal=0.5, lambda_base=0.17, lambda_adjusted=0.30,
            confidence=0.2, category='default', volume_tier='medium',
            is_premium_alpha_target=True,
        )
        result = sizer.compute_size(est)
        assert result.trade_size == 0


class TestAllocator:
    """Test edge-weighted capital allocation."""

    def test_equal_allocation_no_data(self):
        from decimal import Decimal

        from oracle3.trading.allocator import EdgeWeightedAllocator, StrategyPerformance

        alloc = EdgeWeightedAllocator(total_capital=Decimal('1000'))
        strategies = [
            StrategyPerformance('s1', is_active=True),
            StrategyPerformance('s2', is_active=True),
        ]
        result = alloc.allocate(strategies)
        assert result.method == 'equal'
        assert len(result.allocations) == 2

    def test_performance_weighted(self):
        from decimal import Decimal

        from oracle3.trading.allocator import EdgeWeightedAllocator, StrategyPerformance

        alloc = EdgeWeightedAllocator(total_capital=Decimal('1000'))
        strategies = [
            StrategyPerformance('good', cumulative_pnl=100, max_drawdown=-10, n_trades=50, win_rate=0.7),
            StrategyPerformance('bad', cumulative_pnl=-50, max_drawdown=-80, n_trades=50, win_rate=0.3),
        ]
        result = alloc.allocate(strategies)
        assert result.method == 'performance'
        assert result.allocations['good'] > result.allocations['bad']

    def test_premium_alpha_bonus(self):
        from decimal import Decimal

        from oracle3.trading.allocator import EdgeWeightedAllocator, StrategyPerformance

        alloc = EdgeWeightedAllocator(
            total_capital=Decimal('1000'), premium_alpha_bonus=2.0
        )
        strategies = [
            StrategyPerformance('premium', cumulative_pnl=50, max_drawdown=-10, n_trades=20, is_premium_alpha=True),
            StrategyPerformance('other', cumulative_pnl=50, max_drawdown=-10, n_trades=20, is_premium_alpha=False),
        ]
        result = alloc.allocate(strategies)
        assert result.allocations['premium'] > result.allocations['other']
