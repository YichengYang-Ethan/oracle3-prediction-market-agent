"""Tests for the oracle3.pricing module.

Covers:
- Distortion functions (probit, dual power, proportional hazard)
- Fair value estimator
- Online calibrator
- Premium tracker
- Contract scorer
- Distortion-based validation
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import pytest

from oracle3.pricing.calibrator import OnlineCalibrator
from oracle3.pricing.contract_scorer import ContractFeatures, ContractScorer
from oracle3.pricing.distortion import (
    DualPowerDistortion,
    ProbitDistortion,
    ProportionalHazardDistortion,
    _norm_cdf,
    _norm_ppf,
    compare_distortions,
)
from oracle3.pricing.fair_value import FairValueEstimator
from oracle3.pricing.premium_tracker import DecayModel, PremiumTracker

# ── Normal CDF / PPF tests ─���──────────────────────────────────────────────


class TestNormalApproximation:
    """Test the lightweight normal CDF/PPF implementations."""

    def test_cdf_symmetry(self):
        assert abs(_norm_cdf(0.0) - 0.5) < 1e-6

    def test_cdf_tails(self):
        assert _norm_cdf(-5.0) < 0.001
        assert _norm_cdf(5.0) > 0.999

    def test_cdf_known_values(self):
        # Phi(1.96) ~ 0.975
        assert abs(_norm_cdf(1.96) - 0.975) < 0.001

    def test_ppf_inverse_of_cdf(self):
        for x in [-2.0, -1.0, 0.0, 0.5, 1.0, 2.0]:
            p = _norm_cdf(x)
            x_recovered = _norm_ppf(p)
            assert abs(x - x_recovered) < 1e-4, f'Failed at x={x}'

    def test_ppf_known_values(self):
        assert abs(_norm_ppf(0.5) - 0.0) < 1e-5
        assert abs(_norm_ppf(0.975) - 1.96) < 0.01

    def test_ppf_extreme(self):
        # Should not raise
        assert _norm_ppf(0.001) < -2.5
        assert _norm_ppf(0.999) > 2.5


# ── Probit distortion tests ──────────────────────────────────────────────


class TestProbitDistortion:
    """Test the probit (Wang Transform) distortion."""

    def test_identity_at_zero_lambda(self):
        d = ProbitDistortion(lam=0.0)
        for p in [0.1, 0.3, 0.5, 0.7, 0.9]:
            assert abs(d.distort(p) - p) < 1e-5

    def test_overpricing_positive_lambda(self):
        d = ProbitDistortion(lam=0.2)
        for p in [0.1, 0.3, 0.5, 0.7, 0.9]:
            assert d.distort(p) > p + 1e-6, f'Failed at p={p}'

    def test_underpricing_negative_lambda(self):
        d = ProbitDistortion(lam=-0.2)
        for p in [0.1, 0.3, 0.5, 0.7, 0.9]:
            assert d.distort(p) < p - 1e-6, f'Failed at p={p}'

    def test_inverse_roundtrip(self):
        d = ProbitDistortion(lam=0.15)
        for p in [0.1, 0.3, 0.5, 0.7, 0.9]:
            distorted = d.distort(p)
            recovered = d.inverse(distorted)
            assert abs(recovered - p) < 1e-5, f'Failed at p={p}'

    def test_risk_premium_positive(self):
        d = ProbitDistortion(lam=0.2)
        for p in [0.1, 0.3, 0.5, 0.7, 0.9]:
            assert d.risk_premium(p) > 0

    def test_overpricing_ratio_decreasing(self):
        """Favorite-longshot bias: overpricing ratio decreases with p."""
        d = ProbitDistortion(lam=0.2)
        ratios = [d.overpricing_ratio(p) for p in [0.1, 0.3, 0.5, 0.7, 0.9]]
        for i in range(1, len(ratios)):
            assert ratios[i] < ratios[i - 1], (
                f'Overpricing ratio not decreasing: {ratios}'
            )

    def test_implied_lambda(self):
        d = ProbitDistortion(lam=0.15)
        p_phys = 0.5
        p_mkt = d.distort(p_phys)
        implied = d.implied_lambda(p_mkt, p_phys)
        assert abs(implied - 0.15) < 1e-4

    def test_boundary_handling(self):
        d = ProbitDistortion(lam=0.1)
        # Should not raise or return NaN
        assert 0 < d.distort(0.001) < 1
        assert 0 < d.distort(0.999) < 1
        assert 0 < d.inverse(0.001) < 1
        assert 0 < d.inverse(0.999) < 1

    def test_properties(self):
        d = ProbitDistortion(lam=0.15)
        assert d.param == 0.15
        assert d.family == 'probit'


# ── Dual power distortion tests ──────────────────────────────────────────


class TestDualPowerDistortion:
    def test_identity_at_zero_rho(self):
        d = DualPowerDistortion(rho=0.0)
        for p in [0.1, 0.3, 0.5, 0.7, 0.9]:
            assert abs(d.distort(p) - p) < 1e-5

    def test_overpricing_positive_rho(self):
        # Dual power distortion g(p) = 1 - (1-p)^{1/(1+rho)} overprices
        # in the upper range of p where the survival function distortion
        # dominates. At low p, the distortion can actually underprice.
        d = DualPowerDistortion(rho=0.3)
        for p in [0.5, 0.7, 0.9]:
            assert d.distort(p) > p + 1e-6, f'Failed at p={p}'

    def test_inverse_roundtrip(self):
        d = DualPowerDistortion(rho=0.25)
        for p in [0.1, 0.3, 0.5, 0.7, 0.9]:
            recovered = d.inverse(d.distort(p))
            assert abs(recovered - p) < 1e-5

    def test_invalid_rho(self):
        with pytest.raises(ValueError):
            DualPowerDistortion(rho=-1.5)


# ── Proportional hazard distortion tests ─────────────────────────────────


class TestProportionalHazardDistortion:
    def test_identity_at_zero_rho(self):
        d = ProportionalHazardDistortion(rho=0.0)
        for p in [0.1, 0.3, 0.5, 0.7, 0.9]:
            assert abs(d.distort(p) - p) < 1e-5

    def test_inverse_roundtrip(self):
        d = ProportionalHazardDistortion(rho=0.2)
        for p in [0.1, 0.3, 0.5, 0.7, 0.9]:
            recovered = d.inverse(d.distort(p))
            assert abs(recovered - p) < 1e-5


# ── Distortion comparison ────────────────────────────────────────────────


class TestDistortionComparison:
    def test_compare(self):
        distortions = [
            ProbitDistortion(0.15),
            DualPowerDistortion(0.2),
            ProportionalHazardDistortion(0.2),
        ]
        result = compare_distortions(0.5, distortions)
        assert result.p_physical == 0.5
        assert len(result.results) == 3
        assert len(result.premiums) == 3
        # Probit and proportional hazard show positive premium at p=0.5
        assert result.premiums['probit(0.150)'] > 0
        assert result.premiums['proportional_hazard(0.200)'] > 0


# ── Online calibrator tests ──────────────────────────────────────────────


class TestOnlineCalibrator:
    def _make_calibrator(self) -> OnlineCalibrator:
        """Create a calibrator with a temp cache path."""
        tmp = tempfile.mkdtemp()
        return OnlineCalibrator(
            alpha=0.1,
            default_lambda=0.10,
            cache_path=Path(tmp) / 'cal.json',
        )

    def test_default_lambda(self):
        cal = self._make_calibrator()
        assert abs(cal.get_lambda('unknown') - 0.10) < 1e-6

    def test_update_shifts_lambda(self):
        cal = self._make_calibrator()
        # Feed a balanced mix: price always 0.60 but outcome is 50/50
        # outcome=1, price=0.60 -> lambda = ppf(0.60) - ppf(0.999) < 0 (underpriced)
        # outcome=0, price=0.60 -> lambda = ppf(0.60) - ppf(0.001) > 0 (overpriced)
        # Net effect: positive lambda (market overprices on average)
        for _ in range(20):
            cal.update(outcome=1, final_price=0.60, category='test')
            cal.update(outcome=0, final_price=0.60, category='test')
        lam = cal.get_lambda('test')
        # Lambda should be positive (average overpricing at p=0.60)
        assert lam > 0, f'Expected positive lambda, got {lam}'

    def test_category_independence(self):
        cal = self._make_calibrator()
        for _ in range(20):
            cal.update(outcome=1, final_price=0.70, category='sports')
            cal.update(outcome=0, final_price=0.40, category='politics')
        lam_sports = cal.get_lambda('sports')
        lam_politics = cal.get_lambda('politics')
        assert lam_sports != lam_politics

    def test_confidence_increases_with_data(self):
        cal = self._make_calibrator()
        conf_0 = cal.get_confidence('test')
        for _ in range(50):
            cal.update(outcome=1, final_price=0.55, category='test')
        conf_50 = cal.get_confidence('test')
        assert conf_50 > conf_0

    def test_report(self):
        cal = self._make_calibrator()
        cal.update(outcome=1, final_price=0.60, category='test')
        report = cal.report()
        assert report.global_n == 1
        assert 'test' in report.categories

    def test_reset(self):
        cal = self._make_calibrator()
        cal.update(outcome=1, final_price=0.60, category='test')
        cal.reset('test')
        assert cal.get_estimate('test') is None

    def test_persistence(self):
        tmp = tempfile.mkdtemp()
        path = Path(tmp) / 'cal.json'
        cal1 = OnlineCalibrator(cache_path=path, default_lambda=0.10)
        for _ in range(10):
            cal1.update(outcome=1, final_price=0.60, category='test')
        lam1 = cal1.get_lambda('test')

        # Create a new calibrator from the same path
        cal2 = OnlineCalibrator(cache_path=path, default_lambda=0.10)
        lam2 = cal2.get_lambda('test')
        assert abs(lam1 - lam2) < 1e-6


# ── Fair value estimator tests ────────────────────────────────────────────


class TestFairValueEstimator:
    def test_basic_estimate(self):
        est = FairValueEstimator()
        result = est.estimate(0.57, category='default')
        # With default lambda ~0.10, fair value should be below market price
        assert result.fair_value < result.market_price
        assert result.risk_premium > 0

    def test_fair_value_below_market(self):
        """Under positive lambda, fair value should always be below market price."""
        cal = OnlineCalibrator(default_lambda=0.15)
        est = FairValueEstimator(calibrator=cal)
        for p in [0.2, 0.4, 0.5, 0.6, 0.8]:
            result = est.estimate(p)
            assert result.fair_value < p, f'Failed at p={p}'

    def test_volume_lowers_lambda(self):
        est = FairValueEstimator()
        # High volume should lower lambda (negative coefficient in Yang 2026)
        r_high = est.estimate(0.60, volume=50000)
        r_low = est.estimate(0.60, volume=100)
        assert r_high.lambda_adjusted < r_low.lambda_adjusted

    def test_batch_estimate(self):
        est = FairValueEstimator()
        prices = {'A': 0.55, 'B': 0.70, 'C': 0.30}
        results = est.estimate_batch(prices)
        assert len(results) == 3
        for sym in prices:
            assert sym in results
            assert results[sym].market_price == prices[sym]

    @pytest.mark.parametrize('price', [0.02, 0.2, 0.5, 0.8, 0.98])
    def test_decomposition_does_not_imply_mispricing(self, price):
        result = FairValueEstimator().estimate(price)
        assert result.mispricing_signal == 0.0
        assert result.reference_probability is None
        assert result.benchmark_market_price is None

    def test_rank_excludes_decomposition_only_estimates(self):
        est = FairValueEstimator()
        prices = {'A': 0.55, 'B': 0.70, 'C': 0.30}
        results = est.estimate_batch(prices)
        ranked = est.rank_opportunities(
            results, min_confidence=0.0, premium_alpha_only=False
        )
        assert ranked == []

    @pytest.mark.parametrize('price', [0.5, 0.65])
    def test_signal_compares_independent_forecast_with_model_price(
        self, tmp_path, price
    ):
        cal = OnlineCalibrator(default_lambda=0.2, cache_path=tmp_path / 'cal.json')
        est = FairValueEstimator(calibrator=cal, use_paper_coefficients=False)
        decomposition = est.estimate(price)
        result = est.estimate(price, reference_probability=0.5)

        # Independent oracle: Wang(0.5, 0.2) = Phi(0.2).
        expected_price = 0.5 * (1.0 + math.erf(0.2 / math.sqrt(2.0)))
        assert result.reference_probability == 0.5
        assert result.benchmark_market_price == pytest.approx(expected_price, abs=1e-7)
        assert result.mispricing_signal == pytest.approx(
            price - expected_price, abs=1e-7
        )
        assert result.fair_value == decomposition.fair_value
        assert result.risk_premium == decomposition.risk_premium

    def test_identity_model_ranks_only_supplied_nonzero_forecasts(self, tmp_path):
        cal = OnlineCalibrator(default_lambda=0.0, cache_path=tmp_path / 'cal.json')
        est = FairValueEstimator(calibrator=cal, use_paper_coefficients=False)
        results = est.estimate_batch(
            {'over': 0.6, 'under': 0.4, 'equal': 0.5, 'missing': 0.9},
            reference_probabilities={'over': 0.4, 'under': 0.7, 'equal': 0.5},
        )

        assert results['over'].mispricing_signal == pytest.approx(0.2, abs=1e-7)
        assert results['under'].mispricing_signal == pytest.approx(-0.3, abs=1e-7)
        assert results['equal'].mispricing_signal == 0.0
        assert results['missing'].mispricing_signal == 0.0
        ranked = est.rank_opportunities(
            results,
            min_confidence=0.0,
            premium_alpha_only=False,
        )
        assert [symbol for symbol, _ in ranked] == ['under', 'over']

    @pytest.mark.parametrize('price', [0.0, 0.005, 0.995, 1.0])
    def test_tail_quote_matches_identity_forecast_without_clipping(
        self, tmp_path, price
    ):
        cal = OnlineCalibrator(default_lambda=0.0, cache_path=tmp_path / 'cal.json')
        est = FairValueEstimator(calibrator=cal, use_paper_coefficients=False)
        result = est.estimate(price, reference_probability=price)
        assert result.market_price == price
        assert result.benchmark_market_price == pytest.approx(price, abs=1e-7)
        assert result.fair_value == pytest.approx(price, abs=1e-7)
        assert result.mispricing_signal == 0.0
        if price in (0.0, 1.0):
            assert result.fair_value == price
            assert result.risk_premium == 0.0

    def test_roundtrip_approximation_is_not_a_signal(self):
        est = FairValueEstimator()
        decomposition = est.estimate(0.8)
        result = est.estimate(0.8, reference_probability=decomposition.fair_value)
        assert result.mispricing_signal == 0.0

    @pytest.mark.parametrize('reference', [0.0, 1.0])
    def test_reference_probability_endpoints(self, reference):
        result = FairValueEstimator().estimate(0.5, reference_probability=reference)
        assert result.benchmark_market_price == reference
        assert result.mispricing_signal == 0.5 - reference

    @pytest.mark.parametrize('reference', [-0.01, 1.01, math.nan, math.inf, -math.inf])
    def test_invalid_reference_is_rejected(self, reference):
        with pytest.raises(ValueError, match='reference_probability'):
            FairValueEstimator().estimate(0.5, reference_probability=reference)

    @pytest.mark.parametrize('price', [-0.2, 2.0, math.nan, math.inf, -math.inf])
    def test_invalid_market_price_is_rejected(self, price):
        with pytest.raises(ValueError, match='market_price'):
            FairValueEstimator().estimate(price)

    @pytest.mark.parametrize('lam', [math.nan, math.inf, -math.inf])
    def test_nonfinite_calibration_is_rejected(self, tmp_path, lam):
        cal = OnlineCalibrator(default_lambda=lam, cache_path=tmp_path / 'cal.json')
        est = FairValueEstimator(calibrator=cal, use_paper_coefficients=False)
        with pytest.raises(ValueError, match='lambda'):
            est.estimate(0.5, reference_probability=0.4)


# ── Premium tracker tests ────────────────────────────────────────────────


class TestPremiumTracker:
    def test_record_and_snapshot(self):
        tracker = PremiumTracker(min_observations_for_fit=5)
        # Simulate decaying premium
        for i in range(20):
            tau = 1.0 - i / 20.0
            premium = 0.05 * tau  # linear decay
            tracker.record(
                ticker_symbol='TEST',
                market_price=0.50 + premium,
                fair_value=0.50,
                tau=tau,
            )
        snap = tracker.get_snapshot('TEST')
        assert snap is not None
        assert snap.n_observations == 20
        assert snap.current_premium < snap.initial_premium

    def test_predict_premium(self):
        tracker = PremiumTracker(min_observations_for_fit=5)
        for i in range(30):
            tau = 1.0 - i / 30.0
            premium = 0.04 * tau
            tracker.record(
                ticker_symbol='TEST',
                market_price=0.50 + premium,
                fair_value=0.50,
                tau=tau,
            )
        # Predict premium at tau=0.5 (should be ~0.02)
        predicted = tracker.predict_premium('TEST', tau=0.5)
        assert predicted is not None
        assert 0.01 < predicted < 0.04

    def test_category_model(self):
        tracker = PremiumTracker(min_observations_for_fit=5)
        for ticker_id in range(5):
            for i in range(10):
                tau = 1.0 - i / 10.0
                premium = 0.05 * tau
                tracker.record(
                    ticker_symbol=f'T{ticker_id}',
                    market_price=0.50 + premium,
                    fair_value=0.50,
                    tau=tau,
                    category='politics',
                )
        model = tracker.get_category_model('politics')
        assert model is not None
        assert model.n_obs >= 30


class TestDecayModel:
    def test_predict(self):
        model = DecayModel(gamma_1=0.04, gamma_2=0.01, n_obs=100)
        # At tau=1: 0.04 + 0.01 = 0.05
        assert abs(model.predict(1.0) - 0.05) < 1e-6
        # At tau=0: 0
        assert abs(model.predict(0.0)) < 1e-6

    def test_half_life(self):
        model = DecayModel(gamma_1=0.04, gamma_2=0.0, n_obs=100)
        # Linear decay: half-life should be at tau=0.5
        hl = model.half_life_fraction
        assert abs(hl - 0.5) < 0.01


# ── Contract scorer tests ───────────────���─────────────────────────────���──


class TestContractScorer:
    def test_extract_features(self):
        scorer = ContractScorer()
        f = scorer.extract_features(
            ticker_symbol='TEST',
            volume=50000,
            best_bid=0.48,
            best_ask=0.52,
            duration_days=30,
        )
        assert f.mid_price == pytest.approx(0.50)
        assert f.spread == pytest.approx(0.04)
        assert f.extremity == pytest.approx(0.0)
        assert f.log_volume is not None

    def test_score(self):
        scorer = ContractScorer()
        f = scorer.extract_features(
            volume=50000,
            best_bid=0.48,
            best_ask=0.52,
            duration_days=30,
        )
        score = scorer.score(f)
        assert 0 < score.tradability < 1
        assert 0 < score.liquidity < 1

    def test_is_tradeable(self):
        scorer = ContractScorer(min_volume=1000, max_spread=0.10)
        f_good = ContractFeatures(volume=5000, spread=0.03)
        f_bad_vol = ContractFeatures(volume=100, spread=0.03)
        f_bad_spread = ContractFeatures(volume=5000, spread=0.20)
        assert scorer.is_tradeable(f_good)
        assert not scorer.is_tradeable(f_bad_vol)
        assert not scorer.is_tradeable(f_bad_spread)


# ── Validation extensions tests ──────────────────────────────────────────


class TestDistortionValidation:
    def test_estimate_risk_premium(self):
        from oracle3.market.validation import estimate_risk_premium

        # Overpriced contracts: price=0.6 but outcome=1 half the time
        prices = [0.60] * 100
        outcomes = [1] * 50 + [0] * 50
        mean_lam, std_lam = estimate_risk_premium(prices, outcomes)
        # With price always at 0.6 and 50% realisation, lambda should be positive
        assert mean_lam > 0

    def test_cross_platform_premium(self):
        from oracle3.market.validation import cross_platform_premium_test

        # Platform A prices slightly higher than B
        prices_a = [0.55, 0.60, 0.65, 0.70, 0.75]
        prices_b = [0.50, 0.55, 0.60, 0.65, 0.70]
        result = cross_platform_premium_test(prices_a, prices_b)
        assert result['probit_gap'] > 0  # A is more overpriced
        assert result['n_pairs'] == 5

    def test_favorite_longshot_test(self):
        from oracle3.market.validation import favorite_longshot_test

        # Simulate contracts with overpricing bias
        prices = []
        outcomes = []
        import random

        random.seed(42)
        for _ in range(200):
            p_true = random.uniform(0.1, 0.9)
            # Add overpricing proportional to 1/p (longshot bias)
            p_market = min(0.98, p_true * (1.0 + 0.1 / max(p_true, 0.1)))
            outcome = 1 if random.random() < p_true else 0
            prices.append(p_market)
            outcomes.append(outcome)

        result = favorite_longshot_test(prices, outcomes, n_bins=4)
        assert len(result['bins']) == 4
