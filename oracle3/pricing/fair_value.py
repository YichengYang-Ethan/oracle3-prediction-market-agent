"""Real-time fair value estimation using calibrated Wang Transform.

Combines the probit-offset pricing model with empirically estimated
coefficients from Yang (2026) to produce fair value estimates for
live prediction market contracts.

The hierarchical model adjusts the risk premium per-contract:

    lambda_i = beta_0 + beta_1 * ln(1+Volume) + beta_2 * ln(1+Duration_hrs)
               + beta_3 * |price - 0.5| + beta_4 * Spread

Using empirical estimates from 13,274 Polymarket contracts:
    beta_0 =  0.2590  (constant)
    beta_1 = -0.0716  (volume: more liquid → lower premium)
    beta_2 =  0.1431  (duration: longer → higher premium)
    beta_3 = -0.4772  (extremity: near 50% → highest premium)
    beta_4 =  0.1273  (spread: not significant at p=0.76)

Critical trading insight from the volume stratification:
    Very-high-volume markets (>$10K) have lambda ≈ 0 — the risk premium
    is already competed away by informed traders. The alpha is in
    medium-liquidity markets (volume $500-$10K) where lambda = 0.25-0.35
    and prices still embed substantial risk premium.

The time-varying extension models premium decay over contract lifetime:
    lambda(tau) = gamma_1*tau + gamma_2*tau^2 + beta_0 + covariates
    where tau in [0,1] is fraction of lifetime elapsed (0=new, 1=resolution).

    gamma_1 = -0.1555 (linear decay)
    gamma_2 =  0.0744 (quadratic curvature)
    Half-life: 33-77% of contract lifetime.

Reference:
    Yang, Y. (2026). "Pricing Prediction Markets: Risk Premiums,
    Incomplete Markets, and a Decomposition Framework." Working Paper, UIUC.

Usage::

    estimator = FairValueEstimator()
    est = estimator.estimate(market_price=0.57, category='crypto', volume=5000)
    print(f'Physical prob: {est.fair_value:.3f}, premium: {est.risk_premium:.4f}')
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from oracle3.pricing.calibrator import OnlineCalibrator
from oracle3.pricing.contract_scorer import ContractFeatures, ContractScorer
from oracle3.pricing.distortion import ProbitDistortion

# ── Empirical coefficients from Yang (2026) ──────────────────────────────
# Table 3: Hierarchical MLE, full 28-day sample, N=13,274

_BETA_CONSTANT = 0.2590
_BETA_LN_VOLUME = -0.0716
_BETA_LN_DURATION = 0.1431
_BETA_EXTREMITY = -0.4772
_BETA_SPREAD = 0.1273

# Table 5: Time-varying model (stacked panel, N=111,889 obs)
_TV_GAMMA_TAU = -0.1555
_TV_GAMMA_TAU_SQ = 0.0744
_TV_BETA_CONSTANT = 0.2531
_TV_BETA_LN_VOLUME = -0.0568
_TV_BETA_LN_DURATION = 0.1085
_TV_BETA_EXTREMITY = -0.2902

# Volume thresholds for alpha targeting
_VOLUME_HIGH_THRESHOLD = 10000.0  # lambda ≈ 0 above this
_VOLUME_MEDIUM_RANGE = (500.0, 10000.0)  # sweet spot for premium alpha
_SIGNAL_TOLERANCE = 1e-6  # dollars; above the normal-function approximation error


@dataclass(frozen=True)
class FairValueEstimate:
    """Model-implied decomposition and optional independent-forecast comparison.

    ``mispricing_signal`` is market price minus ``benchmark_market_price``,
    in dollars per contract. Without a reference probability it is exactly
    zero: no independent comparison was made, not evidence of fair pricing.
    """

    market_price: float
    fair_value: float  # estimated physical probability
    risk_premium: float  # market_price - fair_value
    mispricing_signal: float  # signed price residual; not a fee-adjusted trade edge
    lambda_base: float  # base lambda from calibrator
    lambda_adjusted: float  # lambda after covariate adjustment
    confidence: float  # model confidence [0, 1]
    category: str
    volume_tier: str  # 'low', 'medium', 'high', 'very_high'
    is_premium_alpha_target: bool  # whether this is in the alpha sweet spot
    reference_probability: float | None = None
    benchmark_market_price: float | None = None


def _compare_reference(
    market_price: float,
    distortion: ProbitDistortion,
    reference_probability: float | None,
) -> tuple[float | None, float]:
    """Compare with an explicit forecast, never the inverse-derived probability."""
    if reference_probability is None:
        return None, 0.0
    if (
        not math.isfinite(reference_probability)
        or not 0.0 <= reference_probability <= 1.0
    ):
        raise ValueError('reference_probability must be finite and in [0, 1]')
    benchmark = (
        reference_probability
        if reference_probability in (0.0, 1.0)
        else distortion.distort(reference_probability)
    )
    residual = market_price - benchmark
    return benchmark, residual if abs(residual) > _SIGNAL_TOLERANCE else 0.0


class FairValueEstimator:
    """Fair value estimation with empirically calibrated Wang Transform.

    Uses the exact coefficients from Yang (2026) for covariate adjustment,
    with the online calibrator providing platform/category-level base lambda.

    Parameters
    ----------
    calibrator:
        Online calibrator for base lambda. Uses empirical priors if None.
    scorer:
        Contract scorer for feature extraction.
    use_paper_coefficients:
        If True (default), uses the exact empirical coefficients from
        Yang (2026). If False, uses the calibrator's generic adjustment.
    time_varying:
        If True, uses the time-varying model for contracts with known
        lifecycle position (tau). Default True.
    """

    def __init__(
        self,
        calibrator: OnlineCalibrator | None = None,
        scorer: ContractScorer | None = None,
        use_paper_coefficients: bool = True,
        time_varying: bool = True,
    ) -> None:
        self._calibrator = calibrator or OnlineCalibrator()
        self._scorer = scorer or ContractScorer()
        self._use_paper = use_paper_coefficients
        self._time_varying = time_varying

    @property
    def calibrator(self) -> OnlineCalibrator:
        return self._calibrator

    def estimate(
        self,
        market_price: float,
        category: str = 'default',
        volume: float | None = None,
        duration_hours: float | None = None,
        spread: float | None = None,
        tau: float | None = None,
        features: ContractFeatures | None = None,
        *,
        reference_probability: float | None = None,
    ) -> FairValueEstimate:
        """Estimate fair value for a contract.

        Parameters
        ----------
        market_price: finite observed market price in [0, 1], preserved as quoted
        category: contract category for calibration lookup
        volume: trading volume (USD)
        duration_hours: contract duration in hours
        spread: bid-ask spread
        tau: lifecycle position [0=new, 1=resolution] for time-varying model
        features: pre-extracted contract features (overrides individual params)
        reference_probability: optional independent physical-probability forecast
            in [0, 1], available at the decision time. Do not derive it by
            inverting this same market price. Its Wang-transformed value is
            the benchmark for the signed price residual ``mispricing_signal``.

        The inverse-derived ``fair_value`` and ``risk_premium`` remain a model
        decomposition. Without a reference, the signal is exactly zero and
        does not assess mispricing. With one, a positive signal means the
        market is above the model-adjusted forecast; it does not guarantee
        convergence or profit. Residuals within 1e-6 dollars are zeroed to
        avoid interpreting the normal-function approximation error as a signal.
        """
        if not math.isfinite(market_price) or not 0.0 <= market_price <= 1.0:
            raise ValueError('market_price must be finite and in [0, 1]')

        # Extract features if not provided
        if features is not None:
            volume = features.volume
            spread = features.spread
            if features.duration_days is not None:
                duration_hours = features.duration_days * 24.0

        # Get base lambda from calibrator
        base_lambda = self._calibrator.get_lambda(category)

        # Compute adjusted lambda using paper coefficients
        if self._use_paper and self._time_varying and tau is not None:
            adjusted_lambda = self._time_varying_lambda(
                market_price, volume, duration_hours, spread, tau
            )
        elif self._use_paper:
            adjusted_lambda = self._hierarchical_lambda(
                market_price, volume, duration_hours, spread
            )
        else:
            adjusted_lambda = base_lambda

        if not math.isfinite(base_lambda) or not math.isfinite(adjusted_lambda):
            raise ValueError('calibrated and adjusted lambda must be finite')

        # Preserve the quote, including the exact probability endpoints.
        # The distortion handles numerical clipping internally for its probit.
        distortion = ProbitDistortion(adjusted_lambda)
        fair_value = (
            market_price
            if market_price in (0.0, 1.0)
            else distortion.inverse(market_price)
        )
        risk_premium = market_price - fair_value

        # Inverting and then reapplying the same model gives no independent
        # evidence of mispricing. Only compare against an explicit forecast.
        benchmark_market_price, mispricing_signal = _compare_reference(
            market_price, distortion, reference_probability
        )

        # Volume tier classification
        volume_tier = self._classify_volume(volume)
        is_target = volume_tier in ('medium', 'high')

        # Confidence
        cal_confidence = self._calibrator.get_confidence(category)
        data_quality = self._data_quality_score(volume, duration_hours, spread)
        confidence = cal_confidence * data_quality

        # Reduce confidence for very-high-volume (premium already competed away)
        if volume_tier == 'very_high':
            confidence *= 0.3

        return FairValueEstimate(
            market_price=market_price,
            fair_value=fair_value,
            risk_premium=risk_premium,
            mispricing_signal=mispricing_signal,
            lambda_base=base_lambda,
            lambda_adjusted=adjusted_lambda,
            confidence=confidence,
            category=category,
            volume_tier=volume_tier,
            is_premium_alpha_target=is_target,
            reference_probability=reference_probability,
            benchmark_market_price=benchmark_market_price,
        )

    def estimate_batch(
        self,
        prices: dict[str, float],
        category: str = 'default',
        volumes: dict[str, float] | None = None,
        durations: dict[str, float] | None = None,
        *,
        reference_probabilities: dict[str, float] | None = None,
    ) -> dict[str, FairValueEstimate]:
        """Estimate values with optional independent forecasts keyed by symbol.

        Symbols without a reference retain a decomposition-only, zero signal.
        """
        results: dict[str, FairValueEstimate] = {}
        for sym, price in prices.items():
            vol = (volumes or {}).get(sym)
            dur = (durations or {}).get(sym)
            results[sym] = self.estimate(
                market_price=price,
                category=category,
                volume=vol,
                duration_hours=dur,
                reference_probability=(reference_probabilities or {}).get(sym),
            )
        return results

    def rank_opportunities(
        self,
        estimates: dict[str, FairValueEstimate],
        min_confidence: float = 0.25,
        premium_alpha_only: bool = True,
    ) -> list[tuple[str, FairValueEstimate]]:
        """Rank independent-forecast residuals by magnitude times confidence.

        Decomposition-only estimates and zero residuals are excluded. The
        optional volume filter and model confidence do not validate the input
        forecast, account for fees, or establish that a residual is tradable.
        """
        filtered = [
            (sym, est)
            for sym, est in estimates.items()
            if est.reference_probability is not None
            and est.benchmark_market_price is not None
            and abs(est.mispricing_signal) > _SIGNAL_TOLERANCE
            and est.confidence >= min_confidence
            and (not premium_alpha_only or est.is_premium_alpha_target)
        ]
        # Sort by |mispricing_signal| * confidence (quality-weighted)
        filtered.sort(
            key=lambda x: abs(x[1].mispricing_signal) * x[1].confidence,
            reverse=True,
        )
        return filtered

    # ── Internal: lambda computation ─────────────────────────────────────

    def _hierarchical_lambda(
        self,
        market_price: float,
        volume: float | None,
        duration_hours: float | None,
        spread: float | None,
    ) -> float:
        """Compute lambda_i using Yang (2026) Table 3 coefficients.

        lambda_i = 0.259 - 0.072*ln(1+V) + 0.143*ln(1+D) - 0.477*|p-0.5| + 0.127*S
        """
        lam = _BETA_CONSTANT

        if volume is not None:
            lam += _BETA_LN_VOLUME * math.log(1 + max(0, volume))

        if duration_hours is not None:
            lam += _BETA_LN_DURATION * math.log(1 + max(0, duration_hours))

        lam += _BETA_EXTREMITY * abs(market_price - 0.5)

        if spread is not None:
            lam += _BETA_SPREAD * max(0, spread)

        return lam

    def _time_varying_lambda(
        self,
        market_price: float,
        volume: float | None,
        duration_hours: float | None,
        spread: float | None,
        tau: float,
    ) -> float:
        """Compute time-varying lambda using Yang (2026) Table 5 coefficients.

        lambda(tau) = gamma_1*tau + gamma_2*tau^2 + beta_0 + covariates
        """
        tau = max(0.0, min(1.0, tau))

        lam = _TV_BETA_CONSTANT
        lam += _TV_GAMMA_TAU * tau
        lam += _TV_GAMMA_TAU_SQ * tau * tau

        if volume is not None:
            lam += _TV_BETA_LN_VOLUME * math.log(1 + max(0, volume))

        if duration_hours is not None:
            lam += _TV_BETA_LN_DURATION * math.log(1 + max(0, duration_hours))

        lam += _TV_BETA_EXTREMITY * abs(market_price - 0.5)

        return lam

    def _classify_volume(self, volume: float | None) -> str:
        """Classify volume tier (from Yang 2026 Table 4 stratification)."""
        if volume is None:
            return 'unknown'
        if volume < 500:
            return 'low'
        if volume < 2000:
            return 'medium'
        if volume < _VOLUME_HIGH_THRESHOLD:
            return 'high'
        return 'very_high'

    def _data_quality_score(
        self,
        volume: float | None,
        duration_hours: float | None,
        spread: float | None,
    ) -> float:
        """Score data quality based on available features (0 to 1)."""
        n_available = sum(1 for v in [volume, duration_hours, spread] if v is not None)
        base = 0.4 + 0.2 * n_available  # 0.4 to 1.0
        return min(1.0, base)
