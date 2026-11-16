"""Model sensitivity computations (Greeks) for the Wang pricing model.

Provides analytic derivatives of the Wang Transform pricing model,
essential for:

1. **Dynamic position sizing** — scale positions by dp/dlambda to
   account for premium sensitivity at different price levels.
2. **Risk decomposition** — separate premium risk from event risk.
3. **Hedge ratios** — compute cross-market hedging weights.
4. **Edge decay estimation** — predict how fast mispricing evaporates.

All Greeks are derived analytically from:

    p_mkt = Phi(Phi^{-1}(p*) + lambda)

Reference:
    Yang, Y. (2026). "Pricing Prediction Markets." Working Paper, UIUC.
"""

from __future__ import annotations

import math

from oracle3.pricing.distortion import _norm_cdf, _norm_ppf

_EPS = 1e-10
_LO = _EPS
_HI = 1.0 - _EPS


def _clamp(p: float) -> float:
    return max(_LO, min(_HI, p))


def _norm_pdf(x: float) -> float:
    """Standard normal PDF."""
    return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


# ── First-order Greeks ───────────────────────────────────────────────────


def delta_lambda(p_physical: float, lam: float) -> float:
    """dp_mkt / dlambda — sensitivity of market price to risk premium.

    = phi(Phi^{-1}(p*) + lambda)

    This is always positive and maximized near p* = Phi(-lambda/2) ≈ 0.5.
    At the extremes (p* near 0 or 1), the sensitivity is near zero.

    Use case: Scale position size inversely with delta_lambda — larger
    positions when premium sensitivity is low (near extremes), smaller
    when it's high (near 50%).
    """
    z = _norm_ppf(_clamp(p_physical))
    return _norm_pdf(z + lam)


def delta_physical(p_physical: float, lam: float) -> float:
    """dp_mkt / dp* — amplification factor from physical to market space.

    = phi(Phi^{-1}(p*) + lambda) / phi(Phi^{-1}(p*))

    This measures how much a change in the true probability is amplified
    (or dampened) in the market price. Values > 1 mean the market
    over-reacts to probability changes; values < 1 mean it under-reacts.
    """
    z = _norm_ppf(_clamp(p_physical))
    phi_z = _norm_pdf(z)
    if phi_z < _EPS:
        return 1.0  # degenerate case
    return _norm_pdf(z + lam) / phi_z


def premium_amount(p_physical: float, lam: float) -> float:
    """Absolute risk premium: p_mkt - p*.

    The dollar amount of risk compensation per contract.
    Maximized near p* = 0.5 for lambda > 0.
    """
    p_mkt = _norm_cdf(_norm_ppf(_clamp(p_physical)) + lam)
    return p_mkt - _clamp(p_physical)


def premium_ratio(p_physical: float, lam: float) -> float:
    """Relative overpricing ratio: p_mkt / p*.

    This IS the favorite-longshot bias. For lambda > 0:
    - Longshots (low p*): high ratio (large proportional overpricing)
    - Favorites (high p*): low ratio (small proportional overpricing)
    - Monotonically decreasing in p* (Theorem 1 in Yang 2026)
    """
    p_phys = _clamp(p_physical)
    p_mkt = _norm_cdf(_norm_ppf(p_phys) + lam)
    return p_mkt / p_phys


def implied_edge(p_market: float, lam: float) -> float:
    """Model-implied edge: |p_mkt - p*_model|.

    The expected profit per contract from trading against the
    risk premium, assuming the model is correctly calibrated.
    """
    p_mkt = _clamp(p_market)
    p_star = _norm_cdf(_norm_ppf(p_mkt) - lam)
    return abs(p_mkt - p_star)


def edge_at_percentile(lam: float, percentile: float = 0.5) -> float:
    """Edge for a contract at the given physical probability percentile.

    Quick helper: what is the expected edge for a 50% event?
    Or a 20% event? Useful for back-of-envelope P&L estimation.
    """
    return premium_amount(_clamp(percentile), lam)


# ── Second-order Greeks ──────────────────────────────────────────────────


def gamma_lambda(p_physical: float, lam: float) -> float:
    """d²p_mkt / dlambda² — convexity of market price in lambda.

    = -(Phi^{-1}(p*) + lambda) * phi(Phi^{-1}(p*) + lambda)

    Negative near p*=0.5 (premium is concave in lambda there),
    positive for extreme probabilities.
    """
    z = _norm_ppf(_clamp(p_physical))
    arg = z + lam
    return -arg * _norm_pdf(arg)


def edge_decay_rate(p_physical: float, lam: float, dlambda_dt: float) -> float:
    """Expected rate of edge decay over time.

    If lambda decays at rate dlambda/dt (from the time-varying model),
    the edge decays at rate: d(edge)/dt = delta_lambda * dlambda/dt.

    The time-varying model from Yang (2026) gives:
        lambda(tau) = gamma_1*tau + gamma_2*tau^2 + beta_0 + covariates
        dlambda/dtau = gamma_1 + 2*gamma_2*tau

    Use case: Estimate how quickly the premium decay trade will
    lose its edge, informing exit timing.
    """
    return delta_lambda(p_physical, lam) * dlambda_dt


# ── Kelly-optimal sizing with model Greeks ───────────────────────────────


def kelly_fraction(
    p_market: float,
    lam: float,
    fee_rate: float = 0.005,
) -> float:
    """Kelly-optimal fraction for a single Wang-model trade.

    The model says the physical probability is p* = Phi(Phi^{-1}(p_mkt) - lambda).
    The market offers odds at p_mkt.

    For a YES bet at price p_mkt with true probability p*:
        f* = (p* * (1 - p_mkt) - (1 - p*) * p_mkt) / (1 - p_mkt)
           = (p* - p_mkt) / (1 - p_mkt)

    For a NO bet at price (1 - p_mkt) with true probability (1 - p*):
        f* = ((1-p*) - (1-p_mkt)) / p_mkt
           = (p_mkt - p*) / p_mkt

    After fees, the edge is reduced by the round-trip fee cost.

    Non-finite inputs (NaN/inf for ``p_market``, ``lam`` or ``fee_rate``)
    return 0.0 so that NaN can never reach position sizing; a negative
    ``fee_rate`` raises ValueError.

    Returns the optimal fraction of bankroll to wager (can be negative
    for NO bets). Capped at [-0.25, 0.25] for safety.
    """
    if fee_rate < 0:
        raise ValueError(f'fee_rate must be non-negative, got {fee_rate}')
    if not (math.isfinite(p_market) and math.isfinite(lam) and math.isfinite(fee_rate)):
        # NaN/inf inputs carry no sizing information; betting nothing is
        # safer than letting NaN clamp into a spurious full-size position.
        return 0.0

    p_mkt = _clamp(p_market)
    p_star = _norm_cdf(_norm_ppf(p_mkt) - lam)
    fee_cost = 2 * fee_rate

    edge = p_star - p_mkt  # negative = overpriced (sell / buy NO)

    if abs(edge) <= fee_cost:
        return 0.0  # no edge after fees

    if edge > 0:
        # Underpriced: buy YES
        net_edge = edge - fee_cost
        kelly = net_edge / (1 - p_mkt) if p_mkt < 1 - _EPS else 0.0
    else:
        # Overpriced: buy NO
        net_edge = -edge - fee_cost
        kelly = -net_edge / p_mkt if p_mkt > _EPS else 0.0

    # Cap for safety (quarter-Kelly is common in practice)
    return max(-0.25, min(0.25, kelly))


# ── Batch computations ───────────────────────────────────────────────────


def compute_greeks_table(
    prices: list[float],
    lam: float,
) -> list[dict[str, float]]:
    """Compute all Greeks for a list of market prices.

    Returns list of dicts with all Greek values per contract.
    """
    results = []
    for p_mkt in prices:
        p_mkt_c = _clamp(p_mkt)
        p_star = _norm_cdf(_norm_ppf(p_mkt_c) - lam)
        results.append({
            'p_market': p_mkt,
            'p_physical': p_star,
            'premium': p_mkt_c - p_star,
            'premium_ratio': premium_ratio(p_star, lam),
            'delta_lambda': delta_lambda(p_star, lam),
            'delta_physical': delta_physical(p_star, lam),
            'gamma_lambda': gamma_lambda(p_star, lam),
            'kelly': kelly_fraction(p_mkt, lam),
            'implied_edge': implied_edge(p_mkt, lam),
        })
    return results
