"""Wang Transform Maximum Likelihood Estimation for prediction markets.

Implements the core estimator from:

    Yang, Y. (2026). "Pricing Prediction Markets: Risk Premiums,
    Incomplete Markets, and a Decomposition Framework." Working Paper,
    University of Illinois Urbana-Champaign.

The model decomposes prediction market prices into physical probability
and risk premium via a probit-space constant shift:

    p_mkt = Phi(Phi^{-1}(p*) + lambda)

where lambda > 0 implies systematic overpricing (risk compensation).

This module provides:
- **Pooled MLE**: single lambda for a homogeneous sample
- **Hierarchical MLE**: lambda_i = X_i * beta with contract-level covariates
  (ln(volume), ln(duration), |price - 0.5|, spread)
- **Analytic gradients** for efficient optimization
- **Three SE estimators**: Fisher, sandwich robust, clustered sandwich
- **Model diagnostics**: LR tests, AIC/BIC, pseudo-R^2, calibration checks

The estimator uses the probit-offset representation:

    Pr(y_i = 1 | p_i) = Phi(z_i - X_i * beta)

where z_i = Phi^{-1}(p_i) is the probit-transformed market price.

This is computationally equivalent to a probit regression with a known
offset, making it numerically stable and fast (N=300K in ~2 seconds).

Empirical priors from the paper (usable as warm starts or defaults):
    Polymarket lambda = 0.166, Kalshi = 0.187, Metaculus = 0.287,
    Manifold (play-money) = -0.218, Pooled (291K contracts) = 0.183.

Usage::

    from oracle3.pricing.wang_mle import WangMLE
    import numpy as np

    # Pooled estimation
    mle = WangMLE()
    result = mle.fit(prices=prices, outcomes=outcomes)
    print(f'lambda = {result.beta[0]:.4f} (SE = {result.se_robust[0]:.4f})')

    # Hierarchical with covariates
    X = mle.build_design_matrix(volumes=vol, durations=dur, prices=prices, spreads=sp)
    result = mle.fit(prices=prices, outcomes=outcomes, covariates=X)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# ── Numerical constants ──────────────────────────────────────────────────

_PRICE_CLIP = 1e-6  # clip prices before probit to avoid +/-inf
_PHI_CLIP = 1e-12  # clip Phi to avoid log(0)
_HESS_EPS = 1e-5  # step size for numerical Hessian
_CHUNK = 30000  # chunk size for BLAS-safe matrix ops (numpy+Accelerate bug)

# ── Empirical priors from Yang (2026) ────────────────────────────────────

LAMBDA_POLYMARKET = 0.166
LAMBDA_KALSHI = 0.187
LAMBDA_METACULUS = 0.287
LAMBDA_MANIFOLD = -0.218  # play-money: underpriced
LAMBDA_POOLED = 0.183

# Hierarchical coefficients (Table 3, full 28-day sample, N=13,274)
HIER_BETA_CONSTANT = 0.2590
HIER_BETA_LN_VOLUME = -0.0716
HIER_BETA_LN_DURATION = 0.1431
HIER_BETA_EXTREMITY = -0.4772
HIER_BETA_SPREAD = 0.1273  # not significant (p=0.76)

# Category-level lambda (Polymarket)
LAMBDA_BY_CATEGORY = {
    'sports': 0.070,
    'politics': 0.054,
    'crypto': 0.253,
    'science': 0.268,
    'tech': 0.268,
    'other': 0.282,
}

# Volume-stratified lambda (Polymarket)
# Key insight: very-high-volume markets have lambda ≈ 0
LAMBDA_BY_VOLUME_TIER = {
    'low': 0.354,  # <$500
    'medium': 0.285,  # $500-2K
    'high': 0.316,  # $2K-10K
    'very_high': -0.031,  # >$10K (not significant, p=0.47)
}


def _require_numpy():
    try:
        import numpy as np

        return np
    except ImportError:
        raise ImportError('numpy is required for WangMLE. pip install numpy') from None


def _require_scipy():
    try:
        import scipy
        from scipy import optimize, stats

        return scipy, optimize, stats
    except ImportError:
        raise ImportError(
            'scipy is required for WangMLE. pip install scipy'
        ) from None


# ── Result dataclass ─────────────────────────────────────────────────────


@dataclass
class MLEResult:
    """Result of Wang Transform MLE estimation.

    Attributes:
        beta: estimated parameter vector (length p)
        se_fisher: Fisher (observed information) standard errors
        se_robust: Huber-White sandwich robust standard errors
        se_cluster: clustered sandwich SEs (if cluster IDs provided)
        log_likelihood: maximized log-likelihood value
        n_obs: number of observations
        n_params: number of parameters
        converged: whether the optimizer converged
        aic: Akaike Information Criterion
        bic: Bayesian Information Criterion
        pseudo_r2: McFadden pseudo R-squared
        covariate_names: names of covariates (if provided)
        hessian: numerical Hessian at the optimum
        vcov_robust: robust variance-covariance matrix
    """

    beta: list[float] = field(default_factory=list)
    se_fisher: list[float] = field(default_factory=list)
    se_robust: list[float] = field(default_factory=list)
    se_cluster: list[float] = field(default_factory=list)
    log_likelihood: float = 0.0
    n_obs: int = 0
    n_params: int = 0
    converged: bool = False
    aic: float = 0.0
    bic: float = 0.0
    pseudo_r2: float = 0.0
    covariate_names: list[str] = field(default_factory=list)
    hessian: object = None  # numpy array
    vcov_robust: object = None  # numpy array

    @property
    def lambda_hat(self) -> float:
        """The estimated constant (first element of beta)."""
        return self.beta[0] if self.beta else 0.0

    def z_stat(self, idx: int = 0, se_type: str = 'robust') -> float:
        """Compute z-statistic for a coefficient."""
        se = self.se_robust if se_type == 'robust' else self.se_fisher
        if idx >= len(se) or se[idx] <= 0:
            return 0.0
        return self.beta[idx] / se[idx]

    def p_value(self, idx: int = 0, se_type: str = 'robust') -> float:
        """Two-sided p-value for a coefficient."""
        _, _, stats = _require_scipy()
        z = abs(self.z_stat(idx, se_type))
        return float(2 * (1 - stats.norm.cdf(z)))

    def predict_interval(
        self,
        p_market: float,
        alpha: float = 0.05,
        covariates: object | None = None,
    ) -> tuple[float, float, float]:
        """Confidence interval for the fair-value probability.

        The fair value is the physical probability implied by the fitted Wang
        transform, ``p_star = Phi(Phi^{-1}(p_market) - lambda_hat)``. The
        interval is constructed with the delta method from the robust
        covariance matrix stored on the result. It is a first-order
        approximation and treats ``p_market`` as fixed.

        Parameters
        ----------
        p_market:
            Observed market price in ``(0, 1)``.
        alpha:
            Significance level in ``(0, 1)``. The interval has coverage
            ``1 - alpha``.
        covariates:
            For a hierarchical model, a vector of covariates with one value
            per estimated parameter. Omit for a pooled model.

        Returns
        -------
        tuple[float, float, float]
            ``(p_star_hat, lower, upper)``.

        Raises
        ------
        ValueError
            If the model has not been fitted, ``alpha`` or ``p_market`` is
            outside its valid range, covariates are missing for a
            hierarchical model, or the covariate length does not match the
            number of parameters.
        """
        np = _require_numpy()
        _, _, stats = _require_scipy()

        if self.vcov_robust is None:
            raise ValueError(
                'vcov_robust is None; fit the model before calling predict_interval'
            )
        if not 0.0 < alpha < 1.0:
            raise ValueError(f'alpha must be in (0, 1); got {alpha}')
        if not 0.0 < p_market < 1.0:
            raise ValueError(f'p_market must be in (0, 1); got {p_market}')

        beta = np.asarray(self.beta, dtype=np.float64)
        vcov = np.asarray(self.vcov_robust, dtype=np.float64)
        if beta.size == 0:
            raise ValueError('beta is empty; fit the model first')
        if not np.isfinite(beta).all():
            raise ValueError('beta contains non-finite values; fit the model again')
        if vcov.shape != (beta.size, beta.size):
            raise ValueError(
                'vcov_robust must have shape '
                f'({beta.size}, {beta.size}); got {vcov.shape}'
            )
        if not np.isfinite(vcov).all():
            raise ValueError(
                'vcov_robust contains non-finite values; fit the model again'
            )

        z_market = float(stats.norm.ppf(p_market))
        if covariates is None:
            if beta.size != 1:
                raise ValueError(
                    'covariates are required for a hierarchical model '
                    f'with {beta.size} parameters'
                )
            eta = z_market - float(beta[0])
            var_eta = float(vcov[0, 0])
        else:
            x = np.asarray(covariates, dtype=np.float64)
            if x.ndim != 1:
                raise ValueError('covariates must be a 1-dimensional array')
            if x.size != beta.size:
                raise ValueError(
                    f'covariates must have length {beta.size}; got {x.size}'
                )
            if beta.size == 1:
                raise ValueError('covariates are not supported for a pooled model')
            eta = z_market - float(x @ beta)
            var_eta = float(x @ vcov @ x)

        if not math.isfinite(var_eta):
            raise ValueError('fair-value variance is not finite')
        if var_eta < 0.0:
            if var_eta > -1e-12:
                var_eta = 0.0
            else:
                raise ValueError('fair-value variance is negative')

        se_eta = math.sqrt(var_eta)
        z = float(stats.norm.ppf(1.0 - alpha / 2.0))
        p_star = float(stats.norm.cdf(eta))
        lower = float(stats.norm.cdf(eta - z * se_eta))
        upper = float(stats.norm.cdf(eta + z * se_eta))
        return p_star, lower, upper

    def summary_table(self) -> str:
        """Generate a formatted summary table."""
        lines = [
            f'Wang Transform MLE (N={self.n_obs}, p={self.n_params})',
            f'Log-likelihood: {self.log_likelihood:.2f}  '
            f'AIC: {self.aic:.2f}  BIC: {self.bic:.2f}  '
            f'Pseudo-R²: {self.pseudo_r2:.4f}',
            '',
            f'{"Variable":<20} {"Coef":>10} {"SE(Fisher)":>12} '
            f'{"SE(Robust)":>12} {"z":>8} {"p":>10}',
            '-' * 74,
        ]
        names = self.covariate_names or [
            f'beta_{i}' for i in range(len(self.beta))
        ]
        for i, name in enumerate(names):
            b = self.beta[i]
            se_f = self.se_fisher[i] if i < len(self.se_fisher) else float('nan')
            se_r = self.se_robust[i] if i < len(self.se_robust) else float('nan')
            z = self.z_stat(i)
            p = self.p_value(i)
            stars = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
            lines.append(
                f'{name:<20} {b:>10.4f} {se_f:>12.4f} {se_r:>12.4f} '
                f'{z:>8.2f} {p:>9.2e} {stars}'
            )
        return '\n'.join(lines)


# ── Main estimator class ─────────────────────────────────────────────────


class WangMLE:
    """Wang Transform Maximum Likelihood Estimator.

    Parameters
    ----------
    optimizer:
        scipy optimizer method (default 'L-BFGS-B').
    ftol:
        Function tolerance for convergence (default 1e-12).
    gtol:
        Gradient tolerance for convergence (default 1e-10).
    max_iter:
        Maximum optimizer iterations (default 500).
    """

    def __init__(
        self,
        optimizer: str = 'L-BFGS-B',
        ftol: float = 1e-12,
        gtol: float = 1e-10,
        max_iter: int = 500,
    ) -> None:
        self._optimizer = optimizer
        self._ftol = ftol
        self._gtol = gtol
        self._max_iter = max_iter

    def fit(
        self,
        prices: object,
        outcomes: object,
        covariates: object | None = None,
        cluster_ids: object | None = None,
        covariate_names: list[str] | None = None,
        initial_beta: object | None = None,
    ) -> MLEResult:
        """Estimate the Wang Transform model via MLE.

        Parameters
        ----------
        prices:
            Array of market prices p_i in (0, 1). Shape (N,).
        outcomes:
            Array of binary outcomes y_i in {0, 1}. Shape (N,).
        covariates:
            Optional design matrix X_i. Shape (N, p).
            If None, estimates a pooled model (single lambda).
        cluster_ids:
            Optional array of cluster IDs for clustered SEs.
            Shape (N,). Typically contract IDs.
        covariate_names:
            Optional names for covariates (for summary table).
        initial_beta:
            Initial parameter values. If None, uses empirical priors.

        Returns
        -------
        MLEResult with all estimation outputs.
        """
        np = _require_numpy()
        scipy, optimize, stats = _require_scipy()

        # Prepare data
        p = np.asarray(prices, dtype=np.float64)
        y = np.asarray(outcomes, dtype=np.float64)
        n = len(p)

        # Clip prices for numerical stability
        p = np.clip(p, _PRICE_CLIP, 1 - _PRICE_CLIP)
        z = stats.norm.ppf(p)  # probit-transformed prices

        # Design matrix
        if covariates is not None:
            X = np.asarray(covariates, dtype=np.float64)
            if X.ndim == 1:
                X = X.reshape(-1, 1)
        else:
            X = np.ones((n, 1), dtype=np.float64)

        n_params = X.shape[1]

        # Infer covariate names from last build_design_matrix call
        if covariate_names is None and hasattr(self, '_last_covariate_names'):
            covariate_names = self._last_covariate_names

        # Initial values
        if initial_beta is not None:
            beta0 = np.asarray(initial_beta, dtype=np.float64)
        else:
            beta0 = np.zeros(n_params, dtype=np.float64)
            beta0[0] = LAMBDA_POLYMARKET  # warm start from empirical prior

        # Set up bounds
        bounds = [(-3.0, 3.0)] + [(-10.0, 10.0)] * (n_params - 1)

        # Optimize negative log-likelihood
        def neg_ll(beta):
            return self._neg_log_likelihood(beta, z, y, X)

        def neg_grad(beta):
            return self._neg_gradient(beta, z, y, X)

        result = optimize.minimize(
            neg_ll,
            beta0,
            jac=neg_grad,
            method=self._optimizer,
            bounds=bounds,
            options={
                'ftol': self._ftol,
                'gtol': self._gtol,
                'maxiter': self._max_iter,
            },
        )

        beta_hat = result.x
        ll = -result.fun
        converged = result.success

        if not converged:
            logger.warning('WangMLE: optimizer did not converge: %s', result.message)

        # Null model log-likelihood (for pseudo-R^2)
        ll_null = float(np.sum(
            y * np.log(np.clip(p, _PHI_CLIP, 1 - _PHI_CLIP))
            + (1 - y) * np.log(np.clip(1 - p, _PHI_CLIP, 1 - _PHI_CLIP))
        ))

        # Compute standard errors
        H = self._numerical_hessian(beta_hat, z, y, X)
        se_fisher = self._fisher_se(H)
        se_robust, V_robust = self._sandwich_se(beta_hat, z, y, X, H)
        se_cluster_arr: list[float] = []
        if cluster_ids is not None:
            se_cluster_arr = self._clustered_se(
                beta_hat, z, y, X, H, np.asarray(cluster_ids)
            )

        # Model comparison statistics
        aic = -2 * ll + 2 * n_params
        bic = -2 * ll + n_params * math.log(n)
        pseudo_r2 = 1 - ll / ll_null if ll_null != 0 else 0.0

        # Covariate names
        if covariate_names is None:
            if n_params == 1:
                covariate_names = ['lambda']
            else:
                covariate_names = ['constant'] + [
                    f'x_{i}' for i in range(1, n_params)
                ]

        return MLEResult(
            beta=beta_hat.tolist(),
            se_fisher=se_fisher,
            se_robust=se_robust,
            se_cluster=se_cluster_arr,
            log_likelihood=ll,
            n_obs=n,
            n_params=n_params,
            converged=converged,
            aic=aic,
            bic=bic,
            pseudo_r2=pseudo_r2,
            covariate_names=covariate_names,
            hessian=H,
            vcov_robust=V_robust,
        )

    def build_design_matrix(
        self,
        volumes: object | None = None,
        durations_hours: object | None = None,
        prices: object | None = None,
        spreads: object | None = None,
        n: int | None = None,
    ) -> object:
        """Build the hierarchical covariate design matrix.

        Constructs X = [1, ln(1+Volume), ln(1+Duration), |p-0.5|, Spread]
        following Yang (2026) specification.

        Parameters
        ----------
        volumes: trading volumes (USD or shares)
        durations_hours: contract duration in hours
        prices: market prices (for extremity |p - 0.5|)
        spreads: bid-ask spreads
        n: number of observations (inferred from arrays if not given)

        Returns
        -------
        numpy array of shape (N, p) where p depends on available covariates.
        """
        np = _require_numpy()

        # Determine N
        for arr in [volumes, durations_hours, prices, spreads]:
            if arr is not None:
                n = len(np.asarray(arr))
                break
        if n is None:
            raise ValueError('At least one covariate array or n must be provided')

        cols = [np.ones(n)]  # intercept
        names = ['constant']

        if volumes is not None:
            v = np.asarray(volumes, dtype=np.float64)
            cols.append(np.log(1 + np.clip(v, 0, None)))
            names.append('ln(1+volume)')

        if durations_hours is not None:
            d = np.asarray(durations_hours, dtype=np.float64)
            cols.append(np.log(1 + np.clip(d, 0, None)))
            names.append('ln(1+duration)')

        if prices is not None:
            p = np.asarray(prices, dtype=np.float64)
            cols.append(np.abs(p - 0.5))
            names.append('|p-0.5|')

        if spreads is not None:
            s = np.asarray(spreads, dtype=np.float64)
            cols.append(s)
            names.append('spread')

        X = np.column_stack(cols)
        # Store names separately (numpy arrays don't support custom attrs)
        self._last_covariate_names = names
        return X

    def lr_test(self, result_full: MLEResult, result_restricted: MLEResult) -> tuple[float, float, int]:
        """Likelihood ratio test comparing two nested models.

        Returns (chi2_stat, p_value, df).
        """
        _, _, stats = _require_scipy()
        df = result_full.n_params - result_restricted.n_params
        if df <= 0:
            return 0.0, 1.0, 0
        chi2 = 2 * (result_full.log_likelihood - result_restricted.log_likelihood)
        chi2 = max(0.0, chi2)
        p_val = float(1 - stats.chi2.cdf(chi2, df))
        return chi2, p_val, df

    # ── Core likelihood and gradient ─────────────────────────────────────

    def _neg_log_likelihood(self, beta, z, y, X):
        """Negative log-likelihood (to be minimized)."""
        np = _require_numpy()
        _, _, stats = _require_scipy()

        eta = z - self._safe_matmul(X, beta)
        Phi = np.clip(stats.norm.cdf(eta), _PHI_CLIP, 1 - _PHI_CLIP)
        ll = np.sum(y * np.log(Phi) + (1 - y) * np.log(1 - Phi))
        return -ll

    def _neg_gradient(self, beta, z, y, X):
        """Analytic gradient of the negative log-likelihood.

        d(-ll)/d(beta) = sum_i [ (y_i * phi/Phi - (1-y_i) * phi/(1-Phi)) * X_i ]

        The sign follows from the offset being z - X*beta (negative X*beta).
        """
        np = _require_numpy()
        _, _, stats = _require_scipy()

        eta = z - self._safe_matmul(X, beta)
        Phi = np.clip(stats.norm.cdf(eta), _PHI_CLIP, 1 - _PHI_CLIP)
        phi = stats.norm.pdf(eta)

        score_i = y * phi / Phi - (1 - y) * phi / (1 - Phi)
        grad_ll = self._safe_matmul(X.T, score_i)
        return grad_ll  # positive because we want d(-ll)/d(beta) = -d(ll)/d(beta) -> this gives d(ll)/d(beta) which we negate

    def _safe_matmul(self, A, B):
        """Chunked matrix multiplication to avoid numpy+Accelerate BLAS bug.

        numpy 1.26.3 + Apple Accelerate BLAS produces NaN for N >= 32768.
        Workaround: process in chunks of 30000.
        """
        np = _require_numpy()
        A = np.asarray(A)
        B = np.asarray(B)

        if A.ndim == 2 and A.shape[0] > _CHUNK:
            # A is (N, p), B is (p,) -> result is (N,)
            n = A.shape[0]
            result = np.empty(n, dtype=np.float64)
            for start in range(0, n, _CHUNK):
                end = min(start + _CHUNK, n)
                result[start:end] = A[start:end] @ B
            return result
        elif A.ndim == 2 and B.ndim == 2 and B.shape[0] > _CHUNK:
            # A is (p, N), B is (N,) via transposed call
            return A @ B
        else:
            return A @ B

    # ── Standard errors ──────────────────────────────────────────────────

    def _numerical_hessian(self, beta, z, y, X):
        """Numerical Hessian via central differences with eigenvalue regularization."""
        np = _require_numpy()

        p = len(beta)
        H = np.zeros((p, p), dtype=np.float64)

        for j in range(p):
            beta_plus = beta.copy()
            beta_minus = beta.copy()
            beta_plus[j] += _HESS_EPS
            beta_minus[j] -= _HESS_EPS
            g_plus = self._neg_gradient(beta_plus, z, y, X)
            g_minus = self._neg_gradient(beta_minus, z, y, X)
            H[j, :] = (g_plus - g_minus) / (2 * _HESS_EPS)

        # Symmetrize
        H = (H + H.T) / 2

        # Eigenvalue regularization: ensure positive definite
        eigvals = np.linalg.eigvalsh(H)
        min_eig = eigvals.min()
        if min_eig <= 0:
            ridge = max(-min_eig + 1e-6, 1e-6)
            H += ridge * np.eye(p)
            logger.debug('Hessian regularized: min_eig=%.6f, ridge=%.6f', min_eig, ridge)

        return H

    def _fisher_se(self, H) -> list[float]:
        """Fisher standard errors from the observed information matrix."""
        np = _require_numpy()
        try:
            V = np.linalg.inv(H)
            return [math.sqrt(max(0, V[i, i])) for i in range(V.shape[0])]
        except np.linalg.LinAlgError:
            logger.warning('Hessian not invertible for Fisher SEs')
            return [float('nan')] * H.shape[0]

    def _sandwich_se(self, beta, z, y, X, H) -> tuple[list[float], object]:
        """Huber-White sandwich robust standard errors.

        V_robust = H^{-1} @ meat @ H^{-1}
        where meat = S^T @ S and S is the (N, p) score matrix.
        """
        np = _require_numpy()
        _, _, stats = _require_scipy()

        eta = z - self._safe_matmul(X, beta)
        Phi = np.clip(stats.norm.cdf(eta), _PHI_CLIP, 1 - _PHI_CLIP)
        phi = stats.norm.pdf(eta)

        score_i = y * phi / Phi - (1 - y) * phi / (1 - Phi)
        S = score_i[:, None] * X  # (N, p) score matrix

        try:
            H_inv = np.linalg.inv(H)
            meat = S.T @ S
            V_robust = H_inv @ meat @ H_inv
            se = [math.sqrt(max(0, V_robust[i, i])) for i in range(V_robust.shape[0])]
            return se, V_robust
        except np.linalg.LinAlgError:
            logger.warning('Hessian not invertible for sandwich SEs')
            p = H.shape[0]
            return [float('nan')] * p, np.full((p, p), float('nan'))

    def _clustered_se(self, beta, z, y, X, H, cluster_ids) -> list[float]:
        """Liang-Zeger clustered sandwich standard errors.

        Clusters scores by ID, then computes sandwich with
        finite-sample correction: (G/(G-1)) * ((N-1)/(N-p)).
        """
        np = _require_numpy()
        _, _, stats = _require_scipy()

        n = len(y)
        p = X.shape[1]
        eta = z - self._safe_matmul(X, beta)
        Phi = np.clip(stats.norm.cdf(eta), _PHI_CLIP, 1 - _PHI_CLIP)
        phi = stats.norm.pdf(eta)

        score_i = y * phi / Phi - (1 - y) * phi / (1 - Phi)
        S = score_i[:, None] * X  # (N, p)

        # Cluster-level score sums
        unique_ids = np.unique(cluster_ids)
        G = len(unique_ids)
        meat = np.zeros((p, p), dtype=np.float64)
        for uid in unique_ids:
            mask = cluster_ids == uid
            s_bar = S[mask].sum(axis=0)  # (p,)
            meat += np.outer(s_bar, s_bar)

        # Finite-sample correction
        correction = (G / (G - 1)) * ((n - 1) / (n - p)) if G > 1 else 1.0

        try:
            H_inv = np.linalg.inv(H)
            V_cluster = correction * H_inv @ meat @ H_inv
            return [math.sqrt(max(0, V_cluster[i, i])) for i in range(p)]
        except np.linalg.LinAlgError:
            logger.warning('Hessian not invertible for clustered SEs')
            return [float('nan')] * p
