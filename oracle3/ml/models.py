"""Model zoo: eight learners and the baselines, each fitted only on the rows it is given.

Learners (tuned on purged inner walk-forward blocks of the training rows)
    logistic           L2-penalized logistic regression on standardized features
    spline_logistic    penalized logistic regression on cubic-spline bases of each feature (an additive model)
    random_forest      bagged decision trees
    lightgbm           gradient-boosted trees, early-stopped on each inner validation block
    lightgbm_monotone  the same, constrained so the probability never falls as the market price rises
    xgboost            gradient-boosted trees (histogram), early-stopped
    catboost           gradient-boosted oblivious trees with ordered boosting, early-stopped
    mlp                two-layer neural network

Baselines
    market         the displayed market price itself
    wang           the paper's probit-shift model, Pr(y=1) = Phi(Phi^-1(p) - lambda),
                   one lambda per venue, estimated by ``oracle3.pricing.WangMLE``
    wang_hier      the paper's hierarchical form, lambda_i = X_i beta with the paper's
                   covariates (volume and spread where the venue reports them)
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from oracle3.ml.features import FEATURES, NUMERIC_FEATURES
from oracle3.pricing.wang_mle import WangMLE

P_CLIP = 1e-6


def _sk():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return (
        RandomForestClassifier,
        SimpleImputer,
        LogisticRegression,
        MLPClassifier,
        make_pipeline,
        StandardScaler,
    )


@dataclass
class Learner:
    name: str
    grid: list[dict[str, Any]]
    make: Callable[[dict[str, Any]], Any]
    early_stopping: bool = False
    notes: str = ''
    # fits with early stopping on (X_val, y_val) and returns the number of trees kept
    fit_early: Callable[..., int] | None = None


def _logistic(params: dict[str, Any]) -> Any:
    _, SimpleImputer, LogisticRegression, _, make_pipeline, StandardScaler = _sk()
    return make_pipeline(
        SimpleImputer(
            strategy='constant',
            fill_value=0.0,
            add_indicator=True,
            keep_empty_features=True,
        ),
        StandardScaler(),
        LogisticRegression(C=params['C'], max_iter=3000),
    )


def _random_forest(params: dict[str, Any]) -> Any:
    RandomForestClassifier, SimpleImputer, _, _, make_pipeline, _ = _sk()
    return make_pipeline(
        SimpleImputer(
            strategy='constant',
            fill_value=-1.0,
            add_indicator=True,
            keep_empty_features=True,
        ),
        RandomForestClassifier(
            n_estimators=300,
            max_depth=params['max_depth'],
            min_samples_leaf=params['min_samples_leaf'],
            max_features=params['max_features'],
            n_jobs=-1,
            random_state=0,
        ),
    )


def _mlp(params: dict[str, Any]) -> Any:
    _, SimpleImputer, _, MLPClassifier, make_pipeline, StandardScaler = _sk()
    return make_pipeline(
        SimpleImputer(
            strategy='constant',
            fill_value=0.0,
            add_indicator=True,
            keep_empty_features=True,
        ),
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=params['hidden'],
            alpha=params['alpha'],
            learning_rate_init=1e-3,
            batch_size=512,
            max_iter=200,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
            random_state=0,
        ),
    )


def _lightgbm(params: dict[str, Any]) -> Any:
    import lightgbm as lgb

    return lgb.LGBMClassifier(
        objective='binary',
        n_estimators=params.get('n_estimators', 2000),
        learning_rate=params['learning_rate'],
        num_leaves=params['num_leaves'],
        min_child_samples=params['min_child_samples'],
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=0,
        verbose=-1,
        n_jobs=-1,
    )


def _lightgbm_fit_early(est: Any, X: Any, y: Any, X_val: Any, y_val: Any) -> int:
    import lightgbm as lgb

    est.fit(
        X,
        y,
        eval_set=[(X_val, y_val)],
        eval_metric='binary_logloss',
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )
    return int(max(1, est.best_iteration_ or est.n_estimators))


MONOTONE_UP = (
    'price',
    'logit_price',
)  # other things equal, a higher price never means a lower probability


def _lightgbm_monotone(params: dict[str, Any]) -> Any:
    est = _lightgbm(params)
    est.set_params(monotone_constraints=[int(f in MONOTONE_UP) for f in FEATURES])
    return est


def _xgboost(params: dict[str, Any]) -> Any:
    import xgboost as xgb

    early = 'n_estimators' not in params
    return xgb.XGBClassifier(
        n_estimators=params.get('n_estimators', 2000),
        learning_rate=params['learning_rate'],
        max_depth=params['max_depth'],
        min_child_weight=params['min_child_weight'],
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        tree_method='hist',
        objective='binary:logistic',
        eval_metric='logloss',
        early_stopping_rounds=100 if early else None,
        n_jobs=-1,
        random_state=0,
        verbosity=0,
    )


def _xgboost_fit_early(est: Any, X: Any, y: Any, X_val: Any, y_val: Any) -> int:
    est.fit(X, y, eval_set=[(X_val, y_val)], verbose=False)
    return int(est.best_iteration) + 1


def _catboost(params: dict[str, Any]) -> Any:
    from catboost import CatBoostClassifier

    early = 'n_estimators' not in params
    return CatBoostClassifier(
        iterations=params.get('n_estimators', 2000),
        learning_rate=params['learning_rate'],
        depth=params['depth'],
        l2_leaf_reg=3.0,
        loss_function='Logloss',
        od_type='Iter' if early else None,
        od_wait=100 if early else None,
        random_seed=0,
        verbose=False,
        thread_count=-1,
        allow_writing_files=False,
    )


def _catboost_fit_early(est: Any, X: Any, y: Any, X_val: Any, y_val: Any) -> int:
    est.fit(X, y, eval_set=(X_val, y_val), use_best_model=True, verbose=False)
    return int(est.get_best_iteration()) + 1


def _spline_logistic(params: dict[str, Any]) -> Any:
    """An additive model: each numeric feature enters through its own cubic-spline basis."""
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import QuantileTransformer, SplineTransformer

    _, SimpleImputer, LogisticRegression, _, make_pipeline, _ = _sk()
    numeric = list(NUMERIC_FEATURES)
    indicators = [f for f in FEATURES if f not in NUMERIC_FEATURES]
    basis = ColumnTransformer(
        [
            (
                'splines',
                make_pipeline(
                    SimpleImputer(strategy='median', keep_empty_features=True),
                    # ranks first, so ties and heavy tails do not crowd the knots
                    QuantileTransformer(
                        n_quantiles=200, subsample=200_000, random_state=0
                    ),
                    SplineTransformer(n_knots=6, degree=3, extrapolation='constant'),
                ),
                numeric,
            ),
            ('indicators', 'passthrough', indicators),
        ]
    )
    return make_pipeline(basis, LogisticRegression(C=params['C'], max_iter=2000))


LEARNERS: tuple[Learner, ...] = (
    Learner('logistic', [{'C': c} for c in (0.01, 0.1, 1.0, 10.0)], _logistic),
    Learner(
        'lightgbm',
        [
            {'num_leaves': 15, 'learning_rate': 0.05, 'min_child_samples': 100},
            {'num_leaves': 31, 'learning_rate': 0.05, 'min_child_samples': 50},
            {'num_leaves': 63, 'learning_rate': 0.03, 'min_child_samples': 100},
        ],
        _lightgbm,
        early_stopping=True,
        fit_early=_lightgbm_fit_early,
    ),
    Learner(
        'random_forest',
        [
            {'max_depth': 8, 'min_samples_leaf': 50, 'max_features': 'sqrt'},
            {'max_depth': 16, 'min_samples_leaf': 20, 'max_features': 'sqrt'},
            {'max_depth': None, 'min_samples_leaf': 50, 'max_features': 0.5},
        ],
        _random_forest,
    ),
    Learner(
        'mlp',
        [
            {'hidden': (64, 32), 'alpha': 1e-3},
            {'hidden': (64, 32), 'alpha': 1e-2},
            {'hidden': (128,), 'alpha': 1e-3},
        ],
        _mlp,
    ),
    Learner('spline_logistic', [{'C': c} for c in (0.1, 1.0)], _spline_logistic),
    Learner(
        'lightgbm_monotone',
        [
            {'num_leaves': 15, 'learning_rate': 0.05, 'min_child_samples': 100},
            {'num_leaves': 31, 'learning_rate': 0.05, 'min_child_samples': 50},
        ],
        _lightgbm_monotone,
        early_stopping=True,
        fit_early=_lightgbm_fit_early,
    ),
    Learner(
        'xgboost',
        [
            {'max_depth': 6, 'learning_rate': 0.05, 'min_child_weight': 50},
            {'max_depth': 8, 'learning_rate': 0.05, 'min_child_weight': 20},
        ],
        _xgboost,
        early_stopping=True,
        fit_early=_xgboost_fit_early,
    ),
    Learner(
        'catboost',
        [{'depth': 6, 'learning_rate': 0.08}, {'depth': 8, 'learning_rate': 0.08}],
        _catboost,
        early_stopping=True,
        fit_early=_catboost_fit_early,
    ),
)
# Learners the served bundle may use: their libraries are in the base ``oracle3[ml]`` install.
SERVABLE = (
    'logistic',
    'spline_logistic',
    'random_forest',
    'lightgbm',
    'lightgbm_monotone',
    'mlp',
)


def design(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[:, list(FEATURES)].astype(float)


def fit_learner(
    learner: Learner,
    params: dict[str, Any],
    X: pd.DataFrame,
    y: np.ndarray,
    X_val: pd.DataFrame | None = None,
    y_val: np.ndarray | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Fit one configuration; LightGBM early-stops on the validation rows when they are given."""
    est = learner.make(params)
    used = dict(params)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        if learner.early_stopping and learner.fit_early and X_val is not None:
            used['n_estimators'] = learner.fit_early(est, X, y, X_val, y_val)
        else:
            est.fit(X, y)
    return est, used


def predict(est: Any, X: pd.DataFrame) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return est.predict_proba(X)[:, 1]


# ── baselines ────────────────────────────────────────────────────────────


def _probit(p: np.ndarray) -> np.ndarray:
    return stats.norm.ppf(np.clip(np.asarray(p, float), P_CLIP, 1 - P_CLIP))


def _hier_design(df: pd.DataFrame, platform: str) -> np.ndarray:
    cols = [
        np.ones(len(df)),
        df['log_life_hours'].to_numpy(float),
        df['extremity'].to_numpy(float),
    ]
    if platform == 'kalshi':
        cols += [
            df['log_volume_to_date'].fillna(0.0).to_numpy(float),
            df['spread'].fillna(1.0).to_numpy(float),
        ]
    return np.column_stack(cols)


@dataclass
class WangBaseline:
    """The paper's model: a probit shift of the market price, fitted per venue."""

    hierarchical: bool = False
    beta: dict[str, list[float]] = field(default_factory=dict)

    def fit(self, df: pd.DataFrame) -> WangBaseline:
        for platform, part in df.groupby('platform'):
            X = _hier_design(part, platform) if self.hierarchical else None
            res = WangMLE().fit(
                prices=part['price'].to_numpy(float),
                outcomes=part['label'].to_numpy(float),
                covariates=X,
            )
            self.beta[platform] = [float(b) for b in res.beta]
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        out = np.empty(len(df))
        for platform, part in df.groupby('platform'):
            beta = np.asarray(self.beta.get(platform, [0.0]), float)
            X = (
                _hier_design(part, platform)
                if self.hierarchical
                else np.ones((len(part), 1))
            )
            pos = df.index.get_indexer(part.index)
            out[pos] = stats.norm.cdf(_probit(part['price'].to_numpy(float)) - X @ beta)
        return out
