"""Model zoo: four learners and two baselines, each fitted only on the rows it is given.

Learners (tuned on an inner purged split of the training rows)
    logistic       L2-penalized logistic regression on standardized features
    lightgbm       gradient-boosted trees, early-stopped on the inner validation block
    random_forest  bagged decision trees
    mlp            two-layer neural network

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

from oracle3.ml.features import FEATURES
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
        if learner.early_stopping and X_val is not None:
            import lightgbm as lgb

            est.fit(
                X,
                y,
                eval_set=[(X_val, y_val)],
                eval_metric='binary_logloss',
                callbacks=[lgb.early_stopping(100, verbose=False)],
            )
            used['n_estimators'] = int(max(1, est.best_iteration_ or est.n_estimators))
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
