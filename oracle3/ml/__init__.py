"""Machine-learning layer: calibrated probabilities for Kalshi and Polymarket contracts.

Modules
-------
collect    cached, read-only download of resolved contracts and their price histories
dataset    leakage-safe prediction snapshots built from the raw cache
features   the feature function shared by training and live serving
cv         event-grouped, purged walk-forward splits
metrics    Brier score, log loss, expected calibration error, bootstrap intervals
models     penalized logistic regression, LightGBM, random forest, MLP, market and Wang baselines
benchmark  the full out-of-sample comparison, written to an evidence directory
serve      the trained bundle behind the ``model_probability`` MCP tool
"""
