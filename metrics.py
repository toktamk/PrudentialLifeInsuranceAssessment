"""Evaluation metrics. Every function expects 0-based labels 0..N-1 and
probability matrices of shape (n_samples, N) whose columns are classes 0..N-1."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.metrics import cohen_kappa_score, log_loss, roc_auc_score


def _check(y, P):
    y = np.asarray(y, dtype=int)
    P = np.asarray(P, dtype=float)
    if y.min() < 0 or y.max() >= P.shape[1]:
        raise ValueError("Labels must be 0..N-1 and match the probability columns")
    return y, P


def onehot(y, n_classes):
    return np.eye(n_classes)[np.asarray(y, dtype=int)]


def expected_class(P):
    """Rounded expected rating: usually a better point prediction for QWK than argmax."""
    P = np.asarray(P, dtype=float)
    return np.clip(np.rint(P @ np.arange(P.shape[1])), 0, P.shape[1] - 1).astype(int)


def multiclass_brier(y, P):
    """Multi-class Brier score: mean over cases of sum_k (p_k - 1[y=k])^2. Range 0-2; lower is better."""
    y, P = _check(y, P)
    return float(np.mean(np.sum((P - onehot(y, P.shape[1])) ** 2, axis=1)))


def ece_top_label(y, P, n_bins=15):
    """Expected calibration error of the top-label confidence (equal-width bins)."""
    y, P = _check(y, P)
    conf = P.max(axis=1)
    correct = (P.argmax(axis=1) == y).astype(float)
    bins = np.minimum((conf * n_bins).astype(int), n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        sel = bins == b
        if sel.any():
            ece += sel.mean() * abs(correct[sel].mean() - conf[sel].mean())
    return float(ece)


def reliability_table(y, P, n_bins=10):
    y, P = _check(y, P)
    conf = P.max(axis=1)
    correct = (P.argmax(axis=1) == y).astype(float)
    bins = np.minimum((conf * n_bins).astype(int), n_bins - 1)
    rows = []
    for b in range(n_bins):
        sel = bins == b
        if sel.any():
            rows.append({"bin": b, "mean_confidence": conf[sel].mean(),
                         "accuracy": correct[sel].mean(), "n": int(sel.sum())})
    return pd.DataFrame(rows)


def evaluate(y, P) -> dict:
    """All headline metrics for one model on one set."""
    y, P = _check(y, P)
    N = P.shape[1]
    Pc = np.clip(P, 1e-15, 1.0)
    Pc = Pc / Pc.sum(axis=1, keepdims=True)
    out = {
        "QWK (argmax)": cohen_kappa_score(y, P.argmax(axis=1), weights="quadratic"),
        "QWK (expected rating)": cohen_kappa_score(y, expected_class(P), weights="quadratic"),
        "Accuracy": float(np.mean(P.argmax(axis=1) == y)),
        "Log loss": log_loss(y, Pc, labels=np.arange(N)),
        "Brier (multi-class)": multiclass_brier(y, P),
        "ECE (top-label)": ece_top_label(y, P),
    }
    try:
        out["Macro AUROC (OvR)"] = roc_auc_score(onehot(y, N), P, average="macro")
    except ValueError:  # a class absent from this sample
        out["Macro AUROC (OvR)"] = np.nan
    return {k: float(v) for k, v in out.items()}


def bootstrap_ci(y, P, n_boot=1000, seed=0, alpha=0.05) -> pd.DataFrame:
    """Point estimate and percentile bootstrap CI for every metric in evaluate()."""
    y, P = _check(y, P)
    rng = np.random.default_rng(seed)
    point = evaluate(y, P)
    samples = {k: [] for k in point}
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        for k, v in evaluate(y[idx], P[idx]).items():
            samples[k].append(v)
    rows = []
    for k, v in point.items():
        s = np.array(samples[k])
        s = s[~np.isnan(s)]
        rows.append({"metric": k, "estimate": v,
                     "ci_low": np.quantile(s, alpha / 2) if len(s) else np.nan,
                     "ci_high": np.quantile(s, 1 - alpha / 2) if len(s) else np.nan})
    return pd.DataFrame(rows)


def mcnemar(y, pred_a, pred_b) -> dict:
    """McNemar test (with continuity correction) on exact-class correctness of two models."""
    y = np.asarray(y)
    a = np.asarray(pred_a) == y
    b = np.asarray(pred_b) == y
    n01 = int(np.sum(a & ~b))   # A right, B wrong
    n10 = int(np.sum(~a & b))   # A wrong, B right
    if n01 + n10 == 0:
        return {"a_right_b_wrong": n01, "a_wrong_b_right": n10, "chi2": 0.0, "p_value": 1.0}
    stat = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)
    return {"a_right_b_wrong": n01, "a_wrong_b_right": n10,
            "chi2": float(stat), "p_value": float(chi2.sf(stat, 1))}


def paired_bootstrap_diff(y, P_a, P_b, metric="QWK (expected rating)", n_boot=1000, seed=0):
    """Bootstrap distribution of metric(A) - metric(B) on the same resampled cases."""
    y, P_a = _check(y, P_a)
    _, P_b = _check(y, P_b)
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        diffs.append(evaluate(y[idx], P_a[idx])[metric] - evaluate(y[idx], P_b[idx])[metric])
    diffs = np.array(diffs)
    point = evaluate(y, P_a)[metric] - evaluate(y, P_b)[metric]
    return {"metric": metric, "difference": float(point),
            "ci_low": float(np.quantile(diffs, 0.025)), "ci_high": float(np.quantile(diffs, 0.975)),
            "share_of_resamples_A_better": float(np.mean(diffs > 0))}
