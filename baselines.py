"""Reference models. All return probabilities with columns ordered 0..N-1."""
from __future__ import annotations

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover
    XGBClassifier = None


def make_baselines(seed=42) -> dict:
    models = {
        "Logistic regression": Pipeline([
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=3000, C=1.0)),
        ]),
        "Gaussian naive Bayes": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("clf", GaussianNB()),
        ]),
        "Decision tree (depth 6)": Pipeline([
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("clf", DecisionTreeClassifier(max_depth=6, min_samples_leaf=50, random_state=seed)),
        ]),
    }
    if XGBClassifier is not None:
        # XGBoost handles NaN natively, so no imputation.
        models["XGBoost"] = XGBClassifier(
            n_estimators=400, learning_rate=0.05, max_depth=6, subsample=0.8,
            colsample_bytree=0.6, objective="multi:softprob", eval_metric="mlogloss",
            tree_method="hist", random_state=seed, n_jobs=-1)
    else:
        print("xgboost not installed: skipping the XGBoost baseline (pip install xgboost)")
    return models


def fit_baselines(X_train, y_train, seed=42, verbose=True) -> dict:
    fitted = {}
    for name, model in make_baselines(seed).items():
        if verbose:
            print(f"Fitting baseline: {name}")
        model.fit(X_train, y_train)
        fitted[name] = model
    return fitted
