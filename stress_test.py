"""Missing-data stress test: remove extra observed values completely at random (MCAR)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from metrics import evaluate


def inject_mcar(X: pd.DataFrame, frac: float, seed: int, columns=None) -> pd.DataFrame:
    """Set a fraction `frac` of the currently OBSERVED cells to NaN."""
    rng = np.random.default_rng(seed)
    Xc = X.copy()
    cols = list(X.columns) if columns is None else list(columns)
    vals = Xc[cols].to_numpy(dtype=float, copy=True)
    drop = (~np.isnan(vals)) & (rng.random(vals.shape) < frac)
    vals[drop] = np.nan
    Xc[cols] = vals
    return Xc


def missing_data_stress_test(X_test, y_test, models: dict, fractions=(0.0, 0.1, 0.3, 0.5),
                             seed=42, columns=None) -> pd.DataFrame:
    """models: name -> fitted object with predict_proba(X). The same corrupted copy of
    the test set is given to every model at each level, so the comparison is paired."""
    rows = []
    for frac in fractions:
        Xc = X_test if frac == 0 else inject_mcar(X_test, frac, seed, columns)
        for name, m in models.items():
            res = evaluate(y_test, m.predict_proba(Xc))
            rows.append({"extra_missing": frac, "model": name, **res})
        print(f"stress test: {int(frac * 100)}% extra missing done")
    return pd.DataFrame(rows)
