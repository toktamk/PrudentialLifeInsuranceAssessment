"""Accept automatically vs refer to an underwriter, based on the model's top belief."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score


def referral_tradeoff(y, P, thresholds=(0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)) -> pd.DataFrame:
    """Refer a case when max belief < threshold; report what happens to the rest.

    y must be 0-based and P's columns must be classes 0..N-1 (no +1 anywhere).
    """
    y = np.asarray(y, dtype=int)
    P = np.asarray(P, dtype=float)
    conf = P.max(axis=1)
    pred = P.argmax(axis=1)
    rows = []
    for t in thresholds:
        auto = conf >= t
        n_auto = int(auto.sum())
        row = {"threshold": t, "referral_rate": float(1 - auto.mean()), "n_auto_decided": n_auto}
        if n_auto > 0:
            row["exact_error_auto"] = float(np.mean(pred[auto] != y[auto]))
            row["mean_abs_rating_error_auto"] = float(np.mean(np.abs(pred[auto] - y[auto])))
            row["share_auto_off_by_2plus"] = float(np.mean(np.abs(pred[auto] - y[auto]) >= 2))
            row["qwk_auto"] = (float(cohen_kappa_score(y[auto], pred[auto], weights="quadratic"))
                               if len(np.unique(y[auto])) > 1 else np.nan)
        rows.append(row)
    return pd.DataFrame(rows)
