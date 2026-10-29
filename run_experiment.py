"""End-to-end experiment on the Prudential Life Insurance Assessment data.

    python run_experiment.py --data data/train.csv --out results
    python run_experiment.py --data data/train.csv --out results_quick --quick   # fast check

Every reported number comes from the held-out TEST set (20%). The validation set
(20%) is used only for early stopping of the ER model.
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd

from baselines import fit_baselines
from data import load_prudential, split_60_20_20
from decision_rule import referral_tradeoff
from er_engine import EvidentialReasoningClassifier
from explanations import underwriter_sheet
from metrics import bootstrap_ci, mcnemar, paired_bootstrap_diff, reliability_table
from stress_test import missing_data_stress_test


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/train.csv")
    ap.add_argument("--out", default="results")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--max-bins", type=int, default=10)
    ap.add_argument("--quick", action="store_true", help="10k-row subsample, 100 bootstraps")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    X, y, groups = load_prudential(args.data)
    if args.quick:
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(len(y), size=min(10000, len(y)), replace=False)
        X, y = X.iloc[idx].reset_index(drop=True), y[idx]
        args.n_boot = min(args.n_boot, 100)
    (X_tr, y_tr), (X_va, y_va), (X_te, y_te) = split_60_20_20(X, y, args.seed)
    print(f"train {len(y_tr)} | val {len(y_va)} | test {len(y_te)} | features {X.shape[1]}")
    print("class counts (0-based):", np.bincount(y, minlength=8).tolist())

    # ---- models ----
    models = fit_baselines(X_tr, y_tr, seed=args.seed)
    print("Fitting binned naive Bayes (ER rule with w = r = 1)")
    models["Binned naive Bayes (ER, w=r=1)"] = EvidentialReasoningClassifier(
        max_bins=args.max_bins, learn_params=False, random_state=args.seed, verbose=False).fit(X_tr, y_tr)
    print("Fitting ER rule with learned weights and reliabilities")
    er = EvidentialReasoningClassifier(max_bins=args.max_bins, learn_params=True,
                                       random_state=args.seed, verbose=True).fit(X_tr, y_tr, X_va, y_va)
    models["ER rule (learned w, r)"] = er
    pd.DataFrame(er.history_).to_csv(out / "er_training_history.csv", index=False)

    # ---- RQ1: test-set metrics with bootstrap CIs ----
    probs = {name: m.predict_proba(X_te) for name, m in models.items()}
    tables = []
    for name, P in probs.items():
        print(f"bootstrap CIs: {name}")
        t = bootstrap_ci(y_te, P, n_boot=args.n_boot, seed=args.seed)
        t.insert(0, "model", name)
        tables.append(t)
    metrics_long = pd.concat(tables, ignore_index=True)
    metrics_long.to_csv(out / "test_metrics_with_ci.csv", index=False)
    summary = metrics_long.assign(
        value=lambda d: d.apply(lambda r: f"{r.estimate:.3f} ({r.ci_low:.3f}-{r.ci_high:.3f})", axis=1)
    ).pivot(index="model", columns="metric", values="value")
    summary.to_csv(out / "test_metrics_summary.csv")
    print(summary.to_string())

    # paired comparisons: ER vs every other model
    comps = []
    er_P = probs["ER rule (learned w, r)"]
    for name, P in probs.items():
        if name == "ER rule (learned w, r)":
            continue
        mc = mcnemar(y_te, er_P.argmax(1), P.argmax(1))
        diff = paired_bootstrap_diff(y_te, er_P, P, n_boot=args.n_boot, seed=args.seed)
        comps.append({"comparison": f"ER vs {name}", **mc,
                      **{f"QWK_diff_{k}": v for k, v in diff.items() if k != "metric"}})
    pd.DataFrame(comps).to_csv(out / "paired_tests_er_vs_others.csv", index=False)

    # calibration tables (plot later from these CSVs)
    pd.concat([reliability_table(y_te, P).assign(model=n) for n, P in probs.items()]) \
        .to_csv(out / "reliability_tables.csv", index=False)

    # ---- RQ2: missing-data stress test ----
    stress = missing_data_stress_test(X_te, y_te, models, seed=args.seed)
    stress.to_csv(out / "missing_data_stress_test.csv", index=False)

    # ---- RQ4: accept / refer ----
    pd.concat([referral_tradeoff(y_te, P).assign(model=n) for n, P in probs.items()]) \
        .to_csv(out / "referral_tradeoff.csv", index=False)

    # ---- RQ3: explanations + learned parameters ----
    params = er.parameters_frame()
    params.to_csv(out / "er_learned_weights_reliabilities.csv", index=False)
    inv = {c: g for g, cols in groups.items() for c in cols}
    params.assign(group=params["evidence"].map(inv).fillna("class prior")) \
        .groupby("group")[["weight", "reliability"]].agg(["mean", "min", "max", "count"]) \
        .to_csv(out / "er_parameters_by_group.csv")
    sheets = [underwriter_sheet(er, X_te.iloc[[i]], groups, applicant_id=f"test row {i}") for i in (0, 1, 2)]
    (out / "underwriter_explanations.md").write_text("\n\n".join(sheets), encoding="utf-8")

    import sklearn
    import torch
    info = {"seed": args.seed, "quick": args.quick, "n_boot": args.n_boot, "max_bins": args.max_bins,
            "n_train": int(len(y_tr)), "n_val": int(len(y_va)), "n_test": int(len(y_te)),
            "n_features": int(X.shape[1]), "er_epochs_run": len(er.history_),
            "python": platform.python_version(), "torch": torch.__version__,
            "sklearn": sklearn.__version__, "runtime_minutes": round((time.time() - t0) / 60, 1)}
    (out / "run_info.json").write_text(json.dumps(info, indent=2))
    print(f"Done in {info['runtime_minutes']} min. Results in {out.resolve()}")


if __name__ == "__main__":
    main()
