"""Loading and splitting the Prudential Life Insurance Assessment data (Kaggle, train.csv)."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

GROUP_PREFIXES = ["Product_Info", "Employment_Info", "InsuredInfo", "Insurance_History",
                  "Family_Hist", "Medical_History", "Medical_Keyword"]


def load_prudential(path="data/train.csv"):
    """Return X (all numeric, NaN = missing), y (0-based risk rating 0..7) and feature groups.

    Product_Info_2 (e.g. 'D3') is the only text column; it is split into a letter
    code and a digit so every column is numeric. Nothing is imputed here.
    """
    df = pd.read_csv(path)
    y = (df["Response"].astype(int) - 1).to_numpy()  # Response 1..8 -> 0..7, the ONLY shift anywhere
    X = df.drop(columns=["Id", "Response"]).copy()
    if "Product_Info_2" in X.columns:
        s = X.pop("Product_Info_2").astype(str)
        X["Product_Info_2_letter"] = s.str[0].map(lambda c: ord(c) - ord("A")).astype(float)
        X["Product_Info_2_digit"] = pd.to_numeric(s.str[1:], errors="coerce")
    non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        raise ValueError(f"Unexpected non-numeric columns: {non_numeric}")
    X = X.astype(float)
    return X, y, feature_groups(X.columns)


def feature_groups(columns) -> dict[str, list[str]]:
    """Map Prudential columns to their named groups (Ins_Age, Ht, Wt and BMI stand alone)."""
    groups: dict[str, list[str]] = {}
    for c in columns:
        g = next((p for p in GROUP_PREFIXES if c.startswith(p + "_")), None)
        if g is None:
            g = re.sub(r"_\d+$", "", c)
        groups.setdefault(g, []).append(c)
    return groups


def split_60_20_20(X, y, seed=42):
    """Stratified train / validation / test split. Validation is used only for early
    stopping of the ER model; every reported number comes from the test set."""
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.4, random_state=seed, stratify=y)
    X_va, X_te, y_va, y_te = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=seed, stratify=y_tmp)
    return (X_tr, y_tr), (X_va, y_va), (X_te, y_te)
