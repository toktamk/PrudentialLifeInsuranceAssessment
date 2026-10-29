"""Underwriter-facing explanation: what each group of evidence did to one applicant's rating."""
from __future__ import annotations

import numpy as np
import pandas as pd


def underwriter_sheet(er_model, x_row: pd.DataFrame, groups: dict, applicant_id=None,
                      top_k=6, rating_offset=1) -> str:
    """Markdown sheet for one applicant (x_row: a one-row DataFrame).

    The 'effect' of a group is how the expected risk rating would change if that
    group's evidence were removed and everything else were combined again with
    the ER rule. A positive shift means the group was pulling the rating DOWN.
    Ratings are shown on the original 1-8 scale (rating_offset=1).
    """
    P = er_model.predict_proba(x_row)[0]
    levels = np.arange(len(P))
    pred = int(P.argmax())
    eff = er_model.removal_effect(x_row, groups)
    eff = eff.assign(abs_shift=eff["expected_rating_shift_if_removed"].abs())
    eff = eff.sort_values("abs_shift", ascending=False).head(top_k)
    params = er_model.parameters_frame().set_index("evidence")

    lines = [f"### Applicant {applicant_id if applicant_id is not None else ''}".rstrip(),
             f"- Most-believed risk rating: **{pred + rating_offset}** "
             f"(belief {P[pred]:.2f}); expected rating {P @ levels + rating_offset:.2f}",
             "- Belief across ratings: " + ", ".join(f"{k + rating_offset}: {p:.2f}" for k, p in enumerate(P)),
             "",
             "| Evidence group | observed / missing | mean weight | mean reliability | rating shift if removed |",
             "|---|---|---|---|---|"]
    for _, r in eff.iterrows():
        cols = [c for c in groups[r["group"]] if c in params.index]
        lines.append(f"| {r['group']} | {r['n_observed']} / {r['n_missing']} | "
                     f"{params.loc[cols, 'weight'].mean():.2f} | {params.loc[cols, 'reliability'].mean():.2f} | "
                     f"{r['expected_rating_shift_if_removed']:+.2f} |")
    lines += ["", "Missing inputs were treated as absence of evidence, not imputed. "
              "A positive shift means removing that group would raise the expected rating."]
    return "\n".join(lines)
