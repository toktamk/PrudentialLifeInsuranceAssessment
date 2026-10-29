"""Evidential Reasoning (ER) rule classifier with data-driven evidence.

What this module implements
---------------------------
1. Evidence acquisition from data (in the spirit of MAKER, Yang & Xu 2025).
   Every input variable is discretised into referential bins on the training set.
   For a bin k of variable j, the class-conditional likelihoods L(k | theta) are
   estimated with Laplace smoothing and normalised across classes, giving a
   belief distribution p_j(theta | k). A missing value gives no evidence
   (vacuous evidence); it is never imputed.

2. The ER rule (Yang & Xu 2013, Artificial Intelligence 205, 1-29) for combining
   evidence that carries a weight w_j and a reliability r_j. For singleton
   hypotheses plus the frame of discernment Theta, the recursive combination is

       m_hat(theta) = (1 - r_i) m(theta) + m_P a_i(theta)
                      + m(theta) a_i(theta) + m(theta) a_i(Theta) + m(Theta) a_i(theta)
       m_hat(Theta) = (1 - r_i) m(Theta) + m_P a_i(Theta) + m(Theta) a_i(Theta)
       m_hat(P)     = (1 - r_i) m_P

   where a_i(.) = w_i p_i(.) and m_P is the residual support on the power set.
   Each step is normalised by the non-conflicting mass, and the final belief is
   p(theta) = m(theta) / (sum_theta m(theta) + m(Theta)).

   Special cases, verified in tests/test_er_engine.py:
     * r_i = 1 for all i                      -> Dempster's rule
     * r_i = w_i = 1, likelihood evidence,
       class prior as one extra evidence      -> Bayes' rule (naive Bayes posterior)
     * w_i = r_i = 0                          -> evidence has no effect

3. Weights and reliabilities learned from data by maximum likelihood
   (minimising the negative log-likelihood of the observed class with Adam,
   early-stopped on a validation set).

What it does NOT implement (state these as limitations in any write-up)
-----------------------------------------------------------------------
* MAKER's joint evidence for interdependent inputs: each variable is a separate
  piece of evidence, so inputs are treated as conditionally independent, as in
  naive Bayes. Learned weights and reliabilities only partly compensate.
* MAKER's bi-objective optimisation and optimisation of referential values:
  bins are fixed quantile or category bins and the objective is the likelihood only.
* Belief on subsets other than singletons and Theta.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

DTYPE = torch.float64


# ---------------------------------------------------------------------------
# Core ER rule (batched, differentiable)
# ---------------------------------------------------------------------------
def er_combine(a: torch.Tensor, a_frame: torch.Tensor, b: torch.Tensor):
    """Combine L pieces of evidence for a batch of B cases with the ER rule.

    Parameters
    ----------
    a       : (B, L, N) weighted singleton beliefs w_i * p_i(theta); 0 for missing evidence
    a_frame : (B, L)    weighted belief on the frame w_i * p_i(Theta); 0 for missing evidence
    b       : (B, L)    1 - r_i; set to 1 for missing evidence

    Returns
    -------
    p       : (B, N) combined belief degrees on the singleton classes
    p_frame : (B,)   combined belief on Theta (global ignorance)
    """
    B, L, N = a.shape
    m = torch.zeros(B, N, dtype=a.dtype, device=a.device)
    m_f = torch.zeros(B, dtype=a.dtype, device=a.device)
    m_p = torch.ones(B, dtype=a.dtype, device=a.device)  # start from vacuous evidence
    for i in range(L):
        ai, afi, bi = a[:, i, :], a_frame[:, i], b[:, i]
        m_new = (bi[:, None] * m + m_p[:, None] * ai + m * ai
                 + m * afi[:, None] + m_f[:, None] * ai)
        m_f_new = bi * m_f + m_p * afi + m_f * afi
        m_p_new = m_p * bi
        z = m_new.sum(dim=1) + m_f_new + m_p_new
        z = torch.clamp(z, min=1e-300)
        m, m_f, m_p = m_new / z[:, None], m_f_new / z, m_p_new / z
    total = m.sum(dim=1) + m_f
    no_evidence = total <= 1e-300
    safe_total = torch.where(no_evidence, torch.ones_like(total), total)
    p = m / safe_total[:, None]
    p_frame = m_f / safe_total
    # With no evidence at all, return total ignorance.
    p = torch.where(no_evidence[:, None], torch.zeros_like(p), p)
    p_frame = torch.where(no_evidence, torch.ones_like(p_frame), p_frame)
    return p, p_frame


def er_rule(beliefs, weights, reliabilities, frame_beliefs=None):
    """Combine L pieces of evidence for ONE case (numpy in, numpy out).

    beliefs       : (L, N) singleton belief degrees, rows sum to <= 1
    weights       : (L,)
    reliabilities : (L,)
    frame_beliefs : (L,) belief on Theta; default 1 - row sums of `beliefs`
    """
    beliefs = np.asarray(beliefs, dtype=float)
    w = np.asarray(weights, dtype=float)
    r = np.asarray(reliabilities, dtype=float)
    if frame_beliefs is None:
        frame_beliefs = np.clip(1.0 - beliefs.sum(axis=1), 0.0, 1.0)
    a = torch.tensor((w[:, None] * beliefs)[None], dtype=DTYPE)
    af = torch.tensor((w * np.asarray(frame_beliefs, dtype=float))[None], dtype=DTYPE)
    b = torch.tensor((1.0 - r)[None], dtype=DTYPE)
    p, pf = er_combine(a, af, b)
    return p[0].numpy(), float(pf[0])


# ---------------------------------------------------------------------------
# Discretisation helpers
# ---------------------------------------------------------------------------
def _make_edges(values: np.ndarray, max_bins: int) -> np.ndarray:
    """Bin edges: one bin per category for low-cardinality variables, else quantiles."""
    uniq = np.unique(values)
    if len(uniq) <= max_bins:
        return (uniq[:-1] + uniq[1:]) / 2.0
    qs = np.quantile(values, np.linspace(0, 1, max_bins + 1)[1:-1])
    return np.unique(qs)


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------
class EvidentialReasoningClassifier:
    """ER-rule classifier with likelihood-based evidence and learned weights/reliabilities.

    Labels must be integers 0..n_classes-1.

    learn_params=False fixes every weight and reliability at 1. With
    use_prior=True that model is exactly a naive Bayes classifier on the binned
    inputs (the Bayes special case), which makes a fair reference model.
    """

    PRIOR_NAME = "__class_prior__"

    def __init__(self, n_classes=8, max_bins=10, alpha=1.0, use_prior=True,
                 learn_params=True, init_weight=0.5, init_reliability=0.5,
                 lr=0.05, max_epochs=60, batch_size=4096, patience=8,
                 random_state=42, verbose=True):
        self.n_classes = n_classes
        self.max_bins = max_bins
        self.alpha = alpha
        self.use_prior = use_prior
        self.learn_params = learn_params
        self.init_weight = init_weight
        self.init_reliability = init_reliability
        self.lr = lr
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.patience = patience
        self.random_state = random_state
        self.verbose = verbose

    # ---------------- evidence acquisition ----------------
    def _fit_evidence(self, X: pd.DataFrame, y: np.ndarray):
        N = self.n_classes
        self.feature_names_ = []
        self.edges_ = []
        tables = []
        for col in X.columns:
            v = X[col].to_numpy(dtype=float)
            obs = ~np.isnan(v)
            if obs.sum() == 0:
                continue  # never observed in training: carries no evidence
            edges = _make_edges(v[obs], self.max_bins)
            codes = np.searchsorted(edges, v[obs], side="right")
            K = len(edges) + 1
            counts = np.zeros((K, N))
            np.add.at(counts, (codes, y[obs]), 1.0)
            # likelihood of bin k given class theta, Laplace-smoothed
            lik = (counts + self.alpha) / (counts.sum(axis=0, keepdims=True) + K * self.alpha)
            belief = lik / lik.sum(axis=1, keepdims=True)  # normalise across classes
            self.feature_names_.append(col)
            self.edges_.append(edges)
            tables.append(belief)
        Kmax = max(t.shape[0] for t in tables)
        T = np.full((len(tables), Kmax, N), 1.0 / N)
        for j, t in enumerate(tables):
            T[j, : t.shape[0]] = t
        self.tables_ = torch.tensor(T, dtype=DTYPE)  # (L, Kmax, N)
        prior = np.bincount(y, minlength=N).astype(float) + self.alpha
        self.class_prior_ = prior / prior.sum()

    def _codes(self, X: pd.DataFrame) -> np.ndarray:
        """Bin codes (n, L); -1 marks a missing value (no evidence)."""
        missing_cols = [c for c in self.feature_names_ if c not in X.columns]
        if missing_cols:
            raise ValueError(f"Columns missing from X: {missing_cols[:5]}")
        out = np.full((len(X), len(self.feature_names_)), -1, dtype=np.int64)
        for j, (col, edges) in enumerate(zip(self.feature_names_, self.edges_)):
            v = X[col].to_numpy(dtype=float)
            obs = ~np.isnan(v)
            out[obs, j] = np.searchsorted(edges, v[obs], side="right")
        return out

    def _evidence_tensors(self, codes: np.ndarray):
        c = torch.as_tensor(codes)
        obs = (c >= 0).to(DTYPE)
        L = c.shape[1]
        beliefs = self.tables_[torch.arange(L)[None, :], c.clamp(min=0)]  # (B, L, N)
        if self.use_prior:
            prior = torch.tensor(self.class_prior_, dtype=DTYPE)
            beliefs = torch.cat([prior.expand(c.shape[0], 1, -1), beliefs], dim=1)
            obs = torch.cat([torch.ones(c.shape[0], 1, dtype=DTYPE), obs], dim=1)
        return beliefs, obs

    # ---------------- parameters ----------------
    def _params(self):
        if not self.learn_params:
            n = self.n_evidence_
            return torch.ones(n, dtype=DTYPE), torch.ones(n, dtype=DTYPE)
        return torch.sigmoid(self.w_logit_), torch.sigmoid(self.r_logit_)

    def _forward(self, codes: np.ndarray, extra_mask: np.ndarray | None = None):
        beliefs, obs = self._evidence_tensors(codes)
        if extra_mask is not None:  # extra_mask: (B, L_total) 1 keeps, 0 removes evidence
            obs = obs * torch.as_tensor(extra_mask, dtype=DTYPE)
        w, r = self._params()
        a = w[None, :, None] * beliefs * obs[:, :, None]
        a_frame = torch.zeros_like(obs)  # likelihood evidence is complete
        b = 1.0 - r[None, :] * obs
        return er_combine(a, a_frame, b)

    @staticmethod
    def _nll(p, y):
        return -torch.log(p[torch.arange(len(y)), y].clamp(min=1e-12)).mean()

    # ---------------- public API ----------------
    def fit(self, X: pd.DataFrame, y, X_val: pd.DataFrame | None = None, y_val=None):
        y = np.asarray(y, dtype=np.int64)
        if y.min() < 0 or y.max() >= self.n_classes:
            raise ValueError("Labels must be integers 0..n_classes-1")
        rng = np.random.default_rng(self.random_state)
        torch.manual_seed(self.random_state)
        self._fit_evidence(X, y)
        self.n_evidence_ = len(self.feature_names_) + int(self.use_prior)
        codes = self._codes(X)

        if not self.learn_params:
            self.history_ = []
            return self

        if X_val is None:  # hold out 10% of the training data for early stopping
            idx = rng.permutation(len(y))
            cut = int(0.9 * len(y))
            tr, va = idx[:cut], idx[cut:]
            codes_tr, y_tr, codes_va, y_va = codes[tr], y[tr], codes[va], y[va]
        else:
            codes_tr, y_tr = codes, y
            codes_va, y_va = self._codes(X_val), np.asarray(y_val, dtype=np.int64)

        def logit(x):
            return float(np.log(x / (1 - x)))

        self.w_logit_ = torch.full((self.n_evidence_,), logit(self.init_weight), dtype=DTYPE, requires_grad=True)
        self.r_logit_ = torch.full((self.n_evidence_,), logit(self.init_reliability), dtype=DTYPE, requires_grad=True)
        opt = torch.optim.Adam([self.w_logit_, self.r_logit_], lr=self.lr)
        yt_va = torch.as_tensor(y_va)

        best, best_state, bad = np.inf, None, 0
        self.history_ = []
        for epoch in range(self.max_epochs):
            perm = rng.permutation(len(y_tr))
            train_loss = 0.0
            for s in range(0, len(perm), self.batch_size):
                bidx = perm[s: s + self.batch_size]
                opt.zero_grad()
                p, _ = self._forward(codes_tr[bidx])
                loss = self._nll(p, torch.as_tensor(y_tr[bidx]))
                loss.backward()
                opt.step()
                train_loss += loss.item() * len(bidx)
            train_loss /= len(perm)
            with torch.no_grad():
                p_va = self._predict_codes(codes_va)
                val_loss = self._nll(p_va, yt_va).item()
            self.history_.append({"epoch": epoch + 1, "train_nll": train_loss, "val_nll": val_loss})
            if self.verbose:
                print(f"epoch {epoch + 1:3d}  train NLL {train_loss:.4f}  val NLL {val_loss:.4f}")
            if val_loss < best - 1e-5:
                best, bad = val_loss, 0
                best_state = (self.w_logit_.detach().clone(), self.r_logit_.detach().clone())
            else:
                bad += 1
                if bad >= self.patience:
                    break
        if best_state is not None:
            with torch.no_grad():
                self.w_logit_.copy_(best_state[0])
                self.r_logit_.copy_(best_state[1])
        return self

    def _predict_codes(self, codes, extra_mask=None, batch=8192):
        outs = []
        with torch.no_grad():
            for s in range(0, len(codes), batch):
                em = None if extra_mask is None else extra_mask[s: s + batch]
                p, _ = self._forward(codes[s: s + batch], em)
                # cases with no evidence at all fall back to uniform beliefs
                empty = p.sum(dim=1) <= 0
                p[empty] = 1.0 / self.n_classes
                outs.append(p)
        return torch.cat(outs)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self._predict_codes(self._codes(X)).numpy()

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Most-believed class (0-based)."""
        return self.predict_proba(X).argmax(axis=1)

    @property
    def evidence_names_(self):
        return ([self.PRIOR_NAME] if self.use_prior else []) + list(self.feature_names_)

    def parameters_frame(self) -> pd.DataFrame:
        w, r = self._params()
        return pd.DataFrame({"evidence": self.evidence_names_,
                             "weight": w.detach().numpy(),
                             "reliability": r.detach().numpy()})

    def removal_effect(self, X: pd.DataFrame, groups: dict[str, list[str]]) -> pd.DataFrame:
        """Effect of removing each group of evidence, for every row of X.

        Returns a long DataFrame with the change in expected risk rating and in the
        belief of the originally predicted class when the group's evidence is removed.
        """
        codes = self._codes(X)
        base = self._predict_codes(codes).numpy()
        levels = np.arange(self.n_classes)
        base_exp = base @ levels
        pred = base.argmax(axis=1)
        offset = int(self.use_prior)
        col_index = {c: j + offset for j, c in enumerate(self.feature_names_)}
        rows = []
        for g, cols in groups.items():
            idx = [col_index[c] for c in cols if c in col_index]
            if not idx:
                continue
            mask = np.ones((len(X), self.n_evidence_))
            mask[:, idx] = 0.0
            p = self._predict_codes(codes, extra_mask=mask).numpy()
            n_obs = (codes[:, [i - offset for i in idx]] >= 0).sum(axis=1)
            for k in range(len(X)):
                rows.append({"row": k, "group": g, "n_observed": int(n_obs[k]),
                             "n_missing": int(len(idx) - n_obs[k]),
                             "expected_rating_shift_if_removed": float(p[k] @ levels - base_exp[k]),
                             "belief_in_predicted_class_shift_if_removed": float(p[k, pred[k]] - base[k, pred[k]])})
        return pd.DataFrame(rows)
