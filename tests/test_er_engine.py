"""Real tests of the ER rule implementation. Run with:  python -m pytest -q"""
import itertools
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from er_engine import EvidentialReasoningClassifier, er_rule  # noqa: E402

RNG = np.random.default_rng(0)


def random_mass(n, frame_share=0.3):
    """Random mass function on singletons {0..n-1} plus the frame Theta."""
    m = RNG.random(n)
    m = (1 - frame_share) * m / m.sum()
    return m, frame_share


def dempster_reference(masses, n):
    """Independent implementation of Dempster's rule over explicit frozensets."""
    theta = frozenset(range(n))

    def as_dict(m, mf):
        d = {frozenset([k]): v for k, v in enumerate(m)}
        d[theta] = d.get(theta, 0.0) + mf
        return d

    comb = as_dict(*masses[0])
    for m, mf in masses[1:]:
        nxt, conflict = {}, 0.0
        for (A, va), (B, vb) in itertools.product(comb.items(), as_dict(m, mf).items()):
            C = A & B
            if C:
                nxt[C] = nxt.get(C, 0.0) + va * vb
            else:
                conflict += va * vb
        comb = {k: v / (1 - conflict) for k, v in nxt.items()}
    return np.array([comb.get(frozenset([k]), 0.0) for k in range(n)]), comb.get(theta, 0.0)


def test_reduces_to_dempster_when_fully_reliable():
    n, L = 5, 4
    masses = [random_mass(n, RNG.uniform(0, 0.5)) for _ in range(L)]
    beliefs = np.array([m for m, _ in masses])
    frames = np.array([f for _, f in masses])
    p, pf = er_rule(beliefs, weights=np.ones(L), reliabilities=np.ones(L), frame_beliefs=frames)
    ref_p, ref_f = dempster_reference(masses, n)
    np.testing.assert_allclose(p, ref_p, atol=1e-10)
    assert abs(pf - ref_f) < 1e-10


def test_weights_do_not_matter_when_fully_reliable():
    n, L = 4, 3
    beliefs = RNG.dirichlet(np.ones(n), size=L)
    p1, _ = er_rule(beliefs, np.ones(L), np.ones(L))
    p2, _ = er_rule(beliefs, RNG.uniform(0.1, 1, L), np.ones(L))
    np.testing.assert_allclose(p1, p2, atol=1e-10)


def test_reduces_to_bayes_with_likelihood_evidence():
    n, L = 6, 5
    prior = RNG.dirichlet(np.ones(n))
    lik = RNG.random((L, n))                        # L(x_j | theta), unnormalised
    evidence = lik / lik.sum(axis=1, keepdims=True)  # normalised likelihoods = MAKER-style evidence
    beliefs = np.vstack([prior, evidence])
    p, pf = er_rule(beliefs, np.ones(L + 1), np.ones(L + 1))
    posterior = prior * lik.prod(axis=0)
    posterior /= posterior.sum()
    np.testing.assert_allclose(p, posterior, atol=1e-10)
    assert pf < 1e-12


def test_unweighted_unreliable_evidence_has_no_effect():
    n = 5
    beliefs = RNG.dirichlet(np.ones(n), size=3)
    w = np.array([0.7, 0.4, 0.9])
    r = np.array([0.6, 0.8, 0.5])
    p_base, _ = er_rule(beliefs, w, r)
    extra = np.vstack([beliefs, RNG.dirichlet(np.ones(n))])
    p_extra, _ = er_rule(extra, np.append(w, 0.0), np.append(r, 0.0))
    np.testing.assert_allclose(p_base, p_extra, atol=1e-12)


def test_order_invariance():
    n, L = 6, 5
    beliefs = RNG.dirichlet(np.ones(n), size=L)
    w, r = RNG.uniform(0.1, 1, L), RNG.uniform(0.1, 1, L)
    p1, _ = er_rule(beliefs, w, r)
    perm = RNG.permutation(L)
    p2, _ = er_rule(beliefs[perm], w[perm], r[perm])
    np.testing.assert_allclose(p1, p2, atol=1e-10)


def test_output_is_a_distribution():
    n, L = 8, 10
    beliefs = RNG.dirichlet(np.ones(n), size=L)
    p, pf = er_rule(beliefs, RNG.uniform(0, 1, L), RNG.uniform(0, 1, L))
    assert np.all(p >= 0)
    assert abs(p.sum() + pf - 1) < 1e-10


def _toy_data(n=3000, seed=1):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 4, n)
    X = pd.DataFrame({
        "a": y + rng.normal(0, 0.8, n),
        "b": (y >= 2).astype(float) + rng.normal(0, 0.5, n),
        "c": rng.normal(0, 1, n),  # pure noise
    })
    X.loc[rng.random(n) < 0.2, "a"] = np.nan
    return X, y


def test_missing_value_equals_removing_the_evidence():
    X, y = _toy_data()
    m = EvidentialReasoningClassifier(n_classes=4, learn_params=False, verbose=False).fit(X, y)
    row = X.iloc[[0]].copy()
    row["a"] = np.nan
    p_missing = m.predict_proba(row)
    mask = np.ones((1, m.n_evidence_))
    mask[0, 1 + m.feature_names_.index("a")] = 0.0  # +1 for the prior evidence
    p_removed = m._predict_codes(m._codes(X.iloc[[0]]), extra_mask=mask).numpy()
    np.testing.assert_allclose(p_missing, p_removed, atol=1e-12)


def test_fixed_parameters_equal_binned_naive_bayes():
    X, y = _toy_data()
    m = EvidentialReasoningClassifier(n_classes=4, learn_params=False, max_bins=5, alpha=1.0,
                                      verbose=False).fit(X, y)
    codes = m._codes(X.iloc[:50])
    T = m.tables_.numpy()
    for i in range(50):
        post = m.class_prior_.copy()
        for j, k in enumerate(codes[i]):
            if k >= 0:
                post = post * T[j, k]
        post /= post.sum()
        np.testing.assert_allclose(m.predict_proba(X.iloc[[i]])[0], post, atol=1e-10)


def test_learning_beats_chance_and_improves_likelihood():
    X, y = _toy_data(n=4000)
    Xv, yv = _toy_data(n=2000, seed=2)
    learned = EvidentialReasoningClassifier(n_classes=4, max_epochs=40, batch_size=1024,
                                            verbose=False).fit(X, y, Xv, yv)
    start = EvidentialReasoningClassifier(n_classes=4, max_epochs=0, verbose=False)
    acc = np.mean(learned.predict(Xv) == yv)
    assert acc > 0.45  # chance is 0.25

    def nll(model):
        P = model.predict_proba(Xv)
        return -np.mean(np.log(np.clip(P[np.arange(len(yv)), yv], 1e-12, 1)))

    # learned weights/reliabilities must fit validation data better than the initial values
    start.fit(X, y, Xv, yv)
    assert nll(learned) < nll(start)
