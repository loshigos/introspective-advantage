"""Exact unbounded introspective advantage in the two-leaves model (Example 1).

A leaf with probability mass w receives n ~ Binomial(m, w) training cases, k of
which are positive, with leaf rate theta ~ Beta(a, b). The agent answers
yhat = 1[k/n >= 1/2] and is correct on a fresh case from the leaf with
probability P(y = yhat | k, n) = (k+a)/(n+a+b) or its complement.

Accesses (all computed exactly by enumeration):
  white: (w, k, n)            -- reads the leaf statistics
  score: (w, k/n)             -- outside observer sees the leaf score and knows the cell mass
  score-blind: (k/n)          -- observer that cannot measure the cell mass
  label: (w, yhat)            -- observer sees only the label
Leaf masses w are drawn from a discrete set, so that the observer either can or
cannot use the mass to infer n.
"""
import itertools, json
import numpy as np
from fractions import Fraction


def auroc_from_posterior(weights, eta):
    """Exact AUROC_2 of the score eta over a finite set of atoms with probability
    weights; each atom contributes weight*eta correct and weight*(1-eta) errors."""
    order = np.argsort(eta)
    eta, w = eta[order], weights[order]
    pos, neg = w * eta, w * (1 - eta)
    P, N = pos.sum(), neg.sum()
    # group ties
    auc, cum_neg = 0.0, 0.0
    i = 0
    while i < len(eta):
        j = i
        while j + 1 < len(eta) and np.isclose(eta[j + 1], eta[i]):
            j += 1
        p_blk, n_blk = pos[i:j + 1].sum(), neg[i:j + 1].sum()
        auc += p_blk * cum_neg + 0.5 * p_blk * n_blk
        cum_neg += n_blk
        i = j + 1
    return auc / (P * N)


def run(m=200, masses=(0.015, 0.03, 0.06), a=1.0, b=1.0, min_n=3):
    atoms = []   # (prob, w, k, n, eta_white)
    from scipy.stats import binom, betabinom
    for w in masses:
        pw = 1.0 / len(masses)
        for n in range(min_n, m + 1):
            pn = binom.pmf(n, m, w)
            if pn < 1e-12:
                continue
            for k in range(n + 1):
                pk = betabinom.pmf(k, n, a, b)
                yhat = 1 if 2 * k >= n else 0
                p1 = (k + a) / (n + a + b)
                eta = p1 if yhat == 1 else 1 - p1
                atoms.append((pw * pn * pk, w, k, n, yhat, eta))
    A = np.array([x[0] for x in atoms]); A /= A.sum()
    eta_w = np.array([x[5] for x in atoms])

    def coarsen(keyf):
        keys = [keyf(x) for x in atoms]
        groups = {}
        for i, kk in enumerate(keys):
            groups.setdefault(kk, []).append(i)
        eta = np.empty(len(atoms))
        for idx in groups.values():
            idx = np.array(idx)
            eta[idx] = (A[idx] * eta_w[idx]).sum() / A[idx].sum()
        return eta

    res = {
        "white": auroc_from_posterior(A, eta_w),
        "score": auroc_from_posterior(A, coarsen(lambda x: (x[1], Fraction(x[2], x[3])))),
        "score_blind": auroc_from_posterior(A, coarsen(lambda x: Fraction(x[2], x[3]))),
        "label": auroc_from_posterior(A, coarsen(lambda x: (x[1], x[4]))),
        "public": 0.5,
    }
    return res


if __name__ == "__main__":
    out = {}
    for name, kw in {
        "one mass, uniform prior": dict(masses=(0.03,)),
        "three masses, uniform prior": dict(masses=(0.015, 0.03, 0.06)),
        "three masses, informative prior Beta(5,5)": dict(masses=(0.015, 0.03, 0.06), a=5, b=5),
        "three masses, skewed prior Beta(0.5,0.5)": dict(masses=(0.015, 0.03, 0.06), a=0.5, b=0.5),
    }.items():
        r = run(**kw)
        r["adv_score"] = r["white"] - r["score"]
        r["adv_score_blind"] = r["white"] - r["score_blind"]
        r["adv_label"] = r["white"] - r["label"]
        out[name] = r
        print(name, {k: round(v, 4) for k, v in r.items()})
    json.dump(out, open("../results/results_exact.json", "w"), indent=1)
