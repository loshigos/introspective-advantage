"""Checks the closed form and the information bound in the two-leaves model."""
import json, numpy as np
from fractions import Fraction
import exact_leaves as X

def gmd(w, eta):
    # E|eta1 - eta2| for independent draws from the atoms
    o = np.argsort(eta); e, p = eta[o], w[o]
    cw = np.cumsum(p); ce = np.cumsum(p * e)
    # sum_i p_i * sum_{j<i} p_j (e_i - e_j)
    s = np.sum(p[1:] * (e[1:] * cw[:-1] - ce[:-1]))
    return 2 * s

def run(**kw):
    from scipy.stats import binom, betabinom
    m, masses, a, b, min_n = 200, kw.get("masses", (0.015, 0.03, 0.06)), kw.get("a", 1.0), kw.get("b", 1.0), 3
    atoms = []
    for w in masses:
        for n in range(min_n, m + 1):
            pn = binom.pmf(n, m, w)
            if pn < 1e-12: continue
            for k in range(n + 1):
                yh = 1 if 2 * k >= n else 0; p1 = (k + a) / (n + a + b)
                atoms.append((pn * betabinom.pmf(k, n, a, b) / len(masses), w, k, n, yh, p1 if yh else 1 - p1))
    A = np.array([t[0] for t in atoms]); A /= A.sum(); eW = np.array([t[5] for t in atoms])
    keys = [(t[1], Fraction(t[2], t[3])) for t in atoms]; g = {}
    for i, k in enumerate(keys): g.setdefault(k, []).append(i)
    eB = np.empty(len(A))
    for idx in g.values():
        idx = np.array(idx); eB[idx] = (A[idx] * eW[idx]).sum() / A[idx].sum()
    pi = (A * eW).sum()
    closedW = 0.5 + gmd(A, eW) / (4 * pi * (1 - pi)); closedB = 0.5 + gmd(A, eB) / (4 * pi * (1 - pi))
    enumW = X.auroc_from_posterior(A, eW); enumB = X.auroc_from_posterior(A, eB)
    lower = 0.0
    for idx in g.values():
        idx = np.array(idx); pc = A[idx].sum()
        lower += pc ** 2 * gmd(A[idx] / pc, eW[idx])
    lower /= (4 * pi * (1 - pi))
    l1 = (A * np.abs(eW - eB)).sum()
    kl = (A * (eW * np.log(eW / eB) + (1 - eW) * np.log((1 - eW) / (1 - eB)))).sum()
    return dict(pi=pi, enumW=enumW, closedW=closedW, enumB=enumB, closedB=closedB, adv=enumW - enumB, lower=lower,
                bound_l1=l1 / (2 * pi * (1 - pi)), cmi_nats=kl, bound_info=np.sqrt(kl / 2) / (2 * pi * (1 - pi)))

out = {}
for name, kw in {"uniform, one mass": dict(masses=(0.03,)), "uniform": {}, "Beta(5,5)": dict(a=5, b=5), "Beta(0.5,0.5)": dict(a=0.5, b=0.5)}.items():
    r = run(**kw); out[name] = r; print(name, {k: round(v, 5) for k, v in r.items()})
json.dump(out, open("../results/results_exact_bounds.json", "w"), indent=1)
