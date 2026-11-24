"""Robustness checks for the tanh networks.

(1) Per-agent probe: for each test agent, a logistic probe is fitted on half of the
    agent's own evaluation inputs and scored on the other half (two-fold cross-fitting),
    once with behavioural features and once with behavioural features plus the full
    hidden layer and logit. This is the usual probing set-up, in which the monitor sees
    labelled examples from the agent it judges.
(2) Monitor seeds: the cross-agent monitors of the main protocol are refitted with three
    seeds to measure the variability that comes from the monitor learner itself.

Usage: python probe_control.py --out ../results_probe
"""
import argparse, json, os, warnings
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import HistGradientBoostingClassifier
import ia_experiments as E
warnings.filterwarnings("ignore")


def hidden(model, X):
    W1, W2 = model.coefs_; b1, b2 = model.intercepts_
    H = np.tanh(X @ W1 + b1); logit = (H @ W2 + b2).ravel()
    return np.column_stack([H, logit])


def agent(fam_cls, seed, n_train, n_eval):
    rng = np.random.default_rng(seed)
    task = fam_cls(rng)
    Xtr = task.sample_x(n_train, rng); ytr = task.label(Xtr, rng)
    Xev = task.sample_x(n_eval, rng); yev = task.label(Xev, rng)
    if len(np.unique(ytr)) < 2:
        return None
    m = E.make_model("mlp", seed).fit(Xtr, ytr)
    yhat = m.predict(Xev); Z = (yhat == yev).astype(int)
    if not (0.05 < Z.mean() < 0.98):
        return None
    p0, Q = E.query_features(m, Xev, rng, qmax=384)
    conf0 = np.column_stack([p0, np.abs(p0 - 0.5)])
    ps = E.p1(m, Xev)
    beh = np.column_stack([Xev, conf0, Q[384]])
    white_small = np.column_stack([Xev, np.column_stack([ps, np.abs(ps - 0.5), p0]), Q[384],
                                   E.white_box_features("mlp", m, Xev, Xtr)])
    white_full = np.column_stack([beh, hidden(m, Xev), ps])
    return dict(Z=Z, beh=beh, white_small=white_small, white_full=white_full)


def crossfit(X, Z, seed):
    r = np.random.default_rng(seed); idx = r.permutation(len(Z)); half = len(Z) // 2
    s = np.empty(len(Z))
    for a, b in [(idx[:half], idx[half:]), (idx[half:], idx[:half])]:
        if Z[a].min() == Z[a].max():
            s[b] = Z[a].mean(); continue
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)).fit(X[a], Z[a])
        s[b] = clf.predict_proba(X[b])[:, 1]
    return roc_auc_score(Z, s)


def boot(v, B=2000):
    v = np.asarray(v); v = v[~np.isnan(v)]; rng = np.random.default_rng(0)
    bs = rng.choice(v, (B, len(v))).mean(1)
    return float(v.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="../results_probe"); a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True); rows = []
    for fam in ["teacher-net", "teacher-tree", "digits-partition"]:
        n_train, n_eval = (200, 400) if fam != "digits-partition" else (300, 600)
        for ch in ["label", "score"]:
            E.CHANNEL = ch
            agents, s = [], 0
            while len(agents) < 100:
                r = agent(E.FAMILIES[fam], 50000 + 1000 * ["teacher-net", "teacher-tree", "digits-partition"].index(fam) + s, n_train, n_eval)
                s += 1
                if r is not None: agents.append(r)
            tr, te = agents[:60], agents[60:]
            # (1) per-agent cross-fitted probes
            pb = np.array([crossfit(t["beh"], t["Z"], 0) for t in te])
            pw = np.array([crossfit(t["white_full"], t["Z"], 0) for t in te])
            for name, v in [("probe_beh", pb), ("probe_white", pw), ("probe_adv", pw - pb)]:
                m_, lo, hi = boot(v); rows.append(dict(family=fam, channel=ch, level=name, auroc=m_, lo=lo, hi=hi))
            # (2) cross-agent monitors with three seeds
            advs = []
            for seed in range(3):
                res = {}
                for lv in ["beh", "white_small"]:
                    X = np.vstack([t[lv] for t in tr]); Zm = np.concatenate([t["Z"] for t in tr])
                    mon = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15,
                                                         l2_regularization=1.0, random_state=seed).fit(X, Zm)
                    res[lv] = np.array([roc_auc_score(t["Z"], mon.predict_proba(t[lv])[:, 1]) for t in te])
                advs.append(float(np.mean(res["white_small"] - res["beh"])))
            rows.append(dict(family=fam, channel=ch, level="crossagent_adv_seeds", auroc=float(np.mean(advs)),
                             sd=float(np.std(advs)), values=advs))
            print(fam, ch, "probe beh %.3f white %.3f adv %.3f | cross-agent adv %.4f sd %.4f" %
                  (pb.mean(), pw.mean(), (pw - pb).mean(), np.mean(advs), np.std(advs)), flush=True)
    json.dump(rows, open(os.path.join(a.out, "summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
