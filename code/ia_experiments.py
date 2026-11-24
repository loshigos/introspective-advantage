"""Introspective advantage: extensional vs intensional self-knowledge.

Runs on CPU in minutes. For each task family and object-model family, it trains
one object model per task, computes monitor features at five access levels, trains
one monitor per level across meta-train tasks and evaluates type-2 AUROC on
held-out meta-test tasks.

Usage: python ia_experiments.py --out results/ [--quick]
"""
import argparse, json, os, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

Q_LEVELS = [0, 6, 24, 96, 384]
RADII = [0.05, 0.25, 1.0]          # multiplied by sqrt(d)


# ---------------------------------------------------------------- task families
class TeacherNet:
    name = "teacher-net"
    d = 6
    def __init__(self, rng):
        self.W1 = rng.normal(size=(self.d, 8)) * 1.2
        self.b1 = rng.normal(size=8) * 0.5
        self.w2 = rng.normal(size=8)
        self.b2 = 0.0
    def sample_x(self, n, rng):
        return rng.normal(size=(n, self.d))
    def label(self, X, rng, noise=0.10):
        s = np.tanh(X @ self.W1 + self.b1) @ self.w2
        y = (s > np.median(s)).astype(int) if False else (s > 0).astype(int)
        flip = rng.random(len(y)) < noise
        return np.where(flip, 1 - y, y)


class TeacherTree:
    name = "teacher-tree"
    d = 6
    def __init__(self, rng, depth=4):
        self.depth = depth
        n_int = 2 ** depth - 1
        self.feat = rng.integers(0, self.d, size=n_int)
        self.thr = rng.normal(size=n_int) * 0.7
        self.leaf = rng.integers(0, 2, size=2 ** depth)
    def sample_x(self, n, rng):
        return rng.normal(size=(n, self.d))
    def label(self, X, rng, noise=0.10):
        node = np.zeros(len(X), dtype=int)
        for _ in range(self.depth):
            go_right = X[np.arange(len(X)), self.feat[node]] > self.thr[node]
            node = 2 * node + 1 + go_right
        y = self.leaf[node - (2 ** self.depth - 1)]
        flip = rng.random(len(y)) < noise
        return np.where(flip, 1 - y, y)


_DIG = None
def _digits():
    global _DIG
    if _DIG is None:
        X, c = load_digits(return_X_y=True)
        X = (X - X.mean(0)) / (X.std(0) + 1e-6)
        _DIG = (X, c)
    return _DIG


class DigitsPartition:
    """Real inputs (8x8 handwritten digits); each task is a random balanced
    partition of the ten digit classes into two labels."""
    name = "digits-partition"
    d = 64
    def __init__(self, rng):
        X, c = _digits()
        pos = rng.choice(10, size=5, replace=False)
        self.map = np.isin(np.arange(10), pos).astype(int)
        self.perm = rng.permutation(len(X))
        self.ptr = 0
    def sample_x(self, n, rng):
        # draw without replacement from this task's permutation of the pool
        X, c = _digits()
        idx = self.perm[self.ptr:self.ptr + n]
        self.ptr += n
        self._last = idx
        return X[idx]
    def label(self, X, rng, noise=0.0):
        _, c = _digits()
        return self.map[c[self._last]]


FAMILIES = {"teacher-net": TeacherNet, "teacher-tree": TeacherTree,
            "digits-partition": DigitsPartition}


# ---------------------------------------------------------------- object models
def make_model(kind, seed):
    if kind == "mlp":
        return MLPClassifier(hidden_layer_sizes=(32,), activation="tanh", alpha=1e-3,
                             max_iter=800, random_state=seed)
    if kind == "tree":
        return DecisionTreeClassifier(min_samples_leaf=3, random_state=seed)
    if kind == "forest":
        return RandomForestClassifier(n_estimators=50, min_samples_leaf=3,
                                      random_state=seed, n_jobs=1)
    if kind == "knn":
        return KNeighborsClassifier(n_neighbors=15)
    raise ValueError(kind)


def p1(model, X):
    P = model.predict_proba(X)
    return P[:, 1] if P.shape[1] == 2 else np.full(len(X), float(model.classes_[0]))


# ---------------------------------------------------------------- features
CHANNEL = "score"

def out(model, X):
    """What a black-box observer receives per query: the score or only the label."""
    return p1(model, X) if CHANNEL == "score" else model.predict(X).astype(float)


def query_features(model, X, rng, qmax=384):
    """Black-box features from score queries at three radii; nested prefixes give
    the features for every budget q <= qmax."""
    n, d = X.shape
    m = qmax // len(RADII)
    U = rng.normal(size=(m, d)); U /= np.linalg.norm(U, axis=1, keepdims=True)
    p0 = out(model, X)
    res = {}
    per_r = []
    for r in RADII:
        Xq = (X[:, None, :] + r * np.sqrt(d) * U[None, :, :]).reshape(-1, d)
        per_r.append(out(model, Xq).reshape(n, m))
    for q in Q_LEVELS:
        if q == 0:
            res[q] = np.zeros((n, 0)); continue
        k = q // len(RADII)
        cols = []
        for P in per_r:
            D = P[:, :k] - p0[:, None]
            flips = ((P[:, :k] > 0.5) != (p0[:, None] > 0.5)).mean(1)
            cols += [np.abs(D).mean(1), D.std(1), D.mean(1), flips]
        res[q] = np.column_stack(cols)
    return p0, res


def white_box_features(kind, model, X, Xtr, counts_override=None):
    if kind == "mlp":
        W1, W2 = model.coefs_; b1, b2 = model.intercepts_
        H = np.tanh(X @ W1 + b1)
        logit = (H @ W2 + b2).ravel()
        # analytic input gradient of the logit
        G = ((1 - H ** 2) * W2.ravel()[None, :]) @ W1.T
        return np.column_stack([np.abs(logit), np.linalg.norm(H, axis=1),
                                np.abs(H).mean(1), np.linalg.norm(G, axis=1),
                                np.abs(logit) / (np.linalg.norm(G, axis=1) + 1e-9)])
    if kind == "tree":
        return _tree_feats(model, X, counts_override)
    if kind == "forest":
        F = [_tree_feats(est, X, None if counts_override is None else counts_override[i])
             for i, est in enumerate(model.estimators_)]
        F = np.stack(F, 0)                      # trees x n x f
        probs = np.stack([est.predict_proba(X)[:, -1] for est in model.estimators_], 0)
        return np.column_stack([F[:, :, 0].mean(0), F[:, :, 0].min(0),
                                np.log1p(F[:, :, 0]).std(0), F[:, :, 1].mean(0),
                                probs.std(0)])
    if kind == "knn":
        # The representation of a nearest-neighbour classifier is its training set,
        # so a white-box monitor can read the stored points and their labels.
        dist, idx = model.kneighbors(X, n_neighbors=15)
        lab = model.classes_[model._y[idx]]
        yhat = model.predict(X)
        agree = (lab == yhat[:, None])
        w = 1.0 / (dist + 1e-6)
        return np.column_stack([dist.mean(1), dist[:, 0], dist[:, -1], dist.std(1),
                                agree[:, :5].mean(1), agree[:, 0],
                                (agree * w).sum(1) / w.sum(1)])
    raise ValueError(kind)


def _tree_feats(est, X, counts=None):
    t = est.tree_
    leaf = est.apply(X)
    n = t.n_node_samples if counts is None else counts
    depth = np.zeros(t.node_count)
    stack = [(0, 0)]
    parent = np.zeros(t.node_count, dtype=int)
    while stack:
        node, dpt = stack.pop(); depth[node] = dpt
        for ch in (t.children_left[node], t.children_right[node]):
            if ch != -1:
                parent[ch] = node; stack.append((ch, dpt + 1))
    return np.column_stack([n[leaf], depth[leaf], n[parent[leaf]]])


def trace_features(X, Xtr, ytr, yhat):
    nn = NearestNeighbors(n_neighbors=15).fit(Xtr)
    dist, idx = nn.kneighbors(X)
    agree15 = (ytr[idx] == yhat[:, None]).mean(1)
    agree5 = (ytr[idx[:, :5]] == yhat[:, None]).mean(1)
    return np.column_stack([dist[:, 0], dist[:, 4], dist.mean(1), agree5, agree15,
                            ytr[idx[:, 0]] == yhat])


def unlabeled_counts(est, Xu):
    """Leaf counts of an unlabeled sample: same function, label-free statistics."""
    leaf_all = est.decision_path(Xu)
    return np.asarray(leaf_all.sum(0)).ravel().astype(float) * (est.tree_.n_node_samples[0] / len(Xu))


# ---------------------------------------------------------------- one task
def run_task(fam_cls, kind, seed, n_train, n_eval, quick=False):
    rng = np.random.default_rng(seed)
    task = fam_cls(rng)
    Xtr = task.sample_x(n_train, rng); ytr = task.label(Xtr, rng)
    Xev = task.sample_x(n_eval, rng); yev = task.label(Xev, rng)
    Xu = task.sample_x(n_train if fam_cls is DigitsPartition else 2000, rng)
    if len(np.unique(ytr)) < 2:
        return None
    model = make_model(kind, seed).fit(Xtr, ytr)
    yhat = model.predict(Xev)
    Z = (yhat == yev).astype(int)
    p0, Q = query_features(model, Xev, rng, qmax=96 if quick else 384)
    conf0 = np.column_stack([p0, np.abs(p0 - 0.5)])
    ps = p1(model, Xev)
    conf_white = np.column_stack([ps, np.abs(ps - 0.5), p0])
    wb = white_box_features(kind, model, Xev, Xtr)
    feats = {"public": Xev}
    for q in Q_LEVELS:
        if q in Q:
            feats[f"beh_q{q}"] = np.column_stack([Xev, conf0, Q[q]])
    qmax = max(Q)
    feats["white"] = np.column_stack([Xev, conf_white, Q[qmax], wb])
    feats["trace"] = np.column_stack([feats["white"], trace_features(Xev, Xtr, ytr, yhat)])
    if kind in ("tree", "forest"):
        if kind == "tree":
            cu = unlabeled_counts(model, Xu)
        else:
            cu = [unlabeled_counts(est, Xu) for est in model.estimators_]
        wb_twin = white_box_features(kind, model, Xev, Xtr, counts_override=cu)
        feats["twin"] = np.column_stack([Xev, conf_white, Q[qmax], wb_twin])
    return {"Z": Z, "feats": feats, "acc": Z.mean()}


# ---------------------------------------------------------------- evaluation
def auroc2(Z, s):
    if Z.min() == Z.max():
        return np.nan
    return roc_auc_score(Z, s)


def evaluate(tasks_train, tasks_test, level, seed=0):
    Xm = np.vstack([t["feats"][level] for t in tasks_train])
    Zm = np.concatenate([t["Z"] for t in tasks_train])
    mon = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05,
                                         max_leaf_nodes=15, l2_regularization=1.0,
                                         random_state=seed).fit(Xm, Zm)
    return np.array([auroc2(t["Z"], mon.predict_proba(t["feats"][level])[:, 1])
                     for t in tasks_test])


def boot_ci(v, B=2000, rng=None):
    rng = rng or np.random.default_rng(0)
    v = v[~np.isnan(v)]
    bs = rng.choice(v, size=(B, len(v)), replace=True).mean(1)
    return float(v.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--families", default="teacher-net,teacher-tree,digits-partition")
    ap.add_argument("--models", default="mlp,tree,forest,knn")
    ap.add_argument("--channel", default="score", choices=["score", "label"])
    a = ap.parse_args()
    global CHANNEL
    CHANNEL = a.channel
    os.makedirs(a.out, exist_ok=True)
    n_tr_tasks, n_te_tasks = (12, 8) if a.quick else (60, 40)
    rows = []
    for fam in a.families.split(","):
        n_train, n_eval = (200, 400) if fam != "digits-partition" else (300, 600)
        for kind in a.models.split(","):
            t0 = time.time()
            tasks = []
            s = 0
            while len(tasks) < n_tr_tasks + n_te_tasks:
                r = run_task(FAMILIES[fam], kind, {"teacher-net": 10000, "teacher-tree": 20000, "digits-partition": 30000}[fam] + s, n_train, n_eval, a.quick)
                s += 1
                if r is not None and 0.05 < r["acc"] < 0.98:
                    tasks.append(r)
            tr, te = tasks[:n_tr_tasks], tasks[n_tr_tasks:]
            levels = [k for k in tasks[0]["feats"]]
            res = {lv: evaluate(tr, te, lv) for lv in levels}
            for lv in levels:
                m, lo, hi = boot_ci(res[lv])
                rows.append(dict(family=fam, model=kind, level=lv, auroc=m, lo=lo, hi=hi,
                                 acc=float(np.mean([t["acc"] for t in te]))))
            # paired advantages
            for lv in levels:
                if lv.startswith("beh_q") or lv in ("twin", "public"):
                    m, lo, hi = boot_ci(res["white"] - res[lv])
                    rows.append(dict(family=fam, model=kind, level=f"adv_vs_{lv}",
                                     auroc=m, lo=lo, hi=hi, acc=np.nan))
            m, lo, hi = boot_ci(res["trace"] - res["white"])
            rows.append(dict(family=fam, model=kind, level="trace_minus_white",
                             auroc=m, lo=lo, hi=hi, acc=np.nan))
            np.save(os.path.join(a.out, f"per_task_{fam}_{kind}.npy"), res, allow_pickle=True)
            print(f"{fam:18s} {kind:7s} acc={np.mean([t['acc'] for t in te]):.3f} "
                  + " ".join(f"{lv}={np.nanmean(res[lv]):.3f}" for lv in levels)
                  + f"  ({time.time()-t0:.0f}s)", flush=True)
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
