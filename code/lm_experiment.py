"""Small transformer language models trained from scratch on two task families.

facts:      [subject, template] -> attribute. Each world has random facts, and each
            subject appears in training a random number of times (including zero).
arithmetic: [a, op, b] -> (a + b) mod p under a random relabelling of the number
            tokens, trained on a random 55% of the pairs and tested on the rest.

The output channel is sampled text: an outside observer receives sampled answers
from the agent. q is the number of samples it may draw, spread over the prompt and
related prompts (paraphrases for facts; shifted and swapped operands for arithmetic).
The white-box monitor reads logits, residual-stream norms, attention entropies,
token-embedding norms and the input gradient. The trace monitor also reads the
training data.

Usage: python lm_experiment.py --out ../results_lm [--quick]
"""
import argparse, json, os, time, warnings
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")
torch.set_num_threads(2)
Q_LEVELS = [0, 4, 16, 64, 256]


# ------------------------------------------------------------------ model
class TinyLM(nn.Module):
    def __init__(self, vocab, n_out, d=64, heads=4, layers=2, seq=3):
        super().__init__()
        self.emb = nn.Embedding(vocab, d); self.pos = nn.Parameter(torch.randn(seq, d) * 0.02)
        self.layers = nn.ModuleList()
        for _ in range(layers):
            self.layers.append(nn.ModuleDict(dict(
                ln1=nn.LayerNorm(d), att=nn.MultiheadAttention(d, heads, batch_first=True),
                ln2=nn.LayerNorm(d), mlp=nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d)))))
        self.lnf = nn.LayerNorm(d); self.head = nn.Linear(d, n_out)

    def forward(self, ids, emb_override=None, return_internals=False):
        x = self.emb(ids) if emb_override is None else emb_override
        x = x + self.pos[: ids.shape[1]]
        norms, ents = [], []
        L = ids.shape[1]
        mask = torch.triu(torch.ones(L, L, dtype=torch.bool), 1)
        for blk in self.layers:
            h = blk["ln1"](x)
            a, w = blk["att"](h, h, h, attn_mask=mask, need_weights=True, average_attn_weights=True)
            x = x + a
            x = x + blk["mlp"](blk["ln2"](x))
            norms.append(x[:, -1].norm(dim=-1))
            wl = w[:, -1].clamp_min(1e-9); ents.append(-(wl * wl.log()).sum(-1))
        logits = self.head(self.lnf(x[:, -1]))
        if return_internals:
            return logits, torch.stack(norms, 1), torch.stack(ents, 1)
        return logits


def train(model, X, y, steps, lr=3e-3, wd=0.1, bs=256, seed=0):
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    X = torch.as_tensor(X); y = torch.as_tensor(y)
    for s in range(steps):
        idx = torch.randint(0, len(X), (min(bs, len(X)),), generator=g)
        loss = F.cross_entropy(model(X[idx]), y[idx])
        opt.zero_grad(); loss.backward(); opt.step()
    return model


# ------------------------------------------------------------------ worlds
class Facts:
    name = "facts"
    S, T, A = 150, 4, 16
    def __init__(self, rng):
        self.attr = rng.integers(0, self.A, self.S)
        p = np.array([0.30, 0.20, 0.15, 0.12, 0.09, 0.07, 0.04, 0.03])  # occurrences 0..7
        self.freq = rng.choice(len(p), size=self.S, p=p / p.sum())
        self.vocab = self.S + self.T; self.n_out = self.A
        Xtr, ytr, self.pair_count = [], [], np.zeros((self.S, self.T), int)
        for s in range(self.S):
            for _ in range(self.freq[s]):
                t = rng.integers(0, self.T); Xtr.append([s, self.S + t]); ytr.append(self.attr[s]); self.pair_count[s, t] += 1
        self.Xtr = np.array(Xtr, dtype=np.int64).reshape(-1, 2); self.ytr = np.array(ytr, dtype=np.int64)
        self.Xev = np.array([[s, self.S + t] for s in range(self.S) for t in range(self.T)], dtype=np.int64)
        self.yev = self.attr[self.Xev[:, 0]]
    groups = {"para": [0, 1, 2]}
    def related(self, x):
        s, t = x[0], x[1] - self.S
        return [np.array([s, self.S + u]) for u in range(self.T) if u != t]
    def trace(self, x, yhat):
        s, t = x[0], x[1] - self.S
        seen = self.ytr[self.Xtr[:, 0] == s] if len(self.Xtr) else np.array([])
        agree = float(np.mean(seen == yhat)) if len(seen) else -1.0
        return [self.freq[s], self.pair_count[s, t], agree, float(len(seen) > 0)]
    def public(self, x):
        return [x[1] - self.S]


class Arith:
    name = "arithmetic"
    P = 23
    def __init__(self, rng):
        P = self.P
        self.perm = rng.permutation(P)          # token -> hidden number
        self.inv = np.argsort(self.perm)
        self.vocab = P + 1; self.n_out = P
        pairs = np.array([(a, b) for a in range(P) for b in range(P)])
        lab = self.inv[(self.perm[pairs[:, 0]] + self.perm[pairs[:, 1]]) % P]
        tr = rng.random(len(pairs)) < 0.55
        self.Xtr = np.column_stack([pairs[tr, 0], np.full(tr.sum(), P), pairs[tr, 1]]).astype(np.int64)
        self.ytr = lab[tr].astype(np.int64)
        self.Xev = np.column_stack([pairs[~tr, 0], np.full((~tr).sum(), P), pairs[~tr, 1]]).astype(np.int64)
        self.yev = lab[~tr]
        self.train_set = {(int(a), int(b)): int(c) for (a, _, b), c in zip(self.Xtr, self.ytr)}
    groups = {"swap": [0], "row": [1, 2, 3, 4], "col": [5, 6, 7, 8]}
    def related(self, x):
        a, b = int(x[0]), int(x[2]); P = self.P
        r = np.random.default_rng(a * 101 + b)
        bs = r.choice([v for v in range(P) if v != b], 4, replace=False)
        as_ = r.choice([v for v in range(P) if v != a], 4, replace=False)
        return [np.array([b, P, a])] + [np.array([a, P, v]) for v in bs] + [np.array([u, P, b]) for u in as_]
    def trace(self, x, yhat):
        a, b = int(x[0]), int(x[2])
        sw = self.train_set.get((b, a), -1)
        na = sum(1 for (u, v) in self.train_set if u == a); nbb = sum(1 for (u, v) in self.train_set if v == b)
        return [float(sw == yhat) if sw >= 0 else -1.0, float(sw >= 0), na, nbb]
    def public(self, x):
        return [0.0]


FAM = {"facts": (Facts, 1500, dict(lr=3e-3, wd=0.1, bs=256)), "arithmetic": (Arith, 3000, dict(lr=1e-3, wd=1.0, bs=512))}


# ------------------------------------------------------------------ features
def probs(model, X):
    with torch.no_grad():
        return F.softmax(model(torch.as_tensor(np.asarray(X, dtype=np.int64))), -1).numpy()


def features(world, model, rng):
    Xev = world.Xev; n = len(Xev)
    P0 = probs(model, Xev); yhat = P0.argmax(1); Z = (yhat == world.yev).astype(int)
    # white box
    ids = torch.as_tensor(Xev)
    e = model.emb(ids).detach().requires_grad_(True)
    logits, norms, ents = model(ids, emb_override=e, return_internals=True)
    top = logits.gather(1, torch.as_tensor(yhat)[:, None]).sum()
    grad = torch.autograd.grad(top, e)[0].norm(dim=(1, 2)).numpy()
    lg = logits.detach().numpy(); srt = np.sort(lg, 1)
    emb_norm = model.emb.weight.detach().norm(dim=1).numpy()
    emb_z = (emb_norm - emb_norm.mean()) / (emb_norm.std() + 1e-9)
    ent = -(P0 * np.log(P0 + 1e-12)).sum(1)
    wb = np.column_stack([P0.max(1), srt[:, -1] - srt[:, -2], ent, norms.detach().numpy(), ents.detach().numpy(),
                          emb_z[Xev[:, 0]], emb_z[Xev[:, -1]], grad])
    # behaviour: sampled answers from the prompt and related prompts
    rel = [world.related(x) for x in Xev]; R = len(rel[0])
    Prel = probs(model, np.concatenate([np.stack(r) for r in rel])).reshape(n, R, -1)
    allP = np.concatenate([P0[:, None], Prel], 1)               # n x (R+1) x out
    qmax = max(Q_LEVELS)
    # draw qmax samples, cycling over prompts, prompt itself first
    order = np.arange(qmax) % (R + 1)
    samples = np.empty((n, qmax), dtype=int)
    for i in range(n):
        cum = allP[i].cumsum(1)
        u = rng.random(qmax)
        samples[i] = (u[:, None] > cum[order]).sum(1).clip(0, allP.shape[2] - 1)
    rel_greedy = Prel.argmax(2)
    beh = {}
    for q in Q_LEVELS:
        if q == 0:
            beh[q] = np.zeros((n, 0)); continue
        s, o = samples[:, :q], order[:q]
        own = (o == 0)
        cols = [(s[:, own] == yhat[:, None]).mean(1) if own.any() else np.full(n, -1.0)]
        for g, idx in world.groups.items():
            m = np.isin(o, np.array(idx) + 1)
            cols.append((s[:, m] == yhat[:, None]).mean(1) if m.any() else np.full(n, -1.0))
        cols.append(np.array([len(np.unique(r)) for r in s]) / q)
        beh[q] = np.column_stack(cols)
    trace = np.array([world.trace(x, yh) for x, yh in zip(Xev, yhat)])
    pub = np.array([world.public(x) for x in Xev], dtype=float)
    feats = {"public": pub}
    for q in Q_LEVELS:
        feats[f"beh_q{q}"] = np.column_stack([pub, beh[q]]) if q else pub.copy()
    feats["white"] = np.column_stack([pub, beh[qmax], wb])
    feats["trace"] = np.column_stack([feats["white"], trace])
    return Z, feats


def evaluate(tr, te, level):
    X = np.vstack([t[1][level] for t in tr]); Z = np.concatenate([t[0] for t in tr])
    mon = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15,
                                         l2_regularization=1.0, random_state=0).fit(X, Z)
    out = []
    for Zt, f in te:
        out.append(roc_auc_score(Zt, mon.predict_proba(f[level])[:, 1]) if 0 < Zt.mean() < 1 else np.nan)
    return np.array(out)


def boot(v, B=2000):
    v = v[~np.isnan(v)]; rng = np.random.default_rng(0)
    bs = rng.choice(v, (B, len(v))).mean(1)
    return float(v.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="../results_lm"); ap.add_argument("--quick", action="store_true")
    ap.add_argument("--families", default="facts,arithmetic")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    ntr, nte = (6, 4) if a.quick else (40, 30)
    rows = []
    for fam in a.families.split(","):
        W, steps, hp = FAM[fam]
        if a.quick: steps //= 3
        t0 = time.time(); agents = []
        for j in range(ntr + nte):
            rng = np.random.default_rng(1000 + j); torch.manual_seed(1000 + j)
            w = W(rng)
            m = TinyLM(w.vocab, w.n_out)
            train(m, w.Xtr, w.ytr, steps, seed=j, **hp); m.eval()
            agents.append(features(w, m, rng))
        tr, te = agents[:ntr], agents[ntr:]
        levels = list(agents[0][1].keys())
        res = {lv: evaluate(tr, te, lv) for lv in levels}
        acc = float(np.mean([z.mean() for z, _ in te]))
        for lv in levels:
            m_, lo, hi = boot(res[lv]); rows.append(dict(family=fam, level=lv, auroc=m_, lo=lo, hi=hi, acc=acc))
        for q in Q_LEVELS:
            m_, lo, hi = boot(res["white"] - res[f"beh_q{q}"]); rows.append(dict(family=fam, level=f"adv_vs_beh_q{q}", auroc=m_, lo=lo, hi=hi))
        m_, lo, hi = boot(res["trace"] - res["white"]); rows.append(dict(family=fam, level="trace_minus_white", auroc=m_, lo=lo, hi=hi))
        print(f"{fam:10s} acc={acc:.3f} " + " ".join(f"{lv}={np.nanmean(res[lv]):.3f}" for lv in levels) + f" ({time.time()-t0:.0f}s)", flush=True)
    json.dump(rows, open(os.path.join(a.out, "summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
