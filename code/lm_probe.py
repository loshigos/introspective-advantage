"""Transformer experiment extended: sampling budget up to 1024 and per-agent probes on the residual stream.

Usage: python lm_probe.py --out ../results_lm2
"""
import argparse, json, os, time, warnings
import numpy as np, torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
import lm_experiment as L
torch.set_num_threads(1)
warnings.filterwarnings("ignore")
L.Q_LEVELS = [0, 4, 16, 64, 256, 1024]


@torch.no_grad()
def residual(model, X):
    ids = torch.as_tensor(np.asarray(X, dtype=np.int64))
    x = model.emb(ids) + model.pos[: ids.shape[1]]
    Lq = ids.shape[1]; mask = torch.triu(torch.ones(Lq, Lq, dtype=torch.bool), 1)
    for blk in model.layers:
        h = blk["ln1"](x); a, _ = blk["att"](h, h, h, attn_mask=mask, need_weights=False); x = x + a
        x = x + blk["mlp"](blk["ln2"](x))
    return model.lnf(x[:, -1]).numpy()


def crossfit(X, Z, seed=0):
    r = np.random.default_rng(seed); idx = r.permutation(len(Z)); half = len(Z) // 2; s = np.empty(len(Z))
    for a, b in [(idx[:half], idx[half:]), (idx[half:], idx[:half])]:
        if Z[a].min() == Z[a].max():
            s[b] = Z[a].mean(); continue
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=3000)).fit(X[a], Z[a])
        s[b] = clf.predict_proba(X[b])[:, 1]
    return roc_auc_score(Z, s) if 0 < Z.mean() < 1 else np.nan


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="../results_lm2"); a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True); rows = []
    for fam in ["facts", "arithmetic"]:
        W, steps, hp = L.FAM[fam]; t0 = time.time(); agents = []; probes = []
        for j in range(70):
            rng = np.random.default_rng(1000 + j); torch.manual_seed(1000 + j)
            w = W(rng); m = L.TinyLM(w.vocab, w.n_out); L.train(m, w.Xtr, w.ytr, steps, seed=j, **hp); m.eval()
            Z, F = L.features(w, m, rng); agents.append((Z, F))
            if j >= 40:
                Hd = residual(m, w.Xev)
                beh = F["beh_q1024"]; white = np.column_stack([F["white"], Hd])
                probes.append((crossfit(beh, Z), crossfit(white, Z)))
        tr, te = agents[:40], agents[40:]
        levels = list(agents[0][1].keys())
        res = {lv: L.evaluate(tr, te, lv) for lv in levels}
        acc = float(np.mean([z.mean() for z, _ in te]))
        for lv in levels:
            m_, lo, hi = L.boot(res[lv]); rows.append(dict(family=fam, level=lv, auroc=m_, lo=lo, hi=hi, acc=acc))
        for q in L.Q_LEVELS:
            m_, lo, hi = L.boot(res["white"] - res[f"beh_q{q}"]); rows.append(dict(family=fam, level=f"adv_vs_beh_q{q}", auroc=m_, lo=lo, hi=hi))
        m_, lo, hi = L.boot(res["trace"] - res["white"]); rows.append(dict(family=fam, level="trace_minus_white", auroc=m_, lo=lo, hi=hi))
        pb = np.array([p[0] for p in probes]); pw = np.array([p[1] for p in probes])
        for name, v in [("probe_beh", pb), ("probe_white", pw), ("probe_adv", pw - pb)]:
            m_, lo, hi = L.boot(v); rows.append(dict(family=fam, level=name, auroc=m_, lo=lo, hi=hi))
        print(f"{fam} acc={acc:.3f} " + " ".join(f"{lv}={np.nanmean(res[lv]):.3f}" for lv in levels)
              + f" | probe beh {np.nanmean(pb):.3f} white {np.nanmean(pw):.3f} ({time.time()-t0:.0f}s)", flush=True)
        json.dump(rows, open(os.path.join(a.out, "summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
