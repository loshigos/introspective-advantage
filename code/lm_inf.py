"""Transformer experiment, limit of unlimited sampling.

The behavioural monitor receives the exact answer distributions on the prompt and on the
related prompts, which is what an observer would learn with unlimited samples. This
estimates A*(B) for the sampled-text channel from below, with no sampling noise.

Usage: python lm_inf.py --out ../results_lm_inf [--families facts,arithmetic]
"""
import argparse, json, os, time, warnings
import numpy as np, torch
import lm_experiment as L
torch.set_num_threads(1)
warnings.filterwarnings("ignore")


def exact_behaviour(world, model):
    Xev = world.Xev; n = len(Xev)
    P0 = L.probs(model, Xev); yhat = P0.argmax(1)
    rel = [world.related(x) for x in Xev]; R = len(rel[0])
    Prel = L.probs(model, np.concatenate([np.stack(r) for r in rel])).reshape(n, R, -1)
    own = P0[np.arange(n), yhat]
    ent = -(P0 * np.log(P0 + 1e-12)).sum(1)
    srt = np.sort(P0, 1)
    cols = [own, srt[:, -1] - srt[:, -2], ent]
    for g, idx in world.groups.items():
        pg = Prel[:, idx, :][np.arange(n)[:, None], np.arange(len(idx))[None, :], yhat[:, None]]
        cols += [pg.mean(1), pg.min(1), pg.max(1)]
        mode_agree = (Prel[:, idx, :].argmax(2) == yhat[:, None]).mean(1)
        cols.append(mode_agree)
    return np.column_stack(cols)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="../results_lm_inf")
    ap.add_argument("--families", default="facts,arithmetic"); a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True); rows = []
    for fam in a.families.split(","):
        W, steps, hp = L.FAM[fam]; t0 = time.time(); agents = []
        for j in range(70):
            rng = np.random.default_rng(1000 + j); torch.manual_seed(1000 + j)
            w = W(rng); m = L.TinyLM(w.vocab, w.n_out); L.train(m, w.Xtr, w.ytr, steps, seed=j, **hp); m.eval()
            Z, F = L.features(w, m, rng)
            beh_inf = np.column_stack([F["public"], exact_behaviour(w, m)])
            F2 = {"beh_inf": beh_inf, "white_inf": np.column_stack([beh_inf, F["white"]])}
            agents.append((Z, F2))
        tr, te = agents[:40], agents[40:]
        res = {lv: L.evaluate(tr, te, lv) for lv in ["beh_inf", "white_inf"]}
        for lv in res:
            m_, lo, hi = L.boot(res[lv]); rows.append(dict(family=fam, level=lv, auroc=m_, lo=lo, hi=hi))
        m_, lo, hi = L.boot(res["white_inf"] - res["beh_inf"]); rows.append(dict(family=fam, level="adv_inf", auroc=m_, lo=lo, hi=hi))
        print(f"{fam} beh_inf={np.nanmean(res['beh_inf']):.3f} white={np.nanmean(res['white_inf']):.3f} adv={m_:.3f} [{lo:.3f},{hi:.3f}] ({time.time()-t0:.0f}s)", flush=True)
        json.dump(rows, open(os.path.join(a.out, "summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
