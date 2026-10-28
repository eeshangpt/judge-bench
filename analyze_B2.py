"""#146 B2 analysis exactly as in prereg-B2.md (frozen). usage: .venv/bin/python -I analyze_B2.py > report/B2.md
Labels: 1 = faithful. Judge pred 0 = judged unfaithful at the fixed 0.5 cut. Seed 7, 1000 resamples."""
import itertools
import json

import numpy as np
from scipy.stats import spearmanr

from analyze_AB import NB, SEED, bacc, ci, kappa, tau_b

HIGH = ["mistralai/Mistral-7B-Instruct-v0.3", "microsoft/Phi-3-mini-4k-instruct", "cohere/command-r-08-2024", "Qwen/Qwen2.5-7B-Instruct"]
LOW = ["openai/gpt-4o", "openai/GPT-3.5-Turbo", "meta-llama/Meta-Llama-3.1-8B-Instruct", "google/gemini-1.5-flash-001",
       "Anthropic/claude-3-5-sonnet-20240620", "meta-llama/Meta-Llama-3.1-70B-Instruct"]
GENS = sorted(HIGH + LOW)
M = {"qw": "Qwen_Qwen3-4B-Instruct-2507", "mc": "minicheck"}


def tier_ok(u, gens):
    g = dict(zip(gens, u))
    return min(g[k] for k in HIGH if k in g) > max(g[k] for k in LOW if k in g)


def verdict(t, ok):
    return "KILL" if (not ok or np.isnan(t) or t < 0.3) else "SCALE (worth a run on a third dataset)" if t >= 0.5 else "ITERATE"


def rank_stats(gi, hum, pred, gens, nb=NB, seed=SEED, doc=None):
    """hum, pred: 1 = unfaithful. Returns tau-b point, bootstrap taus (stratified within generator), perm p, H, U."""
    def rates(idx):
        return (np.array([hum[idx[gi[idx] == k]].mean() for k in range(len(gens))]),
                np.array([pred[idx[gi[idx] == k]].mean() for k in range(len(gens))]))
    H, U = rates(np.arange(len(hum)))
    t = tau_b(H, U)
    rng = np.random.default_rng(seed)
    per = [np.where(gi == k)[0] for k in range(len(gens))]
    if doc is None:  # stratified within generator
        draw = lambda: np.concatenate([rng.choice(a, len(a)) for a in per])
    else:  # cluster bootstrap: resample source documents, keep all summaries of a sampled document (matched design)
        byd = {d: np.where(doc == d)[0] for d in np.unique(doc)}
        ds = np.array(list(byd))
        draw = lambda: np.concatenate([byd[d] for d in rng.choice(ds, len(ds))])
    bt = np.array([tau_b(*rates(draw())) for _ in range(nb)])
    return H, U, t, bt


def perm_p(H, U, t):
    """EXACT two-sided permutation p: all n! orderings of U against fixed H. The multiset of U is unchanged by a permutation, so the
    tau-b denominator is constant and tau-b = (P - Q)/den for every ordering. p = share with |tau-b| >= |observed|."""
    n = len(U)
    perms = np.array(list(itertools.permutations(range(n))), dtype=np.int8)
    S = np.zeros(len(perms), np.int16)
    for i in range(n):
        for j in range(i + 1, n):
            S += (np.sign(H[i] - H[j]) * np.sign(U[perms[:, i]] - U[perms[:, j]])).astype(np.int16)
    s_obs = round(sum(np.sign(H[i] - H[j]) * np.sign(U[i] - U[j]) for i in range(n) for j in range(i + 1, n)))
    return float(np.mean(np.abs(S) >= abs(s_obs)))  # |tau-b| >= |obs| iff |P-Q| >= |obs P-Q| (same denominator)


def block(name, tag, g, gens, gi, hum, hum_q, p_yes, trunc, doc):
    pred = (p_yes < 0.5).astype(int)  # 1 = judged unfaithful
    H, U, t, bt = rank_stats(gi, hum, pred, gens, doc=doc)  # PRIMARY CI: cluster bootstrap over the 80 source documents
    _, _, _, bts = rank_stats(gi, hum, pred, gens)  # sensitivity: stratified within generator
    prim = np.where(np.isnan(bt), 0, bt)
    hi = [i for i, n in enumerate(gens) if n in HIGH]
    lo = [i for i, n in enumerate(gens) if n in LOW]
    unc = 1 - p_yes
    alt = np.array([unc[gi == k].mean() for k in range(len(gens))])
    auc = np.mean([(a > b) + 0.5 * (a == b) for a in U[hi] for b in U[lo]])
    ok = tier_ok(U, gens)
    out = [f"## [{tag}] {name}", "", "| generator | tier | human Unwanted rate H | judged unfaithful rate U |", "|---|---|---|---|"]
    out += [f"| {n} | {'High' if n in HIGH else 'Low'} | {H[i]:.3f} | {U[i]:.3f} |" for i, n in enumerate(gens)]
    y = 1 - hum
    out += ["", f"Kendall tau-b = {t:.3f}; PRIMARY CI = cluster bootstrap over the 80 source documents, 95% (undefined counted 0) {ci(prim)}; with undefined dropped {ci(bt)}; undefined resamples {int(np.isnan(bt).sum())}.",
            f"Sensitivity CI, stratified bootstrap within generator (undefined counted 0): {ci(np.where(np.isnan(bts), 0, bts))}.",
            f"Exact two-sided permutation p of tau-b over all 10! orderings (descriptive): {perm_p(H, U, t):.5f}.",
            f"Descriptive: within-tier tau-b High {tau_b(H[hi], U[hi]):.3f}, Low {tau_b(H[lo], U[lo]):.3f}; tier separation AUC (share of High-Low generator pairs with U(High) > U(Low), Mann-Whitney) {auc:.3f}; tau-b using the threshold-free score mean(1 - p_yes) per generator {tau_b(H, alt):.3f}.",
            f"Tier order correct (min U(High) = {min(U[gens.index(k)] for k in HIGH):.3f} > max U(Low) = {max(U[gens.index(k)] for k in LOW):.3f}): {ok}.",
            f"Spearman rho {spearmanr(H, U).statistic:.3f}. Sensitivity H' = Unwanted or Questionable (tiers unchanged): tau-b {tau_b(np.array([hum_q[gi == k].mean() for k in range(len(gens))]), U):.3f}.",
            f"Item level (800, prevalence faithful {y.mean():.3f}): kappa {kappa(y, 1 - pred, np.ones(len(y))):.3f}, bacc {bacc(y, 1 - pred, np.ones(len(y))):.3f}.",
            f"Truncated items (over the context limit): {int(trunc.sum())} of {len(trunc)}.",
            f"**{tag} verdict by rule: {verdict(t, ok)}**", ""]
    return out, verdict(t, ok)


def main():
    items = [json.loads(l) for l in open("data/samples/faithbench.jsonl")]
    gi = np.array([GENS.index(x["generator"]) for x in items])
    hum = np.array([1 - x["label"] for x in items])
    hum_q = np.array([x["q_unf"] for x in items])
    doc = np.array([x["doc"] for x in items])
    out = ["# B2: FaithBench confirmation (800 items, 80 per summarizer)", ""]
    res = {}
    for j, tag in (("qw", "PRIMARY"), ("mc", "EXPLORATORY")):
        rows = {}
        for l in open(f"preds/{M[j]}__faithbench.jsonl"):
            r = json.loads(l)
            rows[r["id"]] = r
        assert len(rows) == 800
        p = np.array([rows[x["id"]]["p_yes"] for x in items])
        tr = np.array([rows[x["id"]]["truncated"] for x in items])
        o, v = block(M[j], tag, items, GENS, gi, hum, hum_q, p, tr, doc)
        out += o
        res[j] = v
        if j == "qw":  # exploratory same-family check: drop the Qwen2.5-7B generator
            keep = np.array([GENS[g] != "Qwen/Qwen2.5-7B-Instruct" for g in gi])
            g9 = [g for g in GENS if g != "Qwen/Qwen2.5-7B-Instruct"]
            gi9 = np.array([g9.index(GENS[g]) for g in gi[keep]])
            pred9 = (p[keep] < 0.5).astype(int)
            H9, U9, t9, bt9 = rank_stats(gi9, hum[keep], pred9, g9, doc=doc[keep])
            out += [f"Exploratory same-family check, Qwen2.5-7B generator dropped (9 generators; High tier = 3): tau-b {t9:.3f}, CI {ci(np.where(np.isnan(bt9), 0, bt9))}, tier order correct {tier_ok(U9, g9)}.", ""]
    # exploratory: drop the detector-filter generators (gpt-4o, GPT-3.5-Turbo) and compute length confound
    drop = {"openai/gpt-4o", "openai/GPT-3.5-Turbo"}
    kp = np.array([GENS[g] not in drop for g in gi])
    g8 = [g for g in GENS if g not in drop]
    gi8 = np.array([g8.index(GENS[g]) for g in gi[kp]])
    prow = {}
    for l in open(f"preds/{M['qw']}__faithbench.jsonl"):
        r = json.loads(l)
        prow[r["id"]] = r["p_yes"]
    pq = np.array([prow[x["id"]] for x in items])
    H8, U8, t8, bt8 = rank_stats(gi8, hum[kp], (pq[kp] < 0.5).astype(int), g8, doc=doc[kp])
    out += [f"Exploratory filter-generator check (Qwen3-4B), gpt-4o and GPT-3.5-Turbo dropped (8 generators, Low tier = 4): tau-b {t8:.3f}, CI {ci(np.where(np.isnan(bt8), 0, bt8))}, tier order correct {tier_ok(U8, g8)}.", ""]
    ln = np.array([len(x["response"].split()) for x in items])
    Uq = np.array([(pq[gi == k] < 0.5).mean() for k in range(len(GENS))])
    Ln = np.array([ln[gi == k].mean() for k in range(len(GENS))])
    out += [f"Exploratory length confound: Spearman rho between U (Qwen3-4B) and mean summary length (words) per generator {spearmanr(Uq, Ln).statistic:.3f}; between human H and length {spearmanr(np.array([hum[gi == k].mean() for k in range(len(GENS))]), Ln).statistic:.3f}.", ""]
    out.append(f"## B2 verdict (primary judge Qwen3-4B): {res['qw']}")
    print("\n".join(out))


if __name__ == "__main__":
    main()
