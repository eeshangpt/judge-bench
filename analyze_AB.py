"""#146 A+B analysis, exactly as in prereg-AB.md (frozen). usage: .venv/bin/python -I analyze_AB.py A|B > report/AB_<part>.md
Labels: 1 = faithful. pred 1 = faithful. Seed 7, 1000 bootstrap resamples."""
import itertools
import json
import sys

import numpy as np
from scipy.stats import kendalltau, spearmanr

SEED, NB, K, FLOOR, KILL_COV = 7, 1000, 5, 30, 0.20
SONNET = {"ragtruth": 0.601, "wice": 0.548}
M = {"mc": "minicheck", "qw": "Qwen_Qwen3-4B-Instruct-2507", "ph": "microsoft_Phi-4-mini-instruct"}


def load(model, name, gen=False):
    g = [json.loads(l) for l in open(f"data/samples/{name}.jsonl")]
    p = {}
    for l in open(f"preds/{M[model]}__{name}.jsonl"):
        r = json.loads(l)
        p[r["id"]] = r["p_yes"]
    return g, np.array([p[x["id"]] for x in g])


def kappa(y, p, w):
    n = w.sum()
    if n <= 0:
        return np.nan
    po = (w * (y == p)).sum() / n
    pe = sum((w * (y == a)).sum() / n * (w * (p == a)).sum() / n for a in (0, 1))
    return np.nan if pe >= 1 else (po - pe) / (1 - pe)


def bacc(y, p, w):
    r = []
    for a in (0, 1):
        d = (w * (y == a)).sum()
        if d > 0:
            r.append((w * ((y == a) & (p == a))).sum() / d)
    return float(np.mean(r)) if r else np.nan


def fit_tau(c, y, pm, w, target):
    """Largest-coverage tau with weighted selective kappa >= target and accepted weight >= FLOOR; inf if none."""
    o = np.argsort(-c, kind="stable")
    cs, ys, ps, ws = c[o], y[o], pm[o], w[o]
    end = np.r_[cs[1:] != cs[:-1], True]  # tie groups are accepted together
    cum = lambda m: np.cumsum(ws * m)
    a = {(i, j): cum((ys == i) & (ps == j)) for i in (0, 1) for j in (0, 1)}
    n = a[0, 0] + a[0, 1] + a[1, 0] + a[1, 1]
    po = (a[0, 0] + a[1, 1]) / np.maximum(n, 1e-12)
    pe = ((a[0, 0] + a[0, 1]) * (a[0, 0] + a[1, 0]) + (a[1, 0] + a[1, 1]) * (a[0, 1] + a[1, 1])) / np.maximum(n, 1e-12) ** 2
    with np.errstate(all="ignore"):
        k = (po - pe) / (1 - pe)
    ok = end & (n >= FLOOR) & (k >= target)
    return cs[np.where(ok)[0].max()] if ok.any() else np.inf  # more accepted = later index


def folds_for(y, rng):
    f = np.empty(len(y), int)
    for a in (0, 1):
        i = np.where(y == a)[0]
        f[rng.permutation(i)] = np.arange(len(i)) % K
    return f


def cross_fit(c, y, pm, pl, w, f, target):
    acc = np.zeros(len(y), bool)
    for k in range(K):
        tau = fit_tau(c[f != k], y[f != k], pm[f != k], w[f != k], target)
        acc |= (f == k) & (c >= tau)
    return acc


def summarize(acc, y, pm, pl, w):
    casc = np.where(acc, pm, pl)
    n = w.sum()
    s = dict(cov=(w * acc).sum() / n, sel_kappa=kappa(y[acc], pm[acc], w[acc]),
             sel_risk=(w[acc] * (y[acc] != pm[acc])).sum() / max(w[acc].sum(), 1e-12),
             a2=kappa(y, casc, w), casc_err=(w * (y != casc)).sum() / n, lm_err=(w * (y != pl)).sum() / n)
    s["err_diff"] = s["casc_err"] - s["lm_err"]
    return s


def conf(acc, y, pm):
    ya, pa = y[acc], pm[acc]
    return dict(tn=int(((ya == 0) & (pa == 0)).sum()), fp=int(((ya == 0) & (pa == 1)).sum()),
                fn=int(((ya == 1) & (pa == 0)).sum()), tp=int(((ya == 1) & (pa == 1)).sum()),
                prev_acc=float(ya.mean()) if len(ya) else np.nan, prev_full=float(y.mean()), bacc=bacc(ya, pa, np.ones(len(ya))))


def aurc(c, y, pm):
    e = (y != pm)[np.argsort(-c, kind="stable")].astype(float)
    r = np.cumsum(e) / np.arange(1, len(e) + 1)
    ro = np.cumsum(np.sort(e)) / np.arange(1, len(e) + 1)
    return r.mean(), r.mean() - ro.mean(), e.mean()


def ci(v):
    v = np.array(v, float)
    v = v[~np.isnan(v)]
    return f"[{np.percentile(v, 2.5):.3f}, {np.percentile(v, 97.5):.3f}] (n={len(v)})" if len(v) else "n/a"


def part_A():
    out, res = ["# A: cascade (MiniCheck accepts, LM takes the rest)", ""], {}
    for s in ("ragtruth", "wice"):
        g, pmc = load("mc", s)
        y = np.array([x["label"] for x in g])
        target = 0.9 * SONNET[s]
        c, pm = np.abs(pmc - 0.5), (pmc >= 0.5).astype(int)
        for lm in ("qw", "ph"):
            _, pq = load(lm, s)
            pl = (pq >= 0.5).astype(int)
            w1 = np.ones(len(y))
            f0 = folds_for(y, np.random.default_rng(SEED))
            acc = cross_fit(c, y, pm, pl, w1, f0, target)
            pt = summarize(acc, y, pm, pl, w1)
            tau_d = fit_tau(c, y, pm, w1, target)
            acc_d = c >= tau_d
            rng = np.random.default_rng(SEED)
            bs = []
            for _ in range(NB):
                w = rng.multinomial(len(y), np.ones(len(y)) / len(y)).astype(float)
                bs.append(summarize(cross_fit(c, y, pm, pl, w, folds_for(y, rng), target), y, pm, pl, w))
            res[s, lm] = (pt, bs, target)
            tag = "PRIMARY" if lm == "qw" else "exploratory"
            lat = np.median([json.loads(l)["latency_ms"] for l in open(f"preds/{M[lm]}__{s}.jsonl")]) / 1000
            hdr = "PRIMARY" if lm == "qw" else "SECONDARY (exploratory)"
            out += [f"## [{hdr}] {s} n={len(y)}, LM = {M[lm]}; Sonnet kappa {SONNET[s]}, target 0.9x = {target:.3f}", "",
                    "| quantity | out-of-fold point | nested 95% CI |", "|---|---|---|"]
            for k, nm in (("cov", "A1 coverage"), ("sel_kappa", "selective kappa (accepted)"), ("sel_risk", "selective risk (accepted)"),
                          ("a2", "A2 cascade kappa (all items)"), ("casc_err", "cascade error"), ("lm_err", "LM-alone error"),
                          ("err_diff", "cascade error minus LM-alone error")):
                out.append(f"| {nm} | {pt[k]:.3f} | {ci([b[k] for b in bs])} |")
            out += ["", f"Cost (prereg: (1 - coverage) x LM p50 latency): LM p50 {lat:.2f} s/item, coverage {pt['cov']:.3f} -> {(1 - pt['cov']) * lat:.2f} s/item expected vs {lat:.2f} s/item LM alone.", "", f"Operating points (confusion rows tn/fp/fn/tp of the accepted subset, y=truth, pred=MiniCheck; prevalence = share faithful):", ""]
            out += ["| point | n accepted | tn | fp | fn | tp | prev accepted | prev full | bacc (descr.) | selective kappa |", "|---|---|---|---|---|---|---|---|---|---|"]
            for nm, a in (("out-of-fold pooled", acc), (f"deployable tau={tau_d:.3f} (fit on all n, in-sample)", acc_d)):
                d = conf(a, y, pm)
                out.append(f"| {nm} | {int(a.sum())} | {d['tn']} | {d['fp']} | {d['fn']} | {d['tp']} | {d['prev_acc']:.3f} | {d['prev_full']:.3f} | {d['bacc']:.3f} | {kappa(y[a], pm[a], np.ones(int(a.sum()))):.3f} |")
            ar, ea, br = aurc(c, y, pm)
            out += ["", f"Descriptive: AURC {ar:.3f}, E-AURC {ea:.3f}, random-confidence baseline (overall error) {br:.3f}; MiniCheck alone kappa {kappa(y, pm, w1):.3f}, LM alone kappa {kappa(y, pl, w1):.3f}.", ""]
    out += ["## Decision (prereg rules, primary LM = Qwen3-4B)", ""]
    covs = {s: res[s, "qw"][0]["cov"] for s in ("ragtruth", "wice")}
    lows = {s: np.nanpercentile([b["cov"] for b in res[s, "qw"][1]], 2.5) for s in covs}
    a2ok = {s: res[s, "qw"][0]["a2"] >= res[s, "qw"][2] for s in covs}
    kill = all(v < KILL_COV for v in covs.values())
    scale = all(v >= KILL_COV for v in covs.values()) and all(a2ok.values()) and all(v > 0 for v in lows.values())
    out.append(f"coverage {covs}; nested lower bounds {lows}; A2 >= target {a2ok}")
    out.append("**A verdict: " + ("KILL" if kill else "SCALE (to be confirmed on fresh items)" if scale else "ITERATE") + "**")
    return "\n".join(out)


def tau_b(h, u):
    if np.ptp(h) == 0 or np.ptp(u) == 0:
        return np.nan
    return kendalltau(h, u, variant="b").statistic


def tier_ok(u, gens):
    g = {n: u[i] for i, n in enumerate(gens)}
    hi, mid, lo = [g["llama-2-7b-chat"]], [g[k] for k in ("llama-2-13b-chat", "llama-2-70b-chat", "mistral-7B-instruct")], [g["gpt-3.5-turbo-0613"], g["gpt-4-0613"]]
    return hi[0] > max(mid) and min(mid) > max(lo)


def part_B():
    out = ["# B: rank generators (RAGTruth QA test, 900 items)", ""]
    gens = None
    verdicts = {}
    for j in ("qw", "mc", "ph"):
        g, p = load(j, "ragtruthgen")
        assert len(g) == 900
        gens = sorted({x["generator"] for x in g})
        gi = np.array([gens.index(x["generator"]) for x in g])
        y = np.array([x["label"] for x in g])
        pred = (p >= 0.5).astype(int)
        def rates(idx):
            H = np.array([1 - y[idx[gi[idx] == k]].mean() for k in range(len(gens))])
            U = np.array([1 - pred[idx[gi[idx] == k]].mean() for k in range(len(gens))])
            return H, U
        idx0 = np.arange(len(y))
        H, U = rates(idx0)
        t = tau_b(H, U)
        rng = np.random.default_rng(SEED)
        per = [np.where(gi == k)[0] for k in range(len(gens))]
        bt = []
        for _ in range(NB):
            idx = np.concatenate([rng.choice(a, len(a)) for a in per])
            h, u = rates(idx)
            bt.append(tau_b(h, u))
        bt = np.array(bt)
        prim = np.where(np.isnan(bt), 0, bt)
        perm = [tau_b(H, np.array(q)) for q in itertools.permutations(U)]
        perm = np.array(perm)
        pval = float(np.mean(np.abs(np.nan_to_num(perm)) >= abs(t) - 1e-12))
        ok = tier_ok(U, gens)
        unc = 1 - p  # alternative generator score
        alt = np.array([unc[gi == k].mean() for k in range(len(gens))])
        verdict = "KILL" if (not ok or t < 0.3) else "SCALE (pilot; worth a run on fresh data)" if t >= 0.5 else "ITERATE"
        tag = "PRIMARY" if j == "qw" else "exploratory"
        out += [f"## {M[j]} ({tag})", "", "| generator | human halluc. rate H | judged unfaithful rate U |", "|---|---|---|"]
        out += [f"| {n} | {H[i]:.3f} | {U[i]:.3f} |" for i, n in enumerate(gens)]
        out += ["", f"Kendall tau-b = {t:.3f}; bootstrap 95% CI (undefined counted 0) {ci(prim)}; CI with undefined dropped {ci(bt)}; undefined resamples {int(np.isnan(bt).sum())}; exact permutation p = {pval:.3f} (descriptive)",
                f"Tier order correct (U(7b) > max U(Mid), min U(Mid) > max U(Low)): {ok}. Spearman rho {spearmanr(H, U).statistic:.3f}; tau-b with mean(1-p_yes) score {tau_b(H, alt):.3f}.",
                f"Item level on 900 (natural mix, prevalence faithful {y.mean():.3f}): kappa {kappa(y, pred, np.ones(len(y))):.3f}, bacc {bacc(y, pred, np.ones(len(y))):.3f}.",
                f"**{tag} verdict by rule: {verdict}**", ""]
        verdicts[j] = verdict
    out.append(f"## B verdict (primary judge Qwen3-4B): {verdicts['qw']}")
    return "\n".join(out)


if __name__ == "__main__":
    print(part_A() if sys.argv[1] == "A" else part_B())
