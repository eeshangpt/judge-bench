"""Job 03 analysis: kappa / bacc / ECE / latency / VRAM per (model, set); bootstrap CIs; RAIM comparison.
usage: .venv/bin/python -I analyze.py   (run from project root)
"""
import csv, glob, json, os
import numpy as np

SETS = ["ragtruth", "wice", "xsum", "cnn", "expertqa"]
MODELS = ["minicheck", "Qwen_Qwen3-4B-Instruct-2507", "microsoft_Phi-4-mini-instruct"]
# RAIM (refs/2609.39229.txt Table 12 + Table 1): kappa on identical balanced instances
RAIM = {  # set: (sonnet, panel_stacked, minicheck_ft5)
    "ragtruth": (0.601, 0.409, 0.262),
    "wice": (0.548, 0.540, 0.459),
    "xsum": (0.418, 0.437, 0.484),
    "cnn": (0.469, 0.349, 0.349),
    "expertqa": (0.165, 0.235, 0.174),
}
RAIM_BACC = {"ragtruth": 80.0, "wice": 77.4, "xsum": 70.9, "cnn": 73.5, "expertqa": 58.2}
B, SEED = 1000, 7


def load(path):
    return {json.loads(l)["id"]: json.loads(l) for l in open(path)}


def gold(s, split):
    d = "calib" if split == "calib" else "samples"
    return {json.loads(l)["id"]: json.loads(l)["label"] for l in open(f"data/{d}/{s}.jsonl")}


def kappa(y, p):
    n = len(y)
    po = np.mean(y == p)
    pe = sum(np.mean(y == c) * np.mean(p == c) for c in (0, 1))
    return float((po - pe) / (1 - pe)) if pe < 1 else 0.0


def bacc(y, p):
    r = []
    for c in (0, 1):
        m = y == c
        if m.any():
            r.append(np.mean(p[m] == c))
    return float(np.mean(r))


def ece(y, pr, bins=10):
    idx = np.minimum((pr * bins).astype(int), bins - 1)
    e = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            e += m.mean() * abs(y[m].mean() - pr[m].mean())
    return float(e)


def tune(y, pr):
    # maximise kappa on calib; ties -> threshold closest to 0.5
    best, bt = -9, 0.5
    for t in np.round(np.arange(0.01, 1.0, 0.01), 2):
        k = kappa(y, (pr >= t).astype(int))
        if k > best + 1e-12 or (abs(k - best) <= 1e-12 and abs(t - 0.5) < abs(bt - 0.5)):
            best, bt = k, float(t)
    return bt, best


def boot(y, pr, t, fn):
    rng = np.random.default_rng(SEED)
    n = len(y)
    v = []
    for _ in range(B):
        i = rng.integers(0, n, n)
        v.append(fn(y[i], (pr[i] >= t).astype(int)))
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


rows = []
for m in MODELS:
    for s in SETS:
        g = gold(s, "samples")
        gc = gold(s, "calib")
        P = load(f"preds/{m}__{s}.jsonl")
        C = load(f"preds/{m}__{s}__calib.jsonl")
        ids = [i for i in g if i in P]
        cids = [i for i in gc if i in C]
        y = np.array([g[i] for i in ids])
        pr = np.array([P[i]["p_yes"] for i in ids])
        yc = np.array([gc[i] for i in cids])
        prc = np.array([C[i]["p_yes"] for i in cids])
        t_tuned, k_cal = tune(yc, prc)
        lat = np.array([P[i]["latency_ms"] for i in ids])
        r = dict(model=m, set=s, n=len(ids), n_expected=len(g), n_calib=len(cids), pos_rate_gold=float(y.mean()),
                 pos_rate_pred_t05=float((pr >= 0.5).mean()), tuned_threshold=t_tuned, calib_kappa_at_tuned=k_cal,
                 truncated=int(sum(P[i]["truncated"] for i in ids)),
                 lat_p50_s=float(np.percentile(lat, 50) / 1000), lat_p95_s=float(np.percentile(lat, 95) / 1000),
                 peak_vram_mb=float(max(P[i]["peak_vram_mb"] for i in ids)), ece10=ece(y, pr))
        for tag, t in (("t05", 0.5), ("tuned", t_tuned)):
            p = (pr >= t).astype(int)
            k, b = kappa(y, p), bacc(y, p)
            kl, kh = boot(y, pr, t, kappa)
            bl, bh = boot(y, pr, t, bacc)
            r.update({f"kappa_{tag}": k, f"kappa_{tag}_lo": kl, f"kappa_{tag}_hi": kh,
                      f"bacc_{tag}": b, f"bacc_{tag}_lo": bl, f"bacc_{tag}_hi": bh,
                      f"ratio_sonnet_{tag}": k / RAIM[s][0], f"ratio_sonnet_{tag}_lo": kl / RAIM[s][0],
                      f"ratio_sonnet_{tag}_hi": kh / RAIM[s][0]})
        r.update(raim_sonnet_kappa=RAIM[s][0], raim_panel_kappa=RAIM[s][1], raim_minicheck_kappa=RAIM[s][2],
                 raim_sonnet_bacc=RAIM_BACC[s])
        rows.append(r)

os.makedirs("report", exist_ok=True)
keys = list(rows[0].keys())
with open("report/results.csv", "w", newline="") as f:
    w = csv.DictWriter(f, keys)
    w.writeheader()
    for r in rows:
        w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})

print(f"{'model':32s} {'set':8s} {'n':>4s} k05 [lo,hi] | ktuned(t) [lo,hi] | bacc05 | ratio05 ratioT | ece | p50 p95 vram")
for r in rows:
    print(f"{r['model'][:30]:32s} {r['set']:8s} {r['n']:4d} {r['kappa_t05']:.3f} [{r['kappa_t05_lo']:.3f},{r['kappa_t05_hi']:.3f}]"
          f" | {r['kappa_tuned']:.3f}(t={r['tuned_threshold']:.2f}) [{r['kappa_tuned_lo']:.3f},{r['kappa_tuned_hi']:.3f}]"
          f" | {r['bacc_t05']:.3f}/{r['bacc_tuned']:.3f} | {r['ratio_sonnet_t05']:.2f} {r['ratio_sonnet_tuned']:.2f}"
          f" | {r['ece10']:.3f} | {r['lat_p50_s']:.2f} {r['lat_p95_s']:.2f} {r['peak_vram_mb']:.0f}"
          f" | n_exp={r['n_expected']} calib={r['n_calib']} predpos={r['pos_rate_pred_t05']:.2f} trunc={r['truncated']}")
