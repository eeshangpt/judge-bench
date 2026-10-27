"""Synthetic positive/negative controls for analyze_AB.py (no real data). usage: .venv/bin/python -I test_AB.py"""
import sys
sys.path.insert(0, ".")
import numpy as np
from analyze_AB import fit_tau, kappa, aurc, tau_b, tier_ok, folds_for, cross_fit

rng = np.random.default_rng(1)
n = 300
y = (np.arange(n) % 2).astype(int)
c = rng.random(n)
wrong = rng.random(n) < (0.45 - 0.4 * c)  # confident -> fewer errors
pm = np.where(wrong, 1 - y, y)
w = np.ones(n)
# brute-force largest-coverage tau meeting kappa >= 0.6 and >= 30 accepted
best = np.inf
for t in np.unique(c):
    a = c >= t
    if a.sum() >= 30 and kappa(y[a], pm[a], w[a]) >= 0.6:
        best = min(best, t)
assert fit_tau(c, y, pm, w, 0.6) == best, (fit_tau(c, y, pm, w, 0.6), best)
assert fit_tau(c, y, pm, w, 0.999) == np.inf  # infeasible target -> inf
# informative confidence has lower AURC than random confidence; E-AURC >= 0
a1, e1, _ = aurc(c, y, pm)
a2, _, _ = aurc(rng.random(n), y, pm)
assert a1 < a2 and e1 >= 0
# tau-b: perfect = 1, reversed = -1, constant = nan
h = np.array([6, 5, 4, 3, 2, 1.0])
assert abs(tau_b(h, h) - 1) < 1e-9 and abs(tau_b(h, -h) + 1) < 1e-9 and np.isnan(tau_b(h, np.ones(6)))
gens = sorted(["gpt-4-0613", "gpt-3.5-turbo-0613", "llama-2-13b-chat", "llama-2-70b-chat", "llama-2-7b-chat", "mistral-7B-instruct"])
good = {"llama-2-7b-chat": .35, "llama-2-13b-chat": .24, "llama-2-70b-chat": .23, "mistral-7B-instruct": .2, "gpt-3.5-turbo-0613": .03, "gpt-4-0613": .01}
assert tier_ok(np.array([good[g] for g in gens]), gens)
bad = dict(good, **{"gpt-4-0613": .22})
assert not tier_ok(np.array([bad[g] for g in gens]), gens)
# cross-fit: folds stratified, no accepted item when target infeasible
f = folds_for(y, np.random.default_rng(7))
assert all(abs(y[f == k].mean() - .5) < .05 for k in range(5))
assert not cross_fit(c, y, pm, pm, w, f, 0.999).any()
print("OK")
