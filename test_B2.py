"""Synthetic controls for analyze_B2.py (no real data). usage: .venv/bin/python -I test_B2.py"""
import sys
sys.path.insert(0, ".")
import numpy as np
from analyze_B2 import GENS, HIGH, LOW, perm_p, rank_stats, tier_ok, verdict

rng = np.random.default_rng(3)
gi = np.repeat(np.arange(10), 80)
rate = np.array([0.7 if GENS[k] in HIGH else 0.5 for k in range(10)])
hum = (rng.random(800) < rate[gi]).astype(int)
# positive control: judge flags exactly the human-unfaithful items -> SCALE
H, U, t, bt = rank_stats(gi, hum, hum.copy(), GENS, nb=50)
assert t > 0.99 and tier_ok(U, GENS) and verdict(t, True).startswith("SCALE")
# negative control: judge flag rates reversed (High generators flagged less) -> KILL
pred = (rng.random(800) < (1.2 - rate[gi])).astype(int)
H, U, t, bt = rank_stats(gi, hum, pred, GENS, nb=50)
assert not tier_ok(U, GENS) and verdict(t, tier_ok(U, GENS)) == "KILL"
# pure noise judge never ranks perfectly
pred = (rng.random(800) < 0.5).astype(int)
H, U, t, bt = rank_stats(gi, hum, pred, GENS, nb=50)
assert t < 0.9
assert verdict(0.4, True) == "ITERATE" and verdict(0.8, False) == "KILL" and verdict(float("nan"), True) == "KILL"
# exact permutation p: n=4 distinct, perfect agreement -> 2 of 24 orderings have |tau| = 1
h4 = np.array([1.0, 2, 3, 4])
assert abs(perm_p(h4, h4.copy(), 1.0) - 2 / 24) < 1e-12
# cluster bootstrap runs and keeps documents whole
doc = np.tile(np.arange(80), 10)
gi2 = np.repeat(np.arange(10), 80)
H, U, t, bt = rank_stats(gi2, hum, hum.copy(), GENS, nb=30, doc=doc)
assert t > 0.99 and np.nanmean(bt) > 0.5
print("OK")
