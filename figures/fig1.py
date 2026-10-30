#!/usr/bin/env python3
"""Figure 1: judged unfaithful rate vs human rate. Reads only the committed tables in
report/B2.md (FaithBench, primary judge) and report/AB_B.md (RAGTruth pilot, primary judge).
Run from repo root: uv run --with matplotlib python figures/fig1.py"""
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def primary_rows(path, header):
    """Rows of the first table after the first '## ' section header (the PRIMARY judge)."""
    sec = open(path).read().split("\n## ")[1]
    return [[c.strip() for c in l.strip("|").split("|")] for l in sec.splitlines()
            if l.startswith("| ") and not l.startswith("| " + header) and "---" not in l]

fb = [(r[0], r[1], float(r[2]), float(r[3])) for r in primary_rows("report/B2.md", "generator")]
rt = [(r[0], float(r[1]), float(r[2])) for r in primary_rows("report/AB_B.md", "generator")]
assert len(fb) == 10 and len(rt) == 6
assert {t for _, t, _, _ in fb} == {"High", "Low"} and sum(t == "High" for _, t, _, _ in fb) == 4
assert (min(u for *_, u in fb), max(u for *_, u in fb)) == (0.087, 0.2)    # fact sheet: 0.087-0.200
assert (min(h for _, _, h, _ in fb), max(h for _, _, h, _ in fb)) == (0.463, 0.725)

HI, LO, GR = "#D9822B", "#3D82C4", "#7A848D"   # W1 palette: orange, blue, grey
fig, (a, b) = plt.subplots(1, 2, figsize=(15, 7.2), dpi=100)
fig.suptitle("Judge (Qwen3-4B) vs human unfaithful rate, per generator", fontsize=19)

def panel(ax, lim, title):
    ax.plot([0, lim], [0, lim], "--", color="#555", lw=1.5, zorder=1)
    ax.set(xlim=(0, lim), ylim=(0, lim), aspect="equal", xlabel="Human rate", ylabel="Judged unfaithful rate")
    ax.set_title(title, fontsize=16)
    ax.xaxis.label.set_size(15); ax.yaxis.label.set_size(15); ax.tick_params(labelsize=13)
    ax.grid(alpha=.25)

panel(a, 0.8, "FaithBench: 10 generators, 80 summaries each")
for tier, col in (("High", HI), ("Low", LO)):
    p = [(h, u) for _, t, h, u in fb if t == tier]
    a.scatter(*zip(*p), s=150, color=col, edgecolor="k", zorder=3, label=f"{tier} tier")
a.text(0.70, 0.745, "y = x", ha="right", fontsize=13, color="#555")
a.legend(loc="upper left", fontsize=14, frameon=False)

panel(b, 0.4, "RAGTruth pilot: 6 generators, 150 items each")
b.scatter([h for _, h, _ in rt], [u for *_, u in rt], s=150, color=GR, edgecolor="k", zorder=3)
for n, h, u in rt:
    if n in ("gpt-4-0613", "llama-2-7b-chat"):
        b.annotate(n, (h, u), xytext=(8, -22 if n.startswith("gpt") else -26), textcoords="offset points",
                   fontsize=13, ha="left" if n.startswith("gpt") else "right")
fig.tight_layout(rect=(0, 0, 1, 0.95))
for ext in ("png", "svg"):
    fig.savefig(f"figures/fig1.{ext}", dpi=100)
