"""B2 input: FaithBench.csv (800 rows, 10 summarizers x 80) -> data/samples/faithbench.jsonl.
label 1 = faithful, 0 = unfaithful, where unfaithful = worst-label == "Unwanted" (primary). Extra fields: generator, worst, q_unf
(q_unf = 1 if worst-label in {Unwanted, Questionable}, the sensitivity definition). Data is CC BY-NC-SA: never commit or publish it.
Also checks the pre-registered human Unwanted rates and stops if any differs."""
import csv
import json
import sys

TIER = {"openai/gpt-4o": ("Low", 0.46), "openai/GPT-3.5-Turbo": ("Low", 0.47), "meta-llama/Meta-Llama-3.1-8B-Instruct": ("Low", 0.55),
        "google/gemini-1.5-flash-001": ("Low", 0.56), "Anthropic/claude-3-5-sonnet-20240620": ("Low", 0.57),
        "meta-llama/Meta-Llama-3.1-70B-Instruct": ("Low", 0.59), "mistralai/Mistral-7B-Instruct-v0.3": ("High", 0.70),
        "microsoft/Phi-3-mini-4k-instruct": ("High", 0.71), "cohere/command-r-08-2024": ("High", 0.71), "Qwen/Qwen2.5-7B-Instruct": ("High", 0.72)}
rows = list(csv.DictReader(open("data/faithbench/FaithBench.csv")))
assert len(rows) == 800
rate = {}
for g in TIER:
    r = [x for x in rows if x["LLM"] == g]
    assert len(r) == 80, g
    rate[g] = sum(x["worst-label"] == "Unwanted" for x in r) / 80
    # prereg lists rates rounded to 2 decimals (half-up on the 4-decimal value); tolerate only rounding
    if abs(rate[g] - TIER[g][1]) > 0.0051:
        sys.exit(f"STOP: rate differs for {g}: {rate[g]} vs {TIER[g][1]}")
assert max(rate[g] for g, t in TIER.items() if t[0] == "Low") < min(rate[g] for g, t in TIER.items() if t[0] == "High")
doc_id = {s: i for i, s in enumerate(dict.fromkeys(x["source"] for x in rows))}
assert len(doc_id) == 80  # matched design: 80 source documents, each summarized by all 10 LLMs
with open("data/samples/faithbench.jsonl", "w") as f:
    for i, x in enumerate(rows):
        f.write(json.dumps({"id": f"faithbench-{i}", "context": x["source"], "question": "", "response": x["summary"],
                            "label": 0 if x["worst-label"] == "Unwanted" else 1, "generator": x["LLM"], "worst": x["worst-label"],
                            "doc": doc_id[x["source"]], "q_unf": int(x["worst-label"] in ("Unwanted", "Questionable"))}) + "\n")
print("ok", len(rows), {g: round(v, 4) for g, v in rate.items()})
