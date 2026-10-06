"""Build samples/calib splits. label: 1 = faithful/supported, 0 = not (inverted vs RAIM).
Mirrors RAIM construction: class-balanced; RAGTruth = wandb/RAGTruth-processed train QA, paired
(first clean + first hallucinated response per query); AggreFact = per sub-dataset, majority cut to minority.
Per set: 300 sample items (150/150) + 50 calib items (25/25), disjoint, seed 7."""
import json, os, random, sys
from datasets import load_dataset

SEED = 7
N_SAMPLE, N_CALIB = 300, 50
os.makedirs("data/samples", exist_ok=True)
os.makedirs("data/calib", exist_ok=True)


def split_write(name, items):
    rng = random.Random(SEED)
    pos = [x for x in items if x["label"] == 1]
    neg = [x for x in items if x["label"] == 0]
    rng.shuffle(pos)
    rng.shuffle(neg)
    cp, cn = N_CALIB // 2, N_CALIB // 2
    sp, sn = N_SAMPLE // 2, N_SAMPLE // 2
    n = min(len(pos), len(neg))
    print(name, "available pos/neg", len(pos), len(neg))
    if n < cp + sp:
        sp = max(0, n - cp)
        sn = sp
    calib = pos[:cp] + neg[:cn]
    samp = pos[cp:cp + sp] + neg[cn:cn + sn]
    rng.shuffle(calib)
    rng.shuffle(samp)
    for d, rows in (("samples", samp), ("calib", calib)):
        with open(f"data/{d}/{name}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    print(name, "samples", len(samp), "calib", len(calib))


def is_hall(r):
    d = r["hallucination_labels_processed"]
    return any(int(v) > 0 for v in d.values())


def ragtruth():
    ds = load_dataset("wandb/RAGTruth-processed", split="train")
    print("ragtruth cols", ds.column_names)
    ds = [r for r in ds if r.get("task_type") == "QA"]
    byq = {}
    for r in ds:
        byq.setdefault(r["query"], []).append(r)
    items = []
    for q, rs in byq.items():
        clean = next((r for r in rs if not is_hall(r)), None)
        hall = next((r for r in rs if is_hall(r)), None)
        if clean is None or hall is None:
            continue
        for tag, r, lab in (("c", clean, 1), ("h", hall, 0)):
            items.append({"id": f"ragtruth-{r.get('id', len(items))}-{tag}",
                          "context": r["context"], "question": q,
                          "response": r["output"], "label": lab})
    print("ragtruth pairs", len(items) // 2)
    split_write("ragtruth", items)


def aggrefact():
    # official lytang/LLM-AggreFact is gated (needs HF login); public mirror used instead
    ds = load_dataset("NinaCalvi/llm_aggrefact_pre_aug9", split="test")
    print("aggrefact cols", ds.column_names)
    for sub, tag in (("Wice", "wice"), ("AggreFact-XSum", "xsum"), ("AggreFact-CNN", "cnn"), ("ExpertQA", "expertqa")):
        rows = [r for r in ds if r["dataset"] == sub]
        items = [{"id": f"{tag}-{i}", "context": r["doc"], "response": r["claim"], "label": int(r["label"])}
                 for i, r in enumerate(rows)]
        split_write(tag, items)


if __name__ == "__main__":
    for fn in (ragtruth, aggrefact):
        try:
            fn()
        except Exception as e:
            print("FAILED", fn.__name__, repr(e)[:500])
