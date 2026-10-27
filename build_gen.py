"""B input: RAGTruth QA, test split, ALL 900 items (150 per generator, natural class mix, no sampling).
label 1 = faithful (no hallucination span), 0 = not. Adds field `generator`. Download-only from wandb/RAGTruth-processed."""
import ast
import json
from datasets import load_dataset


def hall(r):
    d = r["hallucination_labels_processed"]
    d = ast.literal_eval(d) if isinstance(d, str) else d
    return any(int(v) > 0 for v in d.values())


ds = [r for r in load_dataset("wandb/RAGTruth-processed", split="test") if r["task_type"] == "QA"]
with open("data/samples/ragtruthgen.jsonl", "w") as f:
    for r in ds:
        f.write(json.dumps({"id": f"ragtruthgen-{r['id']}-t", "context": r["context"], "question": r["query"],
                            "response": r["output"], "label": 0 if hall(r) else 1, "generator": r["model"]}) + "\n")
print(len(ds))
