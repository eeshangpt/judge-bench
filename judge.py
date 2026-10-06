"""Local judge harness. Backends: minicheck (lytang/MiniCheck-Flan-T5-Large) and hf causal LMs in 4-bit.
Output per (model, set): <out>/<slug>__<set>.jsonl (calib: <slug>__<set>__calib.jsonl) with
id, p_yes (P(supported)), pred (p_yes >= 0.5), latency_ms, peak_vram_mb, plus truncated, n_tokens.
Resumable: ids already present in the output file are skipped.

usage: judge.py run --model M --sets a,b --splits samples,calib [--limit N] [--out preds]
       judge.py check --models M1,M2 --sets a,b   (exit 0 iff all outputs complete)
"""
import argparse, json, os, re, sys, time

MINICHECK = "lytang/MiniCheck-Flan-T5-Large"
MAX_PROMPT_TOKENS = 4096  # RAIM used 8192; reduced for 6 GB VRAM
SETS = ["ragtruth", "wice", "xsum", "cnn", "expertqa"]

# RAIM App. K claim-support frame (LLM-AggreFact sets); answer line replaced by a single-token Yes/No.
CLAIM_PROMPT = """Determine whether the claim below is fully supported by the document. A
claim is SUPPORTED if every piece of information in it is stated in or
directly entailed by the document; it is HALLUCINATED if it adds, alters,
or implies anything the document does not support.

Answer with exactly one word: Yes if the claim is SUPPORTED, No if it is HALLUCINATED.

# Document
{ctx}

# Claim
{resp}"""

# RAIM def_on QA rubric (RAGTruth instantiation), single-token answer.
QA_PROMPT = """You are assessing a candidate answer to a retrieval-augmented question. You
are given the question, retrieved passages, and one candidate answer.

Definition. A HALLUCINATED answer contains information that is incorrect,
fabricated, or not supported by the evidence: e.g. it misreads the
question, reverses a finding, invents a mechanism, or adds specifics
absent from the source. A SUPPORTED answer is fully consistent with the
truth/evidence.

Decide whether the candidate answer is a HALLUCINATION.

Answer with exactly one word: Yes if the answer is SUPPORTED, No if it is HALLUCINATED.

# Question
{q}

# Retrieved passages
{ctx}

# Candidate answer
{resp}"""


def slug(m):
    return m.replace("/", "_")


def out_path(out, model, s, split):
    return os.path.join(out, f"{slug(model)}__{s}" + ("__calib" if split == "calib" else "") + ".jsonl")


def load_items(s, split):
    d = "calib" if split == "calib" else "samples"
    with open(f"data/{d}/{s}.jsonl") as f:
        return [json.loads(l) for l in f]


def done_ids(path):
    if not os.path.exists(path):
        return set()
    ids = set()
    with open(path) as f:
        for l in f:
            try:
                ids.add(json.loads(l)["id"])
            except Exception:
                pass
    return ids


def vram_reset(torch):
    torch.cuda.reset_peak_memory_stats()


def vram_peak(torch):
    return round(torch.cuda.max_memory_allocated() / 2**20, 1)


class MiniCheck:
    CHUNK_WORDS = 400  # MiniCheck default chunk (words) for flan-t5; max over chunks
    MAX_LEN = 2048

    def __init__(self):
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(MINICHECK)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(MINICHECK, torch_dtype=torch.float32).cuda().eval()

    def chunks(self, doc):
        sents = re.split(r"(?<=[.!?])\s+", doc.strip()) or [""]
        cur, n = [], 0
        for s in sents:
            w = len(s.split())
            if n + w > self.CHUNK_WORDS and cur:
                yield " ".join(cur)
                cur, n = [s], w
            else:
                cur.append(s)
                n += w
        if cur:
            yield " ".join(cur)

    def score(self, item):
        torch = self.torch
        texts = ["predict: " + c + self.tok.eos_token + item["response"] for c in self.chunks(item["context"])]
        best, trunc, ntok = 0.0, False, 0
        for t in texts:
            enc = self.tok(t, return_tensors="pt", max_length=self.MAX_LEN, truncation=True)
            full = len(self.tok(t)["input_ids"])
            trunc |= full > self.MAX_LEN
            ntok = max(ntok, enc["input_ids"].shape[1])
            enc = {k: v.cuda() for k, v in enc.items()}
            dec = torch.zeros((1, 1), dtype=torch.long, device="cuda")
            with torch.no_grad():
                lg = self.model(**enc, decoder_input_ids=dec).logits[:, 0, :]
            p = torch.softmax(lg[:, [3, 209]].float(), dim=-1)[0, 1].item()
            best = max(best, p)
        return best, trunc, ntok


class CausalJudge:
    def __init__(self, name):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(name)
        q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                               bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
        self.model = AutoModelForCausalLM.from_pretrained(name, quantization_config=q, device_map={"": 0},
                                                          torch_dtype=torch.float16).eval()
        self.yes = self._tid(["Yes", " Yes"])
        self.no = self._tid(["No", " No"])

    def _tid(self, cands):
        for c in cands:
            ids = self.tok.encode(c, add_special_tokens=False)
            if len(ids) == 1:
                return ids[0]
        raise RuntimeError("no single-token id for " + cands[0])

    def build(self, item, set_name, ctx):
        if set_name == "ragtruth":
            body = QA_PROMPT.format(q=item.get("question", ""), ctx=ctx, resp=item["response"])
        else:
            body = CLAIM_PROMPT.format(ctx=ctx, resp=item["response"])
        msgs = [{"role": "user", "content": body}]
        try:
            return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    def score(self, item, set_name):
        torch = self.torch
        ctx = item["context"]
        text = self.build(item, set_name, ctx)
        ids = self.tok(text, add_special_tokens=False)["input_ids"]
        trunc = False
        if len(ids) > MAX_PROMPT_TOKENS:
            # truncate the context (head kept) so the whole prompt fits the budget
            over = len(ids) - MAX_PROMPT_TOKENS
            cids = self.tok(ctx, add_special_tokens=False)["input_ids"]
            keep = max(64, len(cids) - over - 8)
            ctx = self.tok.decode(cids[:keep])
            text = self.build(item, set_name, ctx)
            ids = self.tok(text, add_special_tokens=False)["input_ids"][-MAX_PROMPT_TOKENS:]
            trunc = True
        x = torch.tensor([ids], device="cuda")
        with torch.no_grad():
            lg = self.model(input_ids=x, use_cache=False, logits_to_keep=1).logits[0, -1].float()
        p = torch.softmax(lg[[self.yes, self.no]], dim=-1)[0].item()
        return p, trunc, len(ids)


def run(a):
    import torch
    os.makedirs(a.out, exist_ok=True)
    judge = MiniCheck() if a.model == "minicheck" else CausalJudge(a.model)
    for split in a.splits.split(","):
        for s in a.sets.split(","):
            path = out_path(a.out, a.model, s, split)
            items = load_items(s, split)
            if a.limit:
                items = items[: a.limit]
            seen = done_ids(path)
            todo = [it for it in items if it["id"] not in seen]
            print(f"[{a.model}] {s}/{split}: {len(todo)} todo, {len(seen)} done", flush=True)
            with open(path, "a") as f:
                for it in todo:
                    vram_reset(torch)
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    try:
                        p, tr, nt = judge.score(it) if a.model == "minicheck" else judge.score(it, s)
                    except torch.cuda.OutOfMemoryError:
                        torch.cuda.empty_cache()
                        print("OOM on", it["id"], flush=True)
                        continue
                    torch.cuda.synchronize()
                    lat = (time.perf_counter() - t0) * 1000
                    f.write(json.dumps({"id": it["id"], "p_yes": p, "pred": int(p >= 0.5), "latency_ms": round(lat, 1),
                                        "peak_vram_mb": vram_peak(torch), "truncated": tr, "n_tokens": nt}) + "\n")
                    f.flush()
            rows = [json.loads(l) for l in open(path)]
            print(f"[{a.model}] {s}/{split}: {len(rows)} rows, truncated={sum(r['truncated'] for r in rows)}", flush=True)


def check(a):
    ok = True
    for m in a.models.split(","):
        for split in ("samples", "calib"):
            for s in a.sets.split(","):
                path = out_path(a.out, m, s, split)
                need = {it["id"] for it in load_items(s, split)}
                have = done_ids(path)
                if need - have:
                    ok = False
                    print(f"INCOMPLETE {path}: {len(need - have)} missing")
    print("ALL COMPLETE" if ok else "NOT COMPLETE")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "check"])
    ap.add_argument("--model")
    ap.add_argument("--models")
    ap.add_argument("--sets", default=",".join(SETS))
    ap.add_argument("--splits", default="samples,calib")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="preds")
    a = ap.parse_args()
    run(a) if a.cmd == "run" else check(a)
