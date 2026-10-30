"""Download model weights with retries (resumable). xet disabled: the xet path stalled with 0 B/s."""
import os, sys, time
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "60"
os.environ["HF_HUB_ETAG_TIMEOUT"] = "60"
from huggingface_hub import hf_hub_download, snapshot_download

JOBS = [("lytang/MiniCheck-Flan-T5-Large", ["*.json", "*.bin", "*.model", "*.txt"]),
        ("Qwen/Qwen3-4B-Instruct-2507", ["*.json", "*.safetensors", "*.txt", "*.jinja"]),
        ("microsoft/Phi-4-mini-instruct", ["*.json", "*.safetensors", "*.txt", "*.jinja", "*.py"])]  # Qwen2.5-3B, SmolLM3 deferred
bad = []
for m, pats in JOBS:
    for i in range(30):
        try:
            p = snapshot_download(m, allow_patterns=pats)
            print("OK", m, p, flush=True)
            break
        except Exception as e:
            print("retry", m, i, repr(e)[:150], flush=True)
            time.sleep(5)
    else:
        bad.append(m)
print("FAILED:", bad, flush=True)
sys.exit(1 if bad else 0)
