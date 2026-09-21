"""Sanity-check finetune data: schema, empties, duplicates, test leakage, token lengths."""
import json, sys
from pathlib import Path

ROOT = Path(__file__).parent
model = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-4B-Instruct-2507"
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(model)
seen = {}
for split in ("train", "val", "test"):
    rows = [json.loads(l) for l in open(ROOT / f"out/{split}.jsonl", encoding="utf-8")]
    lens, bad, pairs = [], 0, set()
    for r in rows:
        m = r["messages"]
        if [x["role"] for x in m] != ["system", "user", "assistant"] or not all(x["content"].strip() for x in m):
            bad += 1
        pairs.add((m[1]["content"], m[2]["content"]))
        lens.append(len(tok.apply_chat_template(m, tokenize=True)))
    seen[split] = {c for _, c in pairs}
    lens.sort()
    print(f"{split:5} n={len(rows):6} bad={bad} dupes={len(rows) - len(pairs)} "
          f"tokens p50={lens[len(lens)//2]} p99={lens[int(len(lens)*.99)]} max={lens[-1]} "
          f">512: {sum(l > 512 for l in lens)}")
print("test targets also in train:", len(seen["test"] & seen["train"]))
