"""Build Thadou (tcz) <-> English parallel + finetuning datasets from the eBible corpus.

Inputs (raw/): vref.txt, tcz-tczchongthu.txt, eng-engwebp.txt, eng-eng-kjv.txt (optional)
Outputs (out/):
  parallel.tsv            ref, tcz, en_web, en_kjv
  {train,val,test}.jsonl  chat-format SFT pairs, both directions
  rag_corpus.jsonl        one doc per verse w/ both languages, for embedding
  lexicon_candidates.tsv  co-occurrence based word alignment hints
Split is by BOOK (not random verse) so test books are unseen -> honest eval.
"""
import json, random, re, collections
from pathlib import Path

ROOT = Path(__file__).parent
RAW, OUT = ROOT / "raw", ROOT / "out"
OUT.mkdir(exist_ok=True)
random.seed(48)

TEST_BOOKS = {"RUT", "JON", "PHP", "JUD", "3JN"}   # held out entirely
VAL_BOOKS = {"OBA", "TIT", "PHM", "2JN", "NAM"}


def load(name):
    p = RAW / name
    return p.read_text(encoding="utf-8").split("\n") if p.exists() else None


vref = load("vref.txt")
tcz = load("tcz-tczchongthu.txt")
web = load("eng-engwebp.txt")
kjv = load("eng-eng-kjv.txt") or [""] * len(vref)
gg = load("tcz-gospelgo.txt") or [""] * len(vref)  # 2nd Thadou translation (partial)


def merge_ranges(rows):
    """eBible marks verses merged into the previous line with '<range>'. Group them."""
    groups, cur = [], None
    for i, ref in enumerate(vref):
        if not ref:
            continue
        t, w, k, g2 = tcz[i].strip(), web[i].strip(), kjv[i].strip(), gg[i].strip()
        is_cont = t == "<range>" or w == "<range>"
        if is_cont and cur:
            cur["refs"].append(ref)
            for key, val in (("tcz", t), ("en_web", w), ("en_kjv", k), ("tcz2", g2)):
                if val and val != "<range>":
                    cur[key] = (cur[key] + " " + val).strip()
            continue
        cur = {"refs": [ref], "tcz": t, "en_web": w, "en_kjv": k if k != "<range>" else "", "tcz2": g2}
        groups.append(cur)
    return groups


rows = []
for g in merge_ranges(vref):
    if not g["tcz"] or not g["en_web"]:
        continue
    ref = g["refs"][0] if len(g["refs"]) == 1 else f'{g["refs"][0]}-{g["refs"][-1].split(":")[-1]}'
    g["ref"], g["book"] = ref, ref.split()[0]
    # drop pathological length mismatches (misalignment)
    r = len(g["tcz"]) / max(1, len(g["en_web"]))
    if 0.4 < r < 3.0:
        rows.append(g)

with open(OUT / "parallel.tsv", "w", encoding="utf-8") as f:
    f.write("ref\ttcz\ten_web\ten_kjv\ttcz_gospelgo\n")
    for g in rows:
        f.write("\t".join(x.replace("\t", " ") for x in (g["ref"], g["tcz"], g["en_web"], g["en_kjv"], g["tcz2"])) + "\n")

SYS = "You are an expert translator between Thadou-Kuki (Thado Chin) and English."
T2E = ["Translate this Thadou-Kuki text into English:", "Thadou-Kuki to English:",
       "What does this mean in English?"]
E2T = ["Translate this English text into Thadou-Kuki:", "English to Thadou-Kuki:",
       "Say this in Thadou-Kuki:"]


def ex(prompt, src, tgt):
    return {"messages": [{"role": "system", "content": SYS},
                         {"role": "user", "content": f"{prompt}\n\n{src}"},
                         {"role": "assistant", "content": tgt}]}


splits = collections.defaultdict(list)
for g in rows:
    s = "test" if g["book"] in TEST_BOOKS else "val" if g["book"] in VAL_BOOKS else "train"
    splits[s].append(ex(random.choice(T2E), g["tcz"], g["en_web"]))
    splits[s].append(ex(random.choice(E2T), g["en_web"], g["tcz"]))
    if s == "train" and g["en_kjv"] and random.random() < 0.3:  # extra English paraphrase variety
        splits[s].append(ex(random.choice(E2T), g["en_kjv"], g["tcz"]))
    if g["tcz2"] and 0.4 < len(g["tcz2"]) / len(g["en_web"]) < 3.0:  # second translation = more variety
        splits[s].append(ex(random.choice(T2E), g["tcz2"], g["en_web"]))
        splits[s].append(ex(random.choice(E2T), g["en_web"], g["tcz2"]))

for s, items in splits.items():
    random.shuffle(items)
    with open(OUT / f"{s}.jsonl", "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

with open(OUT / "rag_corpus.jsonl", "w", encoding="utf-8") as f:
    for g in rows:
        f.write(json.dumps({"id": g["ref"], "book": g["book"], "tcz": g["tcz"], "en": g["en_web"],
                            "text": f'{g["ref"]}\nThadou: {g["tcz"]}\nEnglish: {g["en_web"]}'},
                           ensure_ascii=False) + "\n")

# Cheap lexicon hints: Dice coefficient over verse co-occurrence (seed for a dictionary, needs human review)
tok = lambda s: set(re.findall(r"[a-zA-Z']+", s.lower()))
cz, ce, cp = collections.Counter(), collections.Counter(), collections.Counter()
for g in rows:
    a, b = tok(g["tcz"]), tok(g["en_web"])
    cz.update(a); ce.update(b)
    cp.update((x, y) for x in a for y in b)
best = {}
for (x, y), n in cp.items():
    if n < 8:
        continue
    d = 2 * n / (cz[x] + ce[y])
    if d > best.get(x, (0, ""))[0]:
        best[x] = (d, y)
with open(OUT / "lexicon_candidates.tsv", "w", encoding="utf-8") as f:
    f.write("tcz\ten\tdice\ttcz_freq\n")
    for x, (d, y) in sorted(best.items(), key=lambda kv: -kv[1][0]):
        if d >= 0.3:
            f.write(f"{x}\t{y}\t{d:.3f}\t{cz[x]}\n")

print(f"aligned verses: {len(rows)}")
for s in ("train", "val", "test"):
    print(f"{s}: {len(splits[s])} examples")
