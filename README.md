# Thadou-Kuki (tcz) ↔ English Dataset

Rebuild: `python build_dataset.py` (reads `raw/`, writes `out/`).

## Current contents (from eBible corpus)
| File | What |
|---|---|
| `out/parallel.tsv` | 30,814 verse-aligned rows: ref, Thadou (Chongthu), English (WEB), English (KJV), Thadou (gospelgo, 13k verses) |
| `out/train.jsonl` | 95,241 chat-format SFT examples, both directions |
| `out/val.jsonl` / `out/test.jsonl` | held-out BOOKS (Obad, Titus, Phlm, 2Jn, Nah / Ruth, Jonah, Phil, Jude, 3Jn) |
| `out/rag_corpus.jsonl` | one doc per verse with both languages, for embeddings |
| `out/lexicon_candidates.tsv` | auto word-alignment hints (mostly proper nouns; needs human review) |

Thadou text = **Pathen Lekhabu Theng, Chongthu dialect**, © 2020 Chongthu Bible Translation Team,
licensed **CC BY-SA 4.0** (attribution + share-alike; derived datasets/models must credit it and use the same license).
English = World English Bible + KJV (public domain).

## Other sources (collect manually / with permission)
| Source | Notes |
|---|---|
| YouVersion `THADBSI` Pathen Thutheng BU (BSI 2015), bible.com/versions/1879 | Standard Thadou + audio. © Bible Society of India — ask BSI for permission, don't scrape |
| bibliamundi.com Chin-Thado-All-Bible.pdf | Same BSI text as PDF |
| scriptureearth.org (iso=tcz) | text/audio/video index |
| globalrecordings.net/en/language/tcz | audio (good for STT) |
| gospelgo.com/a/kuki_bible.htm | Unicode Thadou Bible |
| Google Play "Thadou Kuki English Bible" (jaqer) | parallel app |
| UNT Digital Library: *Thadou-Kuki for Students, Teachers and Writers* | orthography, grammar, wordlists |
| *Grammar of Thadou-Kuki* (dokumen.pub), khalvontawi.in | grammar/oral literature |
| CIIL Bhasha Sanchika – Thadou collection | govt. language corpus |
| omniglot.com/writing/thadou.htm, Wikipedia | phrases, alphabet |
| Local bilingual papers (e.g. *The Hills Today*), church hymnals, Facebook/YouTube Kuki pages | modern everyday language — biggest gap |

Two Bibles (Chongthu + BSI) aligned on the same verses = extra target variety for free.
Not found: no Thadou in FLORES-200, NLLB, or any HF dataset yet → a published dataset would be the first.

## Limitation
Bible-only data gives archaic/religious register. For conversational quality add everyday sentences
(native-speaker written, or recorded speech transcribed with STT, then corrected).

## Collected files (`sources/`)
| File | From |
|---|---|
| `ebible_usfm/`, `tczchongthu_usfm.zip` | ebible.org Chongthu Bible, USFM (with footnotes/headings) |
| `ebible_readaloud/` | same, plain text per chapter (1,192 files), good for TTS scripts |
| `gospelgo_kuki_bible.html` → `raw/tcz-gospelgo.txt` | older standard Thadou Bible, parsed by `parse_gospelgo.py` (13,106 verses aligned) |
| `omniglot_thadou.html`, `wikipedia_thadou_language.html` | alphabet, phonology |
| `dimasa_basic_kuki_phrases.html` | everyday phrases |
| `khalvontawi_oral_literature.html` | oral literature article |
| `scriptureearth_tcz.html`, `unt_thadou_for_students.html` | index pages (links only) |

Could not auto-download (do by hand in a browser): BSI PDF on bibliamundi (503/SSL), mchip PDF (dead),
globalrecordings.net (403), UNT book (viewer only), YouVersion THADBSI (copyright, ask BSI).

## Finetuning (Unsloth QLoRA)
```bash
python -m venv .venv && .venv/Scripts/pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
.venv/Scripts/pip install -r requirements-train.txt
.venv/Scripts/python check_dataset.py                                 # schema / length / leakage check
.venv/Scripts/python train_unsloth.py --max_steps 30 --eval_n 20 --no_gguf   # smoke test
.venv/Scripts/python train_unsloth.py                                 # full: 1 epoch, eval, GGUF, Modelfile
ollama create thadou -f runs/thadou-qwen3-4b/Modelfile
```
Default: `unsloth/Qwen3-4B-Instruct-2507`, 4-bit, LoRA r=32, fits 8 GB VRAM. Loss only on assistant turns.
Output in `runs/`: `lora/`, `eval.json` (chrF++/BLEU per direction on unseen books + samples), `gguf/`, `Modelfile`.

### RunPod (48GB GPU, e.g. A40)
Use image `runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04`, then:
```bash
pip install unsloth sacrebleu
# pip pulls cu130 torch; pod driver is CUDA 12.8 -> reinstall matching wheels:
pip install --force-reinstall --no-deps torch torchvision torchaudio triton --index-url https://download.pytorch.org/whl/cu128
nohup python train_unsloth.py --bf16_base --batch 16 --grad_accum 1 > train.log 2>&1 &
```
A40: 5,953 steps ~1h45m (~$1 at $0.49/hr). 16-bit base avoids the 4-bit checkpoint's degenerate output.

## Results: thadou-qwen3-4b (v1, 2026-09-21)
Qwen3-4B-Instruct-2507, 16-bit LoRA r=32, 1 epoch (5,953 steps, batch 16) on RunPod A40: 1h48m, $1.26.
Eval loss 2.64 (step 500) → 1.85 (end). Held-out books (Ruth, Jonah, Philippians, Jude, 3 John), 200 sentences:

| Direction | chrF++ | BLEU |
|---|---|---|
| Thadou → English | 28.8 | 9.7 |
| English → Thadou | 31.2 | 4.4 |

`results/thadou-qwen3-4b/`: eval.json (all 200 outputs), Modelfile, train_metrics.log.
Weights (not in git, too large): `runs/thadou-qwen3-4b/lora/` (264 MB adapter), `gguf_gguf/*.Q4_K_M.gguf` (2.5 GB).
```bash
cd runs/thadou-qwen3-4b/gguf_gguf && ollama create thadou -f Modelfile
ollama run thadou "Thadou-Kuki to English:\n\nKapa le kanu chu inn ah aum uve."
```
Known weakness: Bible-only data, so everyday sentences drift into scripture phrasing.

Hugging Face (private): https://huggingface.co/Prompt48/thadou-kuki-qwen3-4b-lora (LoRA + Q4_K_M GGUF + Modelfile)
