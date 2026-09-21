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
