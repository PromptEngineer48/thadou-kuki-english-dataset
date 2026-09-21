"""Parse gospelgo.com single-page Thadou Bible into vref-aligned lines (raw/tcz-gospelgo.txt)."""
import re, html
from pathlib import Path

ROOT = Path(__file__).parent
BOOKS = ("GEN EXO LEV NUM DEU JOS JDG RUT 1SA 2SA 1KI 2KI 1CH 2CH EZR NEH EST JOB PSA PRO ECC SNG ISA JER "
         "LAM EZK DAN HOS JOL AMO OBA JON MIC NAM HAB ZEP HAG ZEC MAL MAT MRK LUK JHN ACT ROM 1CO 2CO GAL EPH "
         "PHP COL 1TH 2TH 1TI 2TI TIT PHM HEB JAS 1PE 2PE 1JN 2JN 3JN JUD REV").split()

src = (ROOT / "sources" / "gospelgo_kuki_bible.html").read_text(encoding="utf-8")
src = src[src.find("<a>"):]  # skip menu
verses, book_idx, chap, last_name = {}, -1, 0, None
for m in re.finditer(r"<a>([^<]+?)\s+(\d+)</a>|(?:^|\n|>)\s*(\d+)\s+([^\n<]+)", src):
    if m.group(1):
        name, c = m.group(1).strip(), int(m.group(2))
        if name != last_name or c <= chap:
            if name != last_name:
                book_idx += 1
            last_name = name
        chap = c
    elif book_idx >= 0 and chap:
        text = html.unescape(m.group(4)).replace("�", "'").replace("\xa0", " ").strip()
        key = f"{BOOKS[book_idx]} {chap}:{int(m.group(3))}"
        verses[key] = (verses.get(key, "") + " " + text).strip()

vref = (ROOT / "raw" / "vref.txt").read_text(encoding="utf-8").split("\n")
out = [verses.get(r, "") for r in vref]
(ROOT / "raw" / "tcz-gospelgo.txt").write_text("\n".join(out), encoding="utf-8")
print(f"books seen: {book_idx + 1}, verses parsed: {len(verses)}, aligned to vref: {sum(1 for x in out if x)}")
