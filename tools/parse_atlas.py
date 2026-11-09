#!/usr/bin/env python3
"""Parse the Algorithms Atlas PDF into atlas_engine/data/catalog.json.

Usage: parse_atlas.py <atlas.pdf>
Fails loudly if the per-part counts do not match what the PDF's own part headers state.
"""
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "atlas_engine" / "data" / "catalog.json"

HEADER = re.compile(r"^ML, DL & Computer Vision Algorithms Atlas\b")
PAGENO = re.compile(r"^\s+\d{1,3}\s*$")
ENTRY = re.compile(r"^(\d{1,3}) (\S.*)$")
PART = re.compile(r"^\s*PART (\d+)\s*$")
RANGE = re.compile(r"Entries (\d+)[–-](\d+) · (\d+) algorithms")


def body_lines(text):
    for line in text.replace("\f", "\n").splitlines():
        if HEADER.match(line) or PAGENO.match(line):
            continue
        yield line.rstrip()


def main(pdf):
    with tempfile.NamedTemporaryFile(suffix=".txt") as tmp:
        subprocess.run(["pdftotext", "-layout", pdf, tmp.name], check=True)
        lines = list(body_lines(Path(tmp.name).read_text(encoding="utf-8")))

    # Part headers: PART n / title (1-2 lines) / blurb / "Entries a-b · n algorithms"
    parts, part_at = [], {}
    for i, line in enumerate(lines):
        m = PART.match(line)
        if not m:
            continue
        chunk = [l.strip() for l in lines[i + 1 : i + 12] if l.strip()]
        rng_i = next(j for j, l in enumerate(chunk) if RANGE.search(l))
        a, b, n = map(int, RANGE.search(chunk[rng_i]).groups())
        # Three long titles wrap onto a second line in the PDF.
        wrapped = chunk[0].endswith(("Efficient", "(VLA)", "Meta-,"))
        title = " ".join(chunk[:2]) if wrapped else chunk[0]
        parts.append({"part": int(m.group(1)), "title": title, "blurb": " ".join(chunk[1 + wrapped:rng_i]),
                      "first": a, "last": b, "count": n})
        part_at[i] = len(parts) - 1

    entries, expected, cur = [], 1, None
    i = 0
    n = len(lines)
    while i < n:
        m = ENTRY.match(lines[i])
        look = " ".join(lines[i + 1 : i + 3])
        if not (m and int(m.group(1)) == expected and "Why / used for:" in look):
            i += 1
            continue
        name = m.group(2).strip()
        i += 1
        if not lines[i].startswith("Why / used for:"):   # wrapped name
            name += " " + lines[i].strip()
            i += 1
        why, example, code = [lines[i][len("Why / used for:"):].strip()], [], []
        i += 1
        section = "why"
        while i < n:
            line = lines[i]
            nxt = ENTRY.match(line)
            if nxt and int(nxt.group(1)) == expected + 1 and "Why / used for:" in " ".join(lines[i + 1 : i + 3]):
                break
            if PART.match(line) or re.match(r"^A[–-]Z index|^Index\b", line.strip()):
                break
            if line.startswith("Example:"):
                section = "example"
                example.append(line[len("Example:"):].strip())
            elif section == "why" and line.strip():
                why.append(line.strip())
            elif section == "example" and line.strip() and not line.startswith(" "):
                example.append(line.strip())
            elif line.startswith(" ") and line.strip():
                section = "code"
                code.append(line[1:])
            elif section == "code" and line.strip():
                code.append(line)
            i += 1
        p = next(p for p in parts if p["first"] <= expected <= p["last"])
        entries.append({"id": expected, "name": name, "part": p["part"], "category": p["title"],
                        "why": " ".join(why), "example": " ".join(example), "code": "\n".join(code)})
        expected += 1

    for p in parts:
        got = sum(1 for e in entries if e["part"] == p["part"])
        if got != p["count"]:
            sys.exit(f"Part {p['part']} {p['title']}: PDF says {p['count']}, parsed {got}")
    if len(entries) != 830:
        sys.exit(f"Expected 830 entries, parsed {len(entries)} (stopped before #{expected})")
    empty = [e["id"] for e in entries if not e["why"] or not e["example"] or not e["code"]]
    OUT.write_text(json.dumps({"parts": parts, "entries": entries}, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"OK: {len(entries)} entries, {len(parts)} parts -> {OUT}  (entries missing a field: {empty})")


if __name__ == "__main__":
    main(sys.argv[1])
