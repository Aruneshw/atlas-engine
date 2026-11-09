#!/usr/bin/env python3
"""Write a small table in every supported format and read it back through the engine's loader.

Usage: <env>/bin/python tools/test_formats.py [work_dir]
"""
import gzip
import pickle
import sqlite3
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import pyreadstat

from atlas_worker.data import DataError, load_table
from atlas_worker.fileinfo import sniff_delimiter

W = Path(sys.argv[1] if len(sys.argv) > 1 else Path.home() / ".cache/atlas-formats")
W.mkdir(parents=True, exist_ok=True)
df = pd.DataFrame({"temp": [20.5, 21.0, 19.8, 22.1], "kind": ["a", "b", "a", "c"], "fail": [0, 1, 0, 1]})
writers = {
    "a.csv": lambda p: df.to_csv(p, index=False),
    "semi.csv": lambda p: df.to_csv(p, index=False, sep=";"),
    "latin.csv": lambda p: df.assign(kind=["é", "ü", "é", "ñ"]).to_csv(p, index=False, encoding="latin-1"),
    "utf16.csv": lambda p: df.to_csv(p, index=False, encoding="utf-16"),
    "a.tsv": lambda p: df.to_csv(p, index=False, sep="\t"),
    "pipe.psv": lambda p: df.to_csv(p, index=False, sep="|"),
    "table.txt": lambda p: df.to_csv(p, index=False, sep="\t"),
    "a.dat": lambda p: df.to_csv(p, index=False, sep=";"),
    "a.xlsx": lambda p: df.to_excel(p, index=False),
    "two_sheets.xlsx": lambda p: _two_sheets(p),
    "a.ods": lambda p: df.to_excel(p, index=False, engine="odf"),
    "records.json": lambda p: df.to_json(p, orient="records"),
    "wrapped.json": lambda p: Path(p).write_text('{"meta": {"n": 4}, "data": ' + df.to_json(orient="records") + "}"),
    "a.jsonl": lambda p: df.to_json(p, orient="records", lines=True),
    "a.ndjson": lambda p: df.to_json(p, orient="records", lines=True),
    "a.parquet": lambda p: df.to_parquet(p),
    "a.feather": lambda p: df.to_feather(p),
    "a.orc": lambda p: df.to_orc(p),
    "a.h5": lambda p: df.to_hdf(p, key="data", mode="w"),
    "a.xml": lambda p: df.to_xml(p, index=False, parser="etree"),
    "a.html": lambda p: df.to_html(p, index=False),
    "a.dta": lambda p: df.to_stata(p, write_index=False),
    "a.sav": lambda p: pyreadstat.write_sav(df, str(p)),
    "a.xpt": lambda p: pyreadstat.write_xport(df, str(p)),
    "a.sqlite": lambda p: _sqlite(p),
    "a.csv.gz": lambda p: gzip.open(p, "wt").write(df.to_csv(index=False)),
    "a.zip": lambda p: _zip(p),
}


def _two_sheets(p):
    with pd.ExcelWriter(p) as x:
        pd.DataFrame({"note": ["small"]}).to_excel(x, sheet_name="info", index=False)
        df.to_excel(x, sheet_name="data", index=False)


def _sqlite(p):
    Path(p).unlink(missing_ok=True)
    con = sqlite3.connect(p)
    df.to_sql("readings", con, index=False)
    con.close()


def _zip(p):
    with zipfile.ZipFile(p, "w") as z:
        z.writestr("folder/data.csv", df.to_csv(index=False))


failed = 0
for name, write in writers.items():
    path = W / name
    try:
        write(path)
        got = load_table(path)
        ok = len(got) == 4 and list(got.columns) == list(df.columns) and abs(float(got["temp"].sum()) - 83.4) < 1e-3
        print(f"{'ok  ' if ok else 'FAIL'} {name:18} {got.shape} {list(got.columns)}")
        failed += not ok
    except Exception as e:
        print(f"FAIL {name:18} {type(e).__name__}: {str(e)[:120]}")
        failed += 1

prose = W / "notes.txt"
prose.write_text("the first line of plain writing here\nanother sentence with no separators in it\nand a third line of text\n")
got = load_table(prose)
print("ok  " if list(got.columns) == ["text"] and len(got) == 3 else "FAIL", "plain prose .txt", got.shape, sniff_delimiter(prose))
np.save(W / "m.npy", np.arange(12.0).reshape(4, 3))
np.savez(W / "m.npz", a=np.arange(3.0), b=np.arange(20.0).reshape(4, 5))
for f, shape in (("m.npy", (4, 3)), ("m.npz", (4, 5))):
    got = load_table(W / f)
    print("ok  " if got.shape == shape else "FAIL", f, got.shape)
    failed += got.shape != shape
(W / "evil.pkl").write_bytes(pickle.dumps({"x": 1}))
try:
    load_table(W / "evil.pkl")
    print("FAIL pickle was loaded")
    failed += 1
except DataError as e:
    print("ok   pickle refused:", str(e)[:70])
print("ALL FORMATS OK" if not failed else f"{failed} FAILED")
