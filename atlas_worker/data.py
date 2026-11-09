"""Dataset loading, validation and leak-free splitting."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from .fileinfo import REFUSED_EXT, TABLE_EXT, sniff_delimiter
from .protocol import log

MISSING = "__missing__"
MAX_CATEGORIES = 50


class DataError(ValueError):
    """A problem the user can fix in their dataset. Shown without a traceback."""


def _read_text(p):
    sep = sniff_delimiter(p)
    with open(p, "rb") as f:
        wide = f.read(2) in (b"\xff\xfe", b"\xfe\xff")
    encodings = ("utf-16",) if wide else ("utf-8-sig", "cp1252", "latin-1")
    if sep is None:                       # prose: one row per non-empty line
        for enc in encodings:
            try:
                lines = [l.strip() for l in p.read_text(encoding=enc).splitlines() if l.strip()]
                return pd.DataFrame({"text": lines})
            except UnicodeDecodeError:
                continue
    last = None
    for enc in encodings:
        try:
            return pd.read_csv(p, sep=sep, encoding=enc)
        except UnicodeDecodeError as e:
            last = e
    raise DataError(f"Could not decode {p.name}: {last}")


def _read_sqlite(p):
    import sqlite3

    con = sqlite3.connect(f"file:{p}?mode=ro", uri=True)
    try:
        tables = [r[0] for r in con.execute("select name from sqlite_master where type='table' and name not like 'sqlite_%'")]
        if not tables:
            raise DataError(f"{p.name} has no tables.")
        sizes = {t: con.execute(f'select count(*) from "{t}"').fetchone()[0] for t in tables}
        table = max(sizes, key=sizes.get)
        if len(tables) > 1:
            log(f"{p.name} has {len(tables)} tables: using the largest, '{table}'", "warn")
        return pd.read_sql_query(f'select * from "{table}"', con)
    finally:
        con.close()


def _read_numpy(p):
    data = np.load(p, allow_pickle=False)
    if p.suffix.lower() == ".npz":
        name = max(data.files, key=lambda k: data[k].size)
        data = data[name]
    if data.ndim == 1:
        return pd.DataFrame({"value": data})
    if data.ndim != 2:
        raise DataError(f"{p.name} has {data.ndim} dimensions: a table needs 1 or 2.")
    return pd.DataFrame(data, columns=[f"col_{i}" for i in range(data.shape[1])])


def _read_json(p):
    import json

    with open(p, encoding="utf-8-sig") as f:
        doc = json.load(f)
    if isinstance(doc, dict):                      # find the list of records in a wrapped document
        lists = [v for v in doc.values() if isinstance(v, list)]
        doc = max(lists, key=len) if lists else [doc]
    return pd.json_normalize(doc)


def _read_excel(p):
    book = pd.ExcelFile(p)
    sheet = max(book.sheet_names, key=lambda s: len(book.parse(s, nrows=None)))
    if len(book.sheet_names) > 1:
        log(f"{p.name} has {len(book.sheet_names)} sheets: using the largest, '{sheet}'", "warn")
    return book.parse(sheet)


def _read_zip(p):
    import tempfile
    import zipfile

    with zipfile.ZipFile(p) as z:
        members = [i for i in z.infolist() if not i.is_dir() and Path(i.filename).suffix.lower() in TABLE_EXT - {".zip"}
                   and not Path(i.filename).name.startswith(".") and "__MACOSX" not in i.filename]
        if not members:
            raise DataError(f"{p.name} contains no data file.")
        member = max(members, key=lambda i: i.file_size)
        with tempfile.TemporaryDirectory() as td:
            target = Path(td) / Path(member.filename).name          # flattened: an archive path cannot escape td
            target.write_bytes(z.read(member))
            return load_table(target)


def _read_sas(p):
    import pyreadstat

    reader = pyreadstat.read_xport if p.suffix.lower() == ".xpt" else pyreadstat.read_sas7bdat
    return reader(str(p))[0]


READERS = {
    ".xlsx": _read_excel, ".xlsm": _read_excel, ".xltx": _read_excel, ".xls": _read_excel, ".xlsb": _read_excel, ".ods": _read_excel,
    ".json": _read_json, ".jsonl": lambda p: pd.read_json(p, lines=True), ".ndjson": lambda p: pd.read_json(p, lines=True),
    ".parquet": pd.read_parquet, ".pq": pd.read_parquet, ".feather": pd.read_feather, ".arrow": pd.read_feather, ".orc": pd.read_orc,
    ".h5": pd.read_hdf, ".hdf5": pd.read_hdf, ".xml": lambda p: pd.read_xml(p, parser="etree"), ".html": lambda p: max(pd.read_html(p), key=len),
    ".htm": lambda p: max(pd.read_html(p), key=len), ".dta": pd.read_stata, ".sas7bdat": _read_sas, ".xpt": _read_sas,
    ".sav": pd.read_spss, ".zsav": pd.read_spss, ".sqlite": _read_sqlite, ".sqlite3": _read_sqlite, ".db": _read_sqlite,
    ".npy": _read_numpy, ".npz": _read_numpy, ".zip": _read_zip,
}


def load_table(path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise DataError(f"File not found: {p}")
    suffix = p.suffix.lower()
    if suffix in REFUSED_EXT:
        raise DataError(f"{suffix} files can run hidden code when opened, so they are refused. Export the data as CSV or Parquet.")
    try:
        if suffix in (".gz", ".bz2", ".xz"):
            inner = Path(p.stem).suffix.lower()
            return pd.read_json(p, lines=True) if inner in (".jsonl", ".ndjson") else pd.read_csv(p, sep="\t" if inner == ".tsv" else ",")
        df = READERS.get(suffix, _read_text)(p)
    except DataError:
        raise
    except (ValueError, ImportError, OSError, KeyError, TypeError, UnicodeDecodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as e:
        raise DataError(f"Could not read {p.name}: {e}") from e
    df.columns = [str(c).strip() for c in df.columns]
    if df.columns.duplicated().any():
        raise DataError(f"{p.name} has two columns with the same name. Rename one first.")
    if df.empty:
        raise DataError(f"{p.name} has no data rows.")
    return df


def file_hash(path, chunk=1 << 20) -> str:
    h = hashlib.sha256()
    p = Path(path)
    if p.is_dir():
        for f in sorted(x for x in p.rglob("*") if x.is_file()):
            h.update(f"{f.relative_to(p)}:{f.stat().st_size}".encode())
        return h.hexdigest()[:16]
    with open(p, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()[:16]


def coerce(df: pd.DataFrame, features: list[dict]) -> pd.DataFrame:
    """Force columns to the types seen at training. Used for training and every prediction."""
    out = {}
    for f in features:
        name = f["name"]
        if name not in df.columns:
            raise DataError(f"Missing required column: {name}")
        if f["type"] == "num":
            out[name] = pd.to_numeric(df[name], errors="coerce").astype("float64").to_numpy()
        else:
            col = df[name].astype(object)
            out[name] = np.array([MISSING if pd.isna(v) else str(v) for v in col], dtype=object)
    return pd.DataFrame(out, index=df.index)


def select_features(df, target, time_col, chosen=None, required=True):
    """Decide which columns are usable and warn about each one dropped."""
    skip = {c for c in (target, time_col) if c}
    cols = [c for c in (chosen or df.columns) if c not in skip]
    n = len(df)
    features = []
    for c in cols:
        s = df[c]
        uniq = s.nunique(dropna=True)
        if uniq <= 1:
            log(f"dropping '{c}': constant column", "warn")
        elif pd.api.types.is_bool_dtype(s):
            features.append({"name": c, "type": "cat"})
        elif pd.api.types.is_numeric_dtype(s):
            if pd.api.types.is_integer_dtype(s) and uniq == n and n > 20 and _is_row_counter(s):
                log(f"dropping '{c}': looks like a row ID", "warn")
            else:
                features.append({"name": c, "type": "num"})
        elif pd.to_numeric(s, errors="coerce").notna().sum() >= 0.95 * s.notna().sum():
            features.append({"name": c, "type": "num"})
        elif uniq > MAX_CATEGORIES:
            log(f"dropping '{c}': {uniq} distinct text values is too many to encode", "warn")
        else:
            features.append({"name": c, "type": "cat"})
    if not features and required:
        raise DataError("No usable feature columns are left after validation.")
    return features


def _is_row_counter(s) -> bool:
    diffs = s.sort_values().diff().dropna()
    return (diffs == diffs.iloc[0]).all() if len(diffs) else False


def validate(df, job) -> tuple[pd.DataFrame, list[dict]]:
    ds = job["dataset"]
    target, time_col = ds.get("target"), ds.get("time_column")
    if len(df) < 20:
        raise DataError(f"Only {len(df)} rows: at least 20 are needed to split and evaluate.")
    for label, col in (("Target", target), ("Time", time_col)):
        if col and col not in df.columns:
            raise DataError(f"{label} column '{col}' is not in this file.")
    before = len(df)
    df = df.drop_duplicates() if not time_col else df
    if len(df) < before:
        log(f"removed {before - len(df)} exact duplicate rows so they cannot leak into the test set", "warn")
    if target:
        n_missing = int(df[target].isna().sum())
        if n_missing:
            log(f"dropping {n_missing} rows with no target value", "warn")
            df = df[df[target].notna()]
    if time_col:
        ts = pd.to_datetime(df[time_col], errors="coerce") if not pd.api.types.is_numeric_dtype(df[time_col]) else df[time_col]
        bad = int(ts.isna().sum())
        if bad > 0.05 * len(df):
            raise DataError(f"Time column '{time_col}': {bad} of {len(df)} values could not be read as dates.")
        df = df.assign(__t=ts).dropna(subset=["__t"]).sort_values("__t", kind="stable")
        job["_times"] = df["__t"].to_numpy()
        df = df.drop(columns="__t")
        distinct = len(pd.unique(job["_times"]))
        if job["task"] in ("forecasting", "seq_classification") and distinct < len(df):
            raise DataError(f"'{time_col}' has {len(df):,} rows but only {distinct:,} different times, so this is many series mixed together. "
                            "A forecast needs one row per time. Use Regression or Classification with this time column, or keep a single series.")
        gaps = pd.Series(pd.unique(job["_times"])).diff().dropna()
        if job["task"] == "forecasting" and len(gaps) and gaps.max() > 3 * gaps.median():
            log(f"'{time_col}' has gaps: the longest step is {gaps.max() / gaps.median():.0f} times the usual one, and forecasters assume even steps", "warn")
        log(f"sorted by '{time_col}'; every split will follow time order")
    df = df.reset_index(drop=True)
    missing = df.isna().mean()
    for c in missing[missing > 0.5].index:
        log(f"column '{c}' is {missing[c]:.0%} missing", "warn")
    # A forecaster can work from the target series alone.
    features = select_features(df, target, time_col, ds.get("features"), required=job["task"] != "forecasting")
    log(f"{len(df)} rows, {len(features)} features "
        f"({sum(f['type'] == 'num' for f in features)} numeric, {sum(f['type'] == 'cat' for f in features)} categorical)")
    return df, features


def _snap(i, times):
    """Move a split point forward so rows that share a timestamp stay on the same side."""
    while 0 < i < len(times) and times[i] == times[i - 1]:
        i += 1
    return i


class TimeFolds:
    """Time-ordered cross-validation that never puts one timestamp on both sides of a fold."""

    def __init__(self, n_splits, times):
        self.n_splits, self.times = n_splits, np.asarray(times)

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

    def split(self, X, y=None, groups=None):
        from sklearn.model_selection import TimeSeriesSplit

        distinct = np.unique(self.times)
        for a, b in TimeSeriesSplit(n_splits=self.n_splits).split(distinct):
            yield np.where(self.times <= distinct[a[-1]])[0], np.where(np.isin(self.times, distinct[b]))[0]


def split_indices(n, y, job):
    """Train / validation / test indices. Chronological when a time column is set."""
    s = job.get("split") or {}
    val, test, seed = s.get("val", 0.15), s.get("test", 0.15), s.get("seed", 42)
    chrono = bool(job["dataset"].get("time_column")) or job["task"] in ("forecasting", "seq_classification")
    if chrono:
        a, b = int(n * (1 - val - test)), int(n * (1 - test))
        times = job.get("_times")
        if times is not None and len(times) == n:
            a = _snap(a, times)
            b = _snap(max(b, a + 1), times)
            if not 0 < a < b < n:
                raise DataError("Too few different times to keep training, validation and test apart. Add more time points.")
        return np.arange(a), np.arange(a, b), np.arange(b, n), True
    from sklearn.model_selection import train_test_split

    strat = None
    if y is not None and job["task"] == "classification":
        counts = pd.Series(y).value_counts()
        if counts.min() >= 3:
            strat = y
        else:
            log("a class has under 3 rows, so the split cannot be stratified", "warn")
    idx = np.arange(n)
    tr, rest = train_test_split(idx, test_size=val + test, random_state=seed, stratify=strat)
    va, te = train_test_split(rest, test_size=test / (val + test), random_state=seed,
                              stratify=None if strat is None else strat[rest])
    return tr, va, te, False
