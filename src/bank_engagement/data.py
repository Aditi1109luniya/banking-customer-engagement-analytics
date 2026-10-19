"""Exact UCI variant acquisition, validation, provenance, and ordered preparation."""

from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
from pathlib import Path, PurePosixPath
import sqlite3
import tempfile
import time
from urllib.error import URLError
from urllib.request import Request, urlopen
import zipfile

import numpy as np
import pandas as pd

from .constants import DB, RAW, RAW_COLUMNS, SOURCE_URL
from .database import connect

CSV_NAME = "bank-additional-full.csv"
NAMES_NAME = "bank-additional-names.txt"
CITATION = (
    "Moro, S., Cortez, P., & Rita, P. (2014). Bank Marketing [Dataset]. "
    "UCI Machine Learning Repository. https://doi.org/10.24432/C5K306"
)
SQL_RENAMES = {
    "emp.var.rate": "emp_var_rate", "cons.price.idx": "cons_price_idx",
    "cons.conf.idx": "cons_conf_idx", "nr.employed": "nr_employed",
}
CATEGORY_VALUES = {
    "job": {"admin.", "blue-collar", "entrepreneur", "housemaid", "management",
            "retired", "self-employed", "services", "student", "technician",
            "unemployed", "unknown"},
    "marital": {"divorced", "married", "single", "unknown"},
    "education": {"basic.4y", "basic.6y", "basic.9y", "high.school", "illiterate",
                  "professional.course", "university.degree", "unknown"},
    "default": {"yes", "no", "unknown"},
    "housing": {"yes", "no", "unknown"},
    "loan": {"yes", "no", "unknown"},
    "contact": {"cellular", "telephone"},
    "month": {"jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep",
              "oct", "nov", "dec"},
    "day_of_week": {"mon", "tue", "wed", "thu", "fri"},
    "poutcome": {"failure", "nonexistent", "success"},
    "y": {"yes", "no"},
}
INTEGER_COLUMNS = ["age", "duration", "campaign", "pdays", "previous"]
NUMERIC_COLUMNS = [name for name in RAW_COLUMNS if name not in CATEGORY_VALUES]
MAX_ARCHIVE_BYTES = 100 * 1024 * 1024


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def validate_source(df: pd.DataFrame, expected_rows: int = 41188) -> pd.DataFrame:
    """Validate the additional-full contract and return a normalized copy.

    Tiny synthetic tests explicitly override expected_rows. Production callers
    always use 41,188, so the older and sampled variants cannot pass silently.
    """
    if len(df) != expected_rows:
        raise ValueError(f"Expected {expected_rows:,} source rows; found {len(df):,}.")
    if len(df.columns) != len(RAW_COLUMNS) or set(df.columns) != set(RAW_COLUMNS):
        missing = sorted(set(RAW_COLUMNS) - set(df.columns))
        extra = sorted(set(df.columns) - set(RAW_COLUMNS))
        raise ValueError(f"Expected exact 21-column additional-full schema: missing={missing}, "
                         f"unexpected={extra}. Parse the CSV with sep=';'.")
    result = df.copy()
    for column, allowed in CATEGORY_VALUES.items():
        values = result[column].astype("string").str.strip().str.lower()
        invalid = values.isna() | ~values.isin(allowed)
        if invalid.any():
            bad = values.loc[invalid].fillna("<missing>").unique().tolist()
            raise ValueError(f"Undocumented values in {column}: {bad[:8]}")
        result[column] = values.astype(str)
    for column in NUMERIC_COLUMNS:
        numbers = pd.to_numeric(result[column], errors="coerce")
        if not np.isfinite(numbers.to_numpy(dtype=float)).all():
            raise ValueError(f"{column} contains missing, malformed, or nonfinite numeric values")
        if column in INTEGER_COLUMNS:
            if not (numbers == np.floor(numbers)).all():
                raise ValueError(f"{column} must contain integer-valued source values")
            if ((numbers < 1) if column in {"age", "campaign"} else (numbers < 0)).any():
                raise ValueError(f"{column} contains invalid negative or zero values")
            numbers = numbers.astype("int64")
        result[column] = numbers
    return result


def _read_official_members(payload: bytes) -> dict[str, tuple[bytes, str]]:
    """Read only exact wanted basenames, including the UCI nested zip; extract nothing."""
    wanted: dict[str, tuple[bytes, str]] = {}

    def visit(content: bytes, prefix: str = "", depth: int = 0) -> None:
        if depth > 2:
            raise ValueError("Unexpectedly deep nested source archive")
        try:
            with zipfile.ZipFile(BytesIO(content)) as archive:
                for info in archive.infolist():
                    name = info.filename.replace("\\", "/")
                    parts = PurePosixPath(name)
                    relevant = parts.name in {CSV_NAME, NAMES_NAME, "bank-additional.zip"}
                    if not relevant or info.is_dir():
                        continue
                    if parts.is_absolute() or ".." in parts.parts or ":" in name:
                        raise ValueError(f"Unsafe archive member path: {name}")
                    if info.file_size > MAX_ARCHIVE_BYTES:
                        raise ValueError("Source archive member exceeds the 100 MB limit")
                    member = archive.read(info)
                    source_path = f"{prefix}{name}"
                    if parts.name == "bank-additional.zip":
                        visit(member, source_path + "::", depth + 1)
                    elif parts.name in wanted:
                        raise ValueError(f"Ambiguous duplicate {parts.name} members in archive")
                    else:
                        wanted[parts.name] = (member, source_path)
        except zipfile.BadZipFile as exc:
            raise ValueError("The supplied file is not a valid official ZIP archive") from exc

    visit(payload)
    if set(wanted) != {CSV_NAME, NAMES_NAME}:
        raise ValueError("Archive must contain bank-additional-full.csv and "
                         "bank-additional-names.txt; older and sample variants are unsupported")
    return wanted


def _download() -> bytes:
    failures = []
    for attempt in range(3):
        try:
            request = Request(SOURCE_URL, headers={"User-Agent": "bank-engagement-prototype/1.0"})
            with urlopen(request, timeout=30) as response:
                payload = response.read(MAX_ARCHIVE_BYTES + 1)
            if len(payload) > MAX_ARCHIVE_BYTES:
                raise ValueError("UCI archive exceeds the 100 MB download limit")
            return payload
        except (OSError, URLError) as exc:
            failures.append(str(exc))
            if attempt < 2:
                time.sleep(0.5 * (2 ** attempt))
    raise RuntimeError("Could not download the official UCI archive after 3 attempts. "
                       "Check HTTPS/network access or use fetch-data --archive PATH. "
                       f"Last error: {failures[-1]}")


def _atomic_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(content)
    temporary.replace(path)


def fetch_data(archive: str | Path | None = None) -> Path:
    """Cache exact source bytes and establish an immutable checksum/provenance record."""
    csv_path = RAW / CSV_NAME
    names_path = RAW / NAMES_NAME
    provenance_path = RAW / "provenance.json"
    previous = None
    if provenance_path.exists():
        previous = json.loads(provenance_path.read_text(encoding="utf-8"))
    if archive is None and csv_path.exists() and names_path.exists() and previous:
        if sha256_bytes(csv_path.read_bytes()) != previous["source_sha256"]:
            raise ValueError("Cached source checksum mismatch; restore the verified source "
                             "or explicitly establish a new data version")
        if sha256_bytes(names_path.read_bytes()) != previous["readme_sha256"]:
            raise ValueError("Cached source readme checksum mismatch")
        validate_source(pd.read_csv(csv_path, sep=";"))
        return csv_path
    payload = Path(archive).read_bytes() if archive is not None else _download()
    members = _read_official_members(payload)
    csv_bytes, member_path = members[CSV_NAME]
    names_bytes, names_member = members[NAMES_NAME]
    validate_source(pd.read_csv(BytesIO(csv_bytes), sep=";"))
    checksum = sha256_bytes(csv_bytes)
    if previous and checksum != previous["source_sha256"]:
        raise ValueError("Downloaded source checksum differs from the established data version; "
                         "refusing to overwrite verified provenance")
    if previous and sha256_bytes(names_bytes) != previous["readme_sha256"]:
        raise ValueError("Downloaded source readme differs from the established data version")
    provenance = previous or {
        "source_url": SOURCE_URL,
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "retrieval_method": "local_archive" if archive else "https_download",
        "archive_member": member_path,
        "readme_member": names_member,
        "source_bytes": len(csv_bytes),
        "source_sha256": checksum,
        "readme_sha256": sha256_bytes(names_bytes),
        "archive_sha256": sha256_bytes(payload),
        "citation": CITATION,
        "license": "CC BY 4.0",
    }
    RAW.mkdir(parents=True, exist_ok=True)
    _atomic_bytes(csv_path, csv_bytes)
    _atomic_bytes(names_path, names_bytes)
    _atomic_bytes(provenance_path, (json.dumps(provenance, indent=2) + "\n").encode())
    return csv_path


def split_mapping(n_rows: int) -> pd.DataFrame:
    """Create floor(70%)/floor(15%)/remainder partitions without sorting or shuffling."""
    if n_rows < 1:
        raise ValueError("Cannot split an empty source")
    n_train, n_validation = int(0.70 * n_rows), int(0.15 * n_rows)
    return pd.DataFrame({
        "record_id": np.arange(1, n_rows + 1, dtype=int),
        "split": (["train"] * n_train + ["validation"] * n_validation
                  + ["test"] * (n_rows - n_train - n_validation)),
    })


def _quoted(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _schema(connection: sqlite3.Connection) -> None:
    raw_fields = ", ".join(
        f"{_quoted(column)} {'TEXT' if column in CATEGORY_VALUES else 'REAL'} NOT NULL"
        for column in RAW_COLUMNS
    )
    clean_fields = ", ".join(
        f"{_quoted(SQL_RENAMES.get(column, column))} "
        f"{'INTEGER' if column == 'y' or column in INTEGER_COLUMNS else 'TEXT' if column in CATEGORY_VALUES else 'REAL'} "
        "NOT NULL"
        for column in RAW_COLUMNS
    )
    connection.executescript(f"""
        CREATE TABLE raw_campaigns (record_id INTEGER PRIMARY KEY, {raw_fields});
        CREATE TABLE clean_campaigns (
            record_id INTEGER PRIMARY KEY REFERENCES raw_campaigns(record_id),
            {clean_fields},
            age_group TEXT NOT NULL CHECK(age_group IN ('under 30','30–44','45–59','60+')),
            CHECK(y IN (0,1)), CHECK(age > 0), CHECK(previous >= 0), CHECK(pdays >= 0)
        );
        CREATE TABLE dataset_splits (
            record_id INTEGER PRIMARY KEY REFERENCES clean_campaigns(record_id),
            split TEXT NOT NULL CHECK(split IN ('train','validation','test'))
        );
        CREATE INDEX idx_dataset_splits_split ON dataset_splits(split);
        CREATE TABLE model_runs (
            model_id TEXT PRIMARY KEY, manifest_path TEXT NOT NULL,
            source_sha256 TEXT NOT NULL, created_at TEXT NOT NULL
        );
        CREATE TABLE prospect_scores (
            model_id TEXT NOT NULL REFERENCES model_runs(model_id),
            record_id INTEGER NOT NULL REFERENCES clean_campaigns(record_id),
            score REAL NOT NULL CHECK(score >= 0 AND score <= 1),
            rank INTEGER NOT NULL CHECK(rank > 0),
            split TEXT NOT NULL CHECK(split IN ('train','validation','test')),
            PRIMARY KEY(model_id,record_id)
        );
        CREATE INDEX idx_scores_model_split ON prospect_scores(model_id,split);
        CREATE TABLE engagement_simulated (
            record_id INTEGER PRIMARY KEY REFERENCES clean_campaigns(record_id),
            simulated_opt_out INTEGER NOT NULL CHECK(simulated_opt_out IN (0,1)),
            simulated_contact_permission INTEGER NOT NULL
                CHECK(simulated_contact_permission IN (0,1)),
            simulated_recent_contact INTEGER NOT NULL CHECK(simulated_recent_contact IN (0,1)),
            provenance TEXT NOT NULL CHECK(provenance = 'simulated')
        );
    """)


def _simulation(record_ids: pd.Series) -> pd.DataFrame:
    """Store the shared public simulation policy with SQLite integer types."""
    from .engagement import FLAG_COLUMNS, simulate_engagement

    result = simulate_engagement(record_ids)
    result["record_id"] = result["record_id"].astype(int)
    for field in FLAG_COLUMNS:
        result[field] = result[field].astype(int)
    return result


def prepare_data(csv_path: str | Path | None = None, db_path: str | Path = DB) -> dict:
    """Validate and atomically rebuild all canonical data tables; preserve every source row."""
    source_path = Path(csv_path) if csv_path is not None else RAW / CSV_NAME
    if not source_path.exists():
        raise FileNotFoundError("Source CSV is missing; run fetch-data before prepare")
    source_bytes = source_path.read_bytes()
    source_sha256 = sha256_bytes(source_bytes)
    provenance_path = source_path.parent / "provenance.json"
    if not provenance_path.exists():
        raise ValueError("Source CSV requires an adjacent provenance.json. Use fetch-data "
                         "--archive PATH to establish verified source provenance.")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if source_sha256 != provenance.get("source_sha256"):
        raise ValueError("Source CSV checksum does not match its provenance.json")
    raw = pd.read_csv(BytesIO(source_bytes), sep=";")
    normalized = validate_source(raw)
    splits = split_mapping(len(normalized))
    clean = normalized.copy()
    clean["y"] = clean["y"].map({"no": 0, "yes": 1}).astype(int)
    clean.insert(0, "record_id", splits["record_id"])
    clean["age_group"] = pd.cut(
        clean["age"], bins=[0, 29, 44, 59, np.inf],
        labels=["under 30", "30–44", "45–59", "60+"],
    ).astype(str)
    clean = clean.rename(columns=SQL_RENAMES)
    raw.insert(0, "record_id", splits["record_id"])
    fingerprints = pd.util.hash_pandas_object(raw.drop(columns="record_id"), index=False)
    duplicate_split_counts = pd.DataFrame({"fingerprint": fingerprints, "split": splits.split})
    cross_splits = duplicate_split_counts.groupby("fingerprint")["split"].nunique()
    split_csv = splits.to_csv(index=False, lineterminator="\n")
    quality = {
        "source_sha256": source_sha256,
        "source_rows": len(raw), "rows": len(raw),
        "source_url": provenance.get("source_url", SOURCE_URL),
        "split_sha256": sha256_bytes(split_csv.encode("utf-8")),
        "split_counts": {str(k): int(v) for k, v in splits.split.value_counts().items()},
        "split_ranges": {name: [int(group.record_id.min()), int(group.record_id.max())]
                         for name, group in splits.groupby("split", sort=False)},
        "duplicate_rows": int(fingerprints.duplicated(keep=False).sum()),
        "duplicate_extra_rows": int(fingerprints.duplicated().sum()),
        "cross_split_duplicate_fingerprints": int((cross_splits > 1).sum()),
        "unknown_counts": {column: int((normalized[column] == "unknown").sum())
                           for column in CATEGORY_VALUES if column != "y"},
        "prior_history_contradictions": {
            "previous_zero_but_pdays_not_999": int(
                ((clean.previous == 0) & (clean.pdays != 999)).sum()),
            "previous_positive_but_poutcome_nonexistent": int(
                ((clean.previous > 0) & (clean.poutcome == "nonexistent")).sum()),
            "previous_zero_but_poutcome_not_nonexistent": int(
                ((clean.previous == 0) & (clean.poutcome != "nonexistent")).sum()),
            "pdays_999_but_poutcome_success": int(
                ((clean.pdays == 999) & (clean.poutcome == "success")).sum()),
        },
        "sql_renames": SQL_RENAMES,
        "notes": ["All source rows, duplicates, and supplied order preserved.",
                  "IDs identify records, not unique customers.",
                  "Quality checks deliberately omit held-out subscription rates."],
    }
    database_path = Path(db_path)
    database_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".sqlite", dir=database_path.parent,
                                     delete=False) as handle:
        temporary_db = Path(handle.name)
    try:
        with connect(temporary_db) as connection:
            _schema(connection)
            raw.to_sql("raw_campaigns", connection, if_exists="append", index=False)
            clean.to_sql("clean_campaigns", connection, if_exists="append", index=False)
            splits.to_sql("dataset_splits", connection, if_exists="append", index=False)
            _simulation(splits.record_id).to_sql(
                "engagement_simulated", connection, if_exists="append", index=False,
            )
            if connection.execute("PRAGMA foreign_key_check").fetchall():
                raise ValueError("Canonical table foreign-key validation failed")
            if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                raise ValueError("SQLite integrity validation failed")
        temporary_db.replace(database_path)
    finally:
        temporary_db.unlink(missing_ok=True)
    _atomic_bytes(database_path.parent / "quality.json",
                  (json.dumps(quality, indent=2) + "\n").encode())
    _atomic_bytes(database_path.parent / "dataset_splits.csv", split_csv.encode("utf-8"))
    return quality
