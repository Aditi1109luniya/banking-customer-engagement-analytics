"""SQLite storage and named, parameterized historical analysis queries."""

from pathlib import Path
import sqlite3

import pandas as pd

from .constants import DB, ROOT

ANALYSES = {
    "overview", "by_job", "by_age_group", "by_poutcome", "by_prior_contact",
    "by_month", "by_day_of_week", "by_contact", "by_education", "by_previous",
    "by_recency", "by_default", "by_housing", "by_loan", "unknown_coverage",
    "split_overview", "order_blocks", "score_deciles",
}
VALID_SPLITS = {"train", "validation", "test", "all"}


def connect(db_path: str | Path = DB) -> sqlite3.Connection:
    """Open SQLite with foreign-key enforcement for every caller."""
    connection = sqlite3.connect(Path(db_path))
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def _check_split(split: str) -> None:
    if split not in VALID_SPLITS:
        raise ValueError(f"split must be one of {sorted(VALID_SPLITS)}; received {split!r}")


def load_records(split: str = "train", db_path: str | Path = DB) -> pd.DataFrame:
    """Return normalized records in immutable source order, including split metadata.

    Development-stage access control belongs to the CLI. The released application
    needs these same functions to inspect the frozen historical demonstration.
    """
    _check_split(split)
    with connect(db_path) as connection:
        return pd.read_sql_query(
            "SELECT c.*, s.split FROM clean_campaigns c "
            "JOIN dataset_splits s USING(record_id) "
            "WHERE (:split = 'all' OR s.split = :split) ORDER BY c.record_id",
            connection, params={"split": split},
        )


def query_analysis(
    name: str, split: str = "train", db_path: str | Path = DB,
) -> pd.DataFrame:
    """Run a vetted SQL file; never interpolate user SQL or identifiers.

    Every segment includes its count and numerator. Support annotations belong
    to the presentation layer, which should flag denominators below 30.
    """
    _check_split(split)
    if name not in ANALYSES:
        raise ValueError(f"Unknown analysis {name!r}; choose from {sorted(ANALYSES)}")
    sql_path = ROOT / "sql" / f"{name}.sql"
    with connect(db_path) as connection:
        return pd.read_sql_query(
            sql_path.read_text(encoding="utf-8"), connection, params={"split": split},
        )
