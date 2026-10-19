"""Deterministic demonstration flags, kept independent from propensity modeling.

These flags are simulated. They are neither observed consent nor a production
bank policy. An incomplete simulation always suppresses outreach.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable

import pandas as pd


FLAG_COLUMNS = [
    "simulated_opt_out", "simulated_contact_permission", "simulated_recent_contact"
]
ENGAGEMENT_COLUMNS = ["record_id", *FLAG_COLUMNS, "provenance"]


def _normalize_ids(values: pd.Series, source: str) -> pd.Series:
    """Keep identifiers as text, including significant leading zeros."""
    ids = values.map(lambda value: "" if pd.isna(value) else str(value).strip())
    blank = ids.eq("")
    if blank.any():
        rows = [str(i + 1) for i in range(len(ids)) if blank.iloc[i]][:10]
        raise ValueError(f"{source}: blank record_id at data rows {', '.join(rows)}")
    duplicate = ids.duplicated(keep=False)
    if duplicate.any():
        examples = ids[duplicate].drop_duplicates().tolist()[:10]
        raise ValueError(f"{source}: duplicate record_id after trimming: {examples}")
    return ids.astype("string")


def simulate_engagement(record_ids: Iterable[object]) -> pd.DataFrame:
    """Derive reproducible flags solely from normalized record IDs and seed 42."""
    ids = _normalize_ids(pd.Series(list(record_ids), dtype="object"), "Simulation")
    result = pd.DataFrame({"record_id": ids})
    limits = {
        "simulated_opt_out": 10,
        "simulated_contact_permission": 90,
        "simulated_recent_contact": 10,
    }
    for field, limit in limits.items():
        result[field] = ids.map(
            lambda record_id: int.from_bytes(
                hashlib.sha256(f"42:{field}:{record_id}".encode("utf-8")).digest(), "big"
            ) % 100 < limit
        ).astype("boolean")
    result["provenance"] = "simulated"
    return result[ENGAGEMENT_COLUMNS]


def _parse_flag(value: object, column: str, row: int) -> object:
    if pd.isna(value) or (isinstance(value, str) and not value.strip()):
        return pd.NA
    token = str(value).strip().lower()
    if token in {"true", "1"}:
        return True
    if token in {"false", "0"}:
        return False
    raise ValueError(
        f"Engagement: {column} at data row {row} must be true/false or 1/0; got {value!r}"
    )


def validate_engagement(df: pd.DataFrame) -> pd.DataFrame:
    """Validate simulated input without converting the string 'false' to true.

    Absent flag columns and blank flag cells remain nullable and later suppress
    outreach. IDs and the explicit simulation provenance are mandatory.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Engagement input must be a pandas DataFrame")
    if df.columns.duplicated().any():
        raise ValueError("Engagement input has duplicate column names")
    unexpected = sorted(set(df.columns) - set(ENGAGEMENT_COLUMNS))
    if unexpected:
        raise ValueError(f"Unexpected engagement columns: {unexpected}")
    required = {"record_id", "provenance"} - set(df.columns)
    if required:
        raise ValueError(f"Missing required engagement columns: {sorted(required)}")
    result = df.copy().reset_index(drop=True)
    result["record_id"] = _normalize_ids(result["record_id"], "Engagement")
    provenance = result["provenance"].astype("string").str.strip().str.lower()
    invalid = provenance.isna() | provenance.ne("simulated").fillna(True)
    if invalid.any():
        rows = [str(i + 1) for i in range(len(result)) if invalid.iloc[i]][:10]
        raise ValueError(
            "Engagement provenance must be 'simulated' at every row; "
            f"invalid data rows: {', '.join(rows)}"
        )
    for column in FLAG_COLUMNS:
        values = result[column] if column in result else pd.Series(pd.NA, index=result.index)
        result[column] = pd.array(
            [_parse_flag(value, column, i + 1) for i, value in enumerate(values)],
            dtype="boolean",
        )
    result["provenance"] = "simulated"
    return result[ENGAGEMENT_COLUMNS]


def apply_engagement(records: pd.DataFrame, flags: pd.DataFrame | None = None) -> pd.DataFrame:
    """Enrich records with simulation eligibility; never modify model scores.

    A supplied table may contain flags for a larger replay pool. Extra IDs are
    ignored; absent IDs or flag cells are suppressed rather than simulated.
    """
    if "record_id" not in records:
        raise ValueError("Records require record_id to apply engagement simulation")
    result = records.copy().reset_index(drop=True)
    result["record_id"] = _normalize_ids(result["record_id"], "Records")
    validated = (
        simulate_engagement(result["record_id"])
        if flags is None else validate_engagement(flags)
    )
    aligned = validated.set_index("record_id").reindex(result["record_id"])
    aligned = aligned.reset_index(drop=True)
    failure_masks = [
        aligned["simulated_opt_out"].eq(True).fillna(False),
        aligned["simulated_contact_permission"].eq(False).fillna(False),
        aligned["simulated_recent_contact"].eq(True).fillna(False),
        aligned[FLAG_COLUMNS].isna().any(axis=1),
    ]
    reason_labels = [
        "opted_out", "no_simulated_permission", "simulated_recent_contact",
        "missing_simulation_flags",
    ]
    reasons = [
        ";".join(label for label, failed in zip(reason_labels, failures, strict=True) if failed)
        for failures in zip(*failure_masks, strict=True)
    ]
    for column in FLAG_COLUMNS:
        result[column] = pd.array(aligned[column], dtype="boolean")
    result["eligible"] = pd.Series([not reason for reason in reasons], dtype=bool)
    result["suppression_reasons"] = reasons
    result["provenance"] = "simulated"
    result.attrs["engagement_note"] = (
        "Deterministic simulated engagement; no observed consent or real outreach."
        if flags is None else "User-supplied simulated engagement; incomplete flags suppress outreach."
    )
    return result
