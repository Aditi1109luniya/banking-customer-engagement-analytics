"""Strict pre-contact batch scoring and safe, transient outreach exports."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .constants import CATEGORICAL, EXCLUDED, FEATURES, NUMERIC
from .engagement import _normalize_ids, apply_engagement
from .evaluation import budget_count, ranked_order


SCORE_COLUMNS = [
    "record_id", "model_id", "model_score", "global_rank", "eligible",
    "suppression_reasons", "eligible_rank", "selected", "capacity_mode",
    "requested_capacity", "validation_cutoff", "provenance",
    "recorded_previous_outcome", "previous_contacts", "no_previous_contact",
]

KNOWN_CATEGORIES = {
    "job": {"admin.", "blue-collar", "entrepreneur", "housemaid", "management", "retired",
            "self-employed", "services", "student", "technician", "unemployed", "unknown"},
    "marital": {"divorced", "married", "single", "unknown"},
    "education": {"basic.4y", "basic.6y", "basic.9y", "high.school", "illiterate",
                  "professional.course", "university.degree", "unknown"},
    "default": {"no", "yes", "unknown"},
    "housing": {"no", "yes", "unknown"},
    "loan": {"no", "yes", "unknown"},
    "poutcome": {"failure", "nonexistent", "success", "unknown"},
}


def validate_prospects(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Return only record_id plus the ten allowed raw inputs and diagnostics.

    Validation is batch-atomic: malformed rows cause an error, never a silent
    row drop. Target and known excluded source fields are ignored explicitly.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Prospects input must be a pandas DataFrame")
    if df.columns.duplicated().any():
        raise ValueError("Prospects input has duplicate column names")
    allowed = {"record_id", *FEATURES}
    missing = sorted(allowed - set(df.columns))
    unexpected = sorted(set(df.columns) - allowed - set(EXCLUDED) - {"y"})
    if missing or unexpected:
        details = []
        if missing:
            details.append(f"Missing required prospect columns: {missing}")
        if unexpected:
            details.append(f"Unexpected prospect columns (check spelling): {unexpected}")
        raise ValueError("; ".join(details))
    result = df[["record_id", *FEATURES]].copy().reset_index(drop=True)
    result["record_id"] = _normalize_ids(result["record_id"], "Prospects")
    warnings = []
    ignored = [column for column in [*EXCLUDED, "y"] if column in df]
    if ignored:
        warnings.append(f"Ignored excluded/non-feature columns: {', '.join(ignored)}")
    errors = []
    for column in NUMERIC:
        values = pd.to_numeric(result[column], errors="coerce")
        array = values.to_numpy(dtype=float, na_value=np.nan)
        boolean_values = result[column].map(lambda value: isinstance(value, (bool, np.bool_)))
        invalid = ~np.isfinite(array) | boolean_values.to_numpy()
        invalid |= np.isfinite(array) & (array != np.floor(array))
        invalid |= array <= 0 if column == "age" else array < 0
        # The estimator consumes float64; integer precision must not be silently lost.
        invalid |= np.abs(array) > 2**53 - 1
        if invalid.any():
            rows = [str(i + 1) for i in np.flatnonzero(invalid)][:10]
            bound = "positive" if column == "age" else "nonnegative"
            errors.append(
                f"{column}: finite, exactly representable {bound} integer required at data rows "
                f"{', '.join(rows)}"
            )
        else:
            result[column] = array.astype(np.int64)
    if errors:
        raise ValueError("Invalid prospects batch. " + "; ".join(errors))
    for column in CATEGORICAL:
        values = result[column].astype("string").str.strip().str.lower()
        blank = values.isna() | values.eq("").fillna(False)
        if blank.any():
            warnings.append(f"{column}: {int(blank.sum())} blank value(s) normalized to unknown")
        values = values.mask(blank, "unknown")
        unseen = sorted(set(values) - KNOWN_CATEGORIES[column])
        if unseen:
            warnings.append(
                f"{column}: unseen source categories {unseen}; encoder will handle unknown values"
            )
        result[column] = values.astype(str)
    return result, warnings


def _encoder_warnings(pipeline: object, records: pd.DataFrame) -> list[str]:
    """Inspect fitted encoders without fitting or modifying them."""
    known: dict[str, set[object]] = {}

    def visit(transformer: object, columns: list[str] | None = None) -> None:
        if hasattr(transformer, "categories_"):
            names = getattr(transformer, "feature_names_in_", columns)
            if names is not None and len(names) == len(transformer.categories_):
                for name, categories in zip(names, transformer.categories_, strict=True):
                    if name in CATEGORICAL:
                        known[name] = set(categories)
        if hasattr(transformer, "transformers_"):
            for _, child, selected in transformer.transformers_:
                if isinstance(selected, (list, tuple, np.ndarray, pd.Index)):
                    visit(child, list(selected))
        if hasattr(transformer, "steps"):
            for _, child in transformer.steps:
                visit(child, columns)

    visit(pipeline)
    warnings = []
    for column, categories in known.items():
        unseen = sorted(set(records[column]) - categories)
        if unseen:
            warnings.append(f"{column}: categories unseen during training: {unseen}")
    return warnings


def _positive_scores(pipeline: object, inputs: pd.DataFrame) -> np.ndarray:
    classes = np.asarray(getattr(pipeline, "classes_", []))
    positive = [
        i for i, value in enumerate(classes)
        if str(value).strip().lower() in {"1", "1.0", "yes", "true"}
    ]
    if len(classes) != 2 or len(positive) != 1:
        raise ValueError(f"Expected a binary model with an identifiable positive class; got {classes}")
    probabilities = np.asarray(pipeline.predict_proba(inputs), dtype=float)
    if probabilities.shape != (len(inputs), len(classes)):
        raise ValueError("Model predict_proba returned an incompatible shape")
    scores = probabilities[:, positive[0]]
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("Model scores must be finite and between 0 and 1")
    return scores


def score_prospects(
    pipeline: object,
    df: pd.DataFrame,
    model_id: str,
    flags: pd.DataFrame | None = None,
    capacity: int | None = None,
    fraction: float = 0.2,
    min_score: float | None = None,
) -> pd.DataFrame:
    """Score allowed inputs, suppress ineligible records, then apply a contact budget.

    Returns a safe outreach table sorted by global rank. No uploads, labels, or
    source exclusions are persisted. Diagnostics are in ``result.attrs['warnings']``.
    ``min_score`` is the previously frozen validation cutoff, when supplied.
    """
    records, warnings = validate_prospects(df)
    if not str(model_id).strip():
        raise ValueError("model_id must be nonblank")
    if isinstance(fraction, (bool, np.bool_)) or not math.isfinite(float(fraction)):
        raise ValueError("fraction must be a finite number between 0 and 1")
    if not 0 <= float(fraction) <= 1:
        raise ValueError("fraction must be between 0 and 1")
    if capacity is not None and (
        isinstance(capacity, (bool, np.bool_))
        or not isinstance(capacity, (int, np.integer)) or capacity < 0
    ):
        raise ValueError("capacity must be a nonnegative integer")
    if min_score is not None and (
        isinstance(min_score, (bool, np.bool_)) or not math.isfinite(float(min_score))
        or not 0 <= float(min_score) <= 1
    ):
        raise ValueError("min_score must be a finite number between 0 and 1")
    warnings.extend(_encoder_warnings(pipeline, records))
    scores = (
        _positive_scores(pipeline, records[FEATURES]) if len(records) else np.array([], dtype=float)
    )
    order = ranked_order(records["record_id"].tolist(), scores)
    ranks = np.empty(len(records), dtype=np.int64)
    ranks[order] = np.arange(1, len(records) + 1)
    result = pd.DataFrame({
        "record_id": records["record_id"], "model_id": str(model_id).strip(),
        "model_score": scores, "global_rank": ranks,
        "recorded_previous_outcome": records["poutcome"],
        "previous_contacts": records["previous"],
        "no_previous_contact": records["pdays"].eq(999),
    })
    result = apply_engagement(result, flags)
    eligible_positions = [int(i) for i in order if result.at[int(i), "eligible"]]
    eligible_rank = pd.array([pd.NA] * len(result), dtype="Int64")
    for rank, position in enumerate(eligible_positions, start=1):
        eligible_rank[position] = rank
    result["eligible_rank"] = eligible_rank
    k = budget_count(len(eligible_positions), fraction=float(fraction), capacity=capacity)
    above_cutoff = [
        i for i in eligible_positions if min_score is None or scores[i] >= float(min_score)
    ]
    result["selected"] = False
    result.loc[above_cutoff[:k], "selected"] = True
    result["capacity_mode"] = (
        "cutoff_plus_capacity" if min_score is not None
        else "capacity" if capacity is not None else "fraction"
    )
    result["requested_capacity"] = capacity if capacity is not None else k
    result["validation_cutoff"] = pd.array(
        [pd.NA if min_score is None else float(min_score)] * len(result), dtype="Float64"
    )
    result = result[SCORE_COLUMNS].sort_values("global_rank").reset_index(drop=True)
    result.attrs["warnings"] = warnings
    result.attrs["population"] = {
        "valid_records": len(records), "eligible_records": len(eligible_positions),
        "suppressed_records": len(records) - len(eligible_positions),
        "selected_records": int(result["selected"].sum()), "effective_capacity": k,
    }
    result.attrs["engagement_note"] = "Engagement flags are simulated, not observed consent."
    return result


def export_csv(df: pd.DataFrame) -> bytes:
    """Export only the outreach contract, with no outcomes or contact details."""
    missing = set(SCORE_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Scored export missing contract fields: {sorted(missing)}")
    return df[SCORE_COLUMNS].to_csv(index=False).encode("utf-8")
