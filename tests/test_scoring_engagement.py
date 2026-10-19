"""Regression tests for pre-contact validation and simulation-aware ranking."""

import hashlib
from io import BytesIO

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from bank_engagement.constants import CATEGORICAL, FEATURES, NUMERIC
from bank_engagement.engagement import (
    FLAG_COLUMNS,
    apply_engagement,
    simulate_engagement,
    validate_engagement,
)
from bank_engagement.scoring import (
    SCORE_COLUMNS,
    export_csv,
    score_prospects,
    validate_prospects,
)


@pytest.fixture
def prospects():
    return pd.DataFrame({
        "record_id": [f"p{i:02d}" for i in range(12)],
        "age": [20 + 3 * i for i in range(12)],
        "job": ["student", "admin.", "technician"] * 4,
        "marital": ["single", "married"] * 6,
        "education": ["high.school", "university.degree"] * 6,
        "default": ["no"] * 12,
        "housing": ["no", "yes"] * 6,
        "loan": ["no"] * 12,
        "previous": list(range(12)),
        "pdays": [999, 999, 999, 10, 20, 30, 40, 50, 60, 70, 80, 90],
        "poutcome": ["nonexistent"] * 3 + ["failure"] * 3 + ["success"] * 6,
    })


@pytest.fixture
def model(prospects):
    preprocessing = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ], remainder="drop")
    model = Pipeline([("preprocessing", preprocessing), ("classifier", LogisticRegression())])
    model.fit(prospects[FEATURES], [0] * 6 + [1] * 6)
    return model


def permissive_flags(prospects):
    return pd.DataFrame({
        "record_id": prospects["record_id"],
        "simulated_opt_out": False,
        "simulated_contact_permission": True,
        "simulated_recent_contact": False,
        "provenance": "simulated",
    })


def test_simulation_exact_hash_policy_and_order_independence():
    ids = [" 001 ", "alpha", "2"]
    actual = simulate_engagement(ids)
    assert actual["record_id"].tolist() == ["001", "alpha", "2"]
    for field, limit in zip(FLAG_COLUMNS, [10, 90, 10], strict=True):
        for record_id, value in zip(actual["record_id"], actual[field], strict=True):
            bucket = int(hashlib.sha256(f"42:{field}:{record_id}".encode()).hexdigest(), 16) % 100
            assert value == (bucket < limit)
    reordered = simulate_engagement(reversed(ids))
    pd.testing.assert_frame_equal(
        actual.sort_values("record_id").reset_index(drop=True),
        reordered.sort_values("record_id").reset_index(drop=True),
    )
    assert actual["provenance"].eq("simulated").all()


def test_flags_strict_boolean_parser_and_missing_suppression(prospects):
    flags = permissive_flags(prospects).iloc[:3].copy()
    flags["simulated_opt_out"] = [" FALSE ", "0", "true"]
    flags["simulated_contact_permission"] = ["TRUE", "1", "false"]
    flags["simulated_recent_contact"] = ["0", "", "true"]
    validated = validate_engagement(flags)
    assert validated["simulated_opt_out"].tolist() == [False, False, True]
    result = apply_engagement(prospects, validated)
    assert result.loc[0, "eligible"]
    assert result.loc[1, "suppression_reasons"] == "missing_simulation_flags"
    assert result.loc[2, "suppression_reasons"] == (
        "opted_out;no_simulated_permission;simulated_recent_contact"
    )
    assert result.loc[3:, "suppression_reasons"].eq("missing_simulation_flags").all()
    assert not result.loc[3:, "eligible"].any()


@pytest.mark.parametrize("value", ["no", "yes", "falsey", "2", "1.0", 2, 1.0])
def test_flags_reject_non_boolean_tokens(prospects, value):
    flags = permissive_flags(prospects).iloc[:1].copy()
    flags["simulated_opt_out"] = value
    with pytest.raises(ValueError, match="true/false or 1/0"):
        validate_engagement(flags)


def test_flags_reject_observed_provenance_and_ambiguous_ids(prospects):
    flags = permissive_flags(prospects)
    flags.loc[0, "provenance"] = "observed"
    with pytest.raises(ValueError, match="provenance"):
        validate_engagement(flags)
    flags = permissive_flags(prospects)
    flags.loc[1, "record_id"] = " p00 "
    with pytest.raises(ValueError, match="duplicate record_id"):
        validate_engagement(flags)
    flags = permissive_flags(prospects).drop(columns="simulated_opt_out")
    result = apply_engagement(prospects, flags)
    assert not result["eligible"].any()
    assert result["suppression_reasons"].eq("missing_simulation_flags").all()


def test_validate_ids_preserves_zeros_and_rejects_blanks_duplicates(prospects):
    prospects.loc[0, "record_id"] = " 001 "
    clean, _ = validate_prospects(prospects)
    assert clean.loc[0, "record_id"] == "001"
    prospects.loc[1, "record_id"] = "001"
    with pytest.raises(ValueError, match="duplicate record_id"):
        validate_prospects(prospects)
    prospects.loc[1, "record_id"] = "  "
    with pytest.raises(ValueError, match="blank record_id"):
        validate_prospects(prospects)


@pytest.mark.parametrize("column,value", [
    ("age", 0), ("age", -10), ("age", "n/a"), ("age", True),
    ("previous", -1), ("previous", 2.5), ("pdays", np.inf), ("pdays", None),
    ("pdays", -1), ("age", 2**54),
])
def test_validate_rejects_bad_numeric_batch_with_location(prospects, column, value):
    prospects[column] = prospects[column].astype(object)
    prospects.loc[3, column] = value
    with pytest.raises(ValueError, match=rf"{column}:.*data rows 4"):
        validate_prospects(prospects)


def test_validate_schema_and_unknown_categories(prospects):
    with pytest.raises(ValueError, match="Missing required prospect columns"):
        validate_prospects(prospects.drop(columns="age"))
    with pytest.raises(ValueError, match="Unexpected prospect columns"):
        validate_prospects(prospects.assign(agge=20))
    prospects.loc[0, "job"] = "new-profession"
    prospects.loc[1, "education"] = "   "
    clean, warnings = validate_prospects(prospects.assign(duration=999, y="yes"))
    assert list(clean) == ["record_id", *FEATURES]
    assert clean.loc[1, "education"] == "unknown"
    assert any("unseen source categories" in warning for warning in warnings)
    assert any("normalized to unknown" in warning for warning in warnings)
    assert any("duration, y" in warning for warning in warnings)


def test_scores_ignore_leakage_fields_and_engagement(model, prospects):
    flags = permissive_flags(prospects)
    baseline = score_prospects(model, prospects, "test-model", flags=flags)
    altered_flags = flags.copy()
    altered_flags["simulated_opt_out"] = True
    changed = score_prospects(
        model, prospects.assign(y="yes", duration=999999, campaign=900),
        "test-model", flags=altered_flags,
    )
    np.testing.assert_allclose(baseline["model_score"], changed["model_score"], atol=0, rtol=0)
    assert baseline["eligible"].all()
    assert not changed["eligible"].any()
    assert not changed["selected"].any()
    assert changed["eligible_rank"].isna().all()
    assert not {"y", "duration", "campaign"}.intersection(changed)


def test_opt_out_suppression_precedes_capacity(model, prospects):
    flags = permissive_flags(prospects)
    unsuppressed = score_prospects(model, prospects, "model", flags, capacity=2)
    top_id = unsuppressed.iloc[0]["record_id"]
    flags.loc[flags["record_id"].eq(top_id), "simulated_opt_out"] = True
    actual = score_prospects(model, prospects, "model", flags, capacity=2)
    assert int(actual["selected"].sum()) == 2
    assert not actual.loc[actual["record_id"].eq(top_id), "selected"].item()
    assert actual.loc[actual["selected"], "eligible_rank"].tolist() == [1, 2]
    assert actual.attrs["population"]["eligible_records"] == 11
    fractional = score_prospects(model, prospects, "model", flags, fraction=0.2)
    assert fractional["selected"].sum() == 3  # ceil(20% of 11 eligible records)


def test_capacity_zero_oversized_cutoff_and_unseen_training_category(model, prospects):
    flags = permissive_flags(prospects)
    assert not score_prospects(model, prospects, "model", flags, capacity=0)["selected"].any()
    assert score_prospects(model, prospects, "model", flags, capacity=99)["selected"].all()
    assert not score_prospects(
        model, prospects, "model", flags, capacity=99, min_score=1.0
    )["selected"].any()
    prospects.loc[0, "job"] = "retired"  # valid UCI value absent from training fixture
    result = score_prospects(model, prospects, "model", flags)
    assert np.isfinite(result["model_score"]).all()
    assert any("unseen during training" in warning for warning in result.attrs["warnings"])


class ConstantClassifier:
    classes_ = np.array([0, 1])

    def predict_proba(self, inputs):
        assert list(inputs) == FEATURES
        return np.tile([0.7, 0.3], (len(inputs), 1))


def test_ties_stable_after_reordering_and_exact_positive_class_mapping(prospects):
    classifier = ConstantClassifier()
    flags = permissive_flags(prospects)
    first = score_prospects(classifier, prospects, "ties", flags, capacity=4)
    second = score_prospects(classifier, prospects.iloc[::-1], "ties", flags, capacity=4)
    assert first["record_id"].tolist() == second["record_id"].tolist()
    assert first["selected"].tolist() == second["selected"].tolist()

    class ReversedClassifier(ConstantClassifier):
        classes_ = np.array([1, 0])

    reversed_result = score_prospects(ReversedClassifier(), prospects, "reverse", flags)
    np.testing.assert_allclose(reversed_result["model_score"], 0.7)


@pytest.mark.parametrize("value", [np.nan, np.inf, -0.01, 1.01])
def test_invalid_model_probabilities_rejected(prospects, value):
    class InvalidClassifier(ConstantClassifier):
        def predict_proba(self, inputs):
            return np.tile([0.5, value], (len(inputs), 1))

    with pytest.raises(ValueError, match="finite and between 0 and 1"):
        score_prospects(InvalidClassifier(), prospects, "invalid")


@pytest.mark.parametrize("options", [
    {"capacity": -1}, {"capacity": 1.5}, {"capacity": True},
    {"fraction": -0.1}, {"fraction": 1.1}, {"fraction": np.nan},
    {"min_score": np.nan}, {"min_score": -0.1}, {"min_score": 1.1},
])
def test_invalid_capacity_and_cutoffs_rejected(model, prospects, options):
    with pytest.raises(ValueError):
        score_prospects(model, prospects, "invalid", **options)


def test_empty_population_and_export_schema(model, prospects):
    empty = score_prospects(model, prospects.iloc[:0], "empty")
    assert list(empty) == SCORE_COLUMNS
    assert empty.empty
    assert empty.attrs["population"]["effective_capacity"] == 0
    result = score_prospects(model, prospects, "export", permissive_flags(prospects))
    exported = pd.read_csv(BytesIO(export_csv(result.assign(y="yes", duration=999))))
    assert exported.columns.tolist() == SCORE_COLUMNS
    assert exported["provenance"].eq("simulated").all()
    assert result.loc[~result["eligible"], "eligible_rank"].isna().all()
