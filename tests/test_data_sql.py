"""No-network fixtures exercising provenance, ordered storage, and SQL denominators."""

import hashlib
from io import BytesIO
import json
import sqlite3
import zipfile

import numpy as np
import pandas as pd
import pytest

from bank_engagement import data
from bank_engagement.constants import RAW_COLUMNS
from bank_engagement.database import ANALYSES, connect, load_records, query_analysis


@pytest.fixture
def source():
    records = []
    for i in range(20):
        records.append({
            "age": [25, 35, 55, 65][i % 4],
            "job": ["admin.", "unknown"][i % 2], "marital": "married",
            "education": "university.degree", "default": "unknown",
            "housing": "yes", "loan": "no", "contact": "cellular", "month": "may",
            "day_of_week": "mon", "duration": 10, "campaign": 1,
            "pdays": 999 if i % 2 else 5, "previous": 0 if i % 2 else 1,
            "poutcome": "nonexistent" if i % 2 else "success",
            "emp.var.rate": 1.1, "cons.price.idx": 93.994, "cons.conf.idx": -36.4,
            "euribor3m": 4.857, "nr.employed": 5191.0,
            "y": "yes" if i % 3 == 0 else "no",
        })
    return pd.DataFrame(records, columns=RAW_COLUMNS)


@pytest.fixture
def prepared(tmp_path, source, monkeypatch):
    csv_path = tmp_path / data.CSV_NAME
    payload = source.to_csv(sep=";", index=False).encode()
    csv_path.write_bytes(payload)
    (tmp_path / "provenance.json").write_text(json.dumps({
        "source_sha256": hashlib.sha256(payload).hexdigest(),
    }))
    original_validate = data.validate_source
    monkeypatch.setattr(data, "validate_source", lambda frame: original_validate(frame, 20))
    db_path = tmp_path / "prepared.sqlite"
    quality = data.prepare_data(csv_path, db_path)
    return db_path, quality


def test_exact_variant_and_source_numeric_contract(source):
    assert data.validate_source(source, expected_rows=20).equals(source)
    with pytest.raises(ValueError, match="41,188"):
        data.validate_source(source)
    with pytest.raises(ValueError, match="21-column"):
        data.validate_source(source.drop(columns="duration"), expected_rows=20)
    with pytest.raises(ValueError, match="21-column"):
        data.validate_source(pd.DataFrame({"age;job;y": [1] * 20}), expected_rows=20)
    for column, invalid in [("age", 0), ("previous", -1), ("pdays", 2.5),
                            ("duration", np.inf), ("euribor3m", "bad")]:
        broken = source.copy().astype({column: object})
        broken.loc[0, column] = invalid
        with pytest.raises(ValueError, match=column):
            data.validate_source(broken, expected_rows=20)


def test_categories_normalized_unknown_retained_and_undocumented_rejected(source):
    source.loc[0, "job"] = "  ADMIN. "
    normalized = data.validate_source(source, expected_rows=20)
    assert normalized.loc[0, "job"] == "admin."
    assert normalized.loc[1, "job"] == "unknown"
    source.loc[0, "poutcome"] = "other"  # Older variant's vocabulary must not pass.
    with pytest.raises(ValueError, match="poutcome"):
        data.validate_source(source, expected_rows=20)


def test_exact_chronological_partition():
    split = data.split_mapping(41188)
    assert split.split.value_counts().to_dict() == {
        "train": 28831, "validation": 6178, "test": 6179,
    }
    assert split.groupby("split").record_id.agg(["min", "max"]).to_dict("index") == {
        "train": {"min": 1, "max": 28831},
        "validation": {"min": 28832, "max": 35009},
        "test": {"min": 35010, "max": 41188},
    }
    assert split.record_id.is_unique
    assert split.record_id.tolist() == list(range(1, 41189))


def _zip(members):
    output = BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in members.items():
            archive.writestr(name, content)
    return output.getvalue()


def test_nested_acquisition_preserves_bytes_and_frozen_checksum(tmp_path, source, monkeypatch):
    raw_dir = tmp_path / "raw"
    monkeypatch.setattr(data, "RAW", raw_dir)
    original_validate = data.validate_source
    monkeypatch.setattr(data, "validate_source", lambda frame: original_validate(frame, 20))
    payload = source.to_csv(sep=";", index=False, lineterminator="\r\n").encode()
    readme = b"Original readme bytes\r\npdays=999\r\n"
    inner = _zip({f"bank-additional/{data.CSV_NAME}": payload,
                  f"bank-additional/{data.NAMES_NAME}": readme})
    archive_path = tmp_path / "official.zip"
    archive_path.write_bytes(_zip({"bank-additional.zip": inner}))
    csv_path = data.fetch_data(archive_path)
    assert csv_path.read_bytes() == payload
    assert (raw_dir / data.NAMES_NAME).read_bytes() == readme
    provenance = json.loads((raw_dir / "provenance.json").read_text())
    assert provenance["source_sha256"] == hashlib.sha256(payload).hexdigest()
    assert provenance["archive_member"] == (
        "bank-additional.zip::bank-additional/bank-additional-full.csv"
    )
    monkeypatch.setattr(data, "_download", lambda: pytest.fail("Valid cache must not download"))
    assert data.fetch_data() == csv_path
    csv_path.write_bytes(payload + b"\n")
    with pytest.raises(ValueError, match="checksum mismatch"):
        data.fetch_data()
    csv_path.write_bytes(payload)
    changed = payload.replace(b"admin.", b"student", 1)
    archive_path.write_bytes(_zip({data.CSV_NAME: changed, data.NAMES_NAME: readme}))
    with pytest.raises(ValueError, match="differs from the established"):
        data.fetch_data(archive_path)
    assert csv_path.read_bytes() == payload


def test_acquisition_failure_and_unsafe_or_wrong_variant(tmp_path, monkeypatch):
    raw_dir = tmp_path / "raw"
    monkeypatch.setattr(data, "RAW", raw_dir)

    def failed_download():
        raise RuntimeError("Network unavailable")

    monkeypatch.setattr(data, "_download", failed_download)
    with pytest.raises(RuntimeError, match="Network unavailable"):
        data.fetch_data()
    assert not (raw_dir / "provenance.json").exists()
    with pytest.raises(ValueError, match="Unsafe"):
        data._read_official_members(_zip({"../bank-additional-full.csv": b"bad"}))
    with pytest.raises(ValueError, match="must contain"):
        data._read_official_members(_zip({"bank-full.csv": b"wrong variant"}))


def test_storage_preserves_duplicates_order_and_constraints(prepared, source):
    db_path, quality = prepared
    all_records = load_records("all", db_path)
    assert len(all_records) == len(source)
    assert all_records.record_id.tolist() == list(range(1, 21))
    assert all_records.y.tolist() == source.y.map({"yes": 1, "no": 0}).tolist()
    assert all_records.pdays.tolist() == source.pdays.tolist()
    assert all_records.age_group[:4].tolist() == ["under 30", "30–44", "45–59", "60+"]
    assert all_records["default"].eq("unknown").all()
    assert "emp_var_rate" in all_records and "emp.var.rate" not in all_records
    assert quality["duplicate_extra_rows"] == int(source.duplicated().sum())
    assert quality["cross_split_duplicate_fingerprints"] > 0
    assert not any("subscription" in key or "base_rate" in key for key in quality)
    assert quality["split_counts"] == {"train": 14, "validation": 3, "test": 3}
    with connect(db_path) as connection:
        assert connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        raw = pd.read_sql_query("SELECT * FROM raw_campaigns ORDER BY record_id", connection)
        assert "emp.var.rate" in raw
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute("INSERT INTO dataset_splits VALUES (9999, 'train')")
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute("UPDATE clean_campaigns SET y = 2 WHERE record_id = 1")
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute("UPDATE engagement_simulated SET provenance='real'")
        flags = pd.read_sql_query("SELECT * FROM engagement_simulated ORDER BY record_id", connection)
    for field, cutoff in [("simulated_opt_out", 10), ("simulated_contact_permission", 90),
                          ("simulated_recent_contact", 10)]:
        expected = int(hashlib.sha256(f"42:{field}:1".encode()).hexdigest(), 16) % 100 < cutoff
        assert bool(flags.loc[0, field]) == expected


def test_sql_and_pandas_denominators_agree_all_splits(prepared):
    db_path, _ = prepared
    for split in ["train", "validation", "test", "all"]:
        records = load_records(split, db_path)
        overview = query_analysis("overview", split, db_path).iloc[0]
        assert overview.observations == len(records)
        assert overview.subscriptions == records.y.sum()
        assert overview.subscription_rate == pytest.approx(records.y.mean())
        for name, column in [("by_job", "job"), ("by_age_group", "age_group"),
                             ("by_poutcome", "poutcome"), ("by_default", "default")]:
            actual = query_analysis(name, split, db_path).set_index(column).sort_index()
            expected = records.groupby(column).y.agg(observations="count", subscriptions="sum")
            pd.testing.assert_frame_equal(actual[["observations", "subscriptions"]],
                                          expected, check_dtype=False)
            assert actual.observations.sum() == len(records)
    for name in ANALYSES:
        query_analysis(name, "all", db_path)  # All SQL must be valid even with zero scores.
    unknown = query_analysis("unknown_coverage", "all", db_path).set_index("field")
    assert unknown.loc["default", "unknown_rate"] == 1.0
    assert unknown.loc["job", "unknown_count"] == 10
    with pytest.raises(ValueError, match="Unknown analysis"):
        query_analysis("overview; DROP TABLE clean_campaigns", "all", db_path)
    with pytest.raises(ValueError, match="split must be"):
        load_records("all OR 1=1", db_path)


def test_prepare_mismatch_leaves_existing_database_intact(tmp_path, prepared):
    db_path, _ = prepared
    previous_bytes = db_path.read_bytes()
    csv_path = tmp_path / data.CSV_NAME
    csv_path.write_bytes(csv_path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="checksum"):
        data.prepare_data(csv_path, db_path)
    assert db_path.read_bytes() == previous_bytes
