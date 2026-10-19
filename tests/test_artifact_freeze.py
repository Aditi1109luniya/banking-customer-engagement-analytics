import pytest
from bank_engagement.artifacts import write_json, sha256
from bank_engagement import modeling


def test_selection_rejects_changed_source_code_and_model(monkeypatch, tmp_path):
    root = tmp_path / "root"
    artifacts = root / "artifacts"
    artifacts.mkdir(parents=True)
    model = artifacts / "model.joblib"
    model.write_bytes(b"trusted fixture, never deserialized")
    quality = {"source_sha256": "source", "split_sha256": "split"}
    write_json(root / "data/processed/quality.json", quality)
    selection = {**quality, "code_fingerprint": "code",
                 "model_hashes": {"model.joblib": sha256(model)}}
    write_json(artifacts / "selection.json", selection)
    monkeypatch.setattr(modeling, "ROOT", root)
    monkeypatch.setattr(modeling, "ARTIFACTS", artifacts)
    monkeypatch.setattr(modeling, "code_fingerprint", lambda: "code")
    assert modeling.check_selection() == selection
    model.write_bytes(b"changed artifact")
    with pytest.raises(ValueError, match="Frozen artifact"):
        modeling.check_selection()
    model.write_bytes(b"trusted fixture, never deserialized")
    monkeypatch.setattr(modeling, "code_fingerprint", lambda: "changed")
    with pytest.raises(ValueError, match="code changed"):
        modeling.check_selection()
    monkeypatch.setattr(modeling, "code_fingerprint", lambda: "code")
    write_json(root / "data/processed/quality.json", {**quality, "source_sha256": "changed"})
    with pytest.raises(ValueError, match="Prepared data differs"):
        modeling.check_selection()
