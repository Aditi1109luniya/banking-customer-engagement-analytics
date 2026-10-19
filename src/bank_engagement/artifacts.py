"""Coherent JSON and trusted artifact loading."""
import hashlib
import json
import gzip
from pathlib import Path
import tempfile
import joblib
import pandas as pd
from bank_engagement.constants import DEMO


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False, default=str) + "\n")
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text())


def load_bundle(bundle=DEMO):
    bundle = Path(bundle)
    manifest = read_json(bundle / "manifest.json")
    for filename, digest in manifest["bundle_hashes"].items():
        if sha256(bundle / filename) != digest:
            raise ValueError(f"Demo artifact checksum mismatch: {filename}")
    import importlib.metadata
    for package in ["scikit-learn", "numpy", "pandas", "joblib"]:
        actual = importlib.metadata.version(package)
        if actual != manifest["versions"][package]:
            raise ValueError(f"{package} {actual} differs from model environment "
                             f"{manifest['versions'][package]}. Install requirements.txt.")
    temporary = tempfile.TemporaryDirectory(prefix="bank-engagement-")
    db_path = Path(temporary.name) / "bank_marketing.sqlite"
    db_path.write_bytes(gzip.decompress((bundle / "bank_marketing.sqlite.gz").read_bytes()))
    return dict(pipeline=joblib.load(bundle / "champion.joblib"), manifest=manifest,
                metrics=read_json(bundle / "metrics.json"),
                test=pd.read_csv(bundle / "test_replay.csv", dtype={"record_id": str},
                                 float_precision="round_trip"),
                db_path=db_path, _temporary_database=temporary,
                importance=pd.read_csv(bundle / "importance.csv"),
                group_audit=pd.read_csv(bundle / "group_audit.csv"),
                curves=pd.read_csv(bundle / "budget_curve.csv"))
