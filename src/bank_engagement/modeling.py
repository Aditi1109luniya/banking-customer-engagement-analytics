"""Bounded training, validation freeze, final evaluation, and demo packaging."""
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import tomllib
import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from bank_engagement.constants import ROOT, ARTIFACTS, DB, DEMO, FEATURES, EXCLUDED, SPLITS
from bank_engagement.artifacts import read_json, write_json, sha256
from bank_engagement.database import load_records, connect
from bank_engagement.evaluation import (ranking_metrics, positive_scores, ranked_order,
    budget_count, cutoff_metrics, budget_curve, group_audit)
from bank_engagement.features import make_pipeline


def config():
    return tomllib.loads((ROOT / "configs/project.toml").read_text())


def code_fingerprint():
    return hashlib.sha256(b"".join((Path(__file__).parent / name).read_bytes() for name in
        ["constants.py", "features.py", "modeling.py", "evaluation.py"])).hexdigest()


def check_selection():
    selection = read_json(ARTIFACTS / "selection.json")
    quality = read_json(ROOT / "data/processed/quality.json")
    if any(selection[key] != quality[key] for key in ["source_sha256", "split_sha256"]):
        raise ValueError("Prepared data differs from frozen selection")
    if selection["code_fingerprint"] != code_fingerprint():
        raise ValueError("Modeling code changed after selection. Start a disclosed new experiment.")
    for filename, digest in selection["model_hashes"].items():
        if sha256(ARTIFACTS / filename) != digest:
            raise ValueError(f"Frozen artifact changed: {filename}")
    return selection


def better(candidate, incumbent):
    if incumbent is None:
        return True
    for metric in ["precision_at_20", "ap"]:
        delta = candidate[metric] - incumbent[metric]
        if abs(delta) > 1e-12:
            return delta > 0
    if candidate["family"] != incumbent["family"]:
        return candidate["family"] == "Logistic Regression"
    return candidate["model_id"] < incumbent["model_id"]


def train():
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    if (ARTIFACTS / "selection.json").exists():
        selection = check_selection()
        if not (ARTIFACTS / "importance.csv").exists():
            validation = load_records("validation")
            save_importance(joblib.load(ARTIFACTS / "champion.joblib"), validation, config())
        return selection
    quality = read_json(ROOT / "data/processed/quality.json")
    settings = config()
    training, validation = load_records("train"), load_records("validation")
    versions = {name: importlib.metadata.version(name) for name in
                ["scikit-learn", "numpy", "pandas", "scipy", "joblib", "streamlit"]}
    identity = hashlib.sha256(json.dumps(
        [quality["source_sha256"], settings, versions, code_fingerprint()],
        sort_keys=True).encode()).hexdigest()[:10]
    candidates = []
    for c in [.01, .1, 1., 10.]:
        candidates.append((f"lr-c{c:g}", "Logistic Regression",
                           LogisticRegression(C=c, max_iter=2000, solver="lbfgs"), True,
                           {"C": c, "max_iter": 2000, "solver": "lbfgs",
                            "class_weight": None}))
    for depth in [8, None]:
        for leaf in [10, 50]:
            params = dict(n_estimators=settings["forest_estimators"], max_depth=depth,
                          min_samples_leaf=leaf, max_features="sqrt", class_weight=None,
                          random_state=settings["seed"], n_jobs=settings["forest_jobs"])
            candidates.append((f"rf-d{depth}-l{leaf}", "Random Forest",
                               RandomForestClassifier(**params), False, params))
    candidates.append(("prior", "Prior baseline", DummyClassifier(strategy="prior"), False,
                       {"strategy": "prior"}))
    rows, finalists = [], {}
    for short_id, family, estimator, scale, params in candidates:
        mid = f"{short_id}-{identity}"
        print(f"Training {mid}", flush=True)
        pipeline = make_pipeline(estimator, scale)
        pipeline.fit(training[FEATURES], training.y)
        scores = positive_scores(pipeline, validation[FEATURES])
        row = dict(model_id=mid, family=family, split="validation", params=params,
                   **ranking_metrics(validation.y, scores, validation.record_id))
        rows.append(row)
        if better(row, finalists.get(family)):
            finalists[family] = row
            joblib.dump(pipeline, ARTIFACTS / f"{family.replace(' ', '_')}.joblib", compress=3)
            pd.DataFrame({"record_id": validation.record_id, "y": validation.y,
                          "model_score": scores}).to_csv(
                              ARTIFACTS / f"{family.replace(' ', '_')}_validation.csv", index=False)
    champion = None
    for family in ["Logistic Regression", "Random Forest"]:
        if better(finalists[family], champion):
            champion = finalists[family]
    champion_file = f"{champion['family'].replace(' ', '_')}.joblib"
    shutil.copy2(ARTIFACTS / champion_file, ARTIFACTS / "champion.joblib")
    predictions = pd.read_csv(ARTIFACTS / f"{champion['family'].replace(' ', '_')}_validation.csv",
                              float_precision="round_trip")
    k = budget_count(len(validation), settings["outreach_fraction"])
    order = ranked_order(validation.record_id, predictions.model_score)
    cutoff = float(predictions.model_score.iloc[order[k - 1]])
    write_json(ARTIFACTS / "candidates.json", rows)
    model_files = ["champion.joblib"] + [
        f"{name.replace(' ', '_')}.joblib" for name in finalists]
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                         stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    selection = dict(model_id=champion["model_id"], champion_id=champion["model_id"],
        champion_family=champion["family"], version="1.0.0", positive_class=1,
        features=FEATURES, excluded=EXCLUDED, seed=42, default_capacity_fraction=.2,
        tie_rule='descending score; SHA256("42:<record_id>") ascending; ID ascending',
        validation_cutoff=cutoff, validation_selection=champion, finalists=finalists,
        split_counts=quality["split_counts"], split_ranges=SPLITS,
        source_sha256=quality["source_sha256"], split_sha256=quality["split_sha256"],
        code_fingerprint=code_fingerprint(), config=settings, versions=versions,
        code_commit_at_training=commit, created_at=datetime.now(timezone.utc).isoformat(),
        model_hashes={name: sha256(ARTIFACTS / name) for name in model_files},
        selection_rule="validation Precision@20%, then AP, then LR, then candidate ID",
        status="frozen_before_test")
    write_json(ARTIFACTS / "selection.json", selection)
    chosen = joblib.load(ARTIFACTS / "champion.joblib")
    save_importance(chosen, validation, settings)
    print(f"Frozen {champion['family']} ({champion['model_id']}); cutoff={cutoff:.6f}",
          flush=True)
    return selection


def save_importance(chosen, validation, settings):
    importance = permutation_importance(chosen, validation[FEATURES], validation.y,
        scoring="average_precision", n_repeats=settings["permutation_repeats"],
        random_state=42, n_jobs=1)
    pd.DataFrame({"feature": FEATURES, "importance_mean": importance.importances_mean,
                  "importance_std": importance.importances_std}).sort_values(
                      "importance_mean", ascending=False).to_csv(
                          ARTIFACTS / "importance.csv", index=False)


def evaluate():
    selection = check_selection()
    if (ARTIFACTS / "metrics.json").exists():
        previous = read_json(ARTIFACTS / "evaluation_manifest.json")
        if previous["selection_sha256"] != sha256(ARTIFACTS / "selection.json"):
            raise ValueError("Final test was already evaluated under a different selection")
        persist_scores(selection)
        return read_json(ARTIFACTS / "metrics.json")
    test = load_records("test")
    metrics = []
    champion_scores = None
    for family, val in selection["finalists"].items():
        is_champion = val["model_id"] == selection["champion_id"]
        metrics.append({**val, "champion": is_champion})
        pipeline = joblib.load(ARTIFACTS / f"{family.replace(' ', '_')}.joblib")
        scores = positive_scores(pipeline, test[FEATURES])
        row = dict(model_id=val["model_id"], family=family, split="test",
                   champion=is_champion, **ranking_metrics(test.y, scores, test.record_id))
        if is_champion:
            champion_scores = scores
            row.update(cutoff_metrics(test.y, scores, selection["validation_cutoff"]))
        metrics.append(row)
        pd.DataFrame({"record_id": test.record_id, "y": test.y, "model_score": scores}).to_csv(
            ARTIFACTS / f"{family.replace(' ', '_')}_test.csv", index=False)
    test["model_score"] = champion_scores
    test.to_csv(ARTIFACTS / "test_replay.csv", index=False)
    curves = budget_curve(test.y, champion_scores, test.record_id)
    curves.to_csv(ARTIFACTS / "budget_curve.csv", index=False)
    group_audit(test, champion_scores).to_csv(ARTIFACTS / "group_audit.csv", index=False)
    split_summary = []
    for split in SPLITS:
        frame = load_records(split)
        split_summary.append({"split": split, "n": len(frame), "positives": int(frame.y.sum()),
                              "base_rate": float(frame.y.mean())})
    write_json(ARTIFACTS / "split_summary.json", split_summary)
    write_json(ARTIFACTS / "metrics.json", metrics)
    write_json(ARTIFACTS / "evaluation_manifest.json",
               {"selection_sha256": sha256(ARTIFACTS / "selection.json"),
                "evaluated_at": datetime.now(timezone.utc).isoformat(),
                "test_rows": len(test), "test_evaluation": "frozen finalists; no reselection"})
    persist_scores(selection)
    make_figures(test, metrics, curves)
    return metrics


def persist_scores(selection):
    """Restore canonical scores from frozen predictions without re-evaluating test."""
    with connect() as conn:
        for family, val in selection["finalists"].items():
            conn.execute("INSERT OR REPLACE INTO model_runs VALUES (?,?,?,?)",
                (val["model_id"], "artifacts/selection.json", selection["source_sha256"],
                 selection["created_at"]))
        for split in ["validation", "test"]:
            if split == "test":
                pred = pd.read_csv(ARTIFACTS / "test_replay.csv", float_precision="round_trip")
            else:
                pred = pd.read_csv(ARTIFACTS /
                    f"{selection['champion_family'].replace(' ', '_')}_validation.csv",
                    float_precision="round_trip")
            order = ranked_order(pred.record_id, pred.model_score)
            ranks = np.empty(len(pred), dtype=int)
            ranks[order] = np.arange(1, len(pred) + 1)
            conn.executemany("INSERT OR REPLACE INTO prospect_scores VALUES (?,?,?,?,?)",
                [(selection["champion_id"], int(rid), float(score), int(rank), split)
                 for rid, score, rank in zip(pred.record_id, pred.model_score, ranks)])


def make_figures(test, metrics, curves):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, precision_recall_curve
    from sklearn.calibration import calibration_curve
    folder = ROOT / "reports/figures"
    folder.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                         "font.size": 11, "figure.figsize": (8, 4.5)})
    def save(name):
        plt.tight_layout()
        plt.savefig(folder / f"{name}.png", dpi=180)
        plt.close()
    plt.figure()
    plt.plot(curves.fraction, curves.recall, color="#087f8c", linewidth=2.5,
             label="Frozen champion")
    plt.plot([0, 1], [0, 1], "--", color="#929ca4", label="Random contact expectation")
    plt.xlabel("Fraction of records shortlisted")
    plt.ylabel("Fraction of historical subscriptions captured")
    plt.legend()
    save("cumulative_gains")
    plt.figure()
    for row in metrics:
        if row["split"] != "test":
            continue
        pred = pd.read_csv(ARTIFACTS / f"{row['family'].replace(' ', '_')}_test.csv")
        p, r, _ = precision_recall_curve(pred.y, pred.model_score)
        plt.plot(r, p, label=f"{row['family']} (AP {row['ap']:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    save("precision_recall")
    plt.figure()
    for row in metrics:
        if row["split"] != "test":
            continue
        pred = pd.read_csv(ARTIFACTS / f"{row['family'].replace(' ', '_')}_test.csv")
        fpr, tpr, _ = roc_curve(pred.y, pred.model_score)
        plt.plot(fpr, tpr, label=row["family"])
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.legend()
    save("roc")
    observed, predicted = calibration_curve(test.y, test.model_score, n_bins=10,
                                            strategy="quantile")
    plt.figure()
    plt.plot(predicted, observed, "o-", color="#087f8c")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("Mean model score")
    plt.ylabel("Observed subscription rate")
    save("reliability")
    importance = pd.read_csv(ARTIFACTS / "importance.csv").sort_values("importance_mean")
    plt.figure()
    plt.barh(importance.feature, importance.importance_mean,
             xerr=importance.importance_std, color="#087f8c")
    plt.xlabel("Validation AP decrease after raw-column permutation")
    save("feature_importance")
    plt.figure()
    for label in [0, 1]:
        plt.hist(test.loc[test.y == label, "model_score"], bins=25, alpha=.6,
                 density=True, label=f"Observed y={label}")
    plt.xlabel("Champion score")
    plt.ylabel("Density")
    plt.legend()
    save("score_distribution")


def package_demo():
    selection = check_selection()
    if not (ARTIFACTS / "metrics.json").exists():
        raise ValueError("Run final evaluation before packaging")
    DEMO.mkdir(parents=True, exist_ok=True)
    names = ["champion.joblib", "metrics.json", "test_replay.csv", "importance.csv",
             "group_audit.csv", "budget_curve.csv", "split_summary.json",
             "evaluation_manifest.json", "candidates.json"]
    for name in names:
        shutil.copy2(ARTIFACTS / name, DEMO / name)
    # Compress the public analysis database for an efficient GitHub/Cloud bundle.
    # The trusted loader expands it to a disposable process-local file.
    (DEMO / "bank_marketing.sqlite.gz").write_bytes(gzip.compress(DB.read_bytes(), mtime=0))
    names.append("bank_marketing.sqlite.gz")
    manifest = {**selection, "bundle_hashes": {name: sha256(DEMO / name) for name in names}}
    write_json(DEMO / "manifest.json", manifest)
    sample = load_records("test")[["record_id", *FEATURES]].head(8)
    sample.record_id = ["DEMO-" + str(i + 1) for i in range(len(sample))]
    sample.to_csv(ROOT / "samples/prospects.csv", index=False)
    from bank_engagement.engagement import simulate_engagement
    simulate_engagement(sample.record_id).to_csv(
        ROOT / "samples/engagement_simulated.csv", index=False)
    return {"bundle": "demo", "size_mb": round(sum(p.stat().st_size for p in DEMO.iterdir()
                                               if p.is_file()) / 1e6, 2)}
