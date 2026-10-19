"""Reproducible project stages: python -m bank_engagement.cli --help."""
import argparse
from pathlib import Path
import sys
import pandas as pd
from bank_engagement.constants import ROOT, ARTIFACTS
from bank_engagement.artifacts import write_json, load_bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    fetch = sub.add_parser("fetch-data")
    fetch.add_argument("--archive", type=Path)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--csv", type=Path)
    analysis = sub.add_parser("analyze")
    analysis.add_argument("--split", choices=["train", "validation", "test", "all"], default="train")
    sub.add_parser("train")
    evaluation = sub.add_parser("evaluate")
    evaluation.add_argument("--split", choices=["test"], default="test")
    sub.add_parser("package-demo")
    score = sub.add_parser("score")
    score.add_argument("--input", type=Path, required=True)
    score.add_argument("--output", type=Path, required=True)
    score.add_argument("--engagement", type=Path)
    group = score.add_mutually_exclusive_group()
    group.add_argument("--capacity", type=int)
    group.add_argument("--fraction", type=float, default=.2)
    score.add_argument("--min-score", type=float)
    sub.add_parser("run-all")
    args = parser.parse_args()
    try:
        if args.command in ["fetch-data", "run-all"]:
            from bank_engagement.data import fetch_data
            print(fetch_data(getattr(args, "archive", None)), flush=True)
        if args.command in ["prepare", "run-all"]:
            from bank_engagement.data import prepare_data
            print(prepare_data(getattr(args, "csv", None)), flush=True)
        if args.command in ["analyze", "run-all"]:
            split = getattr(args, "split", "train")
            if split != "train" and not (ARTIFACTS / "selection.json").exists():
                raise ValueError("Target-based non-training analysis requires frozen selection")
            from bank_engagement.database import query_analysis
            folder = ROOT / "reports/sql"
            folder.mkdir(parents=True, exist_ok=True)
            for name in ["overview", "by_job", "by_age_group", "by_poutcome",
                         "by_prior_contact", "by_month", "by_day_of_week", "by_contact",
                         "by_education", "by_previous", "by_recency", "unknown_coverage"]:
                query_analysis(name, split).to_csv(folder / f"{name}_{split}.csv", index=False)
            print(f"Saved SQL summaries for {split}", flush=True)
        if args.command in ["train", "run-all"]:
            from bank_engagement.modeling import train
            print(train()["champion_id"], flush=True)
        if args.command in ["evaluate", "run-all"]:
            from bank_engagement.modeling import evaluate
            print(pd.DataFrame(evaluate()).to_string(index=False), flush=True)
        if args.command in ["package-demo", "run-all"]:
            from bank_engagement.modeling import package_demo
            print(package_demo(), flush=True)
        if args.command == "score":
            from bank_engagement.scoring import score_prospects
            bundle = load_bundle()
            prospects = pd.read_csv(args.input, dtype=str)
            flags = pd.read_csv(args.engagement, dtype=str) if args.engagement else None
            scores = score_prospects(bundle["pipeline"], prospects,
                bundle["manifest"]["champion_id"], flags=flags, capacity=args.capacity,
                fraction=args.fraction, min_score=args.min_score)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            scores.to_csv(args.output, index=False)
            write_json(args.output.with_suffix(".metadata.json"),
                       {"model_id": bundle["manifest"]["champion_id"],
                        "records": len(scores), "selected": int(scores.selected.sum()),
                        "warnings": scores.attrs.get("warnings", []),
                        "context": "Public/synthetic demo. IDs are surrogates; engagement simulated."})
            print(f"Saved {len(scores)} safe scores to {args.output}", flush=True)
    except (ValueError, FileNotFoundError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
