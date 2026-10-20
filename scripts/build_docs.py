"""Generate measured README/report/guide sections from frozen artifacts, using stdlib only.

Run after evaluate/package-demo. Optional URLs are for already verified publication.
The original learning guide is copied once; its worked examples remain explicitly hypothetical.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
START = "<!-- BEGIN VERIFIED RESULTS -->"
END = "<!-- END VERIFIED RESULTS -->"
EVIDENCE_START = "<!-- BEGIN IMPLEMENTATION EVIDENCE -->"
EVIDENCE_END = "<!-- END IMPLEMENTATION EVIDENCE -->"


def read(path: str):
    return json.loads((ROOT / path).read_text())


def percent(value, digits=2):
    return "undefined" if value is None else f"{value:.{digits}%}"


def number(value, digits=4):
    return "undefined" if value is None else f"{value:.{digits}f}"


def table(headers, rows):
    return "\n".join([
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
        *("| " + " | ".join(str(cell) for cell in row) + " |" for row in rows),
    ])


def metric_table(metrics):
    return table(
        ["Split / model", "PR-AUC¹", "AP²", "ROC-AUC", "Precision@20%", "Lift@20%",
         "Recall@20%", "Captured / shortlisted", "Brier ↓"],
        [[f"{m['split']} / {m['family']}{' ★' if m.get('champion') else ''}",
          number(m['pr_auc']), number(m['ap']), number(m['roc_auc']),
          percent(m['precision_at_20']), number(m['lift_at_20'], 3),
          percent(m['recall_at_20']), f"{m['tp_k']:,} / {m['k']:,}", number(m['brier'])]
         for m in metrics],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--github-url", help="Verified repository URL; omit before publication")
    parser.add_argument("--demo-url", help="Verified live application URL; omit before deployment")
    parser.add_argument("--checks-passed", action="store_true",
                        help="Set only after the actual ruff and pytest commands passed")
    args = parser.parse_args()
    required = ["artifacts/selection.json", "artifacts/metrics.json",
                "artifacts/split_summary.json", "data/processed/quality.json",
                "artifacts/evaluation_manifest.json"]
    absent = [name for name in required if not (ROOT / name).exists()]
    if absent:
        raise SystemExit("Run prepare/train/evaluate before generating measured documentation: "
                         + ", ".join(absent))
    selection = read("artifacts/selection.json")
    metrics = read("artifacts/metrics.json")
    quality = read("data/processed/quality.json")
    splits = read("artifacts/split_summary.json")
    evaluation = read("artifacts/evaluation_manifest.json")
    champion = next(m for m in metrics if m["split"] == "test" and m.get("champion"))
    validation = next(m for m in metrics if m["split"] == "validation" and m.get("champion"))
    baseline = next(m for m in metrics if m["split"] == "test" and m["family"] == "Prior baseline")
    lr_test = next(m for m in metrics if m["split"] == "test" and m["family"] == "Logistic Regression")
    lr_validation = next(m for m in metrics if m["split"] == "validation"
                         and m["family"] == "Logistic Regression")
    selected_hash = hashlib.sha256((ROOT / "artifacts/selection.json").read_bytes()).hexdigest()
    if selected_hash != evaluation["selection_sha256"]:
        raise SystemExit("Evaluation does not match frozen selection; refusing stale documentation")
    split_rows = [[s["split"], f"{s['n']:,}",
                   f"{selection['split_ranges'][s['split']][0]:,}–"
                   f"{selection['split_ranges'][s['split']][1]:,}",
                   f"{s['positives']:,}", percent(s["base_rate"])] for s in splits]
    split_table = table(["Partition", "Records", "Source row IDs", "Subscriptions", "Rate"],
                        split_rows)
    headline = (
        f"The validation-selected **{selection['champion_family']}** captured "
        f"**{champion['tp_k']:,} of {champion['positives']:,}** historical test subscriptions "
        f"in the highest-scored **{champion['k']:,} of {champion['n']:,} records**. "
        f"Test Precision@20% was **{percent(champion['precision_at_20'])}**, "
        f"Lift@20% **{number(champion['lift_at_20'], 3)}×**, and "
        f"Recall@20% **{percent(champion['recall_at_20'])}**. "
        "These are offline historical ranking results, not incremental sales."
    )
    comparison_text = (
        f"**The held-out comparison matters:** Logistic Regression achieved "
        f"{percent(lr_test['precision_at_20'])} test Precision@20%, "
        f"{number(lr_test['lift_at_20'], 3)}× lift and "
        f"{percent(lr_test['recall_at_20'])} recall. "
        f"The released {selection['champion_family']} remains the choice fixed on validation "
        f"({percent(validation['precision_at_20'])} precision, versus "
        f"{percent(lr_validation['precision_at_20'])} for Logistic Regression); "
        "we do not switch to the better-looking test model after seeing the holdout. "
        "This reversal is a limitation to explain, not a result to hide."
    )
    drift_text = (
        "**Substantial temporal shift:** subscription prevalence was "
        + " → ".join(f"{percent(s['base_rate'])} ({s['split']})" for s in splits)
        + ". A higher later-period precision does not itself show an improved model. "
        "Ranking generalization is limited, and the scores should not be presented as "
        "calibrated personal probabilities."
    )
    cutoff_text = (
        f"The frozen validation cutoff is **{selection['validation_cutoff']:.6f}**. "
        f"Applying `score >= cutoff` on test selected **{champion['fixed_cutoff_count']:,}** "
        f"records ({percent(champion['fixed_cutoff_coverage'])} coverage), with "
        f"{percent(champion['fixed_cutoff_precision'])} precision and "
        f"{percent(champion['fixed_cutoff_recall'])} recall. "
        "This is a separate policy from the exact top-20% shortlist. Ties at the cutoff can "
        "increase its volume; a new population can change it further."
    )
    with (ROOT / "artifacts/importance.csv").open() as handle:
        importance = list(csv.DictReader(handle))
    importance.sort(key=lambda row: float(row["importance_mean"]), reverse=True)
    importance_table = table(["Raw feature", "Mean validation AP decrease", "Repeat SD"],
                             [[r["feature"], number(float(r["importance_mean"])),
                               number(float(r["importance_std"]))] for r in importance])
    with (ROOT / "artifacts/group_audit.csv").open() as handle:
        groups = list(csv.DictReader(handle))
    group_rows = []
    for row in groups:
        if row["group_type"] == "age_group":
            def group_rate(name):
                return percent(float(row[name])) if row[name] else "insufficient support"
            group_rows.append([row["group"], row["n"], row["positives"], row["selected"],
                               group_rate("base_rate"), group_rate("selection_rate"),
                               group_rate("precision"), group_rate("recall")])
    group_table = table(["Age group", "n", "Subscriptions", "Selected", "Base rate",
                         "Selection rate", "Precision", "Recall"], group_rows)
    unknown_rows = [[field, f"{count:,}", percent(count / quality["rows"])]
                    for field, count in quality["unknown_counts"].items() if count]
    unknown_table = table(["Source categorical field", "Unknown count", "Share of all rows"],
                          unknown_rows)
    contradictions = table(["Prior-history check", "Flagged rows"],
                           list(quality["prior_history_contradictions"].items()))
    sql_finding = "See `reports/sql/` for training-only descriptive SQL summaries."
    sql_path = ROOT / "reports/sql/by_poutcome_train.csv"
    if sql_path.exists():
        with sql_path.open() as handle:
            sql_rows = list(csv.DictReader(handle))
        sql_finding = "Training-only previous-outcome summary: " + "; ".join(
            f"`{row['poutcome']}`: {int(row['subscriptions']):,} subscriptions / "
            f"{int(row['observations']):,} observations "
            f"({percent(float(row['subscription_rate']))})" for row in sql_rows
        ) + ". These are descriptive associations with explicit denominators."
    links = []
    if args.github_url:
        links.append(f"[Public source repository]({args.github_url})")
    if args.demo_url:
        links.append(f"[Live interactive demo]({args.demo_url})")
    publication = " · ".join(links) if links else (
        "Run locally with the commands below. Public hosting is not asserted by this document."
    )
    check_status = ("`ruff check .` and `pytest -q` passed in the delivery environment."
                    if args.checks_passed else
                    "Run `ruff check .` and `pytest -q` before claiming verified checks.")
    clone = (
        "```bash\ngit clone " + args.github_url.removesuffix(".git")
        + ".git\ncd banking-customer-engagement-analytics\n```\n"
        if args.github_url else ""
    )
    readme = f"""# Banking Customer Engagement & Deposit Propensity Analytics

**With limited contact capacity, which prospects should a relationship team review first for a term-deposit conversation?**

An independent portfolio project combining Python, Pandas, SQLite/SQL, scikit-learn and Streamlit. It turns historical campaign observations into a reviewable propensity ranking, campaign summaries and a capacity-based outreach simulation. The data is Portuguese telephone campaign history, **not ICICI Bank customer data or digital-channel engagement data**. Product suitability requires a separate needs assessment.

{publication}

## Measured outcome

{headline}

Test ROC-AUC: **{number(champion['roc_auc'])}** · trapezoidal PR-AUC: **{number(champion['pr_auc'])}** · average precision (AP): **{number(champion['ap'])}**. The model was frozen using validation results before test evaluation. It was not reselected on the test scores.

{comparison_text}

{drift_text}

![Historical subscriptions captured across outreach budgets](reports/figures/cumulative_gains.png)

## What the application does

| Workspace | Useful behavior |
|---|---|
| Engagement hub | Default graphical welcome with an illustration, interactive previews from actual campaign/test artifacts, and direct navigation into analysis or outreach. |
| Campaign overview | SQL-backed counts, subscription rates and segments with denominators; training history is the development default. |
| Model results | Validation/test comparison, precision/recall/lift at a budget, gains curves, reliability, drift and descriptive group review. |
| Outreach shortlist | Historical replay, separate simulated suppression, eligible-pool budget controls, factual context, selected CSV and context JSON downloads. |
| Scoring sandbox | Synthetic one-record form and public/synthetic batch scoring with validation, unseen-category warnings and no upload persistence. |

The app loads a checksum-verified demo bundle; it does not train on startup. A simulated opt-out suppresses selection regardless of score. Neither the model nor simulation sends messages or contacts anyone.

The first screen is the **Engagement hub**. Its **Explore analysis →** and **Plan outreach →** buttons open the detailed workspaces. Three chart previews show historical campaign patterns, the frozen champion's test capacity/capture curve, and separately simulated outreach eligibility. The **Preview contact capacity (%)** slider changes the displayed budget point and simulated selected count without changing official metrics or the frozen model; **Open outreach shortlist** and **Plan outreach →** carry that percentage into the shortlist. Detailed pages provide **← Back to engagement hub** navigation; a raw record table is not the initial product experience. Hero artwork is illustrative and independent of ICICI Bank.

A visible navigation bar offers **Home**, **Campaign insights**, **Model results**, **Outreach**, and **Try scoring** on every page; the optional sidebar stays synchronized. Page links use `?view=home`, `?view=analysis`, `?view=model`, `?view=outreach`, or `?view=score`, so a copied URL can open a specific workspace. Changing pages returns to the heading; changing an analytical control preserves the current reading position. Chart previews stack into readable cards on smaller screens.

{check_status}

## Run the prepared demo

Use **Python 3.11** and run from the repository root. Exact runtime and development dependencies are committed in `requirements.txt` and `requirements-dev.txt`.

{clone}

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
python -m streamlit run app/streamlit_app.py
```

On Windows, replace activation with `.venv\\Scripts\\activate` in Command Prompt, or `.venv\\Scripts\\Activate.ps1` in PowerShell. Use `py -3.11` in place of `python3.11` when appropriate. The committed `demo/` bundle provides the data/model needed to use the app offline after dependencies are installed. The loader checks package versions against the model environment. Install the locked requirements when a version mismatch is reported. Load only the repository's trusted model files; never load uploaded pickle/joblib files.

## Reproduce from the official source

```bash
python -m pip install -r requirements-dev.txt
python -m pip install -e . --no-deps
python -m bank_engagement.cli fetch-data
python -m bank_engagement.cli prepare
python -m bank_engagement.cli analyze --split train
python -m bank_engagement.cli train
python -m bank_engagement.cli evaluate --split test
python -m bank_engagement.cli package-demo
python scripts/build_docs.py
ruff check .
pytest -q
```

`python -m bank_engagement.cli run-all` executes fetch → prepare → training SQL analysis → train/freeze → test evaluation → demo packaging in that order. Run `build_docs.py` afterwards for measured documentation. Initial acquisition requires internet; later `fetch-data` reuses verified cached source bytes. For a manually downloaded official ZIP, use `fetch-data --archive /path/to/bank+marketing.zip`. A manual `prepare --csv /path/to/bank-additional-full.csv` requires adjacent checksum-matching `provenance.json`; establish provenance with `fetch-data --archive` first.

Training and evaluation are guarded: an existing frozen selection is validated and reused, and a matching completed test evaluation is reused. Do not delete manifests to hide later changes. For a disclosed new experiment, preserve the old run and its evidence in a separate checkout/output snapshot before rebuilding. Preparing the SQLite database rebuilds its tables; the packaged demo remains a separate snapshot.

### Score a public or synthetic batch

```bash
python -m bank_engagement.cli score --input samples/prospects.csv --engagement samples/engagement_simulated.csv --output artifacts/scored.csv --capacity 5
```

Use `--fraction 0.20` instead of `--capacity` for a fraction of the eligible pool. `--min-score NUMBER` adds a cutoff while capacity remains a hard cap; use the frozen cutoff in `demo/manifest.json` for the documented policy. The CLI score CSV includes all safe scored records and a `selected` column; the app shortlist download contains selected rows only. Required input fields and safe export columns are documented in [the data dictionary](docs/data_dictionary.md). No real customer records or contact details belong in this demo.

## Pipeline and evaluation choices

```mermaid
flowchart LR
    A[Official UCI additional-full CSV] --> B[Validate, hash, preserve order]
    B --> C[SQLite and training SQL analysis]
    B --> D[Ordered train / validation / test]
    D --> E[Ten pre-contact inputs]
    E --> F[Training-only transforms + LR / RF / prior]
    F --> G[Validation choice and manifest freeze]
    G --> H[Locked later-period test evaluation]
    G --> I[Saved full pipeline]
    I --> J[Score current pool]
    K[Separate simulated engagement] --> L[Suppress and rank eligible records]
    J --> L
    C --> M[Streamlit]
    H --> M
    L --> M
```

{split_table}

All 41,188 source observations are retained, including repeats. IDs identify rows rather than unique customers. Training contains the earliest 70% (floor), validation the next 15% (floor), and test the remainder, in supplied order. Complete dates and real customer groups are not available.

The allowlist is `age`, `job`, `marital`, `education`, `default`, `housing`, `loan`, `previous`, `pdays`, `poutcome`. `pdays=999` creates a no-previous-contact indicator and a recency value filled from the training-only observed-recency median. One-hot categories and Logistic Regression scaling are fitted on training only. Duration, current-contact/campaign fields, economic indicators with unverified publication timing, outcome, IDs, split and simulation fields are excluded.

The fixed search has four Logistic Regression C values, four Random Forest depth/leaf combinations and one prior baseline. Selection uses validation Precision@20%, then AP, then Logistic Regression, then candidate ID. Finalists and champion are frozen before the held-out test is evaluated. Ranking ties use a label-independent SHA-256 key; capacity uses ceiling rounding. The no-information prior baseline uses the same tie policy. AP and trapezoidal PR-AUC are reported separately.

## Read and explain the project

- [Project understanding and interview guide](docs/project_understanding_and_interview_guide.md): complete pipeline, measured results, interview questions and pitches.
- [Model report](reports/model_report.md): exact selection, metrics, data quality, explanations and limitations.
- [Data dictionary](docs/data_dictionary.md): every raw field, derived feature, simulation and scoring contract.
- [Five-minute demo script](docs/demo_script.md): page-by-page interview walkthrough.
- [Implementation plan](plan.md): original build specification; historical planning language is retained.
- [Presentation](docs/presentation/): five-slide delivery and notes when generated.

Original source archives and bulk rebuild artifacts are ignored. The compact trusted `demo/` bundle, source, SQL, exact locks, tests, reports and figures are intended for reproducible delivery. The project has no paid API, LLM, CRM or external database dependency.

## Interpret the results responsibly

Historical ranking captures observed subscriptions under historical campaigns. It does not estimate the causal effect of outreach, revenue, product suitability or a digital channel's effectiveness. Subscription prevalence and observed profiles shift over time; scores are not guaranteed calibrated personal probabilities. Age/job group diagnostics are descriptive and support-limited, not a fairness certification. New deployment would require contemporary permissioned data, separate suitability and engagement controls, monitoring and a prospective controlled evaluation.

## Data and license

Use exactly [UCI Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing), DOI [10.24432/C5K306](https://doi.org/10.24432/C5K306), `bank-additional-full.csv`. Dataset creators: Sérgio Moro, Paulo Rita and Paulo Cortez. The accompanying archive readme describes 41,188 ordered telephone-campaign observations from May 2008 to November 2010. The dataset is [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); [DATA_LICENSE.md](DATA_LICENSE.md) describes attribution and transformations. Original code: [MIT](LICENSE), copyright 2026 Aditi Luniya.

This independent portfolio prototype has no ICICI Bank affiliation or endorsement. No campus deck, employer branding, or private conversation is published.
"""
    report = f"""# Model report — Banking Customer Engagement & Deposit Propensity

**Model:** `{selection['champion_id']}`  
**Selection frozen:** {selection['created_at']}  
**Test evaluation recorded:** {evaluation['evaluated_at']}  
**Use:** offline historical prioritization research and a simulated relationship-manager workflow.

## 1. Decision, population and evidence boundary

{headline}

The decision is which observations to review first when a team can discuss term deposits with only a limited fraction of its prospect pool. The output is a ranking and factual customer-history context. It supports a needs-based conversation; it is not a product recommendation or a suitability decision. No customers were contacted and no incremental sales, revenue or production impact were measured.

The exact source variant is UCI's `bank-additional-full.csv`: 41,188 observations, 20 source inputs and binary subscription target `y`, ordered by date from May 2008 to November 2010. These are Portuguese telephone campaigns. The row identifier is a surrogate observation ID, not evidence of a unique customer, household or account. The data lacks complete timestamps, contact permission, digital engagement, product value, contact cost and customer needs. Its selection into historical campaigns limits generalization to arbitrary prospects and present-day Indian banking.

## 2. Data preparation and quality

Acquisition selects exact basenames from the official nested archive without arbitrary extraction, enforces the 21-column/41,188-row contract, preserves original CSV bytes/readme, and records SHA-256 provenance. It rejects alternative older/sample variants, undocumented source categories, missing/nonfinite numeric fields and malformed numeric counts. Source text is normalized, `unknown` is retained, `y` is mapped to 0/1, and fixed age groups are added for descriptions. No row is removed or reordered to improve the result. Raw schema and source-wide quality inspection do not use held-out outcomes for model choices.

Quality inspection found **{quality['duplicate_extra_rows']:,} extra exact duplicate rows**, involving **{quality['duplicate_rows']:,} rows**, and **{quality['cross_split_duplicate_fingerprints']:,} exact duplicate fingerprints spanning more than one split.** Duplicates remain because identical observations need not be erroneous or the same person. An exact-row comparison cannot rule out repeated customers with different attributes across time. Contradictory history combinations are counted rather than corrected without evidence:

{contradictions}

Unknown categories are not interpreted as negative answers:

{unknown_table}

The normalized SQLite database contains raw and clean tables, a one-to-one split map, model-run metadata, canonical validation/test scores, and a clearly separate simulated-engagement table. Primary/foreign keys and integrity checks protect joins. Named SQL queries use bound split parameters; segment outputs retain counts and subscription numerators. Target-based exploratory analysis defaults to training, and the CLI gates held-out/full-history descriptions until selection has been frozen.

{sql_finding}

## 3. Scoring time and leakage policy

Scoring occurs immediately before a proposed contact. The model assumes recorded customer profile and previous-campaign history were then available. This remains a retrospective approximation because feature-event timestamps are absent. The ten raw inputs are age; job, marital status and education; recorded credit default, housing loan and personal loan; prior contact count; prior campaign recency; and prior campaign outcome. Full definitions are in [the data dictionary](../docs/data_dictionary.md).

`duration` is excluded because it is observed after the call and would answer a different question. `contact`, `month`, `day_of_week` and `campaign` refer to the current/last contact; the source does not establish a pre-contact assignment or snapshot. The five macroeconomic fields are excluded in conservative v1 because release timing and revisions are not recorded. Target, row position, IDs, split and all simulation fields are never predictive inputs. The conservative design may sacrifice ranking performance compared with broader-feature published benchmarks; that is preferable to claiming unavailable information could drive the intended decision.

The saved scikit-learn Pipeline includes feature engineering, one-hot encoding, preprocessing and estimator. `pdays=999` creates `no_previous_contact=1`; its numerical recency is replaced with the median among observed training recencies, with a documented zero fallback only if none exist. Logistic Regression standardizes the four resulting numeric columns; the forest leaves their scales unchanged. Categories and scaler statistics are learned solely from training. Unknown/unseen categories remain scoreable through `handle_unknown='ignore'`, with batch diagnostics. The source `unknown` category remains a distinct recorded state.

## 4. Chronology, experiments and freeze

{split_table}

These are contiguous partitions using floor(70%), floor(15%) and the remainder. There is no shuffled stratification, oversampling or train/test mixing. Supplied order approximates future use but does not create exact timestamps or customer groups that the source does not contain. Label prevalence across the blocks is observed only for final reporting after the policy is fixed.

The bounded search was fixed in advance:

| Family | Candidates and fixed controls |
|---|---|
| Logistic Regression | `C ∈ {{0.01, 0.1, 1, 10}}`; L2, `lbfgs`, `max_iter=2000`, no class weighting; training-only scaling and encoding. |
| Random Forest | `max_depth ∈ {{8, None}} × min_samples_leaf ∈ {{10, 50}}`; 300 trees, sqrt feature sampling, no class weighting, seed 42, bounded parallelism. |
| Prior baseline | `DummyClassifier(strategy='prior')`; training subscription prevalence for every row. |

Within each learner family, choose the greatest validation Precision@20%, then AP. Between the two learner finalists, use the same criteria, then prefer Logistic Regression if equal, then stable candidate ID. Metric differences within 1e-12 are treated as ties. The prior baseline is retained as a reference, not a competitor that silently replaces the trained champion. A learner that fails to beat the baseline must still be reported honestly.

The frozen champion is **{selection['champion_family']}**, parameters `{json.dumps(validation['params'], sort_keys=True)}`. Its validation Precision@20% was **{percent(validation['precision_at_20'])}**, AP **{number(validation['ap'])}**. The complete champion and one finalist per family were saved before final evaluation. The champion was not refitted on train plus validation; preserving the exact selected fitted pipeline avoids changing its score/cutoff after the freeze. The test set evaluates those frozen finalists and cannot select another model.

The freeze manifest records exact features/exclusions, configuration, dependency versions, source/split hashes, code fingerprint, candidate identity, model hashes, tie policy and cutoff. Repeated train/evaluate commands validate the manifest and reuse matching results. Changed model code/data/artifacts require a disclosed new experiment; the guards are provenance controls, not proof against deliberate manual tampering.

## 5. Metrics, results and capacity interpretation

{metric_table(metrics)}

{comparison_text}

{drift_text}

★ indicates the validation-selected champion. **¹ PR-AUC** is the trapezoidal integral of points from `precision_recall_curve`; **² AP** is scikit-learn's non-interpolated average precision. They are different summaries and are not substituted for one another. AP is the selection tie-breaker; the requested PR-AUC is separately reported. [Scikit-learn metric definition](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html).

For an all-test budget of 20%, `k = ceil(0.20 × {champion['n']:,}) = {champion['k']:,}`. Sort descending score; resolve ties by ascending SHA-256 of `42:<record_id>`, then ID. Precision is the {champion['tp_k']:,} captured subscriptions divided by k. Recall divides captured subscriptions by all {champion['positives']:,} test subscriptions. Lift divides precision by the {percent(champion['base_rate'])} overall test rate. Actual coverage is {percent(champion['coverage'])} because of rounding.

The prior baseline provides constant scores and therefore ROC-AUC 0.5 on a two-class set; its exact top-k result depends on the fixed label-independent tie subset. In this run baseline top-20% precision is {percent(baseline['precision_at_20'])}, lift {number(baseline['lift_at_20'], 3)} and recall {percent(baseline['recall_at_20'])}. Analytical random-contact expectation is base-rate precision, unit lift and approximately budget-fraction recall; it is shown as an expectation, not a fabricated sampled experiment. A degenerate constant-score curve can yield a trapezoidal PR-AUC unlike its AP, which is another reason to label both definitions.

{cutoff_text}

Official all-test metrics are computed **without simulated suppression**. In the operational demonstration, apply simulated rules first, then compute capacity as a fraction of the remaining eligible pool or a capped integer. An optional frozen cutoff can leave capacity unused. These different denominators must not be mixed. Changing UI capacity, filters or simulation does not rerun the locked model comparison.

No confidence interval or significance claim is made. Without real customer IDs, an independent-row bootstrap could understate uncertainty from repeat customers and temporal dependence. Rounded point estimates are offline summaries. A present-day deployment would require prospective evidence rather than extrapolating this single temporal holdout.

![Historical subscriptions captured at varying budgets](figures/cumulative_gains.png)

## 6. Explanation, reliability, drift and group review

Raw-feature permutation importance shuffles one original input on validation, measures the AP decrease, repeats ten times with seed 42, and uses the frozen champion. It does not refit the model or inspect test data to tune it. Correlated fields can share importance; negative values are valid. These are global predictive sensitivities, not causal effects or individualized reasons:

{importance_table}

The interface instead gives a prospect's factual prior-history context and clear simulation reasons. It does not claim “this customer subscribed because of age” or invent a personalized SHAP explanation. The relationship-manager prompt asks about saving goals, liquidity needs, tenure preference and willingness to discuss a product.

Reliability uses ten quantile bins of test scores versus observed subscription rates; Brier score is {number(champion['brier'])}. No probability calibrator is fitted in v1. These diagnostics distinguish useful ordering from trustworthy probability estimates. The split-rate table documents temporal prevalence shift. Without complete dates or feature-release history, it is not possible to attribute drift to one macroeconomic event or estimate a release-aligned backtest.

![Test score reliability](figures/reliability.png)

Descriptive audits apply the same frozen global top-20% selection across age/job groups, rather than silently allocating a separate 20% to each group. Counts remain visible. Base and selection rates require n≥30; precision requires selected n≥30; recall requires at least ten group subscriptions. Unsupported rates are suppressed rather than overinterpreted. Age results are below; full job results are in `demo/group_audit.csv` and the application.

{group_table}

Differences can reflect historical selection, recorded profiles, outcomes, sparse data or model behavior. They do not certify fairness, nondiscrimination, legal compliance or suitability. The data does not contain every relevant group attribute, and demographic proxies remain possible.

## 7. Engagement policy, artifact safety and reproduction

The simulation is deliberately separate from modeling: SHA-256-derived demonstration flags represent opt-out, permission and recent contact. Opt-out, absent simulated permission, recent contact and missing simulation fields suppress outreach regardless of score; all failed reasons are retained. The nominal hash cutoffs do not imply real consent rates or a real bank policy. No app, push, email or SMS outcome is learned from telephone records.

The app verifies demo-file hashes and exact key package versions before loading its trusted pipeline. It never loads user-uploaded serialized models or trains on startup. Batch validation is atomic, safe exports omit y and personal contact details, and uploads remain in memory. The CLI outputs only the requested safe score table and companion metadata. These prototype controls do not substitute for production privacy, access management, retention rules or independent validation.

Reproduce from the README's Python 3.11 locked environment using `fetch-data`, `prepare`, `analyze --split train`, `train`, `evaluate --split test`, `package-demo`, then `python scripts/build_docs.py`. Execute `ruff check .` and `pytest -q`; CI configuration runs repository checks. Test execution evidence should be read from the actual run/CI, rather than inferred from the presence of tests. Source provenance and model environment are below:

| Evidence | Exact value |
|---|---|
| CSV SHA-256 | `{selection['source_sha256']}` |
| Split map SHA-256 | `{selection['split_sha256']}` |
| Selection JSON SHA-256 | `{selected_hash}` |
| Model code fingerprint | `{selection['code_fingerprint']}` |
| Champion artifact SHA-256 | `{selection['model_hashes']['champion.joblib']}` |
| Seed | 42 |
| Local training commit | `{selection.get('code_commit_at_training') or 'not yet committed at training; code fingerprint above identifies model source'}` |

The training commit was captured in the local authoring checkout. It is not asserted to exist in the public GitHub history or to identify the published release. The source/split hashes, modeling-code fingerprint and serialized-model checksum above identify the exact experimental inputs and outputs.

Dependency versions: `{json.dumps(selection['versions'], sort_keys=True)}`. Timestamps and commit metadata may differ in a fresh disclosed run; model ID incorporates data, configuration, relevant code and dependency versions. Prediction reproduction should be checked with the locked environment and verified input hashes; cross-platform numerical differences are possible. Serialization scores are verified on a fixed sample within 1e-12 by the project checks.

## 8. Limits and next responsible experiment

Historical subscription propensity is not uplift: a high-scoring customer may have subscribed without an additional contact. There is no untreated comparison, randomized assignment, deposit amount or cost record. Consequently, do not turn captured historical labels into incremental conversions, revenue, ROI or a guaranteed campaign gain. Do not claim an optimal digital channel from telephone history.

The project demonstrates a functioning analytics-to-review workflow, SQL discipline, leakage-aware ML and honest communication. For real use, first obtain contemporary permissioned data and timestamped pre-contact snapshots, validate customer-level and temporal splits, review suitability/engagement policies with accountable teams, assess support and disparities, and evaluate contact benefit in a controlled prospective pilot. Monitor input/schema drift, prevalence, ranking, calibration, suppression and outcomes with a versioned rollback path. This is an independent prototype, not a model deployed at or endorsed by ICICI Bank.

Data sources and credit: [UCI Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing), the official additional-dataset readme, and [DATA_LICENSE.md](../DATA_LICENSE.md). The report's numerical sections are generated from frozen project artifacts, not estimates or illustrative examples.
"""
    (ROOT / "README.md").write_text(readme)
    (ROOT / "reports/model_report.md").write_text(report)
    guide_path = ROOT / "docs/project_understanding_and_interview_guide.md"
    if not guide_path.exists():
        source = ROOT.parent / "outputs/project_understanding_and_interview_guide.md"
        if not source.exists():
            raise SystemExit("Companion learning guide missing; restore original outputs guide")
        guide_path.write_text(source.read_text())
    guide = guide_path.read_text()
    if START in guide and END in guide:
        guide = (guide[:guide.index(START)].rstrip() + "\n\n"
                 + guide[guide.index(END) + len(END):].lstrip("\n"))
    original_status = (
        "**Status: design and learning guide. The project has not yet been implemented or "
        "evaluated.** Commands, screenshots, trained artifacts, and measured results must be "
        "added after implementation. Every numerical performance example in this guide is "
        "explicitly hypothetical. Use the accompanying [plan.md](plan.md) as the "
        "implementation specification."
    )
    guide = guide.replace(original_status, "**Status: implemented and evaluated prototype.** "
                          "Read the verified results section above first. Original worked "
                          "arithmetic examples below remain explicitly hypothetical. Use "
                          "[the implementation plan](../plan.md) for design intent and the "
                          "[README](../README.md) for runnable commands.")
    guide = guide.replace("[Build plan](plan.md)", "[Build plan](../plan.md)")
    guide = guide.replace("[Implementation plan](plan.md)", "[Implementation plan](../plan.md)")
    guide = guide.replace("[plan.md](plan.md)", "[plan.md](../plan.md)")
    guide = guide.replace("what the proposed system does", "what the system does")
    for heading in ["What the four dashboard pages should do", "What the four dashboard pages do"]:
        guide = guide.replace(heading, "What the five dashboard pages do")
    for anchor in ["what-the-four-dashboard-pages-should-do", "what-the-four-dashboard-pages-do"]:
        guide = guide.replace(anchor, "what-the-five-dashboard-pages-do")
    guide = guide.replace("Before implementation, mark this slide “results pending.”",
                          "The delivered slides use the frozen metrics.")
    guide = guide.replace("After implementation, name an actual decision and evidence.",
                          "Name an actual implementation decision and its evidence.")
    guide = guide.replace("Fill placeholders from the final evaluation artifact and retain the "
                          "historical context.", "These figures come from the final evaluation "
                          "artifact; retain the historical context.")
    guide = guide.replace(
        "The earlier [ICICI Project Recommendation conversation](chatgpt-conversation://"
        "6ad5e57d-4534-83e8-be94-116f7904bd7a) supplies background. The current specification "
        "narrows its earlier Power BI/clustering suggestions to one manageable Streamlit project.",
        "The final specification uses one manageable Streamlit project for analysis and scoring.")
    guide = guide.replace(
        "My measured results were **[insert verified values, population, and artifact reference]**.",
        f"On the 6,179-record chronological test, the top 1,236 achieved "
        f"{percent(champion['precision_at_20'])} precision, "
        f"{percent(champion['recall_at_20'])} recall and "
        f"{number(champion['lift_at_20'], 3)}× lift; see the frozen metrics and model report.")
    guide = guide.replace("once implemented.", "in the implemented repository.")
    guide = guide.replace(
        "These are answer structures, not claims that work has already been completed. "
        "Replace future tense only when the corresponding evidence exists.",
        "These answer structures explain the completed prototype. Support each claim with "
        "the actual evidence in the verified section above and the repository.")
    section_start = guide.find("## 12. ")
    section_end = guide.find("\n## 13. ", section_start)
    if section_start >= 0 and section_end > section_start:
        dashboard_section = """## 12. What the five dashboard pages do

The default **Engagement hub** is a graphical product welcome. It introduces the business question with an illustration and interactive chart previews, then provides clear routes to the analytical tools. The first visit begins with the decision and visual evidence rather than a raw records table. Hero artwork is illustrative and independent of ICICI Bank. All chart values come from the actual trusted public-data/model bundle, while eligibility is explicitly simulated.

| Page | What the user sees | What you explain in the demo |
|---|---|---|
| Engagement hub | Welcome illustration, campaign-pattern bars, test capacity/capture curve, simulated outreach donut, budget slider and navigation buttons | A quick route from the business question to real historical evidence; simulation stays separate |
| Campaign overview | Data provenance, record counts, target distribution and SQL segment summaries | These are historical associations with visible denominators |
| Model results | Split sizes, feature boundary, baseline/LR/RF comparison, curves, budget trade-off and limitations | Selection used validation; the later slice was held out |
| Outreach shortlist | Capacity, simulated policy controls, eligible/suppressed counts, sorted records and export | Policy is applied before capacity; the score supports prioritization |
| Scoring sandbox | Synthetic single-record input or schema-validated upload using the saved pipeline | New inputs reuse the fitted transformations; no training occurs here |

The hub hero buttons are **Explore analysis →**, which opens Campaign overview, and **Plan outreach →**, which opens Outreach shortlist. The three previews are **Campaign patterns** (full-history previous-outcome subscription rates), **Capacity & capture** (the frozen champion's test cumulative-gains curve, random reference and budget point), and **Simulated outreach** (test-pool eligible/suppressed shares and selected count). Their buttons are **Open campaign analysis**, **Open model results**, and **Open outreach shortlist**. **Preview contact capacity (%)** spans 1–50%, starting at 20%; it changes the preview budget point and simulated count without altering the official frozen metrics or model. **Open outreach shortlist** and **Plan outreach →** carry that chosen percentage into the shortlist control. Analytical pages return through **← Back to engagement hub**. Full-history descriptions are appropriate here because selection is already frozen; development-stage analysis remains training-only.

The visible navigation bar offers **Home**, **Campaign insights**, **Model results**, **Outreach**, and **Try scoring**. It stays in step with the optional sidebar and page URL. Shareable page links use `?view=home`, `?view=analysis`, `?view=model`, `?view=outreach`, or `?view=score`. The header and preview cards adapt to smaller screens. A page change returns to its heading; an ordinary filter or capacity adjustment keeps the reading position.

All five pages identify the historical public-data prototype and model run. Results come from stored evaluation artifacts; a missing/incompatible bundle produces a helpful setup message rather than fabricated numbers. Loading the saved pipeline verifies hashes and compatible package metadata. Only trusted local serialized artifacts are loaded: model files can execute code during deserialization.

The sandbox model receives only approved raw features. Separate optional simulated policy inputs go to the eligibility layer. A single-record form assigns a synthetic session ID; batch CSVs require unique record IDs. Age, previous and pdays must be finite integer-valued inputs with documented bounds. Known excluded source columns such as duration or y are ignored with warnings and never forwarded to the estimator; unexpected or malformed fields reject the batch. Operational-style exports omit historical targets and personal contact details. A clearly labelled historical replay supports teaching and evaluation.

Streamlit delivers the welcome, analytics and scoring in one application. A public demo is available through the verified link in the results section; local execution and the trusted saved bundle provide the interview fallback. The prototype needs no live bank integration or customer-contact service.
"""
        guide = guide[:section_start] + dashboard_section + guide[section_end:]
    guide = guide.replace(
        '| 0:00–0:40 | State problem and source boundary | “The constraint is contact capacity; this is public historical data.” |',
        '| 0:00–0:40 | Begin at Engagement hub: state the problem, inspect chart previews and adjust preview capacity | “These charts use frozen historical artifacts; eligibility is simulated.” |')
    guide = guide.replace(
        '| 0:40–1:25 | Show overview and one SQL segment chart | “I show the denominator because rates from tiny groups are unstable.” |',
        '| 0:40–1:25 | Click Explore analysis → or Open campaign analysis; show one SQL segment chart | “I show the denominator because rates from tiny groups are unstable.” |')
    evidence_start = guide.find("After implementation, complete this evidence sheet")
    evidence_end = guide.find("\n## 20.", evidence_start)
    if EVIDENCE_START in guide and EVIDENCE_END in guide:
        evidence_start = guide.index(EVIDENCE_START)
        evidence_end = guide.index(EVIDENCE_END) + len(EVIDENCE_END)
    if evidence_start >= 0 and evidence_end > evidence_start:
        evidence = table(["Evidence", "Verified implementation", "Where to inspect"], [
            ["Source and readme", "Exact variant and checksum recorded", "`data/raw/provenance.json`; data dictionary"],
            ["Splits and rates", "28,831 / 6,178 / 6,179, with measured rates", "`demo/split_summary.json`; report"],
            ["LR/RF candidates", "Eight learner candidates plus prior, validation ranked", "`demo/candidates.json`; manifest"],
            ["Frozen champion", selection["champion_id"], "`demo/manifest.json`"],
            ["Held-out metrics", "Measured PR-AUC, AP, ROC-AUC, P/R/Lift@20%", "`demo/metrics.json`; report"],
            ["Cutoff and test volume", f"{selection['validation_cutoff']:.6f}; {champion['fixed_cutoff_count']:,} records", "Report and model-results page"],
            ["SQL insight", "Prior-outcome counts and denominators generated", "`reports/sql/by_poutcome_train.csv`"],
            ["Global explanation", "Validation raw-column permutation AP", "`demo/importance.csv`"],
            ["Opt-out and export", "Separate rule and safe schema implemented", "`tests/test_scoring_engagement.py`; demo script"],
            ["Quality checks", check_status, "Actual shell/CI check output"],
            ["Reproduction", "Locked install and run-all documented", "README; model/source hashes"],
            ["Repository", args.github_url or "Publication pending", "Verified link in README when published"],
            ["Hosted demo", args.demo_url or "External hosting pending; local app available", "Verified live link when deployed"],
        ])
        guide = guide[:evidence_start] + (
            EVIDENCE_START + "\nThis evidence sheet is generated from the completed artifact bundle. "
            "Check counts and figures against these files before rehearsal.\n\n" + evidence
            + "\n" + EVIDENCE_END + "\n\n"
        ) + guide[evidence_end:].lstrip("\n")
    guide = guide.replace(
        "selected **[model]** achieved **[measured Precision@20%]**, "
        "**[measured Recall@20%]**, and **[measured Lift@20%]**",
        f"selected **{selection['champion_family']}** achieved "
        f"**{percent(champion['precision_at_20'])} Precision@20%**, "
        f"**{percent(champion['recall_at_20'])} Recall@20%**, and "
        f"**{number(champion['lift_at_20'], 3)}× Lift@20%**")
    verified = f"""{START}
## Verified implementation results — read this first

**Status:** the local pipeline, frozen model artifacts and Streamlit prototype have been implemented. Numerical results below are read directly from the frozen evaluation. Test/hosting evidence must be checked from the actual delivery run; an old planning paragraph below is not evidence of execution.

{check_status}

{publication}

{headline}

{split_table}

{metric_table(metrics)}

{comparison_text}

{drift_text}

The champion was `{selection['champion_id']}` with `{json.dumps(validation['params'], sort_keys=True)}`. Choose validation Precision@20%, then AP, then Logistic Regression, then stable candidate ID. Do not pick a model again from the test table. PR-AUC here is trapezoidal; AP is the separate non-interpolated average precision. The all-test ranking metrics precede simulation and use 1,236 shortlisted records.

{cutoff_text}

The score budget and fixed score cutoff are different policies. In the dashboard's operational simulation, suppress ineligible records first and calculate capacity from the eligible pool. Percentage budgets use ceiling rounding. An opt-out changes eligibility, never the underlying score. Source row IDs are observation IDs, not unique bank customers.

{sql_finding}

The three largest validation permutation-AP decreases are {', '.join(f"{r['feature']} ({float(r['importance_mean']):.4f})" for r in importance[:3])}. This is a global sensitivity check, not causation or an individual customer's explanation. Full measured diagnostics and data quality are in [the model report](../reports/model_report.md). Quality checks retained {quality['duplicate_extra_rows']} extra exact duplicate rows; {quality['cross_split_duplicate_fingerprints']} exact-row fingerprints crossed the split boundary. This cannot rule out repeat customers with differing rows.

### Completed-project 90-second interview answer

“I built an independent banking analytics prototype to help relationship teams prioritize a limited term-deposit contact budget. I used 41,188 public Portuguese telephone-campaign observations, preserving their order in a 70/15/15 split. Python handles preparation, SQLite answers campaign questions, scikit-learn trains models, and Streamlit delivers the workflow.

“Pre-contact scoring uses ten profile and previous-campaign inputs, excluding call duration and current-contact fields. I compared Logistic Regression and Random Forest. {selection['champion_family']} won validation Precision@20%. On 6,179 later test observations, its 1,236-record shortlist captured {champion['tp_k']:,} of {champion['positives']:,} subscriptions: {percent(champion['precision_at_20'])} precision, {percent(champion['recall_at_20'])} recall and {number(champion['lift_at_20'], 2)} times lift.

“Logistic Regression did better on test, at {percent(lr_test['precision_at_20'])} precision. I report the reversal while keeping the validation choice fixed. Prevalence shifted from {percent(splits[0]['base_rate'])} in training to {percent(splits[-1]['base_rate'])} in test, so contemporary validation is essential.

“The dashboard separates scores from simulated opt-out and permission rules, which are not recorded consent. I show reliability, drift and support-limited group checks. Scores support a needs-based conversation, not suitability or causal sales claims. This uses no ICICI customer data and is not an ICICI deployment.”

### Commands and evidence to show

```bash
python -m bank_engagement.cli run-all
python scripts/build_docs.py
python -m streamlit run app/streamlit_app.py
python -m bank_engagement.cli score --input samples/prospects.csv --engagement samples/engagement_simulated.csv --output artifacts/scored.csv --capacity 5
ruff check .
pytest -q
```

Run from the root after the locked Python 3.11 install in [README](../README.md). `demo/manifest.json` identifies the model and bundle checksums; `demo/metrics.json` is the measured comparison; `data/raw/provenance.json` records the exact source. Existing matching train/evaluate stages are reused; do not edit the final holdout to tune the story.

The rest of this guide teaches the reasoning and includes explicitly hypothetical arithmetic. Use the verified numbers above for the completed project, not hypothetical example numbers. Historical planning instructions below explain the original intended workflow.
{END}

"""
    first_break = guide.find("\n") + 1
    guide = guide[:first_break] + "\n" + verified + guide[first_break:]
    guide_path.write_text(guide)
    print("Wrote README.md, reports/model_report.md and measured learning-guide section")


if __name__ == "__main__":
    main()
