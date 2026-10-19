# Five-slide interview speaker notes

## 1. Business problem and alignment

The business question is capacity allocation. If a relationship manager can speak to only a limited number of prospects, the model supplies an ordered starting point. A propensity score estimates subscription patterns under historical campaign conditions. It does not establish whether a term deposit suits the customer. I connect the prototype to customer engagement and sales-oriented relationships in the campus deck. The listed business groups are indicative, and the bank retains discretion over role and location. This independent prototype has no ICICI customer data or bank deployment.

**Sources:** UCI Bank Marketing: https://archive.ics.uci.edu/dataset/222/bank+marketing
User-supplied FY2027 Campus Proposition deck, slides 3, 8 and 10. The deck serves as alignment context only. No campus source PDF or ICICI logo is included in this deliverable.

## 2. Data and pipeline

The exact UCI additional-full variant contains 41,188 observations, 20 original input fields, and the yes/no subscription target. These are campaign observations rather than verified unique customers. The source order is chronological but lacks enough information to reconstruct exact row dates. Python validates and cleans the file while retaining repeated rows. SQLite stores the reproducible record, split, model, and score tables. SQL supports descriptive segment analyses. The model receives a conservative pre-contact feature allowlist. A saved pipeline supplies scores to Streamlit. The app never retrains when a control changes. The final test records support a clearly labeled historical replay. The data cannot establish which app, email, or notification channel works best.

**Sources:** UCI Bank Marketing: https://archive.ics.uci.edu/dataset/222/bank+marketing
Project artifacts: artifacts/selection.json, artifacts/metrics.json, artifacts/budget_curve.csv. Model rf-d8-l10-32e674d224. Source CSV SHA256 74adfc578bf77a7ff4bb1ba4a9f8709d9e3c6907342959c2c8416847e0afb4d8.

## 3. Leakage and evaluation

Duration is known only after a call, so including it would leak future information. The model also excludes contact, month, day_of_week, campaign, the five macro indicators, target, and metadata. Its raw feature allowlist is age, job, marital, education, default, housing, loan, pdays, previous, and poutcome. Feature engineering handles pdays=999 as no previous contact. Numeric scaling, imputation, and encoding fit on training rows only. Four LR and four RF candidates form a bounded search, alongside a constant prior baseline. The validation rule chooses the strongest Precision@20%, uses average precision as the next tie-breaker, then prefers LR on an exact family tie. A deterministic score/ID hash policy settles tied ranks without looking at outcomes. No refit or test-driven reselection follows the freeze. PR-AUC here uses trapezoidal integration, while AP is a separate metric.

train: 28831 rows, 1606 positives, prevalence 5.6%
validation: 6178 rows, 658 positives, prevalence 10.7%
test: 6179 rows, 2376 positives, prevalence 38.5%

Frozen champion: Random Forest. Frozen validation score cutoff: 0.0679219814659131.

**Sources:** UCI Bank Marketing: https://archive.ics.uci.edu/dataset/222/bank+marketing
Project artifacts: artifacts/selection.json, artifacts/metrics.json, artifacts/budget_curve.csv. Model rf-d8-l10-32e674d224. Source CSV SHA256 74adfc578bf77a7ff4bb1ba4a9f8709d9e3c6907342959c2c8416847e0afb4d8.
scikit-learn common pitfalls: https://scikit-learn.org/stable/common_pitfalls.html
Average precision definition: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html

## 4. Measured findings

All displayed figures come from frozen project outputs. The champion is Random Forest, selected on validation Precision@20%=18.7% and AP=0.191. The test shortlist size is ceil(0.20 × 6179) = 1236. The champion captures 632 recorded test subscriptions. Precision is 632/1236=51.1%. Recall is 632/2376=26.6%. Lift divides shortlist precision by overall test prevalence, giving 1.33. The prior baseline assigns every row the training prevalence, so its deterministic tied shortlist can differ from the analytical random-contact expectation. The grey gain reference describes that expectation. The chart samples measured cumulative recall at 10%, 20%, 30%, 50%, 70%, and 100% capacities. The table reports trapezoidal PR-AUC and average precision separately. For a constant-score baseline, trapezoidal endpoint interpolation can inflate PR-AUC, while AP equals the evaluation prevalence. Model scores have calibration and population-shift limits. This historical ranking evaluation cannot estimate incremental conversions caused by outreach.

**Sources:** Project artifacts: artifacts/selection.json, artifacts/metrics.json, artifacts/budget_curve.csv. Model rf-d8-l10-32e674d224. Source CSV SHA256 74adfc578bf77a7ff4bb1ba4a9f8709d9e3c6907342959c2c8416847e0afb4d8.
Logistic Regression achieved test Precision@20% of 60.0%, exceeding the frozen Random Forest's 51.1%. The project preserves the validation-selected Random Forest to avoid test-driven reselection. The gain chart rounds measured values to eight decimal places for portable editable chart storage.

## 5. Demo and judgment

A five-minute demonstration opens the campaign overview, checks denominators, then shows the frozen model comparison. The outreach page labels its test pool as historical replay. Choose a capacity, inspect the simulated eligibility count, and demonstrate that a high-scoring simulated opt-out disappears from the eligible shortlist while keeping its score unchanged. Explain the difference between model score and business permission. Show one record and factual prior-contact context, then download a CSV with surrogate IDs and no y. The scoring sandbox uses only public or synthetic inputs and the same trained pipeline. A relationship manager still checks liquidity needs, risk understanding, product terms, and suitability. The system offers no automatic product recommendation. No digital-channel effectiveness claim follows from the telephone dataset. Before a real deployment, obtain fresh governed data, validated consent and suitability workflows, drift monitoring, and a prospectively designed evaluation of incremental benefit. Future real outreach requires separate authorization.

**Sources:** UCI Bank Marketing: https://archive.ics.uci.edu/dataset/222/bank+marketing
Project artifacts: artifacts/selection.json, artifacts/metrics.json, artifacts/budget_curve.csv. Model rf-d8-l10-32e674d224. Source CSV SHA256 74adfc578bf77a7ff4bb1ba4a9f8709d9e3c6907342959c2c8416847e0afb4d8.
The engagement rules in the demo use explicitly simulated data. Public prototype, no ICICI affiliation or production deployment.
