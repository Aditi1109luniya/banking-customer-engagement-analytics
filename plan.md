# Banking Customer Engagement & Deposit Propensity Analytics
## End-to-end implementation brief for a GPT-Astra build agent

**Status:** original implementation specification. The prototype is now built and evaluated. See README.md and reports/model_report.md for measured offline results and verification. No incremental sales or revenue effects have been established.

**Companion:** [Project understanding and interview guide](docs/project_understanding_and_interview_guide.md).

**Intended pace:** 7–10 focused working days for someone with basic Python and SQL familiarity. The agent should implement incrementally, teach the owner at each milestone, and report evidence rather than declare completion from code alone.


<details>
<summary>Contents</summary>

- [1. Start here: agent instructions and finish line](#1-start-here-agent-instructions-and-finish-line)
- [2. Resolve the recommendations into one manageable project](#2-resolve-the-recommendations-into-one-manageable-project)
- [3. Development environment, GitHub, and working rules](#3-development-environment-github-and-working-rules)
- [4. Architecture and repository contract](#4-architecture-and-repository-contract)
- [5. Dataset acquisition, schema, and provenance](#5-dataset-acquisition-schema-and-provenance)
- [6. Feature availability and leakage policy](#6-feature-availability-and-leakage-policy)
- [7. SQLite data model and SQL campaign analysis](#7-sqlite-data-model-and-sql-campaign-analysis)
- [8. Chronological splits, experiments, and locked evaluation](#8-chronological-splits-experiments-and-locked-evaluation)
- [9. Explanations, group checks, and honest limitations](#9-explanations-group-checks-and-honest-limitations)
- [10. Separate simulated engagement policy](#10-separate-simulated-engagement-policy)
- [11. Scoring and artifact contracts](#11-scoring-and-artifact-contracts)
- [12. Streamlit application specification](#12-streamlit-application-specification)
- [13. Test plan and phase gates](#13-test-plan-and-phase-gates)
- [14. Reporting and interview deliverables](#14-reporting-and-interview-deliverables)
- [15. Local demo packaging and optional deployment](#15-local-demo-packaging-and-optional-deployment)
- [16. Final acceptance and handoff](#16-final-acceptance-and-handoff)
- [17. Source notes and remaining external choices](#17-source-notes-and-remaining-external-choices)

</details>

---

## 1. Start here: agent instructions and finish line

### Copy this prompt into the build agent

> Read this plan and its companion guide completely. Build the project described here, proceeding through the phase gates in order. Inspect the repository and applicable AGENTS.md first. Preserve existing work. Keep a progress checklist with completed steps, evidence, and blockers. Use the fixed data, feature, evaluation, and engagement policies in this plan; do not change them to obtain attractive results. Implement a reproducible Python package, SQLite analysis, two model families, offline evaluation, a Streamlit demo, tests, documentation, and a five-slide interview presentation. Run the prescribed checks, fix material defects, and update the learning guide with verified results and exact commands. Continue local work if an account connection is unavailable. Ask only for genuinely missing credentials, a repository destination/visibility, or an external publishing decision when it becomes necessary. Never fabricate measurements, claim ICICI affiliation, contact people, or upload real customer data. Finish by providing runnable commands, artifact locations, test evidence, measured limitations, and a short owner walkthrough.

### One business question

**With capacity to contact only a limited number of prospects, which records should the team prioritize for a term-deposit conversation?**

The system estimates historical subscription propensity, ranks eligible records, and shows the relationship between shortlist size and historical subscriptions captured. It supports a relationship manager's conversation. It does not establish suitability, customer need, consent, incremental impact, or revenue.

### Required end state

1. A repository that a fresh user can install and reproduce from documented commands.
2. Explicit dataset provenance, schema validation, preserved ordering, and SQLite tables.
3. SQL campaign analysis with counts, denominators, and descriptive interpretations.
4. A training-only preprocessing pipeline; Logistic Regression and Random Forest; a simple baseline.
5. Validation-based model and cutoff selection, followed by one locked chronological test evaluation.
6. Saved preprocessing plus model, an auditable manifest, scores, metrics, and figures.
7. A Streamlit app with four useful pages and a downloadable simulated outreach shortlist.
8. Meaningful unit/integration/app checks and a GitHub Actions workflow.
9. README, data dictionary, short model report, updated learning guide, and five-slide presentation with speaker notes.
10. An interview demo that works locally without an internet connection after preparation.

**A valid result may show limited or negative lift.** Completion means a truthful, functioning, reproducible prototype, not a promised score.

## 2. Resolve the recommendations into one manageable project

The earlier conversation introduced credit risk, Power BI, clustering, and a separate prediction application. The final recommendation changes the emphasis to customer engagement and term-deposit prioritization. Implement **one Streamlit application** for analysis, model results, and scoring. Use descriptive segments and score deciles; do not add clustering merely to increase the tool count.

| Decision | Default for v1 | Reason |
|---|---|---|
| Business framing | Conversation prioritization under a contact budget | Connects analytics with relationship work and sales |
| UI | Streamlit only | One application is achievable and easier to demonstrate |
| Storage | Local SQLite, built from the source CSV | Demonstrates SQL without an unnecessary database service |
| Models | Logistic Regression and Random Forest | Interpretable reference and nonlinear comparison |
| Scoring moment | Immediately before a proposed historical contact | Defines which features are allowed |
| Evaluation | Ordered 70% / 15% / 15% | Tests generalization to later supplied records |
| Core features | Customer attributes and previous-campaign history | Conservative availability policy |
| Engagement | Separate, deterministic simulation | Consent and opt-out are absent from the source |
| Public services | GitHub; optional Streamlit Community Cloud | No cloud backend, paid API, or LLM is required |
| Repository visibility | Decide with owner before creation/publication | No visibility choice was supplied |
| Initial delivery | Fully reproducible local demo | Public hosting must not block completion |

### Connection to the campus deck

The supplied FY2027 Campus Proposition deck was checked at slides 3, 8, and 10. Use these connections in the report and slides:

| Deck theme | Project evidence | Boundary to explain |
|---|---|---|
| Slide 3: iMobile, InstaBIZ, DigiEase, website experience, AI for engagement | A practical digital dashboard delivering analytics to a relationship manager | The model learns from telephone campaigns; it does not evaluate those digital products |
| Slide 8: Customer 360, fairness, risk/compliance, sales and relationships | A record-level context card, opt-out suppression, honest model limits, and a prioritized conversation list | The dataset cannot assemble a true Customer 360 or certify fairness |
| Slide 10: indicative business groups and sales orientation | Python/SQL/ML combined with a business action and customer conversation | These groups do not guarantee a role; the bank retains role/location discretion |

Paraphrase the relevant themes. Do not publish the supplied campus PDF, its screenshots, branding, or logos in the public repository by default. Identify this as an **independent banking analytics prototype aligned with the campus deck**.

## 3. Development environment, GitHub, and working rules

### Minimal toolchain

- Python 3.11, a project virtual environment, Git, and an editor/terminal.
- Runtime dependencies: pandas, numpy, scikit-learn, scipy, joblib, streamlit, matplotlib, and plotly. SQLite is in Python's standard library.
- Development dependencies: pytest, ruff, and pip-tools. Use Streamlit AppTest from the installed Streamlit package.
- Use TOML configuration and Python's tomllib; avoid a configuration framework.
- Add python-pptx only when needed for the slide deliverable, or use an available presentation skill. Follow its rendering/verification workflow if used.
- Choose compatible released dependency versions at implementation time, resolve on Python 3.11, and commit exact lock files. Do not copy version numbers from an unrelated system environment.

### GitHub connection and publication

The GitHub account connection was verified in this planning session for **Aditi1109luniya**. No repository was created or modified. Recheck repository-specific permissions at build time; account connection does not establish write access to every repository.

Suggested repository name: **banking-customer-engagement-analytics**. Before creating or publishing, resolve the owner and visibility with the owner if they remain unspecified. Do not overwrite an existing repository or assume a similarly named repository is the correct destination.

Use Git locally from the first working milestone. Make small commits for data, analysis, modeling, app, and documentation. Use a GitHub connector or authenticated Git/CLI only when available. Do not install unrelated plugins. Do not expose tokens in source, commands, screenshots, notebooks, or logs.

Create a feature branch and draft PR when working in an established repository. For an empty personal repository, use a clear initial history and README. If the hosting environment requires attaching created PRs to the chat, do so.

### Keep generated and private material controlled

- Ignore virtual environments, caches, local secrets, raw archives, bulk intermediate data, and temporary notebooks.
- Commit source, tests, SQL, configuration, locks, attribution, reports, figures, and small public demo assets.
- Treat model files as trusted, versioned build outputs. Never load an uploaded pickle or joblib file.
- Do not copy the campus deck or a private conversation transcript into the public repository.
- Keep modeling logic in the package. Notebooks, if used, are optional exploration records.
- No real outreach, email integration, account linking to a bank, or live customer data is part of v1.

## 4. Architecture and repository contract

~~~mermaid
flowchart TD
    A[Exact UCI source archive] --> B[CSV validation and provenance]
    B --> C[Immutable row IDs and chronological splits]
    C --> D[SQLite tables and SQL analysis]
    C --> E[Core pre-contact features]
    E --> F[Training-only preprocessing]
    F --> G[Logistic Regression and Random Forest]
    G --> H[Validation selection and policy freeze]
    H --> I[Locked test evaluation]
    H --> J[Saved complete model pipeline]
    J --> K[Scores and factual context]
    L[Separate simulated engagement flags] --> M[Eligibility and suppression]
    K --> M
    M --> N[Capacity ranking and CSV export]
    D --> O[Streamlit dashboard]
    I --> O
    N --> O
~~~

The trained model never receives the target, record ID, split, simulated engagement fields, or excluded source columns. Historical labels are joined to predictions only by the evaluation layer.

### Proposed repository structure

~~~text
banking-customer-engagement-analytics/
  README.md
  plan.md
  pyproject.toml
  requirements.in
  requirements.txt
  requirements-dev.in
  requirements-dev.txt
  .gitignore
  .github/workflows/ci.yml
  configs/project.toml
  src/bank_engagement/
    cli.py
    data.py
    database.py
    features.py
    modeling.py
    evaluation.py
    scoring.py
    engagement.py
    artifacts.py
  sql/                         # named, parameterized analysis queries
  app/streamlit_app.py
  tests/                       # includes tiny synthetic fixtures
  samples/prospects.csv
  samples/engagement_simulated.csv
  data/raw/                    # ignored; original archive/CSV/readme
  data/processed/              # ignored; SQLite database and split mapping
  artifacts/                   # ignored bulk outputs; referenced by manifest
  demo/                        # trusted compact public-data demo bundle
  reports/model_report.md
  reports/figures/
  docs/data_dictionary.md
  docs/project_understanding_and_interview_guide.md
  docs/demo_script.md
  docs/presentation/           # five slides, PDF, and notes
~~~

This is a small package, not a distributed system. Do not introduce FastAPI, Docker, Airflow, MLflow, a feature store, a vector database, or a live CRM dependency in v1.

### Required command interface

Implement these entry points and make their defaults come from configs/project.toml:

~~~bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m pip install -e . --no-deps

python -m bank_engagement.cli fetch-data
python -m bank_engagement.cli prepare
python -m bank_engagement.cli analyze --split train
python -m bank_engagement.cli train
python -m bank_engagement.cli evaluate --split test
python -m bank_engagement.cli score --input samples/prospects.csv --output artifacts/scored.csv --capacity 100
python -m bank_engagement.cli package-demo
python -m streamlit run app/streamlit_app.py

ruff check .
pytest -q
~~~

Document Windows activation separately. Also provide:

- fetch-data --archive PATH for a manually downloaded official archive when network access fails.
- prepare --csv PATH for the exact source CSV plus documented provenance.
- score --engagement PATH for explicitly simulated eligibility data.
- score --min-score NUMBER for the optional cutoff mode; capacity remains a hard cap.
- A documented run-all wrapper that executes the stages in order and evaluates test only after the selection manifest has been frozen.

Each stage logs its inputs, output paths, counts, and run/model IDs. Fail with actionable messages on missing inputs or incompatible artifacts. Re-running a stage must replace that stage's outputs coherently, not append duplicate rows.

## 5. Dataset acquisition, schema, and provenance

### Use exactly this variant

Source: [UCI Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing), DOI [10.24432/C5K306](https://doi.org/10.24432/C5K306).

Use **bank-additional-full.csv**: 41,188 observations, 20 input columns, binary y, supplied in chronological order from May 2008 to November 2010. The records concern historical Portuguese telephone campaigns.

**Known trap:** UCI's landing page also displays metadata for older variants. Do not substitute bank-full.csv, bank.csv, the 4,119-row bank-additional.csv sample, or an unverified ucimlrepo fetch. Those alternatives do not satisfy this specification.

The official download linked by UCI is currently:

~~~text
https://archive.ics.uci.edu/static/public/222/bank+marketing.zip
~~~

Download with a timeout, bounded retry, and HTTPS validation. Inspect archive members safely; the required CSV/readme may be inside bank-additional.zip within the outer archive. Select by exact basename and verify schema. Do not extract arbitrary archive paths. If the URL changes, resolve the official link from UCI or accept an owner-supplied official archive; never silently use another dataset.

Save the original CSV and bank-additional-names.txt unchanged. Record source URL, retrieval time in UTC, archive member path, byte size, SHA-256 of the CSV, and dataset citation in a provenance JSON. The first verified build establishes the checksum; later reproduction should fail or request an explicit new data version on mismatch. Do not invent a checksum.

Credit the dataset authors and UCI; include the dataset's CC BY 4.0 attribution separately from the license chosen for the project's original code. UCI's [dataset page](https://archive.ics.uci.edu/dataset/222/bank+marketing) supplies the license and citation.

### Required raw columns

~~~text
age, job, marital, education, default, housing, loan,
contact, month, day_of_week, duration, campaign,
pdays, previous, poutcome, emp.var.rate, cons.price.idx,
cons.conf.idx, euribor3m, nr.employed, y
~~~

Parse with a semicolon delimiter. Validate 41,188 rows and the exact set of 21 columns before renaming any SQL columns. Validate target values yes/no, finite numeric source fields, and documented categorical values. Unexpected structure is an error, not an invitation to guess a new format.

### Cleaning policy

1. Assign **record_id = 1..41,188** from the original physical row order before any sorting. It is a surrogate record identifier, not a real customer ID.
2. Preserve every source row and its order. Report exact duplicates and cross-split duplicate fingerprints; do not drop duplicates by default. The dataset cannot prove whether identical records are accidental duplicates or different/repeated observations. Report contradictory prior-history combinations without inventing corrections.
3. Keep categorical unknown as an explicit category. It is not equivalent to no.
4. Normalize surrounding whitespace and categorical text consistently. Map y from no/yes to 0/1.
5. Treat pdays=999 as “not previously contacted” according to the additional dataset's readme. Do not import the older variant's -1 convention.
6. Keep valid age and other numeric values; do not cap outliers or merge categories using all-data statistics.
7. Create fixed age groups for descriptive analysis: under 30, 30–44, 45–59, and 60+. These do not replace numeric age in the model.
8. Preserve month and day_of_week for historical description only. Do not reconstruct full timestamps, years, customer identities, or calendar durations from them.

Raw quality checks may inspect structure across the file. During model development, target-based analysis is training-only. Full-history outcome summaries are unlocked after model/policy freeze.

## 6. Feature availability and leakage policy

### Scoring moment and assumptions

Score a record immediately before its proposed contact, using customer attributes and previous-campaign information assumed to have been available at that time. This is a retrospective approximation: the source has no full feature-event timestamps or release history.

### V1 allowlist

| Raw feature | Treatment | Availability assumption |
|---|---|---|
| age | Numeric | Customer attribute known before contact |
| job, marital, education | Categorical | Existing customer profile |
| default, housing, loan | Categorical yes/no/unknown | Existing recorded financial attributes |
| previous | Nonnegative numeric count | Contacts before the current campaign |
| pdays | Recency plus no-previous-contact indicator | Previous-campaign timing known before proposed contact |
| poutcome | Categorical | Outcome of the previous campaign |

These are **10 raw inputs**. Seven are categorical and three are numeric before feature engineering.

### Exclusions

| Columns | Why excluded from the trained model |
|---|---|
| duration | Only known after the call; direct leakage for pre-contact use |
| contact, month, day_of_week | Describe the recorded current/last contact; pre-contact assignment is not established |
| campaign | Includes the recorded current contact; no reliable pre-contact snapshot |
| emp.var.rate, cons.price.idx, cons.conf.idx, euribor3m, nr.employed | Economic variables may be useful, but publication timing and revisions are not supplied; exclude in conservative v1 |
| y | Outcome, available only to training/evaluation |
| record_id, row position, split | Identity/order metadata; not predictive inputs |
| Simulated consent, opt-out, contact channel, notes | Business-rule simulation, not learned evidence |

The macro exclusion is a conservative project choice, not a claim that economic indicators are inherently post-contact. A later experiment may add them only with an explicit as-of availability argument and a new validation protocol. Do not introduce that experiment after seeing the final test results and present it as the same untouched evaluation.

### Persisted preprocessing

Build an importable transformer for deterministic feature construction inside the saved scikit-learn Pipeline:

- Derive no_previous_contact = 1 when pdays=999.
- Replace that sentinel with missing in a derived pdays_recency variable.
- Drop raw pdays from the estimator input after deriving the two fields.
- Impute missing derived recency with the non-sentinel **training** median. If none exists, use 0 and record the fallback.
- One-hot encode categorical fields with handle_unknown='ignore'; retain unknown and report unseen categories in scoring diagnostics.
- Standardize numeric features for Logistic Regression. Random Forest uses the same deterministic feature definitions and imputation, with scaling omitted.
- Use ColumnTransformer with an explicit allowlist and remainder='drop'. Do not rely on dropping only a few known leakage columns.

Learn encoder categories, imputation statistics, and scaling from training rows only. Validation/test/new records call transform/predict, never fit. Add a test that injecting duration or changing y cannot alter scores for identical allowed features.

The [scikit-learn leakage guidance](https://scikit-learn.org/stable/common_pitfalls.html) explains the training-only fit requirement and why a complete Pipeline helps preserve it.

## 7. SQLite data model and SQL campaign analysis

### Tables and keys

Use a single database rebuilt transactionally by prepare:

| Table | Required purpose and key |
|---|---|
| raw_campaigns | Original values plus record_id primary key |
| clean_campaigns | Normalized attributes; y as 0/1; fixed age group; record_id primary key |
| dataset_splits | record_id primary/foreign key and split=train/validation/test |
| model_runs | model_id primary key; manifest reference, source checksum, creation time |
| prospect_scores | Canonical-source scores only; composite key model_id + record_id; score, rank within the fixed original split, data split |
| engagement_simulated | record_id primary key; simulated flags and provenance='simulated' |

For SQL, map punctuation-heavy macro columns to snake_case in clean_campaigns and document the mapping. Preserve original names in raw_campaigns and the model input contract. Quote SQL identifiers consistently, including reserved words such as default. Enable foreign keys, enforce uniqueness, and index split and model_id joins.

Persist only canonical UCI-source scores in prospect_scores and canonical simulation flags in engagement_simulated. Filtered rankings, uploaded records, uploaded simulated flags, and form submissions are transient session/batch results with their own export metadata; they never overwrite canonical rows. Their ranks are recomputed for the explicitly displayed population.

Do not build a customer table implying stable cross-campaign identity. Do not infer lifetime value, account balances, deposit amounts, or relationship tenure: they are unavailable.

### Analysis questions

Implement 8–10 named SQL queries with split parameters and stable ordering:

1. Observation count, subscriptions, and subscription rate.
2. Subscription patterns by job and fixed age group.
3. Education and recorded loan/default categories, retaining unknown.
4. Previous-campaign outcome versus subscription.
5. Previously contacted versus no previous contact.
6. Previous-contact count bands and recency bands.
7. Historical month/day and telephone/cellular associations, with descriptive-only captions.
8. Unknown-value coverage by field and segment.
9. After freeze: rates and counts by chronological split and equal-size row-order blocks.
10. After scoring: score deciles, shortlist size, and subscriptions captured.

Every rate displays its numerator and denominator. Avoid averages of segment percentages when an overall rate requires weighted counts. Use joins with unique keys and verify that joins preserve row counts.

Example pattern:

~~~sql
SELECT c.poutcome,
       COUNT(*) AS observations,
       SUM(c.y) AS subscriptions,
       1.0 * SUM(c.y) / COUNT(*) AS subscription_rate
FROM clean_campaigns AS c
JOIN dataset_splits AS s USING (record_id)
WHERE s.split = :split
GROUP BY c.poutcome
ORDER BY observations DESC, c.poutcome;
~~~

Show rates only with adequate support and retain the raw counts. For ordinary segment subscription/selection/precision rates, mark denominator <30 as “insufficient support.” For recall, use a minimum of 10 positive observations. Never interpret association as the effect of a campaign channel or customer attribute.

## 8. Chronological splits, experiments, and locked evaluation

### Exact partition

For N=41,188:

~~~text
n_train = floor(0.70 * N) = 28,831
n_validation = floor(0.15 * N) = 6,178
n_test = N - n_train - n_validation = 6,179

Train record_id:       1 .. 28,831
Validation record_id: 28,832 .. 35,009
Test record_id:       35,010 .. 41,188
~~~

Store this mapping and checksum it. Never shuffle, stratify, randomly resplit, or sort by month names. Chronology is supplied by row order; exact dates are unavailable. Repeated customer identity cannot be eliminated, so document the absence of a customer-disjoint guarantee.

### Bounded candidate grid

Use random_state=42 where supported and an explicit model candidate ID.

| Family | Fixed settings | Candidate settings |
|---|---|---|
| Prior baseline | DummyClassifier(strategy='prior') fit on training | One candidate |
| Logistic Regression | L2, solver='lbfgs', max_iter=2000, class_weight=None | C in [0.01, 0.1, 1.0, 10.0] |
| Random Forest | n_estimators=300, max_features='sqrt', class_weight=None, random_state=42, n_jobs bounded for the machine | max_depth in [8, None] × min_samples_leaf in [10, 50] |

Eight learned candidates are enough for v1. No SMOTE, random cross-validation, unrestricted search, or deep-learning model. Address convergence warnings explicitly; do not hide them. A higher max_iter to reach convergence is a numerical repair, not a new feature search.

Fit every candidate's complete pipeline on training only. Compute validation predictions once per candidate. Save candidate parameters and metrics.

### Selection policy fixed before final evaluation

Within each learned family, and then between the family finalists:

1. Highest validation Precision@20%.
2. If tied within 1e-12, highest validation average precision (AP).
3. If still tied across families, choose Logistic Regression for simplicity.
4. If still tied within a family, use the lexicographically first candidate ID.

The prior baseline is a benchmark, not a deployable personalized model. If no learned model improves on the baseline, still document the prescribed learned winner and label the demo experimental with no demonstrated ranking advantage.

Save both family finalists and the chosen champion. Do not refit on train+validation in v1: the saved demo champion is the same training-only model that receives the reported final evaluation.

Freeze a selection manifest containing model IDs, feature policy, split checksum, candidate results, champion, default capacity, threshold rule, tie rule, and dependency versions. Only then unlock test evaluation. Evaluate the two frozen family finalists and the prior baseline in one final run. Do not use test results to pick a different champion.

### Capacity policy versus score threshold

**Capacity mode is the default product behavior.** For a population of N valid records and requested fraction q:

~~~text
k = 0                         if q = 0 or N = 0
k = min(N, ceil(q * N))        otherwise
~~~

For integer capacity K, k=min(K,N). Require integer K>=0, fraction 0<=q<=1, and any supplied minimum score in [0,1]; reject invalid settings. Sort by descending score, then a stable SHA-256 tie key of the UTF-8 string "42:<record_id>", ascending, then record_id if necessary. Never use Python's randomized hash. The tie key depends only on ID, not y or the observed outcome.

At 20%, validation selects 1,236 records and test selects 1,236 records. The slight fractional difference is due to rounding; print N and k beside metrics.

**Frozen score-cutoff diagnostic:** set t to the score of the kth validation record at q=.20 for the selected champion. Freeze t before test access. Apply score >= t unchanged on test and report the realized selected count, coverage, precision, and recall. Ties and distribution shifts can make that volume differ from 20%. This is a validation-derived operating point, not an optimal profit or suitability threshold.

In the app, optionally offer “validation cutoff plus capacity cap”: eligible records with score >= t are ranked and capped at K. Label this separately from pure capacity mode. Do not let users tune on final test labels and call the tuned result a new validated policy.

### Headline metrics and definitions

Let N be the number of labeled observations in the evaluated split, P=sum(y), k the top-budget count, and TPk the positives in that shortlist:

~~~text
base_rate       = P / N
precision_at_k  = TPk / k
recall_at_k     = TPk / P
lift_at_k       = precision_at_k / base_rate
coverage        = k / N
~~~

Report these for validation and test:

- **ROC-AUC**, using positive-class continuous scores.
- **PR-AUC (trapezoidal)** from precision_recall_curve and auc(recall, precision).
- **Average precision (AP)** using average_precision_score, separately labeled.
- **Precision@20%, Lift@20%, Recall@20%**, with N, positives, k, and TPk.
- Capacity/gain curves at 5%, 10%, 20%, 30%, and 50%, plus a smooth cumulative gains display.
- Score distribution, split base rates, Brier score, and a reliability diagram as diagnostics, without claiming calibrated probabilities.

AP and trapezoidal PR-AUC are not interchangeable. AP is the secondary selection metric. Constant scores can produce an artificially generous trapezoidal PR-AUC from endpoint interpolation; explain that footnote. The [AP documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html) describes the distinction.

For the prior baseline, all scores equal the training subscription rate. Use the same stable tie rule to report its realized shortlist metrics. Separately show the **analytical random-contact reference**: expected precision equals the evaluated population's base rate, expected lift=1, and expected recall=k/N. Do not present these expectations as measured model results.

Guard denominators: no selected records means precision/lift are N/A; no positives means recall/lift and PR summaries are N/A; a single-class set has undefined ROC-AUC. Do not coerce undefined values to zero or write nonstandard JSON NaN.

### Statistical and business interpretation

These are descriptive offline historical results for previously contacted records. They cannot identify who would subscribe without contact, estimate uplift, prove an increase in sales, or forecast deposit revenue.

Do not introduce a pass/fail metric floor. Inspect class-rate drift and score drift after freeze. If performance is weak, say so and explain that conservative features and chronological drift can make the task harder.

## 9. Explanations, group checks, and honest limitations

### Global explanation

Use permutation importance on the champion's **validation** data, permuting raw allowed columns through the complete pipeline, scored by AP, with 10 repeats and seed 42. Keep the engineered pdays fields together by permuting their raw pdays source. Save mean and spread; correlated history fields may share importance. Negative or near-zero importance is a valid finding.

Do not use test importances to revise the feature policy. Do not label an importance chart as causal influence. See [scikit-learn permutation importance](https://scikit-learn.org/stable/modules/permutation_importance.html).

### Record-level display

Each prospect card shows score, rank, recorded profile, previous-campaign facts, simulation status, and eligibility.

- If the champion is Logistic Regression, optional additive log-odds contributions may be shown only if correctly aggregated to raw features, with the intercept/baseline explained. They are not probability-point effects.
- For either model, factual context is required. Example: “Previous campaign outcome recorded as success.” This is evidence from a field, not proof that it caused this record's score.
- If the champion is Random Forest, do not turn global feature importance into invented personal reason codes. A faithful local explainer is a later extension, not a requirement.
- Conversation prompt: ask about saving goals, liquidity needs, preferred tenure, and willingness to discuss. Do not assert suitability or invent current deposit terms/rates.

### Descriptive group review

After freeze, report test counts, positives, selected counts, base rates, selection rates, precision, and recall for the fixed age groups and job categories. Apply the support rules from section 7. Include unknown where present.

State that age/profile variables and proxies can influence selection. These descriptive checks do not certify fairness or legal compliance. Do not implement group-specific thresholds or fairness optimizations in v1.

### Required limitations in report and app

Historical Portugal data; telephone campaigns only; records are not confirmed unique customers; selection into past contact is unobserved; no complete dates or feature-event timestamps; no app/web activity; no consent; no product suitability labels; no deposit amounts/costs; temporal drift; uncertain calibration; no causal or incremental-sales evidence; no production governance.

## 10. Separate simulated engagement policy

### Simulation data

Generate a deterministic engagement_simulated table from record_id using SHA-256 values independent of labels and scores. For each field, hash the UTF-8 string "42:<field_name>:<record_id>", interpret the digest as an unsigned integer, and take modulo 100.

- simulated_opt_out: hash bucket <10 out of 100.
- simulated_contact_permission: hash bucket <90 out of 100.
- simulated_recent_contact: hash bucket <10 out of 100.
- provenance: always "simulated".

These percentages are demonstration assumptions, not source statistics or real bank policies. A recent-contact flag demonstrates a cooldown concept; it does not assert a legally mandated duration.

Eligibility:

~~~text
eligible = simulated_contact_permission
           AND NOT simulated_opt_out
           AND NOT simulated_recent_contact
~~~

If several rules fail, retain all reasons in a stable order: opted_out, no_simulated_permission, simulated_recent_contact. An opt-out suppresses outreach regardless of the model score.

Scoring is independent of these flags. Changing simulated eligibility must not change propensity scores.

### Population and denominator discipline

Model-evaluation metrics use **all untouched labeled test rows**, with no simulated suppression. The engagement demo applies suppression first, then computes capacity within the eligible population.

The demo shows total records, valid records, eligible records, suppressed records, selected records, and capacity. If it displays retrospective conversions after filtering, label them “historical eligible-subset replay” with their own denominator. Never label this as the headline test Precision@20%.

No observed/simulated email, push, SMS, or app channel is added as a model feature. Digital-channel examples may be described in the guide as future simulated delivery workflows; no digital channel is recommended as superior.

## 11. Scoring and artifact contracts

### Batch input

CSV requires a unique nonblank record_id and the 10 raw feature columns from section 6. Trim batch IDs and treat them as strings without numeric coercion; enforce uniqueness after trimming. Labels are optional for a separately invoked evaluation path and are never passed to the estimator. Accept only a CSV here; do not accept model uploads.

Validate age, previous, and pdays as finite integer-valued inputs in their original units, with age >0, previous >=0, and pdays >=0 (including the documented 999 sentinel). Do not impose invented demographic cutoffs. Require these numeric fields rather than silently filling invalid source values. Normalize categorical blanks to unknown with a warning; report unseen categories and allow the trained encoder to handle them.

If known excluded source columns are supplied, ignore them with a clear summary. Fail on unexpected unrecognized columns to catch misspellings. Reject duplicate IDs, malformed numeric fields, or ambiguous schemas as a batch with row/column diagnostics. Do not partially drop records without disclosure.

For engagement input, accept only the required simulated fields, provenance="simulated", and IDs that can be joined unambiguously. Parse CSV flags strictly after trimming/case normalization: true/false or 1/0 only. Reject other nonblank flag values; never use bool("false"). Missing flags suppress outreach with reason missing_simulation_flags. If no engagement file is supplied, deterministic simulation is allowed in the demo and must be prominently labeled.

### Score output

Export:

~~~text
record_id, model_id, model_score, global_rank,
eligible, suppression_reasons, eligible_rank, selected,
capacity_mode, requested_capacity, validation_cutoff,
provenance, recorded_previous_outcome, previous_contacts,
no_previous_contact
~~~

Use nullable eligible_rank for suppressed records. Global rank includes all valid records; eligible rank follows suppression and any chosen population filters. Never include y in an outreach download. Outcome labels remain in a distinct historical evaluation output.

Include a README or metadata sidecar documenting that record IDs are surrogates and engagement is simulated. Do not fabricate names, phone numbers, account numbers, or contact details.

### Model artifact bundle

Save:

- Complete champion Pipeline, compressed with joblib.
- Both frozen family finalists or references to their stored artifacts.
- JSON manifest with model ID; version; source CSV checksum; feature/exclusion lists; positive class mapping; training/validation/test ranges and counts; split checksum; pipeline steps; parameters; seed; validation metrics; selection rule; capacity/tie policy; frozen cutoff; package versions; code commit when available; artifact hashes.
- Frozen validation predictions, final test predictions, metrics JSON, and CSV tables.
- Figures and a small scoring sample.

Use model.classes_ to identify the positive class; do not blindly assume a score column. Validate score values are finite and in [0,1].

Create the manifest first in a temporary location, checksum associated files, and finalize coherently. Refuse to evaluate/package mismatched data/model manifests. If data, feature policy, or code semantics change after final evaluation, create a new experiment ID and disclose that the original test has already been observed.

Joblib/pickle loading can execute code; load only the project's trusted, checksum-verified artifacts and use the recorded compatible environment. The [model persistence documentation](https://scikit-learn.org/stable/model_persistence.html) covers these constraints.

## 12. Streamlit application specification

Use a restrained neutral design, clear labels, readable plots, and no ICICI logo. All pages display the historical/simulated context and source attribution in an accessible place.

Cache data and model loading with keys that include artifact/model identity. Do not train on application startup or fit when a slider changes. Keep pipeline code importable in the deployed environment.

### Page 1 — Campaign overview

- KPI cards: observations, recorded subscriptions, historical subscription rate, and data version.
- Default split=train during development; final released app may show full history with split controls and visible counts.
- SQL-backed subscription patterns by job, age group, previous outcome, and prior-contact status.
- Historical month/day and telephone/cellular views clearly labeled as associations.
- No exact reconstructed dates or statements of unique customer counts.
- Minimum-support labels and segment counts.

### Page 2 — Model results

- Frozen validation/test tables for prior baseline, Logistic Regression finalist, and Random Forest finalist.
- Highlight the validation-chosen champion, even if another model scores better on test.
- PR and ROC plots; AP/trapezoid explanation; cumulative gains and budget curves.
- N, positive count, k, TPk, base rate, and the default 20% result.
- Validation-derived fixed cutoff diagnostics with realized test coverage.
- Global validation importances, split-drift diagnostics, reliability plot, and descriptive group review.
- Test budget slider is exploratory historical replay; never persist its setting as a new validated threshold.

### Page 3 — Outreach shortlist

- Start with the final test rows as a **historical replay pool**. Hide their labels from selection and export.
- Apply deterministic simulated engagement flags; show suppressed records and all reasons in a separate view.
- Select integer K or capacity fraction, default 20% of eligible records.
- Optional frozen-cutoff-plus-capacity mode, visibly distinct.
- Display ranked record IDs, model score, recorded context, and simulation/eligibility status.
- Provide context filters for inspecting the pool. Recompute eligible denominator and rank after filtering; state the resulting scope. No filter changes official test metrics.
- Provide safe CSV export without outcomes or personal contact information.
- Prospect drawer includes factual context and a customer-needs conversation prompt.
- Handle zero eligible records, capacity zero, oversized capacity, tied scores, and missing artifacts clearly.

### Page 4 — Scoring sandbox

- Single-record form for the allowlisted inputs, with glossary/tooltips and a reproducible sample. Assign a session-local synthetic record ID to form submissions; batch inputs supply their own unique IDs.
- Score button; separate simulated engagement toggles; display scores and suppression independently.
- Optional batch CSV scoring using the same validation and scoring functions as the CLI.
- Limit uploads to 5 MB and 10,000 rows; reject oversized batches clearly.
- Use in-memory processing, no persistent uploaded data, and a prominent “public/synthetic data only” label for the public demo.
- Validate failures with specific field messages; do not show stack traces or invented predictions.

## 13. Test plan and phase gates

Tests should protect real failure modes. Do not chase a coverage percentage or encode that a particular model must win.

| Area | Required cases and acceptance |
|---|---|
| Acquisition/schema | Correct 21-column/41,188-row variant accepted; older/sample/wrong delimiter rejected; failed download leaves no falsely valid cache; checksum mismatch detected |
| Row order/splits | Exact counts and ranges; no overlap; source order preserved; no duplicate surrogate IDs |
| Cleaning | unknown retained; pdays999 handled correctly; repeated rows retained and reported; SQL renaming documented |
| Leakage | Excluded columns and y cannot affect scores; test values cannot alter fitted medians/scalers/categories |
| Preprocessing | Unseen categories do not crash; sentinel produces intended features; no nonfinite estimator input |
| Metrics | Hand-computable ranked fixture; expected precision/lift/recall; rounding; ties; zero capacity; zero positives; single-class ROC; AP distinct from trapezoid |
| Baseline | Constant prior scores; deterministic tie shortlist; analytical random-reference formulas separate |
| Selection | Champion based only on validation; tie policy deterministic; test evaluation requires matching frozen manifest |
| Artifacts | Save/load score equality within 1e-12 on a fixed sample; positive class mapping correct; manifest/hash mismatch rejected |
| SQL | Counts/subscriptions match pandas; joins preserve counts; segment denominators and missing groups correct |
| Eligibility | High-score opt-out excluded; missing flags suppressed; scores unchanged when flags change; capacity applied after suppression |
| Scoring | Duplicate IDs, missing columns, malformed numbers, unknown categories, excluded extras, unexpected fields; outcome not exported |
| UI | Four pages render with small trusted fixture; controls update counts; download schema correct; empty/error states readable |
| End-to-end | Fresh setup follows README and reproduces split counts, model identity, and metrics within documented tolerance |

Use tiny synthetic fixtures for routine unit and app tests; no network is needed. Use a temporary SQLite database. Unit tests should verify formulas independently of their implementation.

GitHub Actions: install Python 3.11 and locked dependencies, run ruff and pytest. Add a manual/full-data workflow or documented local integration command for official-data reproduction; do not redownload/retrain for every trivial documentation change.

### Phase-by-phase execution

| Day/phase | Work | Evidence required before advancing |
|---|---|---|
| 1: Foundation | Inspect repo, create environment/package/config/CLI, fetch exact dataset, preserve readme/checksum | Schema/provenance report and verified source file |
| 2: Data/SQL | Cleaning, immutable IDs, split map, SQLite, training-only SQL analyses | Exact split counts; SQL/pandas agreement; no lost rows |
| 3: Features/baseline | Data dictionary, allowlist, pipeline, prior baseline, metric functions | Leakage and hand-calculation tests pass |
| 4: Models | Bounded LR/RF candidates, validation metrics, selection, cutoff and manifest freeze | Auditable champion selection without test metrics |
| 5: Final evaluation | Test finalists once, plots, drift/group diagnostics, model report | Measured tables/figures with counts and limitations |
| 6: Engagement/scoring | Separate simulation, suppression, ranking, CLI score/export | Opt-out and denominator/capacity checks pass |
| 7–8: App | Four pages, fixtures, UI checks, offline demo packaging | No startup training; functional export and empty states |
| 9: Delivery | README, learning-guide results, five slides/notes, CI, reproducibility check | Clean-install walkthrough and verified artifacts |
| 10: Rehearsal/optional hosting | Interview practice, local backup, optional cloud deployment | Five-minute demo and accurate 90-second explanation |

If time is tight, remove optional public hosting, individual LR contributions, and batch-upload UI first. Preserve leakage controls, both model families, SQL, core scoring/suppression, metrics, documentation, and a working local demonstration.

## 14. Reporting and interview deliverables

### README

Include the business question, screenshot, architectural diagram, exact setup/build/run commands, source/license, feature exclusions, split/evaluation protocol, measured result table, offline demo instructions, known limitations, and repository/demo links only after they exist.

### Data dictionary

Cover every raw source field with type, meaning, sentinel/unknown rules, model inclusion/exclusion and reason, SQL mapping, and availability assumption. Add the artificial metadata and simulation fields in separate sections.

### Short model report

Keep the main report to approximately 4–6 readable pages of Markdown plus linked figures/appendix:

1. Business question, intended use, and non-use.
2. Data source/version, quality findings, row/customer distinction.
3. Pre-contact availability, leakage controls, and conservative macro exclusion.
4. Chronological partition, candidates, validation selection, and frozen cutoff.
5. Actual validation/test results, denominators, baseline comparison, and budget curves.
6. Explanations, calibration/drift/group diagnostics, and limitations.
7. Reproduction instructions, artifact/commit references, and future evaluation.

### Update the companion learning guide

The provided guide explains the planned project. After building, add a clearly marked **verified implementation results** section containing champion/configuration, actual split prevalence, measured results, SQL findings with support, feature-importance findings, screenshots, exact reproduction commands, commit ID, and demo link if deployed.

Remove or update design statements that the final implementation changed. Never replace missing measurements with plausible-looking examples.

### Exactly five interview slides

1. **Business problem and alignment:** limited outreach capacity; customer conversation; deck themes.
2. **Data and pipeline:** exact UCI variant, SQL/store/features/model/app flow, historical limits.
3. **Leakage and evaluation:** duration/current-contact exclusion; ordered split; validation freeze.
4. **Measured findings:** baseline and both model families; champion; precision/lift/recall@20; gain chart.
5. **Relationship-manager demo and judgment:** shortlist, simulated opt-out, practical conversation, limitations and next steps.

Use one main visual per slide, short text, and speaker notes. Slide 4 must use actual measured results; if results are weak, explain what was learned. Produce editable PPTX and a readable PDF preview when tools support them. Render and inspect every slide before delivery.

### Resume wording

Use only after the respective work is complete:

- Analyzed 41,188 historical bank marketing records with Python and SQL to examine subscription patterns across customer segments and previous campaign outcomes.
- Built and compared Logistic Regression and Random Forest models for pre-contact subscription propensity, excluding post-contact information and evaluating on a chronological holdout.
- Developed a Streamlit dashboard for campaign analysis and capacity-based prospect prioritization, reporting precision, recall, and lift at a fixed outreach budget.

Add measured numbers only with a linked report and the correct denominator. Never claim the prototype was deployed at ICICI or generated incremental revenue.

## 15. Local demo packaging and optional deployment

### Reliable local delivery

package-demo creates a coherent compact bundle with the trusted champion, manifest, SQLite analysis tables, final test replay features/scores, simulated flags, frozen metrics, and small charts. All records come from the licensed public dataset. No target labels appear in prospect-scoring exports.

The app should load this bundle and work with networking disabled. Keep model training outside the runtime. Document how to rebuild the bundle from source.

Compress joblib artifacts; target a total demo bundle below 25 MB. If it exceeds this target, first remove redundant intermediates/duplicate models from the demo bundle without changing the champion. If the champion itself is large, distribute it as a checksummed release asset and document one preparation download before offline use. Do not change model selection after test evaluation solely for hosting size.

### Optional Streamlit Community Cloud

Once the local app is verified and the owner chooses to publish, follow the current [official deployment guide](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app). Confirm repository access, Python version, dependencies, entrypoint, and artifact paths.

Prefer the self-contained demo bundle so startup does not fetch data or train models. If a release asset is necessary, use its fixed version and checksum; document startup network dependence and retain the offline local backup.

Use no secrets unless the hosting configuration specifically requires one. Do not add a paid service. Verify all pages and a sample download on the deployed URL. Record the URL only after successful deployment; a configuration file is not proof of deployment.

This public demo is stateless. Uploads remain session-local/in-memory; it is not a multi-user CRM or a durable opt-out store.

## 16. Final acceptance and handoff

The build agent must check every item and provide evidence:

- [ ] Correct dataset, readme, attribution, provenance, and checksum.
- [ ] 41,188 preserved records; surrogate IDs; exact 28,831 / 6,178 / 6,179 split.
- [ ] SQL analysis reproducible and denominators correct.
- [ ] Strict feature allowlist; no duration/current-contact leakage; preprocessing training-only.
- [ ] Both model families and prior baseline; bounded candidate search.
- [ ] Validation-selected champion and score cutoff, manifest frozen before test.
- [ ] Test metrics measured honestly; AP/trapezoidal PR-AUC distinguished; k=1,236 at test 20%.
- [ ] Champion kept even if another finalist scores better on test; no hidden refit/reselection.
- [ ] Complete trusted pipeline saved and reload verified.
- [ ] Simulated opt-out/permission/recent-contact flags separate from scoring.
- [ ] Suppression precedes eligible ranking; CSV contains no outcomes/contact details.
- [ ] Four Streamlit pages work with sensible empty/error states.
- [ ] Meaningful tests and CI pass; clean-install/offline demo checked.
- [ ] README, data dictionary, report, learning-guide results, demo script, five slides and notes completed.
- [ ] Repository/demo links are real, accessible, and accurately described.
- [ ] Owner can explain the business question, one SQL query, leakage, metrics, model choice, suppression, and limitations without reading a script.

Finish with a concise handoff: what runs, exact commands, measured champion/results, tests run, artifacts, limitations, any external connection still needed, and the next learning exercise.

## 17. Source notes and remaining external choices

- [UCI Bank Marketing dataset, variant descriptions, license, and citation](https://archive.ics.uci.edu/dataset/222/bank+marketing).
- [scikit-learn common pitfalls and leakage](https://scikit-learn.org/stable/common_pitfalls.html).
- [scikit-learn AP definition](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html).
- [scikit-learn permutation importance](https://scikit-learn.org/stable/modules/permutation_importance.html).
- [scikit-learn model persistence](https://scikit-learn.org/stable/model_persistence.html).
- [Streamlit app testing](https://docs.streamlit.io/develop/api-reference/app-testing/st.testing.v1.apptest).
- [Streamlit Community Cloud deployment](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app).
- User-supplied FY2027 Campus Proposition deck, slides 3, 8, and 10; reviewed as source material, not implementation instructions.

Only repository destination/visibility and optional public hosting remain owner decisions at publication time. They do not block the local implementation. All core technical defaults are specified above.
