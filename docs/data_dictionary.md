# Data dictionary and scoring contract

Use this document to distinguish observed source data, learned features, and simulated business rules. The variant is **bank-additional-full.csv**, a semicolon-separated file with 41,188 rows, 20 raw inputs and target `y`. It follows supplied date order from May 2008 to November 2010. Rows are campaign observations; this project cannot recover unique customer identities or complete dates. Definitions below were checked against the `bank-additional-names.txt` included in the official archive. Some metadata on the UCI landing page describes an older variant; do not substitute its `balance`, `day`, or `pdays=-1` conventions.

## All 21 original fields

| Source field | Type / unit | Meaning and values | V1 model use |
|---|---|---|---|
| `age` | Positive integer, years | Recorded age. No source-based reason to invent upper-age exclusion. | Included numeric |
| `job` | Category | `admin.`, `blue-collar`, `entrepreneur`, `housemaid`, `management`, `retired`, `self-employed`, `services`, `student`, `technician`, `unemployed`, `unknown` | Included one-hot |
| `marital` | Category | `divorced`, `married`, `single`, `unknown`; the source groups widowed with divorced. | Included one-hot |
| `education` | Category | `basic.4y`, `basic.6y`, `basic.9y`, `high.school`, `illiterate`, `professional.course`, `university.degree`, `unknown` | Included one-hot |
| `default` | Category | Recorded credit in default: `yes`, `no`, `unknown`. `unknown` is distinct from `no`. | Included one-hot |
| `housing` | Category | Recorded housing loan: `yes`, `no`, `unknown`. | Included one-hot |
| `loan` | Category | Recorded personal loan: `yes`, `no`, `unknown`. | Included one-hot |
| `contact` | Category | Current campaign's last contact medium: `cellular` or `telephone`. | Excluded: assignment before scoring not established |
| `month` | Category | Last contact month, lowercase English abbreviation. It does not identify a year. | Excluded: current-contact field |
| `day_of_week` | Category | Last contact weekday: `mon` through `fri`. | Excluded: current-contact field |
| `duration` | Nonnegative integer, seconds | Duration of the recorded last call. Only available after that call. | Excluded: direct temporal leakage |
| `campaign` | Positive integer, count | Contacts in this campaign including the recorded last contact. | Excluded: no pre-contact snapshot |
| `pdays` | Nonnegative integer, days / sentinel | Days since previous campaign contact; **999 means not previously contacted**. | Included after sentinel transformation |
| `previous` | Nonnegative integer, count | Contacts before the current campaign. | Included numeric |
| `poutcome` | Category | Previous campaign result: `failure`, `nonexistent`, `success`. | Included one-hot |
| `emp.var.rate` | Numeric, quarterly indicator | Employment variation rate. | Excluded: release/revision availability unknown |
| `cons.price.idx` | Numeric, monthly index | Consumer price index. | Excluded: release/revision availability unknown |
| `cons.conf.idx` | Numeric, monthly index | Consumer confidence index. | Excluded: release/revision availability unknown |
| `euribor3m` | Numeric, daily rate | Three-month Euribor indicator. | Excluded: release/revision availability unknown |
| `nr.employed` | Numeric, quarterly indicator | Source's number-of-employees indicator. Preserve original source units; do not relabel as a particular population unit without evidence. | Excluded: release/revision availability unknown |
| `y` | Category → binary target | `yes` → 1, `no` → 0: recorded term-deposit subscription. | Label only, never a scoring feature |

Ten raw features are allowed: `age`, `job`, `marital`, `education`, `default`, `housing`, `loan`, `previous`, `pdays`, `poutcome`. Seven are categorical; three begin as numeric. Availability is a retrospective assumption, because the source has no feature-event timestamps.

## Derived features and metadata

| Field | Where / definition | Trained input? |
|---|---|---|
| `record_id` | Immutable source row position 1–41,188; external batch uses unique trimmed text IDs, retaining leading zeros. Not a real customer identifier. | No |
| `split` | Rows 1–28,831 train; 28,832–35,009 validation; 35,010–41,188 test. No sorting or shuffling. | No |
| `age_group` | `under 30`, `30–44`, `45–59`, `60+`; fixed descriptive bins. | No |
| `no_previous_contact` | 1 exactly when `pdays == 999`, else 0. | Yes |
| `pdays_recency` | Observed `pdays`, replacing 999 with median among training rows with `pdays != 999`; fallback 0 only if that training subset is empty. | Yes |
| One-hot columns | Learned categories in the seven categorical features; unseen values use `handle_unknown='ignore'`. | Yes |
| `source_sha256` | SHA-256 of exact CSV bytes; stored in provenance and manifests. | No |
| `split_sha256` | SHA-256 of canonical split-map CSV. | No |
| `model_id` | Candidate identity plus data/config/package/code fingerprint. | No |

Logistic Regression standardizes `age`, `previous`, `pdays_recency`, `no_previous_contact` using training-only means and scales. The forest passes those numbers through unchanged. Both learn categorical encodings on training only. The complete transform plus estimator is saved together.

In normalized SQLite storage, `emp.var.rate`, `cons.price.idx`, `cons.conf.idx`, `nr.employed` become `emp_var_rate`, `cons_price_idx`, `cons_conf_idx`, `nr_employed`. `euribor3m` is unchanged. `raw_campaigns` retains original names.

## Database tables

| Table | Key and purpose |
|---|---|
| `raw_campaigns` | `record_id` primary key; original 21 source fields. |
| `clean_campaigns` | `record_id` primary/foreign key; normalized text, binary label, SQL-safe names, age group. |
| `dataset_splits` | `record_id` primary/foreign key; exact ordered split. |
| `model_runs` | `model_id` primary key; manifest path, source hash, creation time. |
| `prospect_scores` | `(model_id, record_id)` primary key; canonical historical scores, split and rank. Uploaded prospects are never inserted. |
| `engagement_simulated` | `record_id` primary/foreign key; separate simulation fields and provenance. |

Foreign keys are enabled per connection; prepare checks SQLite integrity and foreign-key constraints. Analyses use vetted `.sql` files with bound split parameters. Every reported segment keeps its numerator and denominator.

## Separate simulated engagement

The source has no opt-out, permission, email, push, app usage or digital engagement fields. These demonstration columns are **not observed consent, trained inputs, or a legal compliance assessment**.

| Simulation field | Accepted values / meaning |
|---|---|
| `record_id` | Unique trimmed text ID to join with prospects. |
| `simulated_opt_out` | Boolean `true`/`false` or `1`/`0`; true suppresses. |
| `simulated_contact_permission` | Same boolean tokens; false suppresses. |
| `simulated_recent_contact` | Same boolean tokens; true suppresses. |
| `provenance` | Must equal `simulated` after whitespace/case normalization. |

The generated simulation hashes `42:<field>:<record_id>` with SHA-256 and applies modulo-100 cutoffs of 10 for opt-out, 90 for permission, and 10 for recent contact. Realized fractions need not exactly equal these nominal cutoffs. Changing flags never changes scores. All failed reasons are retained. When a supplied flags file lacks an ID, a flag column or a flag cell, that record is suppressed as `missing_simulation_flags`. If no file is supplied, the CLI/sandbox uses deterministic generated flags.

## Batch scoring input and export

Prospects CSV requires `record_id` plus all ten raw model features, with a comma delimiter. Missing or unexpected columns, blank/duplicate IDs, nonfinite or fractional numeric values, negative recency/counts, and nonpositive ages reject the entire batch. Numeric fields must be exactly representable as integers. Blank categorical values become `unknown` with diagnostics. Other unseen categories generate warnings and remain scoreable through the fitted encoder. Recognized excluded source fields and `y` are explicitly ignored and diagnosed; they are never estimator inputs or exports.

The Streamlit batch sandbox accepts at most 5 MB per file and 10,000 prospects. It processes public/synthetic uploads in memory. The CLI writes only the requested score output and metadata; do not supply real customer data.

| Score/export field | Meaning |
|---|---|
| `record_id`, `model_id`, `model_score` | Surrogate identifier, frozen model identity, historical propensity score in [0,1]. |
| `global_rank` | Rank across the current valid pool before suppression. |
| `eligible`, `suppression_reasons` | Simulation eligibility and all failed rule labels. |
| `eligible_rank` | Rank among eligible records only; absent when suppressed. |
| `selected` | Survives simulation, optional cutoff, and capacity. |
| `capacity_mode`, `requested_capacity` | Fraction, integer capacity, or cutoff-plus-capacity; requested count is recorded. |
| `validation_cutoff` | Frozen validation cutoff when enabled; otherwise missing. |
| `provenance` | `simulated`, referring to eligibility. |
| `recorded_previous_outcome`, `previous_contacts`, `no_previous_contact` | Factual prior-history context, not individual causal explanations. |

Tie order is descending score, ascending SHA-256 of `42:<record_id>`, then ID. Capacity fraction uses `ceil(fraction × eligible_count)` and integer capacity is capped at eligible count. Optional `model_score >= validation_cutoff` can leave some capacity unused. The CLI score CSV contains all safe scored records with `selected`; the dashboard's shortlist download filters to selected records. Exports omit outcome labels and personal contact details.

Source and license: [UCI Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing), [dataset attribution](../DATA_LICENSE.md).
