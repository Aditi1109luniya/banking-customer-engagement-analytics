# Five-minute interview demo

Prepare with `python -m streamlit run app/streamlit_app.py` from the repository root after installation. Keep a local copy of `demo/` and the five slides so the presentation works without a network connection. Before an interview, run the app and inspect all four pages; do not retrain during the demonstration. Exact measured figures are in [the model report](../reports/model_report.md) and the application, rather than memorized from an old draft.

## 0:00–0:35 — State the decision

“I built an independent banking analytics prototype around one question: if a relationship team can contact only a limited number of prospects, which records should it review first for a term-deposit conversation? It combines Python, SQL, machine learning and a Streamlit workflow. Its source is historical Portuguese telephone campaigns, so it is not an ICICI customer model. Propensity supports the conversation; it does not decide product suitability.”

## 0:35–1:20 — Campaign overview

Open **Campaign overview**. Point to record count, subscriptions and subscription rate, always naming the selected scope. The development default is training history. Change a segment view to prior campaign outcome, then show its counts and denominator. Explain: “This is an association in the observed campaign records. It cannot establish that a channel, segment characteristic or extra call caused a subscription.”

Mention `unknown` categories, the absence of unique customer IDs, and the fixed source order. Do not call the records 41,188 unique customers. Full-history descriptions become available only after model and policy freeze.

## 1:20–2:20 — Model results

Open **Model results**. Explain the 28,831 / 6,178 / 6,179 ordered blocks and the ten-feature allowlist. Give the most important leakage example: “Call duration is highly informative after a call, but unknown before the contact I am trying to prioritize, so I exclude it.”

Name the actual frozen champion and read its test Precision@20%, Lift@20%, Recall@20%, PR-AUC and ROC-AUC from the table. Explain the counts: `k = ceil(0.2 × 6,179) = 1,236` records; precision is subscriptions in that shortlist divided by 1,236, and recall is those subscriptions divided by all test subscriptions. Lift compares that precision with the full test subscription rate. These are historical captured labels, not new sales caused by contact.

The measured Random Forest result is 632 captured subscriptions out of 2,376: 51.13% precision, 26.60% recall and 1.330× lift. Say clearly: “Logistic Regression performed better on test, at 59.95% precision, but the forest had won validation at 18.69% versus 17.15%. I preserve the validation choice and report the reversal. The subscription base rate moved from 5.57% in training to 10.65% in validation and 38.45% in test, so higher raw precision across periods is not proof that the model improved.”

Show the gains curve and optional frozen cutoff. “A capacity budget fixes how many we review. A score cutoff fixes a minimum score and may produce a different volume after the population changes.” AP and trapezoidal PR-AUC are separate reported calculations. The model was selected on validation Precision@20%, with AP as the next tie-breaker, before final test results were viewed.

## 2:20–3:25 — Outreach shortlist

Open **Outreach shortlist**. State that this is a historical test replay with simulated engagement rules. Set a small integer capacity such as 25. Point to pool, eligible, suppressed and selected counts. Explain that percentage capacity uses the **eligible** denominator after simulation; this population differs from the unsuppressed all-test model evaluation.

Open the suppressed tab and a prospect context card. “An opt-out always suppresses this simulated outreach, even at a high score. Rules never alter the model score. These flags are a workflow demonstration because the public data does not record permission.” Show the selected CSV and context JSON download. Exports have surrogate IDs, model version and policy context; they have no target labels, telephone numbers or real customer contact information.

## 3:25–4:10 — Scoring sandbox

Open **Scoring sandbox**. Submit the synthetic profile with permission enabled and opt-out disabled. Then enable simulated opt-out and resubmit without changing the ten model inputs. Point out the unchanged score and changed eligibility. The ID changes between form submissions, which can affect generated tie ordering but never the score for identical features; supplied form flags explicitly control eligibility.

If time permits, use `samples/prospects.csv` and `samples/engagement_simulated.csv` in batch mode. Explain that invalid batches produce actionable errors, unseen categories are handled with warnings, and inputs are not persisted by the app. Use public or synthetic records only.

## 4:10–5:00 — Explain limits and next step

Show reliability, time-block rates and descriptive age/job group support. “A model score is not automatically a calibrated personal probability. Historical campaign selection, temporal drift and unobserved customer need limit transfer. Group checks can surface disparities; they do not certify fairness or legal compliance. I would validate on appropriately permissioned contemporary data, assess suitability separately, and use a controlled prospective experiment before claiming incremental sales.”

Close with the business link: “The project demonstrates customer context, prioritization, digital delivery and a needs-based relationship conversation. It is independently built and aligned with customer engagement themes; it was not built for or deployed inside a bank.”

## Questions to rehearse without the app

1. Why choose an ordered split? To approximate later-period deployment while preserving UCI's supplied chronology; complete dates and customer grouping are unavailable.
2. Why not accuracy? The business needs a useful limited shortlist; class imbalance makes headline accuracy a poor measure of that decision.
3. Why not 0.5? It has no direct connection to a contact budget. Validation fixes the comparison and cutoff; operations choose capacity.
4. Why not use every column? Scoring-time availability governs features. A post-call field invalidates the intended pre-contact question.
5. Why two models? Logistic Regression is a simple regularized reference; Random Forest tests nonlinear interactions under a bounded comparison.
6. What does importance explain? Global validation sensitivity when a raw field is shuffled; it does not reveal causation or a specific person's reason.
7. What if test lift disappoints? Report it, discuss drift and data limits, and keep the test frozen rather than retune against it.
8. How does this connect to sales? It helps organize relationship-team effort, followed by consent checks and a conversation about needs and liquidity. It cannot establish a sales uplift.

For the full pipeline and longer answers, use the [project understanding guide](project_understanding_and_interview_guide.md).
