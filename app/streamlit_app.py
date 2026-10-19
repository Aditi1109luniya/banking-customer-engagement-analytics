"""A read-only historical analytics demo and in-memory scoring workspace.

Run with ``streamlit run app/streamlit_app.py`` from the repository root.
The application only loads the project's verified artifact bundle; it never fits a model.
"""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import sys

# Streamlit Community Cloud installs runtime requirements, then runs this file.
# Keep the same importable package available without a startup build step.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve, roc_curve

from bank_engagement.artifacts import load_bundle
from bank_engagement.constants import DEMO, FEATURES
from bank_engagement.database import query_analysis
from bank_engagement.engagement import simulate_engagement
from bank_engagement.evaluation import ranked_order
from bank_engagement.scoring import score_prospects

PAGES = ["Campaign overview", "Model results", "Outreach shortlist", "Scoring sandbox"]
COLOR = "#12756c"
SOURCE = "https://archive.ics.uci.edu/dataset/222/bank+marketing"
MAX_UPLOAD_BYTES = 5 * 1024 * 1024
EXPORT_COLUMNS = [
    "record_id", "model_id", "model_score", "global_rank", "eligible", "suppression_reasons",
    "eligible_rank", "selected", "capacity_mode", "requested_capacity", "validation_cutoff",
    "provenance", "recorded_previous_outcome", "previous_contacts", "no_previous_contact",
]


@st.cache_resource(show_spinner="Loading verified model and historical results…")
def cached_bundle(bundle_path: str, manifest_identity: str) -> dict:
    """A new manifest changes the cache identity, including after a deployment."""
    return load_bundle(Path(bundle_path))


@st.cache_data(show_spinner=False)
def cached_analysis(name: str, split: str, db_path: str, manifest_identity: str) -> pd.DataFrame:
    return query_analysis(name, split=split, db_path=Path(db_path))


def percent(value: float) -> str:
    return f"{value:.1%}" if pd.notna(value) else "—"


def plot_layout(fig: go.Figure, *, height: int = 340) -> go.Figure:
    fig.update_layout(
        template="plotly_white", height=height, margin=dict(l=12, r=12, t=25, b=10),
        font=dict(family="Arial, sans-serif", size=13, color="#334155"),
        colorway=[COLOR, "#456789", "#c3924c", "#8a6ca8"],
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    return fig


def dataframe(frame: pd.DataFrame, **kwargs) -> None:
    st.dataframe(frame, hide_index=True, use_container_width=True, **kwargs)


def context_note() -> None:
    st.caption(
        "Public Portuguese telephone-campaign records · May 2008–November 2010 · "
        "Independent prototype · Engagement permissions and cooldown flags are simulated."
    )
    with st.expander("Data source, boundaries and responsible use"):
        st.markdown(
            f"Source: [UCI Bank Marketing]({SOURCE}), `bank-additional-full.csv`. "
            "These are historical observations, not confirmed unique customers. "
            "This project is independent of ICICI Bank. It uses no ICICI customer data, "
            "app/web activity, real consent or personal contact details."
        )
        st.write(
            "A score supports a conversation about needs; it does not establish suitability. "
            "The source cannot validate digital-channel effectiveness, incremental sales, "
            "revenue or the causal benefit of contact. Calibration and temporal stability "
            "are uncertain. Descriptive group checks do not certify fairness or compliance."
        )
        st.write(
            "The model uses only 10 pre-contact profile and previous-campaign fields. "
            "Call duration, current-campaign/contact fields and economic indicators are "
            "excluded under the conservative feature-availability policy. There are no "
            "complete feature-event timestamps, product-suitability labels, deposit amounts "
            "or contact costs. A production service would need separate governance."
        )


def campaign_overview(bundle: dict, identity: str) -> None:
    st.subheader("Understand the recorded campaigns")
    st.write("Explore observed subscription patterns before deciding where to spend outreach time.")
    split = st.selectbox(
        "Historical population", ["all", "train", "validation", "test"],
        format_func=lambda x: {"all": "Full historical dataset", "train": "Training: first 70%",
                               "validation": "Validation: next 15%", "test": "Test: final 15%"}[x],
    )
    overview = cached_analysis("overview", split, str(bundle["db_path"]), identity)
    if overview.empty:
        st.info("There are no observations in this population.")
        return
    row = overview.iloc[0]
    n, positives = int(row["observations"]), int(row["subscriptions"])
    cards = st.columns(4)
    cards[0].metric("Observations", f"{n:,}")
    cards[1].metric("Recorded subscriptions", f"{positives:,}")
    cards[2].metric("Historical subscription rate", percent(float(row["subscription_rate"])))
    cards[3].metric("Data version", str(bundle["manifest"].get("source_sha256", "unavailable"))[:10])
    st.caption("Subscription rates are associations in historical telephone campaigns, not effects.")
    tabs = st.tabs(["Customer profile", "Previous campaigns", "Recorded contact context"])
    with tabs[0]:
        left, right = st.columns(2)
        with left:
            segment_plot(bundle, identity, split, "by_job", "job", "Recorded job")
        with right:
            segment_plot(bundle, identity, split, "by_age_group", "age_group", "Recorded age group")
    with tabs[1]:
        left, right = st.columns(2)
        with left:
            segment_plot(bundle, identity, split, "by_poutcome", "poutcome", "Previous outcome")
        with right:
            segment_plot(bundle, identity, split, "by_prior_contact", "prior_contact", "Prior contact")
    with tabs[2]:
        st.info(
            "The following fields describe the recorded contact. They are used only for "
            "historical analysis and are excluded from model inputs. Exact dates are unavailable."
        )
        left, right = st.columns(2)
        with left:
            segment_plot(bundle, identity, split, "by_month", "month", "Recorded month")
        with right:
            segment_plot(bundle, identity, split, "by_contact", "contact", "Recorded telephone medium")
        segment_plot(bundle, identity, split, "by_day_of_week", "day_of_week", "Recorded weekday")


def segment_plot(bundle: dict, identity: str, split: str, name: str, group: str, title: str) -> None:
    st.markdown(f"**{title}**")
    frame = cached_analysis(name, split, str(bundle["db_path"]), identity)
    if frame.empty:
        st.info("No segment data available.")
        return
    frame = frame.copy()
    frame["support_label"] = np.where(
        frame["observations"] < 30, "insufficient support (<30)", "adequate support",
    )
    frame.loc[frame["observations"] < 30, "subscription_rate"] = np.nan
    if group not in frame.columns:
        group = next(c for c in frame.columns if c not in {
            "observations", "subscriptions", "subscription_rate", "low_support", "support_label",
            "n", "positives", "base_rate",
        })
    fig = px.bar(
        frame, x=group, y="subscription_rate", color_discrete_sequence=[COLOR],
        hover_data=[c for c in ["observations", "subscriptions", "support_label"] if c in frame],
        labels={group: "", "subscription_rate": "Historical subscription rate"},
    )
    fig.update_yaxes(tickformat=".0%", rangemode="tozero")
    st.plotly_chart(plot_layout(fig), use_container_width=True)
    dataframe(frame)
    st.caption("Check counts and support labels before comparing small segments.")


def model_results(bundle: dict, identity: str) -> None:
    manifest = bundle["manifest"]
    test = bundle["test"]
    metrics = pd.DataFrame(bundle["metrics"])
    st.subheader("Frozen evaluation, with a fixed decision rule")
    st.write(
        f"Validation-selected champion: **{manifest['champion_family']}** "
        f"(`{manifest['champion_id']}`). The test set was not used to choose this model."
    )
    st.caption(
        "Rows remain in supplied chronological order: 28,831 train / 6,178 validation / "
        "6,179 test. Preprocessing was fitted on training data only."
    )
    official = metrics[(metrics["split"] == "test") & metrics["champion"].astype(bool)]
    if not official.empty:
        best = official.iloc[0]
        cards = st.columns(4)
        for col, label, key in zip(
            cards, ["Test average precision", "Precision @ 20%", "Recall @ 20%", "Lift @ 20%"],
            ["ap", "precision_at_20", "recall_at_20", "lift_at_20"], strict=True,
        ):
            value = float(best[key])
            col.metric(label, f"{value:.3f}" if key == "ap" else
                       f"{value:.2f}×" if key == "lift_at_20" else percent(value))
        st.caption(
            f"Official test denominator: N={int(best['n']):,}; positives={int(best['positives']):,}; "
            f"top-k={int(best['k']):,}; historical subscriptions in top-k={int(best['tp_k']):,}; "
            f"base rate={percent(float(best['base_rate']))}. No simulated suppression is applied."
        )
    with st.expander("Compare the frozen baseline and model finalists", expanded=True):
        for split in ["validation", "test"]:
            st.markdown(f"**{split.title()}**")
            columns = ["family", "model_id", "champion", "ap", "pr_auc", "roc_auc",
                       "precision_at_20", "recall_at_20", "lift_at_20", "brier"]
            dataframe(metrics.loc[metrics["split"] == split, [c for c in columns if c in metrics]])
        st.caption(
            "AP is average precision: a step-weighted summary of the precision–recall curve. "
            "PR-AUC is trapezoidal area and can differ from AP. The prior baseline predicts "
            "the training prevalence; tied ranks follow the fixed seeded record-ID policy."
        )
    labels = (test["y"].astype(str).str.lower().isin(["yes", "1", "true"])).astype(int)
    scores = test["model_score"].to_numpy()
    left, right = st.columns(2)
    with left:
        precision, recall, _ = precision_recall_curve(labels, scores)
        fig = go.Figure(go.Scatter(x=recall, y=precision, name="Champion", mode="lines"))
        fig.add_hline(y=float(labels.mean()), line_dash="dot", annotation_text="Test prevalence")
        fig.update_layout(title="Test precision–recall", xaxis_title="Recall", yaxis_title="Precision")
        st.plotly_chart(plot_layout(fig), use_container_width=True)
    with right:
        fpr, tpr, _ = roc_curve(labels, scores)
        fig = go.Figure(go.Scatter(x=fpr, y=tpr, name="Champion", mode="lines"))
        fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dot", color="#94a3b8"))
        fig.update_layout(title="Test ROC", xaxis_title="False-positive rate", yaxis_title="True-positive rate")
        st.plotly_chart(plot_layout(fig), use_container_width=True)
    st.markdown("**Capacity and historical outcomes captured**")
    replay = st.slider("Exploratory test outreach budget (%)", 0, 100, 20, 1)
    st.caption(
        "This slider is historical replay. It does not revise the validation-chosen champion "
        "or frozen cutoff, and it cannot establish incremental sales."
    )
    replay_frame = test.assign(_positive=labels).iloc[ranked_order(test["record_id"], scores)]
    k = min(len(test), int(np.ceil(len(test) * replay / 100)))
    tp = int(replay_frame.head(k)["_positive"].sum())
    cols = st.columns(3)
    cols[0].metric("Replay rows selected", f"{k:,} / {len(test):,}")
    cols[1].metric("Replay precision", percent(tp / k) if k else "—")
    cols[2].metric("Replay recall", percent(tp / int(labels.sum())) if labels.sum() else "—")
    curves = bundle["curves"]
    left, right = st.columns(2)
    with left:
        fig = px.line(curves, x="fraction", y="recall", labels={"fraction": "Fraction contacted", "recall": "Historical subscriptions captured"})
        fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dot", color="#94a3b8"))
        fig.update_xaxes(tickformat=".0%")
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(plot_layout(fig), use_container_width=True)
    with right:
        fig = px.line(curves, x="fraction", y="lift", labels={"fraction": "Fraction contacted", "lift": "Lift over test base rate"})
        fig.add_hline(y=1, line_dash="dot")
        fig.update_xaxes(tickformat=".0%")
        st.plotly_chart(plot_layout(fig), use_container_width=True)
    cutoff = float(manifest["validation_cutoff"])
    above = scores >= cutoff
    count, positives_above = int(above.sum()), int(labels[above].sum())
    st.info(
        f"Frozen validation cutoff = {cutoff:.4f}. On test it selects {count:,} rows "
        f"({percent(count / len(test))} coverage), with "
        f"{percent(positives_above / count) if count else 'undefined'} precision and "
        f"{percent(positives_above / int(labels.sum())) if labels.sum() else 'undefined'} recall. "
        "A fixed score cutoff does not guarantee the same contact volume in a later population."
    )
    diagnostics = st.tabs(["Global explanation", "Reliability & drift", "Descriptive group review"])
    with diagnostics[0]:
        importance = bundle["importance"].sort_values("importance_mean", ascending=True)
        fig = px.bar(importance, x="importance_mean", y="feature", orientation="h", error_x="importance_std",
                     labels={"importance_mean": "Validation AP decrease after permutation", "feature": ""})
        st.plotly_chart(plot_layout(fig, height=390), use_container_width=True)
        st.caption(
            "Validation raw-feature permutation importance, 10 repeats, seed 42. Error bars "
            "show repeat spread. Correlated fields share information; near-zero and negative "
            "values are valid. Global importance does not explain a particular record or causality."
        )
    with diagnostics[1]:
        left, right = st.columns(2)
        with left:
            observed, predicted = calibration_curve(labels, scores, n_bins=10, strategy="quantile")
            fig = go.Figure(go.Scatter(x=predicted, y=observed, mode="lines+markers", name="Champion"))
            fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dot", color="#94a3b8"))
            fig.update_layout(title="Test reliability (quantile bins)", xaxis_title="Mean model score", yaxis_title="Observed subscription rate")
            st.plotly_chart(plot_layout(fig), use_container_width=True)
        with right:
            drift = []
            for split in ["train", "validation", "test"]:
                part = cached_analysis("overview", split, str(bundle["db_path"]), identity)
                if not part.empty:
                    drift.append({"split": split, **part.iloc[0].to_dict()})
            dataframe(pd.DataFrame(drift))
            st.write(
                "Changing subscription prevalence across time blocks signals population drift. "
                "These scores are not guaranteed to be calibrated probabilities. The source "
                "does not supply complete dates or feature-release timestamps."
            )
    with diagnostics[2]:
        audit = bundle["group_audit"].copy()
        audit["support_label"] = np.where(audit["n"] < 30, "insufficient support (<30)", "adequate support")
        for field in ["base_rate", "selection_rate"]:
            if field in audit:
                audit.loc[audit["n"] < 30, field] = np.nan
        if "precision" in audit:
            audit.loc[audit["selected"] < 30, "precision"] = np.nan
        if "recall" in audit:
            audit.loc[audit["positives"] < 10, "recall"] = np.nan
        dataframe(audit)
        st.caption(
            "Test groups use the fixed 20% all-test shortlist. Small groups have unstable rates. "
            "Rates are hidden below 30 denominator observations; recall below 10 positives. "
            "Age/profile variables and proxies can influence selection. These descriptive "
            "checks do not certify fairness or legal compliance; there are no group-specific thresholds."
        )


def safe_export(scored: pd.DataFrame) -> pd.DataFrame:
    return scored.loc[:, [c for c in EXPORT_COLUMNS if c in scored]].copy()


def export_metadata(bundle: dict, pool_count: int, note: str) -> str:
    return json.dumps({
        "project": "Independent banking engagement prototype",
        "model_id": bundle["manifest"]["champion_id"],
        "source": SOURCE, "pool_records": pool_count, "context": note,
        "record_ids": "Surrogates identifying observations, not verified unique customers.",
        "engagement": "Simulated; not source consent or a real bank policy.",
        "use": "Conversation support only. No suitability or causal-sales claim.",
    }, indent=2)


def show_downloads(scored: pd.DataFrame, bundle: dict, pool_count: int, prefix: str) -> None:
    export = safe_export(scored)
    st.download_button(
        "Download selected shortlist (CSV)", export.loc[export["selected"]].to_csv(index=False),
        f"{prefix}_shortlist.csv", "text/csv", key=f"{prefix}_download",
    )
    st.download_button(
        "Download export context (JSON)", export_metadata(bundle, pool_count, prefix),
        f"{prefix}_metadata.json", "application/json", key=f"{prefix}_metadata",
    )
    st.caption("Exports contain surrogate IDs and model/policy fields. Outcomes and personal contact details are excluded.")


def outreach_shortlist(bundle: dict) -> None:
    st.subheader("Turn limited capacity into a reviewable shortlist")
    st.warning(
        "Historical replay pool: final test observations. Engagement permissions, opt-outs "
        "and recent-contact flags below are deterministic simulations, not recorded consent."
    )
    pool = bundle["test"].loc[:, ["record_id", *FEATURES]].copy()
    with st.expander("Inspect a smaller pool", expanded=False):
        jobs = st.multiselect("Recorded job filter", sorted(pool["job"].unique().tolist()))
        ages = st.slider("Recorded age range", int(pool.age.min()), int(pool.age.max()),
                         (int(pool.age.min()), int(pool.age.max())))
        if jobs:
            pool = pool.loc[pool["job"].isin(jobs)]
        pool = pool.loc[pool.age.between(*ages)]
    st.caption(
        f"Current replay scope: {len(pool):,} observations. Filters recompute the eligible "
        "denominator and ranking; official all-test evaluation remains frozen."
    )
    if pool.empty:
        st.info("No observations match these filters. Broaden the pool to continue.")
        return
    left, right = st.columns([2, 1])
    with left:
        mode = st.radio("Capacity policy", ["Fraction of eligible pool", "Integer capacity"], horizontal=True)
        if mode == "Integer capacity":
            capacity = st.number_input("Maximum records to select", min_value=0, value=100, step=1)
            fraction = .2
        else:
            fraction = st.slider("Eligible pool capacity (%)", 0, 100, 20, 1) / 100
            capacity = None
    with right:
        use_cutoff = st.checkbox("Also require the frozen validation cutoff", value=False)
        st.caption(
            "Capacity is computed after simulated suppression. An optional cutoff can leave "
            "capacity unused. All failed suppression rules are retained."
        )
    manifest = bundle["manifest"]
    scored = score_prospects(
        bundle["pipeline"], pool, model_id=manifest["champion_id"],
        flags=simulate_engagement(pool["record_id"].tolist()),
        capacity=capacity, fraction=fraction,
        min_score=float(manifest["validation_cutoff"]) if use_cutoff else None,
    )
    scored["validation_cutoff"] = float(manifest["validation_cutoff"])
    for warning in scored.attrs.get("warnings", []):
        st.warning(str(warning))
    eligible = int(scored["eligible"].sum())
    selected = int(scored["selected"].sum())
    cols = st.columns(5)
    for col, label, value in zip(cols, ["Pool / valid", "Eligible", "Suppressed", "Selected", "Requested capacity"],
                                 [len(pool), eligible, len(pool) - eligible, selected,
                                  int(capacity) if capacity is not None else int(np.ceil(eligible * fraction))], strict=True):
        col.metric(label, f"{value:,}")
    if not eligible:
        st.info("No eligible observations remain after simulated suppression.")
    elif not selected:
        st.info("The current capacity or cutoff selects no observations.")
    tabs = st.tabs(["Selected shortlist", "Eligible pool", "Suppressed observations"])
    with tabs[0]:
        dataframe(safe_export(scored.loc[scored["selected"]]))
        show_downloads(scored, bundle, len(pool), "historical_replay")
    with tabs[1]:
        dataframe(safe_export(scored.loc[scored["eligible"]]))
    with tabs[2]:
        dataframe(safe_export(scored.loc[~scored["eligible"]]))
    st.markdown("**Prospect context**")
    ordered_ids = scored["record_id"].astype(str).tolist()
    chosen = st.selectbox("Inspect observation", ordered_ids)
    result = scored.loc[scored["record_id"].astype(str) == chosen].iloc[0]
    profile = pool.loc[pool["record_id"].astype(str) == chosen].iloc[0]
    cols = st.columns(3)
    cols[0].metric("Model score", f"{float(result['model_score']):.4f}")
    cols[1].metric("Global rank in current pool", str(int(result["global_rank"])))
    cols[2].metric("Simulation status", "Eligible" if result["eligible"] else "Suppressed")
    st.write(
        f"Recorded profile: age {int(profile['age'])}; job {profile['job']}; "
        f"previous contacts {int(profile['previous'])}; previous outcome {profile['poutcome']}. "
        f"Days since previous campaign: {'not previously contacted (999 sentinel)' if int(profile['pdays']) == 999 else int(profile['pdays'])}."
    )
    if not result["eligible"]:
        st.write(f"Simulated suppression reasons: {result['suppression_reasons']}.")
    st.caption("Recorded facts provide context; they are not causal or individual score explanations.")
    st.info(
        "Conversation prompt: Ask about saving goals, liquidity needs, preferred tenure "
        "and willingness to discuss a term deposit. Suitability needs a separate assessment."
    )


CATEGORY_OPTIONS = {
    "job": ["admin.", "blue-collar", "entrepreneur", "housemaid", "management", "retired",
            "self-employed", "services", "student", "technician", "unemployed", "unknown"],
    "marital": ["married", "single", "divorced", "unknown"],
    "education": ["university.degree", "high.school", "basic.4y", "basic.6y", "basic.9y",
                  "professional.course", "illiterate", "unknown"],
    "default": ["no", "yes", "unknown"],
    "housing": ["yes", "no", "unknown"],
    "loan": ["no", "yes", "unknown"],
    "poutcome": ["nonexistent", "failure", "success"],
}


def scoring_sandbox(bundle: dict) -> None:
    st.subheader("Try the pre-contact scoring workflow")
    st.warning(
        "Public or synthetic data only. Inputs are processed in memory; this demo does not "
        "persist uploads. Do not enter real customer records or personal contact details."
    )
    st.write(
        "The form starts with a reproducible synthetic profile. Model inputs and simulated "
        "engagement eligibility are separate: an opt-out suppresses selection at any score."
    )
    with st.form("single_prospect"):
        cols = st.columns(3)
        with cols[0]:
            age = st.number_input("Age", min_value=1, value=40, step=1,
                                  help="Recorded age in years; no invented demographic cutoff.")
            previous = st.number_input("Previous contacts", min_value=0, value=0, step=1,
                                       help="Contacts before the current campaign.")
            pdays = st.number_input("Days since previous campaign", min_value=0, value=999, step=1,
                                    help="999 is the source sentinel for not previously contacted.")
        values = {"age": age, "previous": previous, "pdays": pdays}
        for index, (field, options) in enumerate(CATEGORY_OPTIONS.items()):
            with cols[1 + index % 2]:
                values[field] = st.selectbox(field.replace("_", " ").title(), options,
                                              help="Recorded profile or previous-campaign value.")
        st.markdown("**Simulated engagement flags — not trained inputs**")
        flag_cols = st.columns(3)
        permission = flag_cols[0].checkbox("Simulated contact permission", value=True)
        opt_out = flag_cols[1].checkbox("Simulated opt-out", value=False)
        recent = flag_cols[2].checkbox("Simulated recent contact", value=False)
        submitted = st.form_submit_button("Score synthetic prospect", type="primary")
    if submitted:
        next_id = int(st.session_state.get("sandbox_counter", 0)) + 1
        st.session_state["sandbox_counter"] = next_id
        record_id = f"synthetic-session-{next_id:04d}"
        frame = pd.DataFrame([{"record_id": record_id, **values}])
        flags = pd.DataFrame([{
            "record_id": record_id, "simulated_contact_permission": permission,
            "simulated_opt_out": opt_out, "simulated_recent_contact": recent,
            "provenance": "simulated",
        }])
        try:
            st.session_state["sandbox_result"] = score_prospects(
                bundle["pipeline"], frame, bundle["manifest"]["champion_id"],
                flags=flags, capacity=1,
            )
        except ValueError as exc:
            st.error(f"Please correct this input: {exc}")
            st.session_state.pop("sandbox_result", None)
    if "sandbox_result" in st.session_state:
        result = st.session_state["sandbox_result"].iloc[0]
        cols = st.columns(2)
        cols[0].metric("Synthetic model score", f"{float(result['model_score']):.4f}")
        cols[1].metric("Simulated outreach eligibility", "Eligible" if result["eligible"] else "Suppressed")
        if not result["eligible"]:
            st.info(f"Suppressed regardless of score: {result['suppression_reasons']}.")
        st.caption("A model score is not a verified personal probability or a product recommendation.")
        dataframe(safe_export(st.session_state["sandbox_result"]))
    with st.expander("Batch CSV scoring (up to 5 MB / 10,000 rows)"):
        st.write(
            "Supply unique nonblank record_id values and the 10 raw features: "
            + ", ".join(FEATURES) + ". Recognized excluded source fields are ignored; "
            "misspelled or unrecognized columns fail validation. Numeric fields must be "
            "finite integers in source units. Blank categories become unknown."
        )
        sample = pd.DataFrame([{"record_id": "synthetic-example-001", "age": 40,
                                "job": "admin.", "marital": "married",
                                "education": "university.degree", "default": "no",
                                "housing": "yes", "loan": "no", "previous": 0,
                                "pdays": 999, "poutcome": "nonexistent"}])
        st.download_button("Download synthetic input template", sample.to_csv(index=False),
                           "synthetic_prospects.csv", "text/csv")
        upload = st.file_uploader("Public / synthetic prospects CSV", type=["csv"], key="batch_csv")
        flags_upload = st.file_uploader("Optional simulated engagement CSV", type=["csv"], key="batch_flags")
        st.caption(
            "Simulation CSV fields: record_id, simulated_contact_permission, simulated_opt_out, "
            "simulated_recent_contact, provenance. Flags accept true/false or 1/0 only; "
            "provenance must be simulated. Missing joined flags suppress outreach."
        )
        budget = st.slider("Batch eligible capacity (%)", 0, 100, 20, key="batch_capacity")
        if st.button("Score uploaded batch"):
            if upload is None:
                st.error("Choose a prospects CSV first.")
            else:
                try:
                    frame = read_upload(upload)
                    if len(frame) > 10_000:
                        raise ValueError("Batch contains more than 10,000 rows. Upload a smaller file.")
                    flags = read_upload(flags_upload) if flags_upload is not None else None
                    if flags is None:
                        st.info("No engagement file supplied: deterministic simulated engagement is applied.")
                        if "record_id" not in frame:
                            raise ValueError("Missing required column: record_id.")
                        flags = simulate_engagement(frame["record_id"].astype(str).str.strip().tolist())
                    scored = score_prospects(
                        bundle["pipeline"], frame, bundle["manifest"]["champion_id"],
                        flags=flags, fraction=budget / 100,
                    )
                    st.session_state["batch_result"] = scored
                    st.session_state["batch_scope"] = f"{upload.name}; eligible capacity {budget}%"
                    for warning in scored.attrs.get("warnings", []):
                        st.warning(str(warning))
                    st.success(
                        f"{len(scored):,} valid records; {int(scored['eligible'].sum()):,} eligible; "
                        f"{int(scored['selected'].sum()):,} selected."
                    )
                except (ValueError, UnicodeDecodeError, pd.errors.ParserError) as exc:
                    st.session_state.pop("batch_result", None)
                    st.error(f"Batch rejected: {exc}")
        if "batch_result" in st.session_state:
            scored = st.session_state["batch_result"]
            st.caption("Last successfully scored batch: " + st.session_state["batch_scope"])
            dataframe(safe_export(scored))
            show_downloads(scored, bundle, len(scored), "synthetic_batch")


def read_upload(upload) -> pd.DataFrame:
    data = upload.getvalue()
    if len(data) > MAX_UPLOAD_BYTES:
        raise ValueError("Each upload must be no larger than 5 MB.")
    frame = pd.read_csv(io.BytesIO(data), dtype=str)
    if frame.empty:
        raise ValueError("CSV contains no records.")
    return frame


def main() -> None:
    st.set_page_config(page_title="Banking Engagement | Independent Analytics", page_icon="◈", layout="wide")
    st.markdown("""
        <style>
        .block-container { padding-top: 2.2rem; max-width: 1440px; }
        [data-testid="stSidebar"] { background: #f2f5f7; }
        [data-testid="stMetric"] { border: 1px solid #dde5e9; border-radius: 10px;
            background: #f8fafb; padding: 15px; }
        [data-testid="stMetricLabel"] { color: #526271; }
        h1 { letter-spacing: -1px; }
        h2, h3 { letter-spacing: -.3px; }
        </style>
    """, unsafe_allow_html=True)
    with st.sidebar:
        st.markdown("### ◈ Banking engagement")
        st.caption("Decision support, before the conversation")
        page = st.radio("Workspace", PAGES)
        st.divider()
        st.markdown("**One business question**")
        st.write("With limited contact capacity, which prospects should the team review first?")
        st.caption("Python · SQLite · scikit-learn · Streamlit")
        st.markdown(f"[Dataset & provenance]({SOURCE})")
    st.caption("INDEPENDENT BANKING ANALYTICS PROTOTYPE")
    st.title("Customer engagement & deposit propensity")
    context_note()
    try:
        manifest_path = DEMO / "manifest.json"
        identity = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        bundle = cached_bundle(str(DEMO), identity)
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        st.error("The verified demo bundle is unavailable or incompatible.")
        st.write("Build or restore the trusted demo artifacts using the repository README, then restart the application.")
        st.caption(f"Bundle check: {type(exc).__name__}. No model is trained on application startup.")
        st.stop()
    if page == "Campaign overview":
        campaign_overview(bundle, identity)
    elif page == "Model results":
        model_results(bundle, identity)
    elif page == "Outreach shortlist":
        outreach_shortlist(bundle)
    else:
        scoring_sandbox(bundle)
    st.divider()
    st.caption(
        "Historical offline results · Simulated engagement · No real outreach or personal contact data · "
        "Scores support review and a needs-based conversation."
    )


if __name__ == "__main__":
    main()
