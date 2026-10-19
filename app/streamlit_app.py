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
from bank_engagement.constants import DEMO, FEATURES, ROOT
from bank_engagement.database import query_analysis
from bank_engagement.engagement import simulate_engagement
from bank_engagement.evaluation import ranked_order
from bank_engagement.scoring import score_prospects

PAGES = ["Engagement hub", "Campaign overview", "Model results", "Outreach shortlist", "Scoring sandbox"]
COLOR = "#b9512b"
NAVY = "#19364a"
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
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Arial, sans-serif", size=13, color=NAVY),
        colorway=[COLOR, NAVY, "#b99b68", "#6c8f91"],
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    return fig


def navigate_to(page: str, capacity_percent: int | None = None) -> None:
    """Button callbacks run before the keyed sidebar widget is rendered again."""
    st.session_state["workspace"] = page
    if page == "Outreach shortlist" and capacity_percent is not None:
        st.session_state["outreach_capacity"] = capacity_percent


def engagement_hub(bundle: dict, identity: str) -> None:
    """Use frozen results and simulated policy, never invented dashboard KPIs."""
    hero, artwork = st.columns([1.08, 1], gap="large", vertical_alignment="center")
    with hero:
        st.markdown('<p class="hub-eyebrow">CUSTOMER ENGAGEMENT · DECISION SUPPORT</p>', unsafe_allow_html=True)
        st.markdown('<h1 class="hub-title">Better conversations.<br>Smarter prioritisation.</h1>', unsafe_allow_html=True)
        st.markdown(
            '<p class="hub-intro">Give every limited outreach slot a clear purpose. '
            'Explore historical campaign patterns, review deposit propensity and build '
            'a practical shortlist for a needs-based customer conversation.</p>',
            unsafe_allow_html=True,
        )
        st.caption("Independent banking analytics prototype · Public historical data · Simulated engagement")
        actions = st.columns(2)
        actions[0].button("Explore analysis →", type="primary", use_container_width=True,
                          on_click=navigate_to, args=("Campaign overview",), key="hero_campaign")
        actions[1].button("Plan outreach →", use_container_width=True,
                          on_click=navigate_to, args=("Outreach shortlist",), key="hero_shortlist")
    with artwork:
        hero_path = ROOT / "app/assets/banking_hero.png"
        if hero_path.exists():
            st.image(str(hero_path), use_container_width=True)
            st.caption("Illustrative artwork · No real customers or bank affiliation")
        else:
            overview = cached_analysis("overview", "all", str(bundle["db_path"]), identity).iloc[0]
            subscriptions = int(overview["subscriptions"])
            observations = int(overview["observations"])
            fallback = go.Figure(go.Pie(
                values=[subscriptions, observations - subscriptions],
                labels=["Recorded subscription", "No recorded subscription"], hole=.76,
                marker=dict(colors=[COLOR, "#e9e1d4"]), textinfo="none",
                hovertemplate="%{label}<br>%{value:,} observations<extra></extra>",
            ))
            fallback.add_annotation(x=.5, y=.5, text=f"<b>{percent(subscriptions / observations)}</b><br>historical subscription rate",
                                    showarrow=False, font=dict(size=18, color=NAVY))
            fallback.update_layout(showlegend=False)
            st.plotly_chart(plot_layout(fallback, height=300), use_container_width=True, key="hub_hero_chart")
            st.caption("Full historical dataset · Portuguese telephone campaigns")
    st.markdown('<div class="hub-rule"></div>', unsafe_allow_html=True)
    overview = cached_analysis("overview", "all", str(bundle["db_path"]), identity).iloc[0]
    metrics = pd.DataFrame(bundle["metrics"])
    champion = metrics.loc[(metrics["split"] == "test") & metrics["champion"].astype(bool)].iloc[0]
    cards = st.columns([1, 1, 2], gap="large", vertical_alignment="center")
    cards[0].metric("Historical observations", f"{int(overview['observations']):,}",
                    help="Full historical dataset; observations are not confirmed unique customers.")
    cards[1].metric("Test lift @ 20%", f"{float(champion['lift_at_20']):.2f}×",
                    help="Test precision at 20% divided by the overall test subscription rate.")
    with cards[2]:
        st.markdown("**Evidence for a better-informed conversation**")
        st.caption(
            "The lift compares the highest-scored fifth with the full historical test pool. "
            "It measures offline ranking quality, not incremental sales. Explore the model "
            "results for the full comparison and limitations."
        )
    st.markdown("### Your engagement workspace")
    heading, budget_control = st.columns([1.8, 1], gap="large", vertical_alignment="center")
    with heading:
        st.write("Start with the evidence. Follow a chart into the detailed analysis, then review a simulated shortlist.")
    with budget_control:
        capacity_percent = st.slider("Preview contact capacity (%)", 1, 50, 20, 1, key="hub_capacity",
                                     help="Exploratory replay only; official metrics and the frozen validation cutoff do not change.")
    capacity_fraction = capacity_percent / 100
    test = bundle["test"]
    labels = test["y"].astype(str).str.lower().isin(["yes", "1", "true"]).astype(int)
    selected_count = min(len(test), int(np.ceil(capacity_fraction * len(test))))
    ordered = ranked_order(test["record_id"], test["model_score"])
    captured = int(labels.iloc[ordered[:selected_count]].sum())
    replay_recall = captured / int(labels.sum()) if labels.sum() else 0
    preview_cards = st.columns(3, gap="medium")
    with preview_cards[0], st.container(border=True):
        st.caption("01 / UNDERSTAND THE HISTORY")
        st.markdown("#### Campaign patterns")
        st.write("How do recorded subscriptions vary by previous campaign outcome?")
        frame = cached_analysis("by_poutcome", "all", str(bundle["db_path"]), identity).copy()
        frame["display_rate"] = frame["subscription_rate"].where(frame["observations"] >= 30)
        frame = frame.sort_values("display_rate", ascending=True)
        fig = px.bar(frame, x="display_rate", y="poutcome", orientation="h",
                     custom_data=["observations", "subscriptions"], color_discrete_sequence=[COLOR],
                     labels={"display_rate": "Historical subscription rate", "poutcome": ""})
        fig.update_traces(hovertemplate="%{y}<br>Rate: %{x:.1%}<br>Observations: %{customdata[0]:,}<br>Subscriptions: %{customdata[1]:,}<extra></extra>")
        fig.update_xaxes(tickformat=".0%", rangemode="tozero")
        st.plotly_chart(plot_layout(fig, height=250), use_container_width=True, key="hub_campaign_chart")
        st.caption("Full history · Recorded association, not causality · Rates hidden below 30 observations")
        st.button("Open campaign analysis", use_container_width=True, on_click=navigate_to,
                  args=("Campaign overview",), key="hub_open_campaign")
    with preview_cards[1], st.container(border=True):
        st.caption("02 / EXAMINE THE MODEL")
        st.markdown("#### Capacity & capture")
        st.write(f"At {capacity_percent}% capacity, the historical replay captures {percent(replay_recall)} of test subscriptions.")
        curves = bundle["curves"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=curves["fraction"], y=curves["recall"], name="Champion", mode="lines",
                                 line=dict(color=COLOR, width=3),
                                 hovertemplate="Capacity: %{x:.0%}<br>Captured: %{y:.1%}<extra>Champion</extra>"))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="Random reference", mode="lines",
                                 line=dict(color="#9b9a91", dash="dot"), hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=[capacity_fraction], y=[replay_recall], name="Preview budget", mode="markers",
                                 marker=dict(color=NAVY, size=10), showlegend=False,
                                 hovertemplate=f"{selected_count:,} selected<br>{captured:,} historical subscriptions<extra>Preview</extra>"))
        fig.update_layout(xaxis_title="Fraction of test pool contacted", yaxis_title="Test subscriptions captured")
        fig.update_xaxes(tickformat=".0%", range=[0, 1])
        fig.update_yaxes(tickformat=".0%", range=[0, 1])
        st.plotly_chart(plot_layout(fig, height=250), use_container_width=True, key="hub_capacity_chart")
        st.caption(f"All {len(test):,} test rows · No simulated suppression · Exploration does not tune the model")
        st.button("Open model results", use_container_width=True, on_click=navigate_to,
                  args=("Model results",), key="hub_open_model")
    with preview_cards[2], st.container(border=True):
        st.caption("03 / PREPARE THE CONVERSATION")
        st.markdown("#### Simulated outreach")
        flags = simulate_engagement(test["record_id"].tolist())
        eligible = (flags["simulated_contact_permission"] & ~flags["simulated_opt_out"] & ~flags["simulated_recent_contact"])
        eligible_count = int(eligible.sum())
        simulated_selected = int(np.ceil(capacity_fraction * eligible_count))
        st.write(f"Review {simulated_selected:,} records at {capacity_percent}% of the simulated eligible pool.")
        fig = go.Figure(go.Pie(
            labels=["Simulated eligible", "Simulated suppressed"],
            values=[eligible_count, len(test) - eligible_count], hole=.72,
            marker=dict(colors=[NAVY, "#e5d7c7"]), textinfo="none",
            hovertemplate="%{label}<br>%{value:,} observations (%{percent})<extra></extra>",
        ))
        fig.add_annotation(x=.5, y=.5, text=f"<b>{eligible_count:,}</b><br>simulated eligible",
                           showarrow=False, font=dict(size=16, color=NAVY))
        fig.update_layout(showlegend=False)
        st.plotly_chart(plot_layout(fig, height=250), use_container_width=True, key="hub_eligibility_chart")
        st.caption("Final test replay · Permission, opt-out and cooldown are simulated · Opt-outs always suppress")
        st.button("Open outreach shortlist", type="primary", use_container_width=True, on_click=navigate_to,
                  args=("Outreach shortlist", capacity_percent), key="hub_open_shortlist")
    st.markdown('<div class="hub-rule"></div>', unsafe_allow_html=True)
    st.markdown("**A score starts a conversation. It does not establish product suitability.**")
    st.write("Review saving goals, liquidity needs, preferred tenure and willingness to discuss before considering any product.")


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
    cards[3].metric("Data version", str(bundle["manifest"].get("source_sha256", "unavailable"))[:6])
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
            if "outreach_capacity" not in st.session_state:
                st.session_state["outreach_capacity"] = 20
            fraction = st.slider("Eligible pool capacity (%)", 0, 100, step=1, key="outreach_capacity") / 100
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
        .block-container { padding-top: 4rem; max-width: 1440px; }
        [data-testid="stAppViewContainer"], [data-testid="stMain"] { background: #ffffff; }
        [data-testid="stHeader"] { background: #ffffff; border-bottom: 1px solid #edf0f4; }
        [data-testid="stSidebar"] { background: #ffffff; border-right: 1px solid #e3e8ee; }
        [data-testid="stMetric"] { border: 1px solid #e3e8ee; border-radius: 8px;
            background: #ffffff; padding: 18px; }
        [data-testid="stMetricLabel"] { color: #64717d; }
        [data-testid="stMetricLabel"] p { white-space: normal !important;
            overflow: visible !important; text-overflow: clip !important; }
        [data-testid="stMetricValue"] { color: #19364a; }
        [data-testid="stVerticalBlockBorderWrapper"] > div {
            background: #ffffff; border-color: #e3e8ee; border-radius: 8px; }
        [data-testid="stButton"] button { border-radius: 6px; font-weight: 600; }
        [data-testid="stButton"] button[kind="secondary"] { background: #ffffff;
            border-color: #d6dee7; color: #19364a; }
        [data-testid="stButton"] button[kind="secondary"]:hover {
            border-color: #b9512b; color: #b9512b; }
        h1 { color: #19364a; letter-spacing: -1.8px; line-height: 1.12; font-weight: 650; }
        h2, h3, h4 { color: #19364a; letter-spacing: -.4px; }
        .hub-eyebrow { font-size: .72rem; font-weight: 700; letter-spacing: .14em;
            color: #aa4a2a; margin-bottom: .7rem; }
        .hub-title { font-size: clamp(2.25rem, 3.5vw, 3.7rem); }
        .hub-intro { font-size: 1.08rem; line-height: 1.65; color: #51616a;
            max-width: 560px; margin: 1rem 0; }
        .hub-rule { height: 1px; background: #e3e8ee; margin: 1.3rem 0; }
        [data-testid="stImage"] img { border-radius: 16px; }
        </style>
    """, unsafe_allow_html=True)
    with st.sidebar:
        st.markdown("### ◈ Banking engagement")
        st.caption("Decision support, before the conversation")
        page = st.radio("Workspace", PAGES, key="workspace")
        st.divider()
        st.markdown("**One business question**")
        st.write("With limited contact capacity, which prospects should the team review first?")
        st.caption("Python · SQLite · scikit-learn · Streamlit")
        st.markdown(f"[Dataset & provenance]({SOURCE})")
    if page != "Engagement hub":
        st.caption("INDEPENDENT BANKING ANALYTICS PROTOTYPE")
        st.title("Customer engagement & deposit propensity")
        st.button("← Back to engagement hub", on_click=navigate_to, args=("Engagement hub",), key="back_to_hub")
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
    if page == "Engagement hub":
        engagement_hub(bundle, identity)
        context_note()
    elif page == "Campaign overview":
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
