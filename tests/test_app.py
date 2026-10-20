"""Exercise the real Streamlit entry point, including eligibility-independent scoring."""

from pathlib import Path
import io

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from bank_engagement.constants import DEMO, ROOT

APP = ROOT / "app/streamlit_app.py"


def demo_app() -> AppTest:
    if not (DEMO / "manifest.json").exists():
        pytest.skip("Run the reproducible pipeline/package-demo before demo AppTest checks.")
    return AppTest.from_file(str(APP), default_timeout=60).run()


def test_missing_bundle_is_actionable(monkeypatch, tmp_path: Path):
    import bank_engagement.constants as constants

    monkeypatch.setattr(constants, "DEMO", tmp_path / "missing_demo")
    at = AppTest.from_file(str(APP), default_timeout=30).run()
    assert not at.exception
    assert any("verified demo bundle" in item.value for item in at.error)
    assert any("No model is trained" in item.value for item in at.caption)


@pytest.mark.parametrize("page", ["Engagement hub", "Campaign overview", "Model results", "Outreach shortlist", "Scoring sandbox"])
def test_each_page_renders(page):
    at = demo_app()
    at.sidebar.radio[0].set_value(page).run()
    assert not at.exception
    assert not at.error
    assert any("historical" in item.value.lower() for item in at.caption)


def test_zero_capacity_is_handled_without_error():
    at = demo_app()
    at.sidebar.radio[0].set_value("Outreach shortlist").run()
    next(s for s in at.slider if s.label == "Eligible pool capacity (%)").set_value(0).run()
    assert not at.exception
    assert any(m.label == "Selected" and m.value == "0" for m in at.metric)


def test_opt_out_suppresses_but_does_not_change_propensity():
    at = demo_app()
    at.sidebar.radio[0].set_value("Scoring sandbox").run()
    next(b for b in at.button if b.label == "Score synthetic prospect").click().run()
    assert not at.exception
    first_score = next(m.value for m in at.metric if m.label == "Synthetic model score")
    assert any(m.value == "Eligible" for m in at.metric)
    next(c for c in at.checkbox if c.label == "Simulated opt-out").set_value(True)
    next(b for b in at.button if b.label == "Score synthetic prospect").click().run()
    assert not at.exception
    assert next(m.value for m in at.metric if m.label == "Synthetic model score") == first_score
    assert any(m.value == "Suppressed" for m in at.metric)
    assert any("opted_out" in item.value for item in at.info)


def test_upload_retains_text_identifiers_and_strict_flag_tokens():
    from app.streamlit_app import read_upload

    result = read_upload(io.BytesIO(b"record_id,simulated_opt_out\n001,false\n002,0\n"))
    assert result["record_id"].tolist() == ["001", "002"]
    assert result["simulated_opt_out"].tolist() == ["false", "0"]


def test_oversized_upload_and_safe_export():
    from app.streamlit_app import MAX_UPLOAD_BYTES, read_upload, safe_export

    with pytest.raises(ValueError, match="5 MB"):
        read_upload(io.BytesIO(b" " * (MAX_UPLOAD_BYTES + 1)))
    result = safe_export(pd.DataFrame([{"record_id": "001", "model_score": .4,
                                        "selected": True, "y": 1, "phone": "not-real"}]))
    assert list(result) == ["record_id", "model_score", "selected"]


def test_hub_is_default_and_capacity_preview_is_exploratory():
    at = demo_app()
    assert not at.exception
    assert at.sidebar.radio[0].value == "Engagement hub"
    assert not at.dataframe
    official_lift = next(m.value for m in at.metric if m.label == "Test lift @ 20%")
    next(s for s in at.slider if s.label == "Preview contact capacity (%)").set_value(35).run()
    assert not at.exception
    assert next(m.value for m in at.metric if m.label == "Test lift @ 20%") == official_lift
    assert any("At 35% capacity" in item.value for item in at.markdown)
    next(b for b in at.button if b.label == "Open outreach shortlist").click().run()
    assert not at.exception
    assert next(s.value for s in at.slider if s.label == "Eligible pool capacity (%)") == 35


@pytest.mark.parametrize("button,destination", [
    ("Open campaign analysis", "Campaign overview"),
    ("Open model results", "Model results"),
    ("Open outreach shortlist", "Outreach shortlist"),
])
def test_hub_navigation_and_return(button, destination):
    at = demo_app()
    next(b for b in at.button if b.label == button).click().run()
    assert not at.exception
    assert at.sidebar.radio[0].value == destination
    next(b for b in at.button if b.label == "← Back to engagement hub").click().run()
    assert not at.exception
    assert at.sidebar.radio[0].value == "Engagement hub"


@pytest.mark.parametrize("slug,page", [
    ("home", "Engagement hub"), ("analysis", "Campaign overview"),
    ("model", "Model results"), ("outreach", "Outreach shortlist"),
    ("score", "Scoring sandbox"),
])
def test_shared_page_link_opens_the_requested_workspace(slug, page):
    at = AppTest.from_file(str(APP), default_timeout=60)
    at.query_params["view"] = slug
    at.run()
    assert not at.exception
    assert at.sidebar.radio[0].value == page
    assert at.pills[0].value == page


def test_unknown_page_link_falls_back_to_welcome():
    at = AppTest.from_file(str(APP), default_timeout=60)
    at.query_params["view"] = "unrecognised-page"
    at.run()
    assert not at.exception
    assert at.sidebar.radio[0].value == "Engagement hub"


def test_changed_page_url_updates_an_existing_session():
    at = demo_app()
    at.query_params["view"] = "model"
    at.run()
    assert not at.exception
    assert at.sidebar.radio[0].value == "Model results"
    assert at.pills[0].value == "Model results"


def test_visible_navigation_sidebar_and_page_url_stay_in_step():
    at = demo_app()
    at.pills[0].set_value("Model results").run()
    assert not at.exception
    assert at.sidebar.radio[0].value == "Model results"
    assert at.query_params["view"] == "model"
    at.sidebar.radio[0].set_value("Scoring sandbox").run()
    assert not at.exception
    assert at.pills[0].value == "Scoring sandbox"
    assert at.query_params["view"] == "score"


def test_hero_outreach_button_keeps_the_preview_capacity():
    at = demo_app()
    next(s for s in at.slider if s.label == "Preview contact capacity (%)").set_value(35).run()
    next(b for b in at.button if b.label == "Plan outreach →").click().run()
    assert not at.exception
    assert at.sidebar.radio[0].value == "Outreach shortlist"
    assert at.pills[0].value == "Outreach shortlist"
    assert at.query_params["view"] == "outreach"
    assert next(s.value for s in at.slider if s.label == "Eligible pool capacity (%)") == 35
