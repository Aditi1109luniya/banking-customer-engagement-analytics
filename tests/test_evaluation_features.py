import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from bank_engagement.features import make_pipeline
from bank_engagement.evaluation import (budget_count, ranked_order, ranking_metrics,
                                      positive_scores, cutoff_metrics)
from bank_engagement.modeling import better


def frame():
    return pd.DataFrame(dict(age=[20,30,40,50,60,70], previous=[0,0,1,2,3,1],
        pdays=[999,999,4,6,8,10], job=["student","admin."]*3, marital=["single"]*6,
        education=["high.school"]*6, default=["no"]*6, housing=["yes"]*6,
        loan=["no"]*6, poutcome=["nonexistent","failure"]*3))


def test_hand_metrics():
    scores = [.99,.8,.6,.2,.1]
    m = ranking_metrics([1,0,1,0,0], scores, range(5), fraction=.4)
    assert (m["k"], m["tp_k"], m["precision_at_20"], m["recall_at_20"],
            m["lift_at_20"]) == (2,1,.5,.5,1.25)


def test_constant_ap_and_trapezoid():
    m = ranking_metrics([0,0,0,1], [.25]*4, range(4))
    assert m["ap"] == .25
    assert m["pr_auc"] == .625
    assert m["roc_auc"] == .5


def test_edge_cases():
    m = ranking_metrics([0,0], [.1,.2], [1,2], fraction=0)
    assert m["precision_at_20"] is None and m["recall_at_20"] is None
    assert m["roc_auc"] is None and m["ap"] is None
    assert budget_count(6179) == 1236
    assert budget_count(0) == 0
    assert cutoff_metrics([1,0], [.8,.8], .8)["fixed_cutoff_count"] == 2
    with pytest.raises(ValueError):
        budget_count(10, capacity=-1)


def test_ties_independent_of_input_order():
    a = np.array(["a","b","c","d"])
    b = a[::-1]
    assert list(a[ranked_order(a,[.1]*4)]) == list(b[ranked_order(b,[.1]*4)])


def test_pipeline_fitted_once_and_saved(tmp_path):
    X = frame()
    model = make_pipeline(LogisticRegression(max_iter=500), True).fit(X,[0,0,1,0,1,1])
    median = model.named_steps["features"].recency_median_
    before = positive_scores(model,X)
    altered = X.assign(duration=999999,y=1,split="test")
    assert np.array_equal(before, positive_scores(model,altered))
    future = X.copy()
    future["pdays"] = 999
    future["job"] = "never_seen"
    assert np.isfinite(positive_scores(model,future)).all()
    assert model.named_steps["features"].recency_median_ == median == 7
    path=tmp_path/"pipeline.joblib"
    joblib.dump(model,path)
    assert np.allclose(before,positive_scores(joblib.load(path),X),atol=1e-12)


def test_no_recency_fallback():
    from bank_engagement.features import PreContactFeatures
    X=frame().assign(pdays=999)
    transformer=PreContactFeatures().fit(X)
    assert transformer.recency_fallback_ and transformer.recency_median_ == 0
    assert (transformer.transform(X).pdays_recency == 0).all()


def test_selection_uses_budget_then_ap_then_simplicity():
    a={"model_id":"a","family":"Random Forest","precision_at_20":.5,"ap":.7}
    b={**a,"model_id":"b","family":"Logistic Regression"}
    assert better(b,a)
    assert not better({**b,"precision_at_20":.4,"ap":.9},a)
    assert better({**a,"precision_at_20":.6,"ap":.1},b)
