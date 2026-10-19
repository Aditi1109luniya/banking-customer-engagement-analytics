"""Label-independent ranking and explicit offline metric definitions."""
import hashlib
import math
import numpy as np
import pandas as pd
from sklearn.metrics import (roc_auc_score, precision_recall_curve, auc,
                             average_precision_score, brier_score_loss)


def budget_count(n, fraction=.2, capacity=None):
    if capacity is not None:
        if isinstance(capacity, bool) or int(capacity) != capacity or capacity < 0:
            raise ValueError("Capacity must be a nonnegative integer")
        return min(n, int(capacity))
    if not np.isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError("Fraction must be between 0 and 1")
    return min(n, math.ceil(fraction * n))


def ranked_order(ids, scores):
    scores = np.asarray(scores, dtype=float)
    ids = np.asarray([str(v) for v in ids])
    if len(ids) != len(scores) or not np.isfinite(scores).all():
        raise ValueError("IDs and finite scores must have matching lengths")
    ties = np.array([hashlib.sha256(f"42:{v}".encode()).hexdigest() for v in ids])
    return np.lexsort((ids, ties, -scores))


def positive_scores(pipeline, frame):
    classes = list(pipeline.classes_)
    if 1 not in classes:
        raise ValueError("Fitted model has no positive class 1")
    values = pipeline.predict_proba(frame)[:, classes.index(1)]
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("Invalid model scores")
    return values


def ranking_metrics(y, scores, ids, fraction=.2):
    y = np.asarray(y, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if len(y) != len(scores) or not set(y).issubset({0, 1}):
        raise ValueError("Aligned binary outcomes are required")
    n, positives = len(y), int(y.sum())
    k = budget_count(n, fraction)
    selected = ranked_order(ids, scores)[:k]
    tp = int(y[selected].sum())
    prevalence = positives / n if n else None
    precision = tp / k if k else None
    recall = tp / positives if positives else None
    result = dict(n=n, positives=positives, k=k, tp_k=tp, base_rate=prevalence,
                  coverage=k/n if n else 0.0, precision_at_20=precision,
                  recall_at_20=recall,
                  lift_at_20=precision/prevalence if k and positives else None,
                  roc_auc=None, pr_auc=None, ap=None, brier=None)
    if n:
        result["brier"] = float(brier_score_loss(y, scores))
    if len(set(y)) == 2:
        result["roc_auc"] = float(roc_auc_score(y, scores))
    if positives:
        p, r, _ = precision_recall_curve(y, scores)
        result.update(pr_auc=float(auc(r, p)), ap=float(average_precision_score(y, scores)))
    return result


def cutoff_metrics(y, scores, threshold):
    y, scores = np.asarray(y), np.asarray(scores)
    picked = scores >= threshold
    count, positives = int(picked.sum()), int(y.sum())
    return {"fixed_cutoff_count": count, "fixed_cutoff_coverage": count/len(y) if len(y) else 0,
            "fixed_cutoff_precision": float(y[picked].mean()) if count else None,
            "fixed_cutoff_recall": int(y[picked].sum())/positives if positives else None}


def budget_curve(y, scores, ids):
    rows = []
    for q in sorted(set([.01, .05, .1, .2, .3, .5, 1.] + list(np.linspace(.02, .98, 49)))):
        m = ranking_metrics(y, scores, ids, q)
        rows.append(dict(fraction=q, precision=m["precision_at_20"],
                         recall=m["recall_at_20"], lift=m["lift_at_20"]))
    return pd.DataFrame(rows)


def group_audit(frame, scores):
    picked = np.zeros(len(frame), dtype=bool)
    picked[ranked_order(frame.record_id, scores)[:budget_count(len(frame))]] = True
    work = frame.copy()
    work["selected"] = picked
    rows = []
    for column in ["age_group", "job"]:
        for name, part in work.groupby(column, observed=True, dropna=False):
            n, pos, sel = len(part), int(part.y.sum()), int(part.selected.sum())
            tp = int(part.loc[part.selected, "y"].sum())
            rows.append(dict(group_type=column, group=str(name), n=n, positives=pos,
                             selected=sel, base_rate=pos/n if n >= 30 else None,
                             selection_rate=sel/n if n >= 30 else None,
                             precision=tp/sel if sel >= 30 else None,
                             recall=tp/pos if pos >= 10 else None))
    return pd.DataFrame(rows)
