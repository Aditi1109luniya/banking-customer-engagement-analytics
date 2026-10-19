"""Persistable, allowlisted pre-contact feature construction."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from bank_engagement.constants import FEATURES, CATEGORICAL


class PreContactFeatures(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        self.feature_names_in_ = np.array(FEATURES, dtype=object)
        recency = pd.to_numeric(X["pdays"], errors="raise")
        observed = recency[recency != 999]
        self.recency_fallback_ = observed.empty
        self.recency_median_ = 0.0 if observed.empty else float(observed.median())
        return self

    def transform(self, X):
        out = X.loc[:, FEATURES].copy()
        for c in CATEGORICAL:
            out[c] = out[c].fillna("unknown").astype(str).str.strip().str.lower()
            out[c] = out[c].replace("", "unknown")
        out["no_previous_contact"] = (out["pdays"] == 999).astype(int)
        out["pdays_recency"] = out["pdays"].mask(out["pdays"] == 999, self.recency_median_)
        return out.drop(columns="pdays")


def make_pipeline(estimator, scale=False):
    numeric = ["age", "previous", "pdays_recency", "no_previous_contact"]
    transform = ColumnTransformer([
        ("numeric", StandardScaler() if scale else "passthrough", numeric),
        ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
         CATEGORICAL),
    ], remainder="drop")
    return Pipeline([("features", PreContactFeatures()), ("preprocess", transform),
                     ("model", estimator)])
