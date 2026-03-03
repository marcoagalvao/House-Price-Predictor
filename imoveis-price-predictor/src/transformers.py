# src/transformers.py
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class QuantileClipper(BaseEstimator, TransformerMixin):
    def __init__(self, lower_q=0.01, upper_q=0.99):
        self.lower_q = lower_q
        self.upper_q = upper_q
        self.lower_ = None
        self.upper_ = None
        self.columns_ = None

    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            self.columns_ = X.columns
            self.lower_ = X.quantile(self.lower_q)
            self.upper_ = X.quantile(self.upper_q)
        else:
            self.columns_ = None
            self.lower_ = np.quantile(X, self.lower_q, axis=0)
            self.upper_ = np.quantile(X, self.upper_q, axis=0)
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            Xc = X.copy()
            for col in Xc.columns:
                lo = self.lower_.loc[col]
                hi = self.upper_.loc[col]
                Xc[col] = Xc[col].clip(lo, hi)
            return Xc
        return np.clip(X, self.lower_, self.upper_)

    def get_feature_names_out(self, input_features=None):
        return input_features