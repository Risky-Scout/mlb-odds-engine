from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


def _clip_prob(p):
    p = np.asarray(p, dtype=float)
    return np.clip(p, 1e-6, 1 - 1e-6)


class IdentityCalibrator:
    name = "identity"

    def fit(self, p, y):
        return self

    def predict(self, p):
        return _clip_prob(p)


class IsotonicProbCalibrator:
    name = "isotonic"

    def __init__(self):
        self.model = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1 - 1e-6)

    def fit(self, p, y):
        p = _clip_prob(p)
        y = np.asarray(y, dtype=int)
        self.model.fit(p, y)
        return self

    def predict(self, p):
        p = _clip_prob(p)
        out = self.model.predict(p)
        return _clip_prob(out)


class PlattCalibrator:
    def __init__(self, C=1.0):
        self.C = float(C)
        self.name = f"platt_C{self.C:g}"
        self.model = LogisticRegression(C=self.C, solver="lbfgs", max_iter=2000)

    def fit(self, p, y):
        p = _clip_prob(p)
        x = np.log(p / (1 - p)).reshape(-1, 1)
        y = np.asarray(y, dtype=int)
        self.model.fit(x, y)
        return self

    def predict(self, p):
        p = _clip_prob(p)
        x = np.log(p / (1 - p)).reshape(-1, 1)
        out = self.model.predict_proba(x)[:, 1]
        return _clip_prob(out)


class BetaCalibrator:
    def __init__(self, C=1.0):
        self.C = float(C)
        self.name = f"beta_C{self.C:g}"
        self.model = LogisticRegression(C=self.C, solver="lbfgs", max_iter=2000)

    def fit(self, p, y):
        p = _clip_prob(p)
        X = np.column_stack([np.log(p), np.log(1 - p)])
        y = np.asarray(y, dtype=int)
        self.model.fit(X, y)
        return self

    def predict(self, p):
        p = _clip_prob(p)
        X = np.column_stack([np.log(p), np.log(1 - p)])
        out = self.model.predict_proba(X)[:, 1]
        return _clip_prob(out)
