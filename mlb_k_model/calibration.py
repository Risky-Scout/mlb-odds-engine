
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


def _clip_probs(p: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)


@dataclass
class BinaryIsotonicCalibrator:
    model: Optional[IsotonicRegression] = None

    def fit(self, y_true: np.ndarray, p_pred: np.ndarray) -> "BinaryIsotonicCalibrator":
        y_true = np.asarray(y_true, dtype=float)
        p_pred = _clip_probs(p_pred)
        if len(np.unique(p_pred)) < 5:
            self.model = None
            return self
        self.model = IsotonicRegression(y_min=1e-6, y_max=1 - 1e-6, out_of_bounds="clip")
        self.model.fit(p_pred, y_true)
        return self

    def predict(self, p_pred: np.ndarray) -> np.ndarray:
        p_pred = _clip_probs(p_pred)
        if self.model is None:
            return p_pred
        return _clip_probs(self.model.predict(p_pred))


@dataclass
class CountThresholdCalibrator:
    max_k: int
    models: Dict[int, IsotonicRegression] = field(default_factory=dict)

    def fit(self, pmf_matrix: np.ndarray, actual_counts: np.ndarray) -> "CountThresholdCalibrator":
        pmf_matrix = np.asarray(pmf_matrix, dtype=float)
        actual_counts = np.asarray(actual_counts, dtype=int)

        for k in range(self.max_k):
            raw_cdf = pmf_matrix[:, : k + 1].sum(axis=1)
            y = (actual_counts <= k).astype(float)
            if len(np.unique(raw_cdf)) < 5:
                continue
            iso = IsotonicRegression(y_min=1e-6, y_max=1 - 1e-6, out_of_bounds="clip")
            iso.fit(_clip_probs(raw_cdf), y)
            self.models[k] = iso

        return self

    def calibrate_pmf(self, pmf: np.ndarray) -> np.ndarray:
        pmf = np.asarray(pmf, dtype=float)
        pmf = pmf / pmf.sum()

        cdf = np.cumsum(pmf)
        cal_cdf = np.zeros_like(cdf)

        for k in range(len(pmf)):
            if k in self.models:
                cal_cdf[k] = float(self.models[k].predict([float(cdf[k])])[0])
            else:
                cal_cdf[k] = float(cdf[k])

        cal_cdf = np.maximum.accumulate(np.clip(cal_cdf, 1e-6, 1 - 1e-6))
        cal_cdf[-1] = 1.0

        out = np.empty_like(pmf)
        out[0] = cal_cdf[0]
        out[1:] = np.diff(cal_cdf)
        out = np.clip(out, 0.0, 1.0)
        out = out / out.sum()
        return out


@dataclass
class LineProbabilityBlender:
    """
    Optional market-aware calibrator.

    Fits on historical market snapshots if you have them. If not fit, it falls back to the
    model probability.
    """
    model: Optional[LogisticRegression] = None
    feature_names: List[str] = field(default_factory=lambda: ["model_prob", "market_prob", "abs_gap", "vendor_count", "is_live"])

    def fit(self, frame) -> "LineProbabilityBlender":
        if frame is None or len(frame) < 200:
            self.model = None
            return self

        X = frame[self.feature_names].fillna(0.0).to_numpy()
        y = frame["target"].astype(int).to_numpy()
        if len(np.unique(y)) < 2:
            self.model = None
            return self

        self.model = LogisticRegression(C=1.0, max_iter=1000)
        self.model.fit(X, y)
        return self

    def predict(self, frame) -> np.ndarray:
        model_prob = frame["model_prob"].astype(float).to_numpy()
        if self.model is None:
            return _clip_probs(model_prob)
        X = frame[self.feature_names].fillna(0.0).to_numpy()
        return _clip_probs(self.model.predict_proba(X)[:, 1])
