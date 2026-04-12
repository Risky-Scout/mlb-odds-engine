
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import LogisticRegression

from .market import american_to_prob


def decimal_from_american(odds: int) -> float:
    if odds < 0:
        return 1.0 + 100.0 / abs(odds)
    return 1.0 + odds / 100.0


def kelly_fraction(prob: float, american_odds: int) -> float:
    d = decimal_from_american(int(american_odds))
    b = d - 1.0
    q = 1.0 - prob
    frac = (b * prob - q) / b
    return float(max(frac, 0.0))


def bet_ev(prob: float, american_odds: int) -> float:
    d = decimal_from_american(int(american_odds))
    return float(prob * (d - 1.0) - (1.0 - prob))


@dataclass
class BetSelector:
    """
    Rule engine first, optional learned filter second.
    """
    min_edge: float = 0.025
    min_ev: float = 0.015
    max_uncertainty: float = 0.06
    min_vendor_count: int = 1
    logistic: Optional[LogisticRegression] = None
    feature_names: List[str] = field(default_factory=lambda: ["edge", "ev", "uncertainty", "vendor_count", "abs_model_market_gap", "is_live"])

    def fit(self, historical_candidates: pd.DataFrame) -> "BetSelector":
        if historical_candidates is None or len(historical_candidates) < 250:
            self.logistic = None
            return self

        X = historical_candidates[self.feature_names].fillna(0.0).to_numpy()
        y = historical_candidates["realized_profit_positive"].astype(int).to_numpy()
        if len(np.unique(y)) < 2:
            self.logistic = None
            return self

        self.logistic = LogisticRegression(C=0.5, max_iter=1000)
        self.logistic.fit(X, y)
        return self

    def score(self, candidates: pd.DataFrame) -> pd.DataFrame:
        df = candidates.copy()
        df["pass_rules"] = (
            (df["edge"] >= self.min_edge)
            & (df["ev"] >= self.min_ev)
            & (df["uncertainty"] <= self.max_uncertainty)
            & (df["vendor_count"] >= self.min_vendor_count)
        )

        if self.logistic is None:
            df["selection_score"] = np.where(df["pass_rules"], 1.0, 0.0)
            df["selected"] = df["pass_rules"].astype(int)
            return df

        X = df[self.feature_names].fillna(0.0).to_numpy()
        df["selection_score"] = self.logistic.predict_proba(X)[:, 1]
        df["selected"] = ((df["pass_rules"]) & (df["selection_score"] >= 0.55)).astype(int)
        return df


@dataclass
class RiskSizer:
    bankroll: float = 10000.0
    kelly_fraction_cap: float = 0.0125
    daily_cap: float = 0.05
    per_game_cap: float = 0.025
    uncertainty_penalty: float = 1.0
    default_same_game_corr: float = 0.30
    default_same_pitcher_corr: float = 0.65
    empirical_covariance: Optional[np.ndarray] = None

    def candidate_stake_fraction(self, prob: float, odds: int, uncertainty: float) -> float:
        base = kelly_fraction(prob, odds)
        penalized = base * max(0.0, 1.0 - self.uncertainty_penalty * uncertainty)
        return float(min(self.kelly_fraction_cap, max(0.0, penalized)))

    def _covariance_from_candidates(self, df: pd.DataFrame) -> np.ndarray:
        n = len(df)
        if n == 0:
            return np.zeros((0, 0))

        var = np.maximum(df["return_variance"].to_numpy(dtype=float), 1e-8)
        cov = np.diag(var)

        for i in range(n):
            for j in range(i + 1, n):
                corr = 0.05
                if df.iloc[i]["game_id"] == df.iloc[j]["game_id"]:
                    corr = self.default_same_game_corr
                if df.iloc[i]["pitcher_id"] == df.iloc[j]["pitcher_id"]:
                    corr = self.default_same_pitcher_corr
                cov_ij = corr * np.sqrt(var[i] * var[j])
                cov[i, j] = cov_ij
                cov[j, i] = cov_ij

        return cov

    def empirical_fit(self, return_matrix: np.ndarray) -> "RiskSizer":
        if return_matrix.ndim != 2 or return_matrix.shape[0] < 20:
            return self
        lw = LedoitWolf().fit(return_matrix)
        self.empirical_covariance = lw.covariance_
        return self

    def size_portfolio(self, selected: pd.DataFrame) -> pd.DataFrame:
        df = selected[selected["selected"].eq(1)].copy()
        if df.empty:
            return selected.copy()

        df["base_frac"] = [
            self.candidate_stake_fraction(prob, odds, unc)
            for prob, odds, unc in zip(df["bet_prob"], df["book_odds"], df["uncertainty"])
        ]
        df["mu"] = df["ev"].astype(float).to_numpy()
        d = df["book_odds"].astype(int).apply(decimal_from_american).to_numpy()
        p = df["bet_prob"].astype(float).to_numpy()
        df["return_variance"] = p * (d - 1.0) ** 2 + (1 - p) * 1.0 - df["mu"] ** 2

        cov = self._covariance_from_candidates(df)
        caps = df["base_frac"].to_numpy(dtype=float)
        mu = df["mu"].to_numpy(dtype=float)

        def objective(w: np.ndarray) -> float:
            return float(-(mu @ w - 0.5 * w @ cov @ w))

        cons = [{"type": "ineq", "fun": lambda w: self.daily_cap - float(np.sum(w))}]
        bounds = [(0.0, float(c)) for c in caps]
        x0 = np.minimum(caps, self.daily_cap / max(len(caps), 1))

        res = minimize(objective, x0=x0, bounds=bounds, constraints=cons, method="SLSQP")
        weights = np.clip(res.x if res.success else x0, 0.0, caps)

        # Per-game cap after optimization.
        for game_id, idx in df.groupby("game_id").groups.items():
            idx = list(idx)
            total = weights[df.index.get_indexer(idx)].sum()
            if total > self.per_game_cap and total > 0:
                weights[df.index.get_indexer(idx)] *= self.per_game_cap / total

        df["stake_frac"] = weights
        df["stake_dollars"] = df["stake_frac"] * self.bankroll

        merged = selected.copy()
        merged = merged.merge(
            df[["game_id", "pitcher_id", "line_value", "bet_side", "stake_frac", "stake_dollars"]],
            on=["game_id", "pitcher_id", "line_value", "bet_side"],
            how="left",
        )
        merged["stake_frac"] = merged["stake_frac"].fillna(0.0)
        merged["stake_dollars"] = merged["stake_dollars"].fillna(0.0)
        return merged
