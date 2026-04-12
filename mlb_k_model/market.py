
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def american_to_prob(odds: Optional[int]) -> float:
    if odds is None or pd.isna(odds):
        return np.nan
    odds = int(odds)
    if odds < 0:
        return -odds / (-odds + 100.0)
    return 100.0 / (odds + 100.0)


def prob_to_american(p: float) -> int:
    p = float(np.clip(p, 1e-9, 1 - 1e-9))
    if p >= 0.5:
        return int(round(-100.0 * p / (1.0 - p)))
    return int(round(100.0 * (1.0 - p) / p))


def remove_vig_two_way(over_odds: Optional[int], under_odds: Optional[int]) -> Tuple[float, float]:
    p_over = american_to_prob(over_odds)
    p_under = american_to_prob(under_odds)
    if np.isnan(p_over) or np.isnan(p_under) or p_over + p_under <= 0:
        return np.nan, np.nan
    z = p_over + p_under
    return p_over / z, p_under / z


def line_event_probs_from_pmf(pmf: np.ndarray, line_value: float) -> Tuple[float, float, float]:
    ks = np.arange(len(pmf))
    p_over = float(pmf[ks > line_value].sum())
    p_under = float(pmf[ks < line_value].sum())
    p_push = float(pmf[ks == line_value].sum()) if abs(line_value - round(line_value)) < 1e-9 else 0.0
    return p_over, p_under, p_push


def pmf_mean(pmf: np.ndarray) -> float:
    ks = np.arange(len(pmf))
    return float((ks * pmf).sum())


@dataclass
class MarketView:
    consensus: pd.DataFrame
    market_pmf: Optional[np.ndarray]
    efficiency: Dict[str, float]


@dataclass
class MarketModel:
    max_iter: int = 100
    tol: float = 1e-7

    def quotes_to_frame(self, quotes: Sequence) -> pd.DataFrame:
        if not quotes:
            return pd.DataFrame()
        frame = pd.DataFrame([q.__dict__ if hasattr(q, "__dict__") else q for q in quotes]).copy()
        if frame.empty:
            return frame
        frame["line_value"] = frame["line_value"].astype(float)
        frame["fair_over_prob"], frame["fair_under_prob"] = zip(
            *frame.apply(lambda r: remove_vig_two_way(r.get("over_odds"), r.get("under_odds")), axis=1)
        )
        return frame

    def consensus_thresholds(self, quotes: Sequence) -> pd.DataFrame:
        frame = self.quotes_to_frame(quotes)
        if frame.empty:
            return frame

        out = (
            frame.groupby("line_value", as_index=False)
            .agg(
                fair_over_prob=("fair_over_prob", "mean"),
                fair_under_prob=("fair_under_prob", "mean"),
                vendor_count=("vendor", "nunique"),
                vendor_std=("fair_over_prob", "std"),
                updated_at=("updated_at", "max"),
            )
            .sort_values("line_value")
            .reset_index(drop=True)
        )
        out["vendor_std"] = out["vendor_std"].fillna(0.0)
        return out

    def _project_one_constraint(self, q: np.ndarray, mask: np.ndarray, target_prob: float) -> np.ndarray:
        current = float(q[mask].sum())
        target_prob = float(np.clip(target_prob, 1e-6, 1 - 1e-6))
        current = float(np.clip(current, 1e-6, 1 - 1e-6))

        q_new = q.copy()
        q_new[mask] *= target_prob / current
        q_new[~mask] *= (1 - target_prob) / (1 - current)
        q_new = np.clip(q_new, 1e-12, None)
        q_new /= q_new.sum()
        return q_new

    def entropy_pool(self, prior_pmf: np.ndarray, consensus: pd.DataFrame) -> np.ndarray:
        q = np.asarray(prior_pmf, dtype=float)
        q = np.clip(q, 1e-12, None)
        q /= q.sum()

        if consensus.empty:
            return q

        ks = np.arange(len(q))
        for _ in range(self.max_iter):
            old = q.copy()
            for row in consensus.itertuples(index=False):
                mask = ks > float(row.line_value)
                q = self._project_one_constraint(q, mask, float(row.fair_over_prob))
            if np.max(np.abs(q - old)) < self.tol:
                break

        return q

    def build_view(self, quotes: Sequence, prior_pmf: np.ndarray) -> MarketView:
        consensus = self.consensus_thresholds(quotes)
        if consensus.empty:
            return MarketView(consensus=consensus, market_pmf=None, efficiency={})

        market_pmf = self.entropy_pool(prior_pmf, consensus)
        eff = {
            "vendor_count": float(consensus["vendor_count"].max()),
            "threshold_count": float(len(consensus)),
            "mean_vendor_std": float(consensus["vendor_std"].mean()),
            "market_mean": pmf_mean(market_pmf),
            "prior_mean": pmf_mean(prior_pmf),
            "mean_shift": float(abs(pmf_mean(market_pmf) - pmf_mean(prior_pmf))),
            "js_like_divergence": float(0.5 * np.sum((np.sqrt(prior_pmf) - np.sqrt(market_pmf)) ** 2)),
        }
        return MarketView(consensus=consensus, market_pmf=market_pmf, efficiency=eff)

    def line_grid_from_pmf(self, pmf: np.ndarray, line_values: Iterable[float]) -> pd.DataFrame:
        rows: List[Dict] = []
        for line_value in sorted(set(float(x) for x in line_values)):
            p_over, p_under, p_push = line_event_probs_from_pmf(pmf, line_value)
            rows.append(
                {
                    "line_value": line_value,
                    "p_over": p_over,
                    "p_under": p_under,
                    "p_push": p_push,
                    "fair_over_american": prob_to_american(max(p_over, 1e-9)),
                    "fair_under_american": prob_to_american(max(p_under, 1e-9)),
                }
            )
        return pd.DataFrame(rows)
