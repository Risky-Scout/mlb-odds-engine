
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import kstest
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

from .market import line_event_probs_from_pmf, remove_vig_two_way
from .system import StrikeoutBettingSystem


def ranked_probability_score(pmf: np.ndarray, actual: int) -> float:
    pmf = np.asarray(pmf, dtype=float)
    if actual >= len(pmf):
        pmf = np.pad(pmf, (0, actual - len(pmf) + 1))
    obs = np.zeros_like(pmf)
    obs[actual] = 1.0
    return float(np.mean((np.cumsum(pmf) - np.cumsum(obs)) ** 2))


def randomized_pit(pmf: np.ndarray, actual: int, rng: np.random.Generator) -> float:
    pmf = np.asarray(pmf, dtype=float)
    if actual >= len(pmf):
        return 1.0 - rng.random() * 1e-6
    lower = float(pmf[:actual].sum())
    hit = float(pmf[actual])
    return lower + rng.random() * hit


def calibration_table(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    frame = pd.DataFrame({"y": y, "p": p}).dropna()
    if frame.empty:
        return pd.DataFrame()
    frame["bin"] = pd.qcut(frame["p"], q=min(n_bins, frame["p"].nunique()), duplicates="drop")
    out = frame.groupby("bin", as_index=False, observed=False).agg(n=("y", "size"), pred=("p", "mean"), actual=("y", "mean"))
    out["gap"] = out["actual"] - out["pred"]
    return out


def expected_calibration_error(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    tab = calibration_table(y, p, n_bins=n_bins)
    if tab.empty:
        return np.nan
    w = tab["n"] / tab["n"].sum()
    return float((w * tab["gap"].abs()).sum())


def spiegelhalter_z(y: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    numerator = float(np.sum(y - p))
    denominator = float(np.sqrt(np.sum(p * (1 - p))))
    z = numerator / denominator if denominator > 0 else np.nan
    return {"z": z}


@dataclass
class QuantDiagnostics:
    system: StrikeoutBettingSystem

    def feature_signal_report(self, pa_df: pd.DataFrame) -> pd.DataFrame:
        rows: List[Dict] = []
        y = pa_df["is_strikeout"].astype(float)
        for col in self.system.true_model.pa_feature_cols_:
            if col not in pa_df:
                continue
            x = pd.to_numeric(pa_df[col], errors="coerce")
            if x.notna().sum() < 100:
                continue
            corr = float(pd.Series(x).corr(y, method="spearman"))
            rows.append(
                {
                    "feature": col,
                    "spearman_vs_k": corr,
                    "missing_rate": float(x.isna().mean()),
                    "std": float(np.nanstd(x)),
                }
            )
        out = pd.DataFrame(rows).sort_values("spearman_vs_k", ascending=False, key=lambda s: s.abs())
        return out.reset_index(drop=True)

    def permutation_importance_report(self, pa_df: pd.DataFrame) -> pd.DataFrame:
        X = self.system.true_model._ensure_columns(pa_df, self.system.true_model.pa_feature_cols_).fillna(0.0)
        y = pa_df["is_strikeout"].astype(int)
        perm = permutation_importance(self.system.true_model.pa_model, X, y, n_repeats=5, random_state=42, scoring="neg_log_loss")
        return pd.DataFrame({"feature": X.columns, "importance_mean": perm.importances_mean, "importance_std": perm.importances_std}).sort_values("importance_mean", ascending=False)

    def pa_model_report(self, pa_df: pd.DataFrame) -> Dict[str, float]:
        X = self.system.true_model._ensure_columns(pa_df, self.system.true_model.pa_feature_cols_).fillna(0.0)
        y = pa_df["is_strikeout"].astype(int).to_numpy()
        raw = self.system.true_model._predict_binary(self.system.true_model.pa_model, X)
        p = self.system.true_model.pa_calibrator.predict(raw)
        return {
            "log_loss": float(log_loss(y, p)),
            "brier": float(brier_score_loss(y, p)),
            "roc_auc": float(roc_auc_score(y, p)),
            "pr_auc": float(average_precision_score(y, p)),
            "ece": float(expected_calibration_error(y, p)),
            "spiegelhalter_z": float(spiegelhalter_z(y, p)["z"]),
        }

    def pmf_report(self, start_states: pd.DataFrame, lineup_lookup: Dict) -> Dict[str, float]:
        rng = np.random.default_rng(42)
        scores = []
        pits = []
        for row in start_states.itertuples(index=False):
            lineup = lineup_lookup.get((int(row.game_id), int(row.pitcher_id)))
            if lineup is None or len(lineup) == 0:
                continue
            state = {c: getattr(row, c) for c in start_states.columns if c not in {"actual_total_k", "actual_total_bf"}}
            pmf = self.system.true_model.predict_pmf(state=state, future_lineup=lineup)
            actual = int(row.actual_total_k)
            prob = float(pmf[actual]) if actual < len(pmf) else 1e-12
            scores.append(
                {
                    "log_score": -np.log(max(prob, 1e-12)),
                    "rps": ranked_probability_score(pmf, actual),
                }
            )
            pits.append(randomized_pit(pmf, actual, rng))

        score_df = pd.DataFrame(scores)
        ks = kstest(pits, "uniform") if pits else None
        return {
            "pmf_log_score": float(score_df["log_score"].mean()) if not score_df.empty else np.nan,
            "pmf_rps": float(score_df["rps"].mean()) if not score_df.empty else np.nan,
            "pit_ks_stat": float(ks.statistic) if ks else np.nan,
            "pit_ks_pvalue": float(ks.pvalue) if ks else np.nan,
        }

    def market_report(self, market_training: pd.DataFrame) -> pd.DataFrame:
        if market_training.empty:
            return pd.DataFrame()

        rows = []
        for row in market_training.itertuples(index=False):
            fair_over, fair_under = remove_vig_two_way(row.over_odds, row.under_odds)
            rows.append(
                {
                    "line_value": float(row.line_value),
                    "market_over_prob": fair_over,
                    "market_under_prob": fair_under,
                    "actual_over": float(row.over_hit),
                    "actual_under": float(row.under_hit),
                }
            )
        return pd.DataFrame(rows)

    def selection_report(self, candidate_bets: pd.DataFrame) -> Dict[str, float]:
        if candidate_bets.empty:
            return {}
        return {
            "candidate_count": int(len(candidate_bets)),
            "selected_count": int(candidate_bets["selected"].sum()),
            "mean_edge_selected": float(candidate_bets.loc[candidate_bets["selected"].eq(1), "edge"].mean()) if candidate_bets["selected"].sum() else np.nan,
            "mean_ev_selected": float(candidate_bets.loc[candidate_bets["selected"].eq(1), "ev"].mean()) if candidate_bets["selected"].sum() else np.nan,
        }


def write_audit_outputs(
    *,
    out_dir: str | Path,
    feature_signal: pd.DataFrame,
    permutation_report: pd.DataFrame,
    pa_metrics: Dict[str, float],
    pmf_metrics: Dict[str, float],
    pa_calibration: pd.DataFrame,
    market_report: pd.DataFrame,
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    feature_signal.to_csv(out_dir / "feature_signal.csv", index=False)
    permutation_report.to_csv(out_dir / "permutation_importance.csv", index=False)
    pa_calibration.to_csv(out_dir / "pa_calibration.csv", index=False)
    market_report.to_csv(out_dir / "market_report.csv", index=False)

    metrics = {"pa_model": pa_metrics, "pmf": pmf_metrics}
    pd.Series(metrics["pa_model"]).to_json(out_dir / "pa_metrics.json", indent=2)
    pd.Series(metrics["pmf"]).to_json(out_dir / "pmf_metrics.json", indent=2)

    lines = ["# Quant Audit", "", "## PA model", ""]
    lines.extend([f"- {k}: {v}" for k, v in pa_metrics.items()])
    lines.extend(["", "## PMF", ""])
    lines.extend([f"- {k}: {v}" for k, v in pmf_metrics.items()])
    (out_dir / "audit_report.md").write_text("\n".join(lines))
