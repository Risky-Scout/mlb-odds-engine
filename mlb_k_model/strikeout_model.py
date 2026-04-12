
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import joblib
import numpy as np
import pandas as pd

from .calibration import BinaryIsotonicCalibrator, CountThresholdCalibrator
from .market import prob_to_american

try:
    from xgboost import XGBClassifier
    _MODEL_CLASS = XGBClassifier
    _MODEL_BACKEND = "xgboost"
except Exception:  # pragma: no cover
    from sklearn.ensemble import HistGradientBoostingClassifier
    _MODEL_CLASS = HistGradientBoostingClassifier
    _MODEL_BACKEND = "sklearn"


def poisson_binomial_pmf(probs: Sequence[float]) -> np.ndarray:
    pmf = np.array([1.0], dtype=float)
    for p in probs:
        p = float(np.clip(p, 1e-6, 1 - 1e-6))
        nxt = np.zeros(len(pmf) + 1, dtype=float)
        nxt[:-1] += pmf * (1.0 - p)
        nxt[1:] += pmf * p
        pmf = nxt
    return pmf


@dataclass
class ModelConfig:
    date_col: str = "game_date"
    valid_frac: float = 0.20
    random_state: int = 42
    max_future_bf: int = 36
    max_k: int = 20

    pa_features: List[str] = field(
        default_factory=lambda: [
            "inning",
            "outs_start",
            "bf_so_far",
            "k_so_far",
            "walk_so_far",
            "hit_so_far",
            "outs_so_far",
            "pitch_count_so_far",
            "avg_pitch_count_so_far",
            "times_through_order",
            "times_faced_batter_before",
            "runners_on",
            "is_scoring_position",
            "days_rest",
            "is_home_pitcher",
            "same_hand",
            "pitcher_k_rate_15",
            "pitcher_k_rate_30",
            "pitcher_k_rate_60",
            "pitcher_walk_rate_60",
            "pitcher_hit_rate_60",
            "pitcher_out_rate_60",
            "pitcher_pa_pitch_ct_60",
            "pitcher_velo_60",
            "pitcher_spin_60",
            "pitcher_extension_60",
            "pitcher_xba_contact_60",
            "pitcher_barrel_rate_60",
            "batter_k_rate_15",
            "batter_k_rate_30",
            "batter_k_rate_60",
            "batter_walk_rate_60",
            "batter_hit_rate_60",
            "batter_barrel_rate_60",
            "batter_xba_contact_60",
            "lineup_mean_batter_k_rate",
            "lineup_mean_batter_walk_rate",
            "lineup_mean_batter_hit_rate",
            "lineup_mean_batter_barrel_rate",
            "lineup_mean_batter_xba",
            "lineup_same_side_share",
        ]
    )
    survival_features: List[str] = field(
        default_factory=lambda: [
            "inning",
            "outs_start",
            "bf_so_far",
            "k_so_far",
            "walk_so_far",
            "hit_so_far",
            "outs_so_far",
            "pitch_count_so_far",
            "avg_pitch_count_so_far",
            "times_through_order",
            "runners_on",
            "is_scoring_position",
            "days_rest",
            "is_home_pitcher",
            "pitcher_workload_flag",
            "future_bf_step",
            "pitcher_k_rate_15",
            "pitcher_k_rate_30",
            "pitcher_k_rate_60",
            "pitcher_walk_rate_60",
            "pitcher_hit_rate_60",
            "pitcher_out_rate_60",
            "pitcher_pa_pitch_ct_60",
            "pitcher_velo_60",
            "pitcher_spin_60",
            "pitcher_extension_60",
            "pitcher_xba_contact_60",
            "pitcher_barrel_rate_60",
            "lineup_mean_batter_k_rate",
            "lineup_mean_batter_walk_rate",
            "lineup_mean_batter_hit_rate",
            "lineup_mean_batter_barrel_rate",
            "lineup_mean_batter_xba",
            "lineup_same_side_share",
        ]
    )

    pa_model_params: Dict = field(
        default_factory=lambda: dict(
            n_estimators=350,
            max_depth=4,
            learning_rate=0.04,
            min_child_weight=30,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=2.0,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=42,
        )
    )
    survival_model_params: Dict = field(
        default_factory=lambda: dict(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            min_child_weight=25,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=2.0,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=42,
        )
    )


class StrikeoutDistributionModel:
    """
    True-outcome model.

    Layer 1: PA strikeout probability.
    Layer 2: discrete survival / continuation to get remaining BF PMF.
    Layer 3: exact PMF mixture + count calibration.
    """

    def __init__(self, config: Optional[ModelConfig] = None):
        self.config = config or ModelConfig()
        self.pa_model = self._make_model(self.config.pa_model_params)
        self.survival_model = self._make_model(self.config.survival_model_params)
        self.pa_calibrator = BinaryIsotonicCalibrator()
        self.survival_calibrator = BinaryIsotonicCalibrator()
        self.count_calibrator = CountThresholdCalibrator(max_k=self.config.max_k)
        self.pmf_temperature_ = 1.0
        self._is_fit = False
        self.valid_pmfs_: Optional[np.ndarray] = None
        self.valid_actuals_: Optional[np.ndarray] = None
        self.pa_feature_cols_: List[str] = list(self.config.pa_features)
        self.survival_feature_cols_: List[str] = list(self.config.survival_features)

    @staticmethod
    def _make_model(params: Dict):
        if _MODEL_BACKEND == "xgboost":
            return _MODEL_CLASS(**params)
        clean = {
            "max_depth": params.get("max_depth", 4),
            "learning_rate": params.get("learning_rate", 0.05),
            "max_iter": params.get("n_estimators", 300),
            "l2_regularization": params.get("reg_lambda", 1.0),
            "random_state": params.get("random_state", 42),
        }
        return _MODEL_CLASS(**clean)

    def _time_split(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        ordered = pd.Series(pd.to_datetime(df[self.config.date_col]).sort_values().unique())
        cutoff_idx = max(1, int(np.floor((1.0 - self.config.valid_frac) * len(ordered))) - 1)
        cutoff = ordered.iloc[cutoff_idx]
        train = df[pd.to_datetime(df[self.config.date_col]) <= cutoff].copy()
        valid = df[pd.to_datetime(df[self.config.date_col]) > cutoff].copy()
        if valid.empty:
            valid = train.tail(max(200, len(train) // 5)).copy()
            train = train.iloc[:-len(valid)].copy()
        return train, valid


    @staticmethod
    def _resolve_feature_cols(df: pd.DataFrame, base_cols: Sequence[str], *, prefix: str = "ext_") -> List[str]:
        extras = sorted([c for c in df.columns if c.startswith(prefix)])
        cols = list(base_cols) + extras
        return [c for c in cols if c in df.columns]

    @staticmethod
    def _ensure_columns(frame: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
        out = frame.copy()
        for col in cols:
            if col not in out.columns:
                out[col] = 0.0
        return out[list(cols)]

    def fit(self, pa_df: pd.DataFrame, survival_long_df: pd.DataFrame, start_states: pd.DataFrame, lineup_lookup: Dict) -> "StrikeoutDistributionModel":
        pa_train, pa_valid = self._time_split(pa_df)
        surv_train, surv_valid = self._time_split(survival_long_df)
        start_train, start_valid = self._time_split(start_states)

        self.pa_feature_cols_ = self._resolve_feature_cols(pa_df, self.config.pa_features)
        self.survival_feature_cols_ = self._resolve_feature_cols(survival_long_df, self.config.survival_features)

        X_pa_train = self._ensure_columns(pa_train, self.pa_feature_cols_).fillna(0.0)
        y_pa_train = pa_train["is_strikeout"].astype(int)
        X_pa_valid = self._ensure_columns(pa_valid, self.pa_feature_cols_).fillna(0.0)
        y_pa_valid = pa_valid["is_strikeout"].astype(int)

        self.pa_model.fit(X_pa_train, y_pa_train)
        pa_valid_pred = self._predict_binary(self.pa_model, X_pa_valid)
        self.pa_calibrator.fit(y_pa_valid.to_numpy(), pa_valid_pred)

        X_surv_train = self._ensure_columns(surv_train, self.survival_feature_cols_).fillna(0.0)
        y_surv_train = surv_train["stop_here"].astype(int)
        X_surv_valid = self._ensure_columns(surv_valid, self.survival_feature_cols_).fillna(0.0)
        y_surv_valid = surv_valid["stop_here"].astype(int)

        self.survival_model.fit(X_surv_train, y_surv_train)
        surv_valid_pred = self._predict_binary(self.survival_model, X_surv_valid)
        self.survival_calibrator.fit(y_surv_valid.to_numpy(), surv_valid_pred)

        pmfs: List[np.ndarray] = []
        actuals: List[int] = []
        for row in start_valid.itertuples(index=False):
            key = (int(row.game_id), int(row.pitcher_id))
            lineup_df = lineup_lookup.get(key)
            if lineup_df is None or len(lineup_df) == 0:
                continue
            state = {col: getattr(row, col) for col in start_valid.columns if col not in {"actual_total_k", "actual_total_bf"}}
            for _attr in ("is_fit_", "is_fitted_", "_is_fit", "_is_fitted", "fitted_", "_fitted"):
                setattr(self, _attr, True)
            pmf = self.predict_pmf(state=state, future_lineup=lineup_df, apply_count_calibration=False)
            pmfs.append(self._pad_pmf(pmf, self.config.max_k + 1))
            actuals.append(int(row.actual_total_k))

        if pmfs:
            self.valid_pmfs_ = np.vstack(pmfs)
            self.valid_actuals_ = np.asarray(actuals, dtype=int)
            self.count_calibrator.fit(self.valid_pmfs_, self.valid_actuals_)

            def _temp_apply(pmf: np.ndarray, temp: float) -> np.ndarray:
                z = np.asarray(pmf, dtype=float).copy()
                z = np.clip(z, 1e-12, None)
                z /= z.sum()
                if abs(temp - 1.0) < 1e-9:
                    return z
                z = np.exp(np.log(z) / temp)
                z = np.clip(z, 1e-12, None)
                return z / z.sum()

            best_temp = 1.0
            best_score = np.inf
            base_pmfs = np.vstack([self.count_calibrator.calibrate_pmf(p) for p in self.valid_pmfs_])

            for temp in np.linspace(0.70, 1.50, 17):
                score = 0.0
                for pmf, actual in zip(base_pmfs, self.valid_actuals_):
                    adj = _temp_apply(pmf, float(temp))
                    idx = min(int(actual), len(adj) - 1)
                    score += -np.log(max(float(adj[idx]), 1e-12))
                score /= max(len(base_pmfs), 1)
                if score < best_score:
                    best_score = score
                    best_temp = float(temp)

            self.pmf_temperature_ = best_temp

        self._is_fit = True
        return self


    def _predict_binary(self, model, X) -> np.ndarray:
        if hasattr(model, "predict_proba"):
            return model.predict_proba(X)[:, 1]
        return model.predict(X)

    def _row_to_array(self, row: Dict, cols: List[str]) -> np.ndarray:
        arr = np.zeros((1, len(cols)), dtype=float)
        for i, col in enumerate(cols):
            val = row.get(col, 0.0)
            if pd.isna(val):
                val = 0.0
            arr[0, i] = float(val)
        return arr

    def _clip_prob(self, x: float, low: float = 1e-6, high: float = 1 - 1e-6) -> float:
        return float(np.clip(float(x), low, high))

    def _apply_pmf_temperature(self, pmf: np.ndarray) -> np.ndarray:
        temp = float(getattr(self, "pmf_temperature_", 1.0) or 1.0)
        out = np.asarray(pmf, dtype=float).copy()
        out = np.clip(out, 1e-12, None)
        out /= out.sum()
        if not np.isfinite(temp) or abs(temp - 1.0) < 1e-9:
            return out
        out = np.exp(np.log(out) / temp)
        out = np.clip(out, 1e-12, None)
        return out / out.sum()

    def _event_mix(self, state: Dict, batter: Dict | pd.Series, p_k: float) -> Dict[str, float]:
        p_k = self._clip_prob(p_k, 1e-4, 0.95)

        p_bb = float(np.nanmean([
            state.get("pitcher_walk_rate_60", np.nan),
            batter.get("batter_walk_rate_60", np.nan),
        ]))
        if not np.isfinite(p_bb):
            p_bb = 0.08
        p_bb = self._clip_prob(p_bb, 0.01, 0.20)

        p_hit = float(np.nanmean([
            state.get("pitcher_hit_rate_60", np.nan),
            batter.get("batter_hit_rate_60", np.nan),
        ]))
        if not np.isfinite(p_hit):
            p_hit = 0.22
        p_hit = self._clip_prob(p_hit, 0.05, 0.40)

        p_out_base = float(state.get("pitcher_out_rate_60", 0.70))
        if not np.isfinite(p_out_base):
            p_out_base = 0.70
        p_out_base = self._clip_prob(p_out_base, 0.45, 0.90)

        p_inplay_out = max(1e-5, p_out_base - p_k)

        total = p_k + p_bb + p_hit + p_inplay_out
        cap = 0.985
        if total > cap:
            scale = cap / total
            p_k *= scale
            p_bb *= scale
            p_hit *= scale
            p_inplay_out *= scale

        return {
            "k": float(p_k),
            "bb": float(p_bb),
            "hit": float(p_hit),
            "out": float(p_k + p_inplay_out),
        }

    def _advance_expected_state(self, state: Dict, event_mix: Dict[str, float], lineup_len: int) -> Dict:
        nxt = dict(state)

        bf = float(state.get("bf_so_far", 0.0)) + 1.0
        k = float(state.get("k_so_far", 0.0)) + float(event_mix["k"])
        bb = float(state.get("walk_so_far", 0.0)) + float(event_mix["bb"])
        hit = float(state.get("hit_so_far", 0.0)) + float(event_mix["hit"])
        outs = float(state.get("outs_so_far", 0.0)) + float(event_mix["out"])

        avg_ct = float(state.get("avg_pitch_count_so_far", state.get("pitcher_pa_pitch_ct_60", 4.0) or 4.0))
        pitch_ct = float(state.get("pitch_count_so_far", 0.0)) + avg_ct

        runners = float(state.get("runners_on", 0.0))
        runners = runners + 0.85 * (float(event_mix["bb"]) + float(event_mix["hit"])) - 0.60 * float(event_mix["out"])
        runners = float(np.clip(runners, 0.0, 3.0))

        nxt["bf_so_far"] = bf
        nxt["k_so_far"] = k
        nxt["walk_so_far"] = bb
        nxt["hit_so_far"] = hit
        nxt["outs_so_far"] = outs
        nxt["pitch_count_so_far"] = pitch_ct
        nxt["avg_pitch_count_so_far"] = avg_ct
        nxt["runners_on"] = runners
        nxt["is_scoring_position"] = float(runners >= 1.5)
        nxt["outs_start"] = int(np.floor(outs)) % 3
        nxt["inning"] = max(int(state.get("inning", 1)), 1 + int(np.floor(outs)) // 3)
        nxt["times_through_order"] = int((bf - 1) // max(lineup_len, 1))
        nxt["times_faced_batter_before"] = int((bf - 1) // max(lineup_len, 1))
        nxt["pitcher_workload_flag"] = int((pitch_ct >= 75.0) or (bf >= 18.0))
        return nxt

    def _predict_pa_probs(self, state: Dict, future_lineup: pd.DataFrame, max_future_bf: Optional[int] = None) -> np.ndarray:
        max_future_bf = max_future_bf or self.config.max_future_bf
        lineup = future_lineup.sort_values("batting_order").reset_index(drop=True).copy()
        if lineup.empty:
            return np.zeros(max_future_bf, dtype=float)

        lineup_len = len(lineup)
        start_idx = int(state.get("bf_so_far", 0)) % lineup_len
        projected = dict(state)

        lineup_summary = {
            "lineup_mean_batter_k_rate": float(lineup.get("batter_k_rate_60", pd.Series(dtype=float)).mean()),
            "lineup_mean_batter_walk_rate": float(lineup.get("batter_walk_rate_60", pd.Series(dtype=float)).mean()),
            "lineup_mean_batter_hit_rate": float(lineup.get("batter_hit_rate_60", pd.Series(dtype=float)).mean()),
            "lineup_mean_batter_barrel_rate": float(lineup.get("batter_barrel_rate_60", pd.Series(dtype=float)).mean()),
            "lineup_mean_batter_xba": float(lineup.get("batter_xba_contact_60", pd.Series(dtype=float)).mean()),
        }

        same_side_vals = []
        pitcher_hand = str(state.get("pitcher_hand", ""))
        for _, r in lineup.iterrows():
            same_side_vals.append(int(str(r.get("batter_side", "")) == pitcher_hand))
        lineup_summary["lineup_same_side_share"] = float(np.mean(same_side_vals)) if same_side_vals else 0.5

        pa_probs = []

        for step in range(max_future_bf):
            batter = lineup.iloc[(start_idx + step) % lineup_len]

            row = {
                "inning": projected.get("inning", 1),
                "outs_start": projected.get("outs_start", 0),
                "bf_so_far": projected.get("bf_so_far", 0),
                "k_so_far": projected.get("k_so_far", 0),
                "walk_so_far": projected.get("walk_so_far", 0),
                "hit_so_far": projected.get("hit_so_far", 0),
                "outs_so_far": projected.get("outs_so_far", 0),
                "pitch_count_so_far": projected.get("pitch_count_so_far", 0.0),
                "avg_pitch_count_so_far": projected.get("avg_pitch_count_so_far", projected.get("pitcher_pa_pitch_ct_60", 4.0)),
                "times_through_order": projected.get("times_through_order", 0),
                "times_faced_batter_before": projected.get("times_faced_batter_before", 0),
                "runners_on": projected.get("runners_on", 0),
                "is_scoring_position": projected.get("is_scoring_position", 0),
                "days_rest": projected.get("days_rest", 5),
                "is_home_pitcher": projected.get("is_home_pitcher", 0),
                "same_hand": int(str(batter.get("batter_side", "")) == pitcher_hand),
                "pitcher_k_rate_15": projected.get("pitcher_k_rate_15"),
                "pitcher_k_rate_30": projected.get("pitcher_k_rate_30"),
                "pitcher_k_rate_60": projected.get("pitcher_k_rate_60"),
                "pitcher_walk_rate_60": projected.get("pitcher_walk_rate_60"),
                "pitcher_hit_rate_60": projected.get("pitcher_hit_rate_60"),
                "pitcher_out_rate_60": projected.get("pitcher_out_rate_60"),
                "pitcher_pa_pitch_ct_60": projected.get("pitcher_pa_pitch_ct_60"),
                "pitcher_velo_60": projected.get("pitcher_velo_60"),
                "pitcher_spin_60": projected.get("pitcher_spin_60"),
                "pitcher_extension_60": projected.get("pitcher_extension_60"),
                "pitcher_xba_contact_60": projected.get("pitcher_xba_contact_60"),
                "pitcher_barrel_rate_60": projected.get("pitcher_barrel_rate_60"),
                "batter_k_rate_15": batter.get("batter_k_rate_15"),
                "batter_k_rate_30": batter.get("batter_k_rate_30"),
                "batter_k_rate_60": batter.get("batter_k_rate_60"),
                "batter_walk_rate_60": batter.get("batter_walk_rate_60"),
                "batter_hit_rate_60": batter.get("batter_hit_rate_60"),
                "batter_barrel_rate_60": batter.get("batter_barrel_rate_60"),
                "batter_xba_contact_60": batter.get("batter_xba_contact_60"),
                **lineup_summary,
                **{c: projected.get(c, 0.0) for c in self.pa_feature_cols_ if c.startswith("ext_pitcher_")},
                **{c: batter.get(c, 0.0) for c in self.pa_feature_cols_ if c.startswith("ext_batter_")},
            }

            X = self._row_to_array(row, self.pa_feature_cols_)
            raw = float(self._predict_binary(self.pa_model, X)[0])
            p_k = float(self.pa_calibrator.predict(np.asarray([raw]))[0])
            pa_probs.append(p_k)

            mix = self._event_mix(projected, batter, p_k)
            projected = self._advance_expected_state(projected, mix, lineup_len)

        return np.asarray(pa_probs, dtype=float)

    def predict_remaining_bf_pmf(self, state: Dict, max_future_bf: Optional[int] = None) -> np.ndarray:
        max_future_bf = max_future_bf or self.config.max_future_bf
        pmf = np.zeros(max_future_bf + 1, dtype=float)
        survive = 1.0
        projected = dict(state)

        proxy_batter = {
            "batter_walk_rate_60": projected.get("lineup_mean_batter_walk_rate", 0.08),
            "batter_hit_rate_60": projected.get("lineup_mean_batter_hit_rate", 0.22),
            "batter_k_rate_60": projected.get("lineup_mean_batter_k_rate", 0.22),
        }

        for step in range(1, max_future_bf + 1):
            row = dict(projected)
            row["future_bf_step"] = step
            X = self._row_to_array(row, self.survival_feature_cols_)
            raw_hazard = float(self._predict_binary(self.survival_model, X)[0])
            h = float(self.survival_calibrator.predict(np.asarray([raw_hazard]))[0])
            h = self._clip_prob(h, 1e-6, 1 - 1e-6)

            stop_prob = survive * h
            pmf[step] = stop_prob
            survive *= 1.0 - h

            p_k_proxy = float(np.nanmean([
                projected.get("pitcher_k_rate_60", np.nan),
                projected.get("lineup_mean_batter_k_rate", np.nan),
            ]))
            if not np.isfinite(p_k_proxy):
                p_k_proxy = 0.22
            p_k_proxy = self._clip_prob(p_k_proxy, 0.05, 0.50)

            mix = self._event_mix(projected, proxy_batter, p_k_proxy)
            projected = self._advance_expected_state(projected, mix, 9)

        pmf[max_future_bf] += survive
        pmf = np.clip(pmf, 0.0, None)
        return pmf / pmf.sum()

    @staticmethod
    def _pad_pmf(pmf: np.ndarray, n: int) -> np.ndarray:
        if len(pmf) >= n:
            out = pmf[:n].copy()
            out[-1] += pmf[n:].sum()
            return out / out.sum()
        return np.pad(pmf, (0, n - len(pmf)), constant_values=0.0)


    def _apply_mainline_pmf_postcal(self, pmf: np.ndarray) -> np.ndarray:
        tau = getattr(self, "mainline_pmf_postcal_tau_", None)
        lam = getattr(self, "mainline_pmf_postcal_lambda_", None)

        z = np.asarray(pmf, dtype=float).copy()
        z = np.clip(z, 1e-12, None)
        if z.sum() <= 0:
            return z
        z /= z.sum()

        if tau is None or lam is None:
            return z

        ks = np.arange(len(z), dtype=float)
        logits = (np.log(z) / float(tau)) + float(lam) * ks
        logits -= logits.max()
        out = np.exp(logits)
        out = np.clip(out, 1e-12, None)
        return out / out.sum()

    def predict_pmf(
        self,
        *,
        state: Dict,
        future_lineup: pd.DataFrame,
        apply_count_calibration: bool = True,
        max_future_bf: Optional[int] = None,
    ) -> np.ndarray:
        if not self._is_fit and self.valid_pmfs_ is None:
            raise RuntimeError("Model must be fit before predict_pmf is called.")

        max_future_bf = max_future_bf or self.config.max_future_bf
        bf_pmf = self.predict_remaining_bf_pmf(state, max_future_bf=max_future_bf)
        pa_probs = self._predict_pa_probs(state, future_lineup, max_future_bf=max_future_bf)

        max_k = min(self.config.max_k, max_future_bf + int(state.get("k_so_far", 0)))
        final = np.zeros(max_k + 1, dtype=float)

        for n in range(0, len(bf_pmf)):
            weight = bf_pmf[n]
            if weight <= 0:
                continue
            k_pmf = poisson_binomial_pmf(pa_probs[:n])
            k_pmf = self._pad_pmf(k_pmf, max_k + 1)
            final += weight * k_pmf

        final = np.clip(final, 0.0, None)
        final /= final.sum()

        if apply_count_calibration:
            final = self.count_calibrator.calibrate_pmf(final)

        final = self._apply_pmf_temperature(final)

        total_so_far = int(state.get("k_so_far", 0))
        if total_so_far > 0:
            shifted = np.zeros(total_so_far + len(final), dtype=float)
            shifted[total_so_far:] = final
            final = shifted

        return final / final.sum()

    def line_grid(self, pmf: np.ndarray, line_values: Iterable[float]) -> pd.DataFrame:
        rows = []
        ks = np.arange(len(pmf))
        for line_value in sorted(set(float(x) for x in line_values)):
            p_over = float(pmf[ks > line_value].sum())
            p_under = float(pmf[ks < line_value].sum())
            p_push = float(pmf[ks == line_value].sum()) if abs(line_value - round(line_value)) < 1e-9 else 0.0
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

    def exact_count_odds(self, pmf: np.ndarray) -> pd.DataFrame:
        rows = []
        for k, p in enumerate(pmf):
            rows.append({"strikeouts": k, "probability": float(p), "fair_american": prob_to_american(max(float(p), 1e-9))})
        return pd.DataFrame(rows)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "StrikeoutDistributionModel":
        return joblib.load(path)
