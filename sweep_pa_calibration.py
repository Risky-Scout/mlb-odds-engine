from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

from mlb_k_model.calibration_models import (
    BetaCalibrator,
    IdentityCalibrator,
    IsotonicProbCalibrator,
    PlattCalibrator,
)
from mlb_k_model.data_pipeline import FeatureBuilder
from mlb_k_model.external_data import merge_external_features
from mlb_k_model.system import StrikeoutBettingSystem


def clip_prob(p):
    return np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)


def expected_calibration_error(y, p, n_bins=15):
    y = np.asarray(y, dtype=int)
    p = clip_prob(p)
    qs = np.linspace(0, 1, n_bins + 1)
    bins = np.unique(np.quantile(p, qs))
    if len(bins) < 4:
        bins = np.linspace(0, 1, n_bins + 1)
    idx = np.digitize(p, bins[1:-1], right=True)

    ece = 0.0
    for b in range(idx.min(), idx.max() + 1):
        mask = idx == b
        if mask.sum() == 0:
            continue
        ece += (mask.mean()) * abs(y[mask].mean() - p[mask].mean())
    return float(ece)


def spiegelhalter_z(y, p):
    y = np.asarray(y, dtype=int)
    p = clip_prob(p)
    num = np.sum((y - p) * (1.0 - 2.0 * p))
    den = np.sqrt(np.sum(((1.0 - 2.0 * p) ** 2) * p * (1.0 - p)))
    if den <= 0:
        return np.nan
    return float(num / den)


def detect_target_col(df: pd.DataFrame) -> str:
    candidates = [
        "is_strikeout",
        "strikeout",
        "pa_is_strikeout",
        "is_k",
        "k_outcome",
        "target",
        "y",
    ]
    for c in candidates:
        if c in df.columns:
            return c

    for c in df.columns:
        lc = c.lower()
        s = pd.to_numeric(df[c], errors="coerce").dropna()
        if not len(s):
            continue
        vals = set(np.unique(s.astype(int)))
        if vals <= {0, 1} and ("strike" in lc or lc in {"y", "target"} or lc.endswith("_hit")):
            return c

    raise SystemExit("Could not detect PA target column.")


def detect_date_col(df: pd.DataFrame) -> str:
    for c in ["game_date", "date"]:
        if c in df.columns:
            return c
    raise SystemExit("Could not detect game date column.")


def build_merged_frame(data_dir: Path) -> pd.DataFrame:
    games = pd.read_parquet(data_dir / "games.parquet")
    pa = pd.read_parquet(data_dir / "plate_appearances.parquet")
    train_df = FeatureBuilder.build_training_frame(pa, games)

    player_map = pd.read_parquet(data_dir / "external" / "player_id_map.parquet")
    pitcher_ext = pd.read_parquet(data_dir / "external" / "savant_pitcher_features.parquet").copy()
    batter_ext = pd.read_parquet(data_dir / "external" / "savant_batter_features.parquet").copy()

    pitcher_ext["season"] = pd.to_numeric(pitcher_ext["season"], errors="coerce") + 1
    batter_ext["season"] = pd.to_numeric(batter_ext["season"], errors="coerce") + 1

    return merge_external_features(train_df, player_map, pitcher_ext, batter_ext)


def split_out_of_time(df: pd.DataFrame, date_col: str):
    d = pd.to_datetime(df[date_col], utc=True, errors="coerce").dt.normalize()
    df = df.loc[d.notna()].copy()
    df["_date_norm"] = d.loc[d.notna()].values

    dates = np.array(sorted(df["_date_norm"].drop_duplicates().tolist()))
    n_dates = len(dates)
    if n_dates < 30:
        raise SystemExit(f"Need more unique dates for strict sweep, found only {n_dates}.")

    n_eval = max(7, int(round(n_dates * 0.10)))
    n_cal = max(7, int(round(n_dates * 0.10)))

    eval_dates = set(dates[-n_eval:])
    cal_dates = set(dates[-(n_eval + n_cal):-n_eval])

    cal_df = df.loc[df["_date_norm"].isin(cal_dates)].copy()
    eval_df = df.loc[df["_date_norm"].isin(eval_dates)].copy()

    if cal_df.empty or eval_df.empty:
        raise SystemExit("Calibration or evaluation split is empty.")

    return cal_df, eval_df


def metric_pack(y, p):
    y = np.asarray(y, dtype=int)
    p = clip_prob(p)
    return {
        "log_loss": float(log_loss(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "ece": float(expected_calibration_error(y, p)),
        "spiegelhalter_z": float(spiegelhalter_z(y, p)),
    }


def better(a: dict, b: dict) -> bool:
    ll_tol = 1e-4
    ece_tol = 5e-4
    z_tol = 0.05
    br_tol = 1e-4

    if a["log_loss"] < b["log_loss"] - ll_tol:
        return True
    if abs(a["log_loss"] - b["log_loss"]) <= ll_tol:
        if a["ece"] < b["ece"] - ece_tol:
            return True
        if abs(a["ece"] - b["ece"]) <= ece_tol:
            if abs(a["spiegelhalter_z"]) < abs(b["spiegelhalter_z"]) - z_tol:
                return True
            if abs(abs(a["spiegelhalter_z"]) - abs(b["spiegelhalter_z"])) <= z_tol:
                if a["brier"] < b["brier"] - br_tol:
                    return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--system-path", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    system_path = Path(args.system_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    system = StrikeoutBettingSystem.load(system_path)
    model = system.true_model

    df = build_merged_frame(data_dir)
    target_col = detect_target_col(df)
    date_col = detect_date_col(df)

    cal_df, eval_df = split_out_of_time(df, date_col)

    X_cal = model._ensure_columns(cal_df, model.pa_feature_cols_).fillna(0.0)
    X_eval = model._ensure_columns(eval_df, model.pa_feature_cols_).fillna(0.0)

    y_cal = pd.to_numeric(cal_df[target_col], errors="coerce").astype(int).to_numpy()
    y_eval = pd.to_numeric(eval_df[target_col], errors="coerce").astype(int).to_numpy()

    raw_cal = clip_prob(model._predict_binary(model.pa_model, X_cal))
    raw_eval = clip_prob(model._predict_binary(model.pa_model, X_eval))

    candidates = []
    candidates.append(("current", None, clip_prob(model.pa_calibrator.predict(raw_eval))))
    candidates.append(("identity", IdentityCalibrator().fit(raw_cal, y_cal), None))
    candidates.append(("isotonic", IsotonicProbCalibrator().fit(raw_cal, y_cal), None))

    for C in [0.1, 1.0, 10.0]:
        candidates.append((f"platt_C{C:g}", PlattCalibrator(C=C).fit(raw_cal, y_cal), None))
        candidates.append((f"beta_C{C:g}", BetaCalibrator(C=C).fit(raw_cal, y_cal), None))

    rows = []
    best_name = None
    best_cal = None
    best_metrics = None

    for name, cal, precomputed in candidates:
        if name == "current":
            p_eval = precomputed
        else:
            p_eval = clip_prob(cal.predict(raw_eval))

        m = metric_pack(y_eval, p_eval)
        m["name"] = name
        rows.append(m)

        if best_metrics is None or better(m, best_metrics):
            best_name = name
            best_cal = cal
            best_metrics = m

    result = pd.DataFrame(rows).sort_values(["log_loss", "ece", "brier"]).reset_index(drop=True)
    result.to_csv(out_dir / "pa_calibrator_sweep.csv", index=False)

    current_metrics = result.loc[result["name"].eq("current")].iloc[0].to_dict()
    chosen_metrics = result.loc[result["name"].eq(best_name)].iloc[0].to_dict()

    accepted = best_name != "current" and better(chosen_metrics, current_metrics)

    decision = {
        "accepted": bool(accepted),
        "current_name": "current",
        "chosen_name": best_name,
        "current_metrics": {k: current_metrics[k] for k in ["log_loss", "brier", "roc_auc", "pr_auc", "ece", "spiegelhalter_z"]},
        "chosen_metrics": {k: chosen_metrics[k] for k in ["log_loss", "brier", "roc_auc", "pr_auc", "ece", "spiegelhalter_z"]},
    }

    if accepted:
        backup_path = system_path.with_suffix(system_path.suffix + ".bak_before_pa_sweep")
        if not backup_path.exists():
            backup_path.write_bytes(system_path.read_bytes())

        system.true_model.pa_calibrator = best_cal
        system.save(system_path)
        decision["saved_model_path"] = str(system_path)
        decision["backup_path"] = str(backup_path)

    with open(out_dir / "pa_calibrator_decision.json", "w") as f:
        json.dump(decision, f, indent=2)

    print("\nSWEEP RESULTS")
    print(result.to_string(index=False))
    print("\nDECISION")
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
