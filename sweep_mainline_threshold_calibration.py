from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

from mlb_k_model.calibration_models import (
    BetaCalibrator,
    IdentityCalibrator,
    IsotonicProbCalibrator,
    PlattCalibrator,
)
from mlb_k_model.system import StrikeoutBettingSystem


def clip_prob(p):
    return np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)


def expected_calibration_error(y, p, n_bins=12):
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
        ece += mask.mean() * abs(y[mask].mean() - p[mask].mean())
    return float(ece)


def spiegelhalter_z(y, p):
    y = np.asarray(y, dtype=int)
    p = clip_prob(p)
    num = np.sum((y - p) * (1.0 - 2.0 * p))
    den = np.sqrt(np.sum(((1.0 - 2.0 * p) ** 2) * p * (1.0 - p)))
    if den <= 0:
        return np.nan
    return float(num / den)


def split_out_of_time(df: pd.DataFrame, date_col: str):
    d = pd.to_datetime(df[date_col], utc=True, errors="coerce").dt.normalize()
    df = df.loc[d.notna()].copy()
    df["_date_norm"] = d.loc[d.notna()].values

    dates = np.array(sorted(df["_date_norm"].drop_duplicates().tolist()))
    n_dates = len(dates)
    if n_dates < 20:
        raise SystemExit(f"Need more unique dates for strict sweep, found only {n_dates}.")

    n_eval = max(5, int(round(n_dates * 0.20)))
    n_cal = max(5, int(round(n_dates * 0.20)))

    eval_dates = set(dates[-n_eval:])
    cal_dates = set(dates[-(n_eval + n_cal):-n_eval])

    cal_df = df.loc[df["_date_norm"].isin(cal_dates)].copy()
    eval_df = df.loc[df["_date_norm"].isin(eval_dates)].copy()

    if cal_df.empty or eval_df.empty:
        raise SystemExit("Calibration or evaluation split is empty.")

    return cal_df, eval_df


def metric_pack(y, p, market_p):
    y = np.asarray(y, dtype=int)
    p = clip_prob(p)
    market_p = clip_prob(market_p)

    model_ll = float(log_loss(y, p))
    market_ll = float(log_loss(y, market_p))
    model_br = float(brier_score_loss(y, p))
    market_br = float(brier_score_loss(y, market_p))

    return {
        "model_log_loss": model_ll,
        "market_log_loss": market_ll,
        "delta_log_loss_market_minus_model": float(market_ll - model_ll),
        "model_brier": model_br,
        "market_brier": market_br,
        "delta_brier_market_minus_model": float(market_br - model_br),
        "ece": float(expected_calibration_error(y, p)),
        "spiegelhalter_z": float(spiegelhalter_z(y, p)),
        "mean_model_over_prob": float(np.mean(p)),
        "mean_market_over_prob": float(np.mean(market_p)),
    }


def better(a: dict, b: dict) -> bool:
    # Keep only genuine improvements:
    # 1) lower model log loss
    # 2) lower model brier
    # 3) gap vs market moves upward (closer to zero / positive)
    # 4) lower ECE
    # 5) lower |Spiegelhalter z|
    ll_tol = 1e-4
    br_tol = 1e-4
    gap_tol = 1e-4
    ece_tol = 5e-4
    z_tol = 0.05

    if a["model_log_loss"] < b["model_log_loss"] - ll_tol:
        return True
    if abs(a["model_log_loss"] - b["model_log_loss"]) <= ll_tol:
        if a["model_brier"] < b["model_brier"] - br_tol:
            return True
        if abs(a["model_brier"] - b["model_brier"]) <= br_tol:
            if a["delta_log_loss_market_minus_model"] > b["delta_log_loss_market_minus_model"] + gap_tol:
                return True
            if abs(a["delta_log_loss_market_minus_model"] - b["delta_log_loss_market_minus_model"]) <= gap_tol:
                if a["ece"] < b["ece"] - ece_tol:
                    return True
                if abs(a["ece"] - b["ece"]) <= ece_tol:
                    if abs(a["spiegelhalter_z"]) < abs(b["spiegelhalter_z"]) - z_tol:
                        return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows-path", required=True)
    ap.add_argument("--system-path", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    rows_path = Path(args.rows_path)
    system_path = Path(args.system_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(rows_path)
    if df.empty:
        raise SystemExit("Main-line evaluator rows are empty.")

    # use strict main-line comparison rows only
    needed = ["snapshot_ts", "actual_over", "model_over_prob", "market_over_prob"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing required columns: {missing}")

    for c in ["actual_over"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ["model_over_prob", "market_over_prob"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["snapshot_ts", "actual_over", "model_over_prob", "market_over_prob"]).copy()

    cal_df, eval_df = split_out_of_time(df, "snapshot_ts")

    y_cal = cal_df["actual_over"].astype(int).to_numpy()
    y_eval = eval_df["actual_over"].astype(int).to_numpy()

    raw_cal = clip_prob(cal_df["model_over_prob"].to_numpy())
    raw_eval = clip_prob(eval_df["model_over_prob"].to_numpy())
    market_eval = clip_prob(eval_df["market_over_prob"].to_numpy())

    candidates = []
    candidates.append(("current_identity_threshold", None, raw_eval))
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
        if precomputed is not None:
            p_eval = clip_prob(precomputed)
        else:
            p_eval = clip_prob(cal.predict(raw_eval))

        m = metric_pack(y_eval, p_eval, market_eval)
        m["name"] = name
        rows.append(m)

        if best_metrics is None or better(m, best_metrics):
            best_name = name
            best_cal = cal
            best_metrics = m

    result = pd.DataFrame(rows).sort_values(
        ["model_log_loss", "model_brier", "ece"]
    ).reset_index(drop=True)
    result.to_csv(out_dir / "mainline_threshold_calibrator_sweep.csv", index=False)

    current_metrics = result.loc[result["name"].eq("current_identity_threshold")].iloc[0].to_dict()
    chosen_metrics = result.loc[result["name"].eq(best_name)].iloc[0].to_dict()

    accepted = (
        best_name != "current_identity_threshold"
        and better(chosen_metrics, current_metrics)
    )

    decision = {
        "accepted": bool(accepted),
        "current_name": "current_identity_threshold",
        "chosen_name": best_name,
        "current_metrics": {
            k: current_metrics[k]
            for k in [
                "model_log_loss",
                "market_log_loss",
                "delta_log_loss_market_minus_model",
                "model_brier",
                "market_brier",
                "delta_brier_market_minus_model",
                "ece",
                "spiegelhalter_z",
                "mean_model_over_prob",
                "mean_market_over_prob",
            ]
        },
        "chosen_metrics": {
            k: chosen_metrics[k]
            for k in [
                "model_log_loss",
                "market_log_loss",
                "delta_log_loss_market_minus_model",
                "model_brier",
                "market_brier",
                "delta_brier_market_minus_model",
                "ece",
                "spiegelhalter_z",
                "mean_model_over_prob",
                "mean_market_over_prob",
            ]
        },
    }

    if accepted:
        system = StrikeoutBettingSystem.load(system_path)
        backup_path = system_path.with_suffix(system_path.suffix + ".bak_before_mainline_threshold_sweep")
        if not backup_path.exists():
            backup_path.write_bytes(system_path.read_bytes())

        system.mainline_threshold_calibrator = best_cal
        system.mainline_threshold_calibrator_meta = {
            "source_rows_path": str(rows_path),
            "chosen_name": best_name,
            "decision": decision,
        }
        system.save(system_path)

        decision["saved_model_path"] = str(system_path)
        decision["backup_path"] = str(backup_path)

    with open(out_dir / "mainline_threshold_calibrator_decision.json", "w") as f:
        json.dump(decision, f, indent=2)

    print("\nSWEEP RESULTS")
    print(result.to_string(index=False))
    print("\nDECISION")
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
