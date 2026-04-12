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
        "mean_model_over_prob": float(np.mean(p)),
        "mean_market_over_prob": float(np.mean(market_p)),
    }


def better(a: dict, b: dict) -> bool:
    ll_tol = 1e-4
    br_tol = 1e-4
    gap_tol = 1e-4

    if a["model_log_loss"] < b["model_log_loss"] - ll_tol:
        return True
    if abs(a["model_log_loss"] - b["model_log_loss"]) <= ll_tol:
        if a["model_brier"] < b["model_brier"] - br_tol:
            return True
        if abs(a["model_brier"] - b["model_brier"]) <= br_tol:
            if a["delta_log_loss_market_minus_model"] > b["delta_log_loss_market_minus_model"] + gap_tol:
                return True
    return False


def split_group_holdout(rows: pd.DataFrame):
    groups = (
        rows.groupby(["game_id", "player_id"], as_index=False)
        .agg(snapshot_ts=("snapshot_ts", "max"))
        .sort_values(["snapshot_ts", "game_id", "player_id"])
        .reset_index(drop=True)
    )

    n_groups = len(groups)
    if n_groups < 20:
        raise SystemExit(f"Need at least 20 unique pitcher-game groups, found {n_groups}.")

    n_eval = max(10, int(round(n_groups * 0.30)))
    if n_eval >= n_groups:
        n_eval = max(1, n_groups - 1)

    cal_groups = groups.iloc[:-n_eval].copy()
    eval_groups = groups.iloc[-n_eval:].copy()

    cal_keys = set(zip(cal_groups["game_id"], cal_groups["player_id"]))
    eval_keys = set(zip(eval_groups["game_id"], eval_groups["player_id"]))

    overlap = cal_keys & eval_keys
    if overlap:
        raise SystemExit(f"Split leakage detected: {len(overlap)} overlapping groups")

    key_series = list(zip(rows["game_id"], rows["player_id"]))
    cal_mask = pd.Series([k in cal_keys for k in key_series], index=rows.index)
    eval_mask = pd.Series([k in eval_keys for k in key_series], index=rows.index)

    cal_df = rows.loc[cal_mask].copy()
    eval_df = rows.loc[eval_mask].copy()

    if cal_df.empty or eval_df.empty:
        raise SystemExit("Calibration/evaluation split is empty.")

    return cal_df, eval_df


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

    rows = pd.read_parquet(rows_path).copy()
    needed = ["game_id", "player_id", "snapshot_ts", "actual_over", "model_over_prob", "market_over_prob"]
    missing = [c for c in needed if c not in rows.columns]
    if missing:
        raise SystemExit(f"Missing required columns: {missing}")

    rows["snapshot_ts"] = pd.to_datetime(rows["snapshot_ts"], utc=True, errors="coerce")
    for c in ["actual_over", "model_over_prob", "market_over_prob"]:
        rows[c] = pd.to_numeric(rows[c], errors="coerce")
    rows = rows.replace([np.inf, -np.inf], np.nan)
    rows = rows.dropna(subset=needed).copy()

    cal_df, eval_df = split_group_holdout(rows)

    y_cal = cal_df["actual_over"].astype(int).to_numpy()
    y_eval = eval_df["actual_over"].astype(int).to_numpy()

    raw_cal = clip_prob(cal_df["model_over_prob"].to_numpy())
    raw_eval = clip_prob(eval_df["model_over_prob"].to_numpy())
    market_eval = clip_prob(eval_df["market_over_prob"].to_numpy())

    candidates = []
    candidates.append(("current_identity_residual", None, raw_eval))
    candidates.append(("identity", IdentityCalibrator().fit(raw_cal, y_cal), None))
    candidates.append(("isotonic", IsotonicProbCalibrator().fit(raw_cal, y_cal), None))
    for C in [0.1, 1.0, 10.0]:
        candidates.append((f"platt_C{C:g}", PlattCalibrator(C=C).fit(raw_cal, y_cal), None))
        candidates.append((f"beta_C{C:g}", BetaCalibrator(C=C).fit(raw_cal, y_cal), None))

    rows_out = []
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
        rows_out.append(m)

        if best_metrics is None or better(m, best_metrics):
            best_name = name
            best_cal = cal
            best_metrics = m

    sweep = pd.DataFrame(rows_out).sort_values(
        ["model_log_loss", "model_brier", "delta_log_loss_market_minus_model"],
        ascending=[True, True, False]
    ).reset_index(drop=True)
    sweep.to_csv(out_dir / "mainline_residual_calibrator_sweep.csv", index=False)

    current_metrics = sweep.loc[sweep["name"].eq("current_identity_residual")].iloc[0].to_dict()
    chosen_metrics = sweep.loc[sweep["name"].eq(best_name)].iloc[0].to_dict()

    accepted = (
        best_name != "current_identity_residual"
        and better(chosen_metrics, current_metrics)
    )

    decision = {
        "accepted": bool(accepted),
        "current_name": "current_identity_residual",
        "chosen_name": best_name,
        "current_metrics": {
            k: current_metrics[k]
            for k in [
                "model_log_loss", "market_log_loss", "delta_log_loss_market_minus_model",
                "model_brier", "market_brier", "delta_brier_market_minus_model",
                "mean_model_over_prob", "mean_market_over_prob"
            ]
        },
        "chosen_metrics": {
            k: chosen_metrics[k]
            for k in [
                "model_log_loss", "market_log_loss", "delta_log_loss_market_minus_model",
                "model_brier", "market_brier", "delta_brier_market_minus_model",
                "mean_model_over_prob", "mean_market_over_prob"
            ]
        },
        "n_rows_total": int(len(rows)),
        "n_rows_cal": int(len(cal_df)),
        "n_rows_eval": int(len(eval_df)),
    }

    if accepted:
        system = StrikeoutBettingSystem.load(system_path)
        backup_path = system_path.with_suffix(system_path.suffix + ".bak_before_residual_mainline_cal")
        if not backup_path.exists():
            backup_path.write_bytes(system_path.read_bytes())

        # evaluator already knows how to apply this hook
        system.mainline_threshold_calibrator = best_cal
        system.mainline_threshold_calibrator_meta = decision
        system.save(system_path)

        decision["saved_model_path"] = str(system_path)
        decision["backup_path"] = str(backup_path)

    with open(out_dir / "mainline_residual_calibrator_decision.json", "w") as f:
        json.dump(decision, f, indent=2)

    print("\nSWEEP RESULTS")
    print(sweep.to_string(index=False))
    print("\nDECISION")
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
