from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize live eval coverage and prepare calibration-ready rows.")
    parser.add_argument("--date", required=True)
    parser.add_argument("--eval-dir", default="outputs/live_eval")
    args = parser.parse_args()

    root = Path(args.eval_dir) / args.date
    rows_path = root / "live_eval_rows.csv"
    unmatched_path = root / "live_eval_unmatched_quotes.csv"

    if not rows_path.exists():
        raise SystemExit(f"Missing {rows_path}")
    if not unmatched_path.exists():
        raise SystemExit(f"Missing {unmatched_path}")

    valid = pd.read_csv(rows_path)
    unmatched = pd.read_csv(unmatched_path)

    total_rows = int(len(valid) + len(unmatched))
    matched_rate = float(len(valid) / total_rows) if total_rows else None
    unmatched_rate = float(len(unmatched) / total_rows) if total_rows else None

    if "pitcher_active_flag" in valid.columns:
        valid["pitcher_active_flag"] = pd.to_numeric(valid["pitcher_active_flag"], errors="coerce")
    if "state_lag_seconds" in valid.columns:
        valid["state_lag_seconds"] = pd.to_numeric(valid["state_lag_seconds"], errors="coerce")
    if "realized_push" in valid.columns:
        valid["realized_push"] = pd.to_numeric(valid["realized_push"], errors="coerce")
    if "realized_over" in valid.columns:
        valid["realized_over"] = pd.to_numeric(valid["realized_over"], errors="coerce")
    if "realized_under" in valid.columns:
        valid["realized_under"] = pd.to_numeric(valid["realized_under"], errors="coerce")

    active_valid = valid.loc[valid["pitcher_active_flag"].fillna(0).eq(1)].copy() if "pitcher_active_flag" in valid.columns else valid.iloc[0:0].copy()

    calibration_ready = valid.loc[
        valid["pitcher_active_flag"].fillna(0).eq(1)
        & valid["state_lag_seconds"].notna()
        & valid["state_lag_seconds"].between(0, 180, inclusive="both")
        & valid["realized_push"].fillna(0).eq(0)
    ].copy() if len(valid) else valid.copy()

    keep_cols = [c for c in [
        "game_id",
        "mlb_game_pk",
        "player_name",
        "player_id",
        "vendor",
        "line_value",
        "snapshot_ts",
        "capture_run_ts_state",
        "state_lag_seconds",
        "strikeouts_so_far",
        "batters_faced_so_far",
        "pitches_thrown_so_far",
        "innings_completed",
        "pitcher_active_flag",
        "final_strikeouts",
        "market_over_prob",
        "market_under_prob",
        "realized_over",
        "realized_under",
        "realized_push",
    ] if c in calibration_ready.columns]
    calibration_ready = calibration_ready[keep_cols].copy()

    vendor_cov = valid.groupby("vendor", dropna=False).size().reset_index(name="valid_rows") if "vendor" in valid.columns else pd.DataFrame()
    pitcher_cov = valid.groupby("player_name", dropna=False).size().reset_index(name="valid_rows") if "player_name" in valid.columns else pd.DataFrame()

    lag = valid["state_lag_seconds"].dropna() if "state_lag_seconds" in valid.columns else pd.Series(dtype=float)

    summary = {
        "date": args.date,
        "quote_rows_total": total_rows,
        "valid_state_rows": int(len(valid)),
        "unmatched_rows": int(len(unmatched)),
        "matched_rate": matched_rate,
        "unmatched_rate": unmatched_rate,
        "active_valid_rows": int(len(active_valid)),
        "calibration_ready_rows": int(len(calibration_ready)),
        "matched_pitchers": int(valid["player_name"].nunique()) if "player_name" in valid.columns and len(valid) else 0,
        "books_with_valid_rows": int(valid["vendor"].nunique()) if "vendor" in valid.columns and len(valid) else 0,
        "lag_mean_seconds": float(lag.mean()) if len(lag) else None,
        "lag_median_seconds": float(lag.median()) if len(lag) else None,
        "lag_p90_seconds": float(lag.quantile(0.90)) if len(lag) else None,
    }

    (root / "live_eval_metrics_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    vendor_cov.to_csv(root / "live_eval_valid_rows_by_vendor.csv", index=False)
    pitcher_cov.to_csv(root / "live_eval_valid_rows_by_pitcher.csv", index=False)
    calibration_ready.to_csv(root / "live_calibration_prep.csv", index=False)

    print("WROTE:", root / "live_eval_metrics_summary.json")
    print("WROTE:", root / "live_eval_valid_rows_by_vendor.csv")
    print("WROTE:", root / "live_eval_valid_rows_by_pitcher.csv")
    print("WROTE:", root / "live_calibration_prep.csv")
    print(json.dumps(summary, indent=2))

    if len(calibration_ready):
        print("\\nSAMPLE CALIBRATION-READY ROWS:")
        print(calibration_ready.head(20).to_string(index=False))
    else:
        print("\\nNo calibration-ready rows yet under current filters.")

if __name__ == "__main__":
    main()
