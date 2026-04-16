from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, log_loss


def clip01(x):
    x = np.asarray(x, dtype=float)
    return np.clip(x, 1e-6, 1 - 1e-6)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate live eval history and fit a baseline live calibrator when enough rows exist.")
    parser.add_argument("--eval-dir", default="outputs/live_eval")
    parser.add_argument("--archive-dir", default="outputs/live_board_archive")
    parser.add_argument("--out-dir", default="outputs/live_model_eval")
    parser.add_argument("--min-rows", type=int, default=50)
    args = parser.parse_args()

    eval_root = Path(args.eval_dir)
    archive_root = Path(args.archive_dir)
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    pool_parts = []

    for date_dir in sorted(eval_root.iterdir()) if eval_root.exists() else []:
        if not date_dir.is_dir():
            continue

        metrics_path = date_dir / "live_eval_metrics_summary.json"
        eval_rows_path = date_dir / "live_eval_rows.csv"
        if not metrics_path.exists() or not eval_rows_path.exists():
            continue

        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        metrics["date"] = date_dir.name
        summary_rows.append(metrics)

        archive_date_dir = archive_root / date_dir.name
        if not archive_date_dir.exists():
            continue

        eval_rows = pd.read_csv(eval_rows_path)
        if eval_rows.empty:
            continue

        eval_rows["snapshot_ts"] = pd.to_datetime(eval_rows["snapshot_ts"], utc=True, errors="coerce")
        eval_rows["line_value"] = pd.to_numeric(eval_rows["line_value"], errors="coerce")
        eval_rows["realized_over"] = pd.to_numeric(eval_rows["realized_over"], errors="coerce")
        eval_rows["realized_push"] = pd.to_numeric(eval_rows["realized_push"], errors="coerce")
        eval_rows["pitcher_name"] = eval_rows["player_name"]

        cand_parts = []
        for run_dir in sorted(archive_date_dir.iterdir()):
            fp = run_dir / "live_candidate_bets.csv"
            if fp.exists():
                df = pd.read_csv(fp)
                if len(df):
                    cand_parts.append(df)

        if not cand_parts:
            continue

        cand = pd.concat(cand_parts, ignore_index=True, sort=False)
        cand["snapshot_ts"] = pd.to_datetime(cand["snapshot_ts"], utc=True, errors="coerce")
        cand["line_value"] = pd.to_numeric(cand["line_value"], errors="coerce")
        cand["model_over_prob"] = pd.to_numeric(cand["model_over_prob"], errors="coerce")
        cand["market_over_prob"] = pd.to_numeric(cand["market_over_prob"], errors="coerce")

        merged = cand.merge(
            eval_rows[
                [
                    "game_id",
                    "pitcher_name",
                    "vendor",
                    "line_value",
                    "snapshot_ts",
                    "realized_over",
                    "realized_push",
                    "state_lag_seconds",
                    "strikeouts_so_far",
                    "batters_faced_so_far",
                    "pitches_thrown_so_far",
                    "innings_completed",
                    "pitcher_active_flag",
                    "final_strikeouts",
                ]
            ],
            on=["game_id", "pitcher_name", "vendor", "line_value", "snapshot_ts"],
            how="inner",
        )

        if len(merged):
            merged["date"] = date_dir.name
            merged = merged.loc[merged["realized_push"].fillna(0).eq(0)].copy()
            pool_parts.append(merged)

    summary_df = pd.DataFrame(summary_rows).sort_values("date") if summary_rows else pd.DataFrame()
    summary_df.to_csv(out_root / "live_eval_history_summary.csv", index=False)

    pool = pd.concat(pool_parts, ignore_index=True, sort=False) if pool_parts else pd.DataFrame()
    pool.to_csv(out_root / "live_calibration_pool.csv", index=False)

    report = {
        "dates_in_history": int(len(summary_df)),
        "total_quote_rows": int(summary_df["quote_rows_total"].sum()) if "quote_rows_total" in summary_df else 0,
        "total_valid_state_rows": int(summary_df["valid_state_rows"].sum()) if "valid_state_rows" in summary_df else 0,
        "total_active_valid_rows": int(summary_df["active_valid_rows"].sum()) if "active_valid_rows" in summary_df else 0,
        "total_calibration_ready_rows": int(summary_df["calibration_ready_rows"].sum()) if "calibration_ready_rows" in summary_df else 0,
        "calibration_pool_rows_with_model_probs": int(len(pool)),
        "fitted_calibrator": False,
    }

    if len(pool) >= args.min_rows and pool["realized_over"].nunique() >= 2:
        x = clip01(pool["model_over_prob"].values)
        y = pool["realized_over"].astype(int).values

        iso = IsotonicRegression(y_min=1e-6, y_max=1 - 1e-6, out_of_bounds="clip")
        iso.fit(x, y)
        p_cal = clip01(iso.predict(x))

        report["fitted_calibrator"] = True
        report["model_log_loss_pre"] = float(log_loss(y, clip01(x)))
        report["model_log_loss_post"] = float(log_loss(y, p_cal))
        report["model_brier_pre"] = float(brier_score_loss(y, clip01(x)))
        report["model_brier_post"] = float(brier_score_loss(y, p_cal))

        joblib.dump(iso, out_root / "live_over_isotonic.joblib")

        grid = np.linspace(0.01, 0.99, 99)
        curve = pd.DataFrame({
            "raw_model_over_prob": grid,
            "calibrated_model_over_prob": clip01(iso.predict(grid)),
        })
        curve.to_csv(out_root / "live_over_isotonic_curve.csv", index=False)

    (out_root / "live_calibration_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("WROTE:", out_root / "live_eval_history_summary.csv")
    print("WROTE:", out_root / "live_calibration_pool.csv")
    print("WROTE:", out_root / "live_calibration_report.json")
    if (out_root / "live_over_isotonic_curve.csv").exists():
        print("WROTE:", out_root / "live_over_isotonic_curve.csv")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
