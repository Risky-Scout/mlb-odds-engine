
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from mlb_k_model.data_pipeline import FeatureBuilder
from mlb_k_model.external_data import load_all_market_snapshots, merge_external_features
from mlb_k_model.diagnostics import QuantDiagnostics, calibration_table, write_audit_outputs
from mlb_k_model.system import StrikeoutBettingSystem

from train_system import build_lineup_lookup


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a quant audit on the pitcher strikeout system.")
    parser.add_argument("--data-dir", default="data", help="Directory with real parquet files")
    parser.add_argument("--system-path", default="data/pitcher_k_system.joblib", help="Saved system path")
    parser.add_argument("--out-dir", default="outputs/audit", help="Output directory")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    games = pd.read_parquet(data_dir / "games.parquet")
    pa_raw = pd.read_parquet(data_dir / "plate_appearances.parquet")

    train_df = FeatureBuilder.build_training_frame(pa_raw, games)
    player_map_path = data_dir / "external" / "player_id_map.parquet"
    pitcher_ext_path = data_dir / "external" / "savant_pitcher_features.parquet"
    batter_ext_path = data_dir / "external" / "savant_batter_features.parquet"
    if player_map_path.exists():
        player_map = pd.read_parquet(player_map_path)
        pitcher_ext = pd.read_parquet(pitcher_ext_path) if pitcher_ext_path.exists() else pd.DataFrame()
        batter_ext = pd.read_parquet(batter_ext_path) if batter_ext_path.exists() else pd.DataFrame()
        train_df = merge_external_features(train_df, player_map, pitcher_ext, batter_ext)
    start_states = FeatureBuilder.build_start_state_frame(train_df)
    lineup_lookup = build_lineup_lookup(train_df)

    system = StrikeoutBettingSystem.load(args.system_path)
    diag = QuantDiagnostics(system)

    feature_signal = diag.feature_signal_report(train_df)
    permutation_report = diag.permutation_importance_report(train_df.sample(min(len(train_df), 5000), random_state=42))
    pa_metrics = diag.pa_model_report(train_df.sample(min(len(train_df), 10000), random_state=42))
    pmf_metrics = diag.pmf_report(start_states, lineup_lookup)

    X = system.true_model._ensure_columns(train_df, system.true_model.pa_feature_cols_).fillna(0.0)
    raw = system.true_model._predict_binary(system.true_model.pa_model, X)
    p = system.true_model.pa_calibrator.predict(raw)
    pa_cal = calibration_table(train_df["is_strikeout"].to_numpy(), p)

    snapshot_df = load_all_market_snapshots(data_dir)
    if not snapshot_df.empty:
        actuals = FeatureBuilder.build_actual_game_totals(train_df)
        market_training = FeatureBuilder.build_market_training_frame(snapshot_df, actuals)
        market_report = diag.market_report(market_training)
    else:
        market_report = pd.DataFrame()

    write_audit_outputs(
        out_dir=args.out_dir,
        feature_signal=feature_signal,
        permutation_report=permutation_report,
        pa_metrics=pa_metrics,
        pmf_metrics=pmf_metrics,
        pa_calibration=pa_cal,
        market_report=market_report,
    )
    print(f"Wrote audit outputs to {args.out_dir}")


if __name__ == "__main__":
    main()
