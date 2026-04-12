
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from mlb_k_model.data_pipeline import FeatureBuilder
from mlb_k_model.external_data import load_all_market_snapshots, merge_external_features
from mlb_k_model.market import line_event_probs_from_pmf, remove_vig_two_way
from mlb_k_model.portfolio import bet_ev, decimal_from_american
from mlb_k_model.strikeout_model import ModelConfig, StrikeoutDistributionModel
from mlb_k_model.system import StrikeoutBettingSystem


def build_lineup_lookup(train_df: pd.DataFrame) -> dict:
    lookup = {}
    for (game_id, pitcher_id), g in train_df.groupby(["game_id", "pitcher_id"], sort=False):
        lineup = (
            g.sort_values("pa_number")
            .drop_duplicates("batter_id")
            [
                [
                    "batter_id",
                    "batter_side",
                    "batter_k_rate_15",
                    "batter_k_rate_30",
                    "batter_k_rate_60",
                    "batter_walk_rate_60",
                    "batter_hit_rate_60",
                    "batter_barrel_rate_60",
                    "batter_xba_contact_60",
                ]
            ]
            .assign(batting_order=lambda x: np.arange(1, len(x) + 1))
        )
        lookup[(int(game_id), int(pitcher_id))] = lineup
    return lookup


def fit_market_history(
    system: StrikeoutBettingSystem,
    market_training: pd.DataFrame,
    start_states: pd.DataFrame,
    lineup_lookup: dict,
) -> StrikeoutBettingSystem:
    if market_training.empty:
        return system

    base = start_states.set_index(["game_id", "pitcher_id"])
    rows = []

    for row in market_training.itertuples(index=False):
        key = (int(row.game_id), int(row.player_id))
        if key not in base.index or key not in lineup_lookup:
            continue
        state = base.loc[key].to_dict()
        actual_k = int(state.pop("actual_total_k"))
        state.pop("actual_total_bf", None)
        pmf = system.true_model.predict_pmf(state=state, future_lineup=lineup_lookup[key])

        p_over, p_under, _ = line_event_probs_from_pmf(pmf, float(row.line_value))
        fair_over, fair_under = remove_vig_two_way(row.over_odds, row.under_odds)

        for side, model_prob, market_prob, book_odds, won in [
            ("over", p_over, fair_over, row.over_odds, int(actual_k > float(row.line_value))),
            ("under", p_under, fair_under, row.under_odds, int(actual_k < float(row.line_value))),
        ]:
            if book_odds is None or pd.isna(book_odds):
                continue

            d = decimal_from_american(int(book_odds))
            realized_profit = (d - 1.0) if won else -1.0

            rows.append(
                {
                    "model_prob": model_prob,
                    "market_prob": market_prob,
                    "abs_gap": abs(model_prob - market_prob),
                    "vendor_count": 1,
                    "is_live": int(getattr(row, "is_live_snapshot", 0)),
                    "target": won,
                    "edge": model_prob - market_prob,
                    "ev": bet_ev(model_prob, int(book_odds)),
                    "uncertainty": abs(model_prob - market_prob),
                    "abs_model_market_gap": abs(model_prob - market_prob),
                    "realized_profit_positive": int(realized_profit > 0),
                }
            )

    hist = pd.DataFrame(rows)
    if hist.empty:
        return system

    # Fit market-aware line calibrator on over/under events.
    system.blender.fit(hist)
    system.bet_selector.fit(hist)
    return system


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit the full pitcher strikeout betting system on real local data.")
    parser.add_argument("--data-dir", default="data", help="Directory with real parquet files")
    parser.add_argument("--out-path", default="data/pitcher_k_system.joblib", help="Where to save the fitted system")
    parser.add_argument("--max-future-bf", type=int, default=36)
    parser.add_argument("--max-k", type=int, default=20)
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
    live_states = FeatureBuilder.build_state_frame(train_df)
    survival_long = FeatureBuilder.build_survival_long(live_states, max_horizon=args.max_future_bf)
    lineup_lookup = build_lineup_lookup(train_df)

    config = ModelConfig(max_future_bf=args.max_future_bf, max_k=args.max_k)
    true_model = StrikeoutDistributionModel(config=config)
    true_model.fit(train_df, survival_long, start_states, lineup_lookup)

    system = StrikeoutBettingSystem(true_model=true_model)

    snapshot_df = load_all_market_snapshots(data_dir)
    if not snapshot_df.empty:
        actuals = FeatureBuilder.build_actual_game_totals(train_df)
        market_training = FeatureBuilder.build_market_training_frame(snapshot_df, actuals)
        system = fit_market_history(system, market_training, start_states, lineup_lookup)

    system.save(args.out_path)
    print(f"Saved fitted system to {args.out_path}")


if __name__ == "__main__":
    main()
