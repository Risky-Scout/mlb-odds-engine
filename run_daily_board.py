
from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd
from mlb_k_model.current_props import fetch_current_pitcher_strikeouts_snapshot
from mlb_k_model.quote_universe_board import build_quote_universe_pregame_board

from mlb_k_model import BallDontLieClient, ESPNClient, OddsAPIClient
from mlb_k_model.data_pipeline import FeatureBuilder
from mlb_k_model.external_data import merge_external_features
from mlb_k_model.live import DailyPriceBoard, write_daily_board
from mlb_k_model.system import StrikeoutBettingSystem


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the daily pre-game and live pitcher strikeout board.")
    parser.add_argument("--date", required=True, help="Date in YYYY-MM-DD")
    parser.add_argument("--data-dir", default="data", help="Directory with real parquet files")
    parser.add_argument("--out-dir", default="outputs/board", help="Output directory")
    parser.add_argument("--system-path", default="data/pitcher_k_system.joblib", help="Saved system path")
    args = parser.parse_args()

    api_key = os.environ["BDL_API_KEY"]
    data_dir = Path(args.data_dir)

    games = pd.read_parquet(data_dir / "games.parquet")
    pa_raw = pd.read_parquet(data_dir / "plate_appearances.parquet")
    history_pa_df = FeatureBuilder.build_training_frame(pa_raw, games)
    player_map_path = data_dir / "external" / "player_id_map.parquet"
    pitcher_ext_path = data_dir / "external" / "savant_pitcher_features.parquet"
    batter_ext_path = data_dir / "external" / "savant_batter_features.parquet"
    if player_map_path.exists():
        player_map = pd.read_parquet(player_map_path)
        pitcher_ext = pd.read_parquet(pitcher_ext_path) if pitcher_ext_path.exists() else pd.DataFrame()
        batter_ext = pd.read_parquet(batter_ext_path) if batter_ext_path.exists() else pd.DataFrame()
        history_pa_df = merge_external_features(history_pa_df, player_map, pitcher_ext, batter_ext)

    system = StrikeoutBettingSystem.load(args.system_path)
    odds_api = OddsAPIClient(os.environ["ODDS_API_KEY"]) if "ODDS_API_KEY" in os.environ else None
    players_df = pd.read_parquet(data_dir / "players.parquet") if (data_dir / "players.parquet").exists() else pd.DataFrame()
    board = DailyPriceBoard(
        system=system,
        bdl=BallDontLieClient(api_key),
        espn=ESPNClient(),
        odds_api=odds_api,
        players_df=players_df,
    )

    history_pa_df = history_pa_df.copy()
    if "game_date" in history_pa_df.columns:
        history_pa_df["game_date"] = pd.to_datetime(
            history_pa_df["game_date"], utc=True, errors="coerce"
        ).dt.tz_localize(None)
    history_pa_df = history_pa_df.copy()
    if "game_date" in history_pa_df.columns:
        history_pa_df["game_date"] = pd.to_datetime(
            history_pa_df["game_date"], errors="coerce", utc=True
        ).dt.tz_localize(None)
    try:
        current_props = fetch_current_pitcher_strikeouts_snapshot(
            data_dir=args.data_dir,
            target_date=str(args.date),
            regions="us",
        )
        print(f"Fetched current pitcher_strikeouts snapshot rows: {len(current_props)}")
    except Exception as e:
        print(f"Current props fetch warning: {e}")

    if 'current_props' in locals() and current_props is not None and len(current_props) > 0:
        print("Using quote-universe pregame board builder")
        pregame = build_quote_universe_pregame_board(
            data_dir=args.data_dir,
            system_path=args.system_path,
            target_date=str(args.date),
        )
    else:
        pregame = board.build_pregame_board(date=args.date, history_pa_df=history_pa_df)
    live = board.build_live_board(date=args.date, history_pa_df=history_pa_df)

    write_daily_board(date=args.date, pregame=pregame, live=live, out_dir=args.out_dir)
    print(f"Wrote daily board to {Path(args.out_dir) / args.date}")


if __name__ == "__main__":
    main()
