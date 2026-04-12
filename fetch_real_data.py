
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable, List

import pandas as pd

from mlb_k_model import BallDontLieClient, DataBuilder, OddsAPIClient, PyBaseballClient
from mlb_k_model.external_data import (
    build_player_id_map,
    collect_savant_batter_features,
    collect_savant_pitcher_features,
    map_odds_api_quotes_to_bdl,
    normalize_odds_api_event_odds,
)


def date_range(start: str, end: str) -> list[str]:
    return [d.strftime("%Y-%m-%d") for d in pd.date_range(start, end, freq="D")]


def timestamp_range(start: str, end: str, freq_minutes: int) -> list[str]:
    return [d.strftime("%Y-%m-%dT%H:%M:%SZ") for d in pd.date_range(start, end, freq=f"{int(freq_minutes)}min", tz="UTC")]


def chunked(values: Iterable, size: int) -> List[list]:
    vals = list(values)
    return [vals[i : i + size] for i in range(0, len(vals), size)]


def fetch_bdl_backbone(args, root: Path) -> None:
    api_key = os.environ["BDL_API_KEY"]
    client = BallDontLieClient(api_key)
    builder = DataBuilder(client, root)

    if args.seasons:
        games = builder.collect_games(seasons=args.seasons, season_type=args.season_type, save_name="games.parquet")
        if games.empty:
            print("No BDL games fetched.")
            return

        pa = builder.collect_plate_appearances(games["id"].astype(int).tolist(), save_name="plate_appearances.parquet")
        player_ids = sorted(set(pd.concat([pa["pitcher_id"], pa["batter_id"]], axis=0).dropna().astype(int).tolist())) if not pa.empty else []
        if player_ids:
            builder.collect_players(player_ids, save_name="players.parquet")
            builder.collect_stats(player_ids=player_ids, seasons=args.seasons, save_name="stats.parquet")
            builder.collect_injuries(player_ids=player_ids, save_name="injuries.parquet")

        builder.collect_season_stats(args.seasons, season_type=args.season_type)
        builder.collect_team_season_stats(args.seasons, season_type=args.season_type)
        print(f"Fetched BDL backbone for seasons: {args.seasons}")

    if args.with_bdl_snapshots:
        if not args.start_date or not args.end_date:
            raise SystemExit("--with-bdl-snapshots requires --start-date and --end-date")
        for d in date_range(args.start_date, args.end_date):
            builder.snapshot_lineups(d)
            builder.snapshot_game_odds(d)
            builder.snapshot_player_props(d, prop_type="pitcher_strikeouts")
            print(f"Saved BDL snapshots for {d}")


def fetch_odds_api(args, root: Path) -> None:
    api_key = os.environ["ODDS_API_KEY"]
    client = OddsAPIClient(api_key)
    raw_dir = root / "snapshots" / "odds_api_raw"
    norm_dir = root / "snapshots" / "player_props_odds_api"
    raw_dir.mkdir(parents=True, exist_ok=True)
    norm_dir.mkdir(parents=True, exist_ok=True)

    games_path = root / "games.parquet"
    players_path = root / "players.parquet"
    if not games_path.exists() or not players_path.exists():
        raise SystemExit("Odds API mapping needs data/games.parquet and data/players.parquet first. Run BDL fetch first.")

    games = pd.read_parquet(games_path)
    players = pd.read_parquet(players_path)

    bookmakers = args.odds_bookmakers.split(",") if args.odds_bookmakers else None
    markets = args.odds_markets

    if args.start_timestamp and args.end_timestamp:
        timestamps = timestamp_range(args.start_timestamp, args.end_timestamp, args.odds_snapshot_interval_minutes)

        for ts in timestamps:
            hist_events = client.get_historical_events(date=ts, sport=args.odds_sport)
            event_rows = hist_events.get("data", []) if isinstance(hist_events, dict) else hist_events
            pd.json_normalize(event_rows).to_parquet(raw_dir / f"historical_events_{ts.replace(':', '-')}.parquet", index=False)

            frames = []
            for event in event_rows:
                payload = client.get_historical_event_odds(
                    str(event["id"]),
                    date=ts,
                    sport=args.odds_sport,
                    regions=args.odds_regions,
                    markets=markets,
                    odds_format="american",
                    bookmakers=bookmakers,
                )
                raw_path = raw_dir / f"historical_event_odds_{event['id']}_{ts.replace(':', '-')}.json"
                raw_path.write_text(pd.Series(payload).to_json())
                norm = normalize_odds_api_event_odds(payload.get("data", payload), fetched_at=ts, source="historical")
                if not norm.empty:
                    frames.append(norm)

            if frames:
                norm = pd.concat(frames, ignore_index=True)
                mapped = map_odds_api_quotes_to_bdl(norm, games, players)
                if not mapped.empty:
                    mapped.to_parquet(norm_dir / f"{ts.replace(':', '-')}.parquet", index=False)
            print(f"Saved Odds API historical snapshots for {ts}")

    if args.start_date and args.end_date:
        # current / daily pregame snapshots
        for d in date_range(args.start_date, args.end_date):
            events = client.list_events(sport=args.odds_sport)
            event_frame = pd.json_normalize(events)
            if not event_frame.empty:
                event_frame.to_parquet(raw_dir / f"events_{d}.parquet", index=False)

            frames = []
            for event in events:
                payload = client.get_event_odds(
                    str(event["id"]),
                    sport=args.odds_sport,
                    regions=args.odds_regions,
                    markets=markets,
                    odds_format="american",
                    bookmakers=bookmakers,
                )
                raw_path = raw_dir / f"event_odds_{event['id']}_{d}.json"
                raw_path.write_text(pd.Series(payload).to_json())
                norm = normalize_odds_api_event_odds(payload, fetched_at=f"{d}T00:00:00Z", source="live")
                if not norm.empty:
                    frames.append(norm)

            if frames:
                norm = pd.concat(frames, ignore_index=True)
                mapped = map_odds_api_quotes_to_bdl(norm, games, players)
                if not mapped.empty:
                    mapped.to_parquet(norm_dir / f"{d}.parquet", index=False)
            print(f"Saved Odds API current snapshots for {d}")


def fetch_pybaseball(args, root: Path) -> None:
    pyb = PyBaseballClient(cache=True)
    ext_dir = root / "external"
    ext_dir.mkdir(parents=True, exist_ok=True)

    players_path = root / "players.parquet"
    if not players_path.exists():
        raise SystemExit("pybaseball mapping needs data/players.parquet first. Run BDL fetch first.")

    players = pd.read_parquet(players_path)
    player_map = build_player_id_map(players, pyb)
    player_map.to_parquet(ext_dir / "player_id_map.parquet", index=False)

    if args.seasons:
        pitcher_ext = collect_savant_pitcher_features(pyb, args.seasons)
        batter_ext = collect_savant_batter_features(pyb, args.seasons)
        if not pitcher_ext.empty:
            pitcher_ext.to_parquet(ext_dir / "savant_pitcher_features.parquet", index=False)
        if not batter_ext.empty:
            batter_ext.to_parquet(ext_dir / "savant_batter_features.parquet", index=False)

    if args.with_pyb_statcast and args.start_date and args.end_date:
        statcast_dir = ext_dir / "statcast_chunks"
        statcast_dir.mkdir(parents=True, exist_ok=True)
        for rng in chunked(date_range(args.start_date, args.end_date), args.pyb_days_per_chunk):
            start_dt, end_dt = rng[0], rng[-1]
            df = pyb.statcast(start_dt, end_dt)
            if not df.empty:
                df.to_parquet(statcast_dir / f"{start_dt}_{end_dt}.parquet", index=False)
            print(f"Saved pybaseball statcast chunk {start_dt} -> {end_dt}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch real MLB data for the pitcher strikeout system.")
    parser.add_argument("--data-dir", default="data", help="Root data directory")

    parser.add_argument("--seasons", nargs="*", type=int, help="Historical seasons to fetch")
    parser.add_argument("--season-type", default="regular", help="Season type")

    parser.add_argument("--start-date", help="Date range start YYYY-MM-DD")
    parser.add_argument("--end-date", help="Date range end YYYY-MM-DD")

    parser.add_argument("--fetch-bdl", action="store_true", help="Fetch BDL backbone data")
    parser.add_argument("--with-bdl-snapshots", action="store_true", help="Save BDL lineups, odds, and player props snapshots")

    parser.add_argument("--fetch-odds-api", action="store_true", help="Fetch The Odds API event odds and map them to BDL ids")
    parser.add_argument("--odds-sport", default="baseball_mlb", help="The Odds API sport key")
    parser.add_argument("--odds-regions", default="us", help="The Odds API regions")
    parser.add_argument("--odds-markets", default="pitcher_strikeouts,pitcher_strikeouts_alternate,h2h,spreads,totals,totals_1st_5_innings", help="Comma-separated Odds API markets")
    parser.add_argument("--odds-bookmakers", default="", help="Optional comma-separated bookmakers")
    parser.add_argument("--start-timestamp", help="Historical Odds API timestamp start ISO8601 UTC")
    parser.add_argument("--end-timestamp", help="Historical Odds API timestamp end ISO8601 UTC")
    parser.add_argument("--odds-snapshot-interval-minutes", type=int, default=60, help="Historical snapshot spacing in minutes")

    parser.add_argument("--fetch-pybaseball", action="store_true", help="Fetch pybaseball / Savant supplements")
    parser.add_argument("--with-pyb-statcast", action="store_true", help="Also save raw pybaseball statcast chunks")
    parser.add_argument("--pyb-days-per-chunk", type=int, default=3, help="Day span per raw statcast chunk")

    args = parser.parse_args()
    root = Path(args.data_dir)

    if args.fetch_bdl:
        fetch_bdl_backbone(args, root)

    if args.fetch_odds_api:
        fetch_odds_api(args, root)

    if args.fetch_pybaseball:
        fetch_pybaseball(args, root)

    if not any([args.fetch_bdl, args.fetch_odds_api, args.fetch_pybaseball]):
        raise SystemExit("Pick at least one of --fetch-bdl, --fetch-odds-api, --fetch-pybaseball")


if __name__ == "__main__":
    main()
