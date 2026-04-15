from __future__ import annotations

import argparse
import json
from datetime import datetime, date
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

from mlb_k_model.current_props import fetch_current_pitcher_strikeouts_snapshot
from mlb_k_model.live_quote_universe_board import build_quote_universe_live_board


def _json_safe_value(x):
    if x is None:
        return None
    if isinstance(x, pd.Timestamp):
        return x.isoformat()
    if isinstance(x, (datetime, date)):
        return x.isoformat()
    if pd.isna(x):
        return None
    return x


def frame_to_records(df: pd.DataFrame) -> list[dict]:
    if df is None or df.empty:
        return []
    out = df.copy()
    out = out.astype(object).where(pd.notnull(out), None)
    for c in out.columns:
        out[c] = out[c].map(_json_safe_value)
    return out.to_dict(orient="records")


def write_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def append_live_archive(data_dir: Path, target_date: str, snap: pd.DataFrame) -> None:
    archive_dir = data_dir / "snapshots" / "player_props_odds_api" / "live_archive"
    archive_dir.mkdir(parents=True, exist_ok=True)
    archive_path = archive_dir / f"odds_api_live_capture_{target_date}.parquet"

    snap = snap.copy()
    snap["capture_run_ts"] = pd.Timestamp.utcnow()

    if archive_path.exists():
        prev = pd.read_parquet(archive_path)
        out = pd.concat([prev, snap], ignore_index=True, sort=False)
    else:
        out = snap.copy()

    dedupe_cols = [c for c in [
        "event_id", "vendor", "prop_type", "player_name", "line_value",
        "snapshot_ts", "over_odds", "under_odds"
    ] if c in out.columns]

    if dedupe_cols:
        out = out.drop_duplicates(subset=dedupe_cols, keep="last").reset_index(drop=True)

    out.to_parquet(archive_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Live and in-play MLB pitcher strikeout pipeline.")
    parser.add_argument("--date", required=True)
    parser.add_argument("--data-dir", default="data_rebuild")
    parser.add_argument("--system-path", default="data_rebuild/pitcher_k_system.joblib")
    parser.add_argument("--out-dir", default="outputs/board")
    parser.add_argument("--predictions-dir", default="Predictions")
    parser.add_argument("--regions", default="us")
    parser.add_argument("--refresh-board", action="store_true", help="Refresh live/current quote snapshot before building live board.")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    board_dir = Path(args.out_dir) / args.date
    board_dir.mkdir(parents=True, exist_ok=True)

    if args.refresh_board:
        if not os.environ.get("ODDS_API_KEY", "").strip():
            raise SystemExit("Missing ODDS_API_KEY. Export it before using --refresh-board.")
        snap = fetch_current_pitcher_strikeouts_snapshot(
            data_dir=data_dir,
            target_date=args.date,
            regions=args.regions,
        )
        append_live_archive(data_dir, args.date, snap)
        print("Refreshed current pitcher strikeout snapshot rows:", len(snap))

        subprocess.run(
            [
                sys.executable,
                "capture_live_game_state_snapshots.py",
                "--date",
                args.date,
                "--data-dir",
                args.data_dir,
            ],
            check=True,
        )

    live = build_quote_universe_live_board(
        data_dir=data_dir,
        system_path=args.system_path,
        target_date=args.date,
    )

    live.summary.to_csv(board_dir / "live_summary.csv", index=False)
    live.candidate_bets.to_csv(board_dir / "live_candidate_bets.csv", index=False)
    live.line_grid.to_csv(board_dir / "live_line_grid.csv", index=False)
    live.exact_pmf.to_csv(board_dir / "live_exact_pmf.csv", index=False)

    live_card = live.candidate_bets.copy()
    if not live_card.empty:
        if "passes_strict_ev_gate" in live_card.columns:
            live_card = live_card.loc[
                live_card["passes_strict_ev_gate"].astype(str).str.lower().isin(["true", "1", "yes"])
            ].copy()
        elif "action_tier" in live_card.columns:
            live_card = live_card.loc[live_card["action_tier"].astype(str).eq("BET")].copy()

        sort_cols = [c for c in ["best_ev", "gate_conf_dist"] if c in live_card.columns]
        if sort_cols:
            live_card = live_card.sort_values(sort_cols, ascending=False).reset_index(drop=True)

    generated_at = pd.Timestamp.utcnow().isoformat()

    api_root = Path(args.predictions_dir) / "API" / "mlb" / "pitcher_strikeouts" / "live"
    date_root = Path(args.predictions_dir) / "mlb" / "pitcher_strikeouts" / "live" / args.date

    latest_payload = {
        "model": "mlb-odds-engine",
        "market": "pitcher_strikeouts",
        "mode": "live",
        "date": args.date,
        "generated_at_utc": generated_at,
        "summary": frame_to_records(live.summary),
        "live_card": frame_to_records(live_card),
        "candidate_bets": frame_to_records(live.candidate_bets),
        "line_grid": frame_to_records(live.line_grid),
        "exact_pmf": frame_to_records(live.exact_pmf),
    }

    write_json(api_root / "latest.json", latest_payload)
    write_json(date_root / "summary.json", frame_to_records(live.summary))
    write_json(date_root / "live_card.json", frame_to_records(live_card))
    write_json(date_root / "candidate_bets.json", frame_to_records(live.candidate_bets))
    write_json(date_root / "line_grid.json", frame_to_records(live.line_grid))
    write_json(date_root / "exact_pmf.json", frame_to_records(live.exact_pmf))

    print("WROTE:")
    print(board_dir / "live_summary.csv")
    print(board_dir / "live_candidate_bets.csv")
    print(board_dir / "live_line_grid.csv")
    print(board_dir / "live_exact_pmf.csv")
    print(api_root / "latest.json")
    print(date_root / "summary.json")
    print(date_root / "live_card.json")
    print(date_root / "candidate_bets.json")
    print(date_root / "line_grid.json")
    print(date_root / "exact_pmf.json")


if __name__ == "__main__":
    main()
