from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from mlb_k_model.current_props import fetch_current_pitcher_strikeouts_snapshot


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture and archive current MLB pitcher strikeout quote snapshots.")
    parser.add_argument("--date", required=True)
    parser.add_argument("--data-dir", default="data_rebuild")
    parser.add_argument("--regions", default="us")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    snap = fetch_current_pitcher_strikeouts_snapshot(
        data_dir=data_dir,
        target_date=args.date,
        regions=args.regions,
    )

    archive_dir = data_dir / "snapshots" / "player_props_odds_api" / "live_archive"
    archive_dir.mkdir(parents=True, exist_ok=True)
    archive_path = archive_dir / f"odds_api_live_capture_{args.date}.parquet"

    snap = snap.copy()
    snap["capture_run_ts"] = pd.Timestamp.utcnow()

    if archive_path.exists():
        prev = pd.read_parquet(archive_path)
        out = pd.concat([prev, snap], ignore_index=True, sort=False)
    else:
        out = snap.copy()

    dedupe_cols = [c for c in [
        "event_id",
        "vendor",
        "prop_type",
        "player_name",
        "line_value",
        "snapshot_ts",
        "over_odds",
        "under_odds",
    ] if c in out.columns]

    if dedupe_cols:
        out = out.drop_duplicates(subset=dedupe_cols, keep="last").reset_index(drop=True)

    out.to_parquet(archive_path, index=False)

    print("WROTE:", archive_path)
    print("CURRENT SNAPSHOT ROWS:", len(snap))
    if "is_live_snapshot" in snap.columns:
        print("\nCURRENT is_live_snapshot counts:")
        print(snap["is_live_snapshot"].value_counts(dropna=False).to_string())

    if len(snap):
        keep = [c for c in [
            "event_id", "game_id", "player_name", "vendor", "line_value",
            "over_odds", "under_odds", "snapshot_ts", "commence_time", "is_live_snapshot"
        ] if c in snap.columns]
        print("\nCURRENT SNAPSHOT SAMPLE:")
        print(snap[keep].head(40).to_string(index=False))

    print("\nARCHIVE TOTAL ROWS:", len(out))


if __name__ == "__main__":
    main()
