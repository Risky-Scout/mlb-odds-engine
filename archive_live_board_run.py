from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Archive one live board refresh without using extra API calls.")
    parser.add_argument("--date", required=True)
    parser.add_argument("--board-dir", default="outputs/board")
    parser.add_argument("--predictions-dir", default="Predictions")
    parser.add_argument("--archive-dir", default="outputs/live_board_archive")
    args = parser.parse_args()

    board_root = Path(args.board_dir) / args.date
    pred_root = Path(args.predictions_dir) / "mlb" / "pitcher_strikeouts" / "live" / args.date
    latest_json = Path(args.predictions_dir) / "API" / "mlb" / "pitcher_strikeouts" / "live" / "latest.json"

    if latest_json.exists():
        payload = json.loads(latest_json.read_text(encoding="utf-8"))
        ts = pd.to_datetime(payload.get("generated_at_utc"), utc=True, errors="coerce")
    else:
        ts = pd.Timestamp.utcnow()

    run_id = ts.strftime("%Y%m%dT%H%M%SZ")
    out = Path(args.archive_dir) / args.date / run_id
    out.mkdir(parents=True, exist_ok=True)

    copied = []
    for src in [
        board_root / "live_summary.csv",
        board_root / "live_candidate_bets.csv",
        board_root / "live_line_grid.csv",
        board_root / "live_exact_pmf.csv",
        board_root / "live_production_card.csv",
        board_root / "live_watchlist.csv",
        pred_root / "summary.json",
        pred_root / "live_card.json",
        pred_root / "live_production_card.json",
        pred_root / "live_watchlist.json",
        pred_root / "candidate_bets.json",
        pred_root / "line_grid.json",
        pred_root / "exact_pmf.json",
        latest_json,
    ]:
        if src.exists():
            dst = out / src.name
            shutil.copy2(src, dst)
            copied.append(dst.name)

    manifest = {
        "date": args.date,
        "run_id": run_id,
        "copied_files": copied,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("WROTE:", out)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
