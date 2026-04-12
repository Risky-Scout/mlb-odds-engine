from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


def read_csv_safe(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def frame_to_records(df: pd.DataFrame) -> list[dict]:
    if df is None or df.empty:
        return []
    out = df.copy()
    out = out.where(pd.notnull(out), None)
    return out.to_dict(orient="records")


def write_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Live and in-play MLB pitcher strikeout pipeline.")
    parser.add_argument("--date", required=True)
    parser.add_argument("--data-dir", default="data_rebuild")
    parser.add_argument("--system-path", default="data_rebuild/pitcher_k_system.joblib")
    parser.add_argument("--out-dir", default="outputs/board")
    parser.add_argument("--predictions-dir", default="Predictions")
    args = parser.parse_args()

    subprocess.run(
        [
            sys.executable,
            "run_daily_board.py",
            "--date",
            args.date,
            "--data-dir",
            args.data_dir,
            "--system-path",
            args.system_path,
            "--out-dir",
            args.out_dir,
        ],
        check=True,
    )

    board_dir = Path(args.out_dir) / args.date
    live_summary_path = board_dir / "live_summary.csv"
    live_candidates_path = board_dir / "live_candidate_bets.csv"
    live_line_grid_path = board_dir / "live_line_grid.csv"
    live_exact_pmf_path = board_dir / "live_exact_pmf.csv"

    required = {
        "live_summary": live_summary_path,
        "live_candidate_bets": live_candidates_path,
        "live_line_grid": live_line_grid_path,
        "live_exact_pmf": live_exact_pmf_path,
    }
    missing = [name for name, path in required.items() if not path.exists()]
    if missing:
        raise SystemExit(f"Missing expected live board outputs: {missing}")

    summary = read_csv_safe(live_summary_path)
    candidates = read_csv_safe(live_candidates_path)
    line_grid = read_csv_safe(live_line_grid_path)
    exact_pmf = read_csv_safe(live_exact_pmf_path)

    live_card = candidates.copy()
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
        "summary": frame_to_records(summary),
        "live_card": frame_to_records(live_card),
        "candidate_bets": frame_to_records(candidates),
        "line_grid": frame_to_records(line_grid),
        "exact_pmf": frame_to_records(exact_pmf),
    }

    write_json(api_root / "latest.json", latest_payload)
    write_json(date_root / "summary.json", frame_to_records(summary))
    write_json(date_root / "live_card.json", frame_to_records(live_card))
    write_json(date_root / "candidate_bets.json", frame_to_records(candidates))
    write_json(date_root / "line_grid.json", frame_to_records(line_grid))
    write_json(date_root / "exact_pmf.json", frame_to_records(exact_pmf))

    print("WROTE:")
    print(api_root / "latest.json")
    print(date_root / "summary.json")
    print(date_root / "live_card.json")
    print(date_root / "candidate_bets.json")
    print(date_root / "line_grid.json")
    print(date_root / "exact_pmf.json")


if __name__ == "__main__":
    main()
