from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

from mlb_k_model.live_quote_universe_board import build_quote_universe_live_board


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect live quote-universe board state and outputs.")
    parser.add_argument("--date", required=True)
    parser.add_argument("--data-dir", default="data_rebuild")
    parser.add_argument("--system-path", default="data_rebuild/pitcher_k_system.joblib")
    parser.add_argument("--out-dir", default="outputs/live_diagnostics")
    args = parser.parse_args()

    out_dir = Path(args.out_dir) / args.date
    out_dir.mkdir(parents=True, exist_ok=True)

    live = build_quote_universe_live_board(
        data_dir=args.data_dir,
        system_path=args.system_path,
        target_date=args.date,
    )

    live.summary.to_csv(out_dir / "live_summary.csv", index=False)
    live.candidate_bets.to_csv(out_dir / "live_candidate_bets.csv", index=False)
    live.line_grid.to_csv(out_dir / "live_line_grid.csv", index=False)
    live.exact_pmf.to_csv(out_dir / "live_exact_pmf.csv", index=False)

    print("WROTE:")
    print(out_dir / "live_summary.csv")
    print(out_dir / "live_candidate_bets.csv")
    print(out_dir / "live_line_grid.csv")
    print(out_dir / "live_exact_pmf.csv")

    if not live.summary.empty:
        print("\nLIVE SUMMARY:")
        print(live.summary.to_string(index=False))

    if not live.line_grid.empty:
        print("\nLIVE LINE GRID:")
        print(live.line_grid.to_string(index=False))

    if not live.candidate_bets.empty:
        print("\nLIVE CANDIDATE BETS:")
        cols = [c for c in [
            "pitcher_name", "vendor", "line_value", "best_side",
            "model_over_prob", "raw_model_over_prob", "calibrated_model_over_prob",
            "market_over_prob", "best_ev", "passes_strict_ev_gate", "action_tier"
        ] if c in live.candidate_bets.columns]
        print(live.candidate_bets[cols].to_string(index=False))


if __name__ == "__main__":
    main()
