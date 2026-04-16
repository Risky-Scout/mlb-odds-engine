from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path

import pandas as pd


def _norm(s) -> str:
    if s is None:
        return ""
    s = str(s)
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = s.lower().strip()
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s


def american_to_prob(x):
    try:
        x = float(x)
    except Exception:
        return float("nan")
    if x > 0:
        return 100.0 / (x + 100.0)
    return (-x) / ((-x) + 100.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build offline live evaluation dataset from archived quotes and MLB live state snapshots.")
    parser.add_argument("--date", required=True)
    parser.add_argument("--data-dir", default="data_rebuild")
    parser.add_argument("--out-dir", default="outputs/live_eval")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir) / args.date
    out_dir.mkdir(parents=True, exist_ok=True)

    quote_path = data_dir / "snapshots" / "player_props_odds_api" / "live_archive" / f"odds_api_live_capture_{args.date}.parquet"
    state_path = data_dir / "snapshots" / "mlb_statsapi" / "live_state_archive" / f"mlb_live_state_{args.date}.parquet"

    if not quote_path.exists():
        raise SystemExit(f"Missing quote archive: {quote_path}")
    if not state_path.exists():
        raise SystemExit(f"Missing state archive: {state_path}")

    q = pd.read_parquet(quote_path).copy()
    s = pd.read_parquet(state_path).copy()

    if q.empty:
        raise SystemExit("Quote archive is empty.")
    if s.empty:
        raise SystemExit("State archive is empty.")

    if "prop_type" in q.columns:
        q = q.loc[q["prop_type"].astype(str).eq("pitcher_strikeouts")].copy()
    if "is_live_snapshot" in q.columns:
        q = q.loc[q["is_live_snapshot"].fillna(0).astype(int).eq(1)].copy()

    q["snapshot_ts"] = pd.to_datetime(q["snapshot_ts"], utc=True, errors="coerce")
    q["commence_time"] = pd.to_datetime(q["commence_time"], utc=True, errors="coerce")
    q["player_name_norm"] = q["player_name"].map(_norm)
    q["home_team_norm"] = q["home_team"].map(_norm)
    q["away_team_norm"] = q["away_team"].map(_norm)

    s["capture_run_ts"] = pd.to_datetime(s["capture_run_ts"], utc=True, errors="coerce")
    s["game_date"] = pd.to_datetime(s["game_date"], utc=True, errors="coerce")
    s["pitcher_name_norm"] = s["pitcher_name_norm"].map(_norm)
    s["home_team_norm"] = s["home_team"].map(_norm)
    s["away_team_norm"] = s["away_team"].map(_norm)

    state_games = s[["mlb_game_pk", "game_date", "home_team", "away_team", "home_team_norm", "away_team_norm"]].drop_duplicates().copy()
    quote_games = q[["game_id", "home_team", "away_team", "home_team_norm", "away_team_norm", "commence_time"]].drop_duplicates().copy()

    map_rows = []
    for r in quote_games.itertuples(index=False):
        cand = state_games.loc[
            (state_games["home_team_norm"] == r.home_team_norm) &
            (state_games["away_team_norm"] == r.away_team_norm)
        ].copy()
        if cand.empty:
            continue
        if pd.notna(r.commence_time):
            cand["time_diff"] = (cand["game_date"] - r.commence_time).abs()
            cand = cand.sort_values("time_diff")
        map_rows.append({
            "game_id": int(r.game_id),
            "mlb_game_pk": int(cand.iloc[0]["mlb_game_pk"]),
        })

    mapping = pd.DataFrame(map_rows).drop_duplicates()
    q = q.merge(mapping, on="game_id", how="left")

    q["market_over_prob_raw"] = q["over_odds"].map(american_to_prob)
    q["market_under_prob_raw"] = q["under_odds"].map(american_to_prob)
    denom = q["market_over_prob_raw"] + q["market_under_prob_raw"]
    q["market_over_prob"] = q["market_over_prob_raw"] / denom
    q["market_under_prob"] = q["market_under_prob_raw"] / denom

    q = q.rename(columns={"player_name_norm": "pitcher_name_norm"}).copy()

    left = q.sort_values("snapshot_ts").copy()
    right = s.sort_values("capture_run_ts").copy()

    matched = pd.merge_asof(
        left,
        right,
        left_on="snapshot_ts",
        right_on="capture_run_ts",
        by=["mlb_game_pk", "pitcher_name_norm"],
        direction="backward",
        tolerance=pd.Timedelta(minutes=3),
        suffixes=("", "_state"),
    )

    state_ts_col = "capture_run_ts_state" if "capture_run_ts_state" in matched.columns else "capture_run_ts"
    matched["has_valid_state"] = matched[state_ts_col].notna()
    matched["state_lag_seconds"] = (matched["snapshot_ts"] - matched[state_ts_col]).dt.total_seconds()

    final_k = (
        s.groupby(["mlb_game_pk", "pitcher_name_norm"], as_index=False)["strikeouts_so_far"]
        .max()
        .rename(columns={"strikeouts_so_far": "final_strikeouts"})
    )

    matched = matched.merge(final_k, on=["mlb_game_pk", "pitcher_name_norm"], how="left")

    matched["realized_over"] = (matched["final_strikeouts"] > matched["line_value"]).astype("float")
    matched["realized_under"] = (matched["final_strikeouts"] < matched["line_value"]).astype("float")
    matched["realized_push"] = (matched["final_strikeouts"] == matched["line_value"]).astype("float")

    valid = matched.loc[matched["has_valid_state"].fillna(False)].copy()
    unmatched = matched.loc[~matched["has_valid_state"].fillna(False)].copy()

    valid_path_csv = out_dir / "live_eval_rows.csv"
    valid_path_parquet = out_dir / "live_eval_rows.parquet"
    unmatched_path_csv = out_dir / "live_eval_unmatched_quotes.csv"
    summary_path_json = out_dir / "live_eval_summary.json"

    valid.to_csv(valid_path_csv, index=False)
    valid.to_parquet(valid_path_parquet, index=False)
    unmatched.to_csv(unmatched_path_csv, index=False)

    summary = {
        "date": args.date,
        "quote_rows_total": int(len(matched)),
        "valid_state_rows": int(len(valid)),
        "unmatched_rows": int(len(unmatched)),
        "matched_pitchers": int(valid["pitcher_name"].nunique()) if "pitcher_name" in valid.columns and len(valid) else 0,
        "books": int(valid["vendor"].nunique()) if "vendor" in valid.columns and len(valid) else 0,
        "mean_state_lag_seconds": float(valid["state_lag_seconds"].dropna().mean()) if len(valid) else None,
    }
    summary_path_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("WROTE:", valid_path_csv)
    print("WROTE:", valid_path_parquet)
    print("WROTE:", unmatched_path_csv)
    print("WROTE:", summary_path_json)
    print(json.dumps(summary, indent=2))

    if len(valid):
        show_cols = [c for c in [
            "game_id", "mlb_game_pk", "pitcher_name", "vendor", "line_value",
            "snapshot_ts", state_ts_col, "state_lag_seconds",
            "strikeouts_so_far", "batters_faced_so_far", "pitches_thrown_so_far",
            "innings_completed", "pitcher_active_flag",
            "final_strikeouts", "market_over_prob", "market_under_prob",
            "realized_over", "realized_under", "realized_push"
        ] if c in valid.columns]
        print("\\nSAMPLE VALID ROWS:")
        print(valid[show_cols].head(20).to_string(index=False))

    if len(unmatched):
        show_cols = [c for c in [
            "game_id", "pitcher_name", "vendor", "line_value",
            "snapshot_ts", "home_team", "away_team"
        ] if c in unmatched.columns]
        print("\\nSAMPLE UNMATCHED ROWS:")
        print(unmatched[show_cols].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
