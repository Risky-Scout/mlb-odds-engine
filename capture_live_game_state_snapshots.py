from __future__ import annotations

import argparse
import math
import re
import unicodedata
from pathlib import Path

import pandas as pd
import requests


SCHEDULE_URL = "https://statsapi.mlb.com/api/v1/schedule"
LIVE_FEED_URL = "https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live"


def _norm(s) -> str:
    if s is None:
        return ""
    s = str(s)
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = s.lower().strip()
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s


def _safe_int(x, default: int = 0) -> int:
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return default
        return int(x)
    except Exception:
        return default


def _parse_innings_pitched(x) -> float:
    if x is None:
        return 0.0
    s = str(x).strip()
    if not s:
        return 0.0
    if "." not in s:
        try:
            return float(int(s))
        except Exception:
            return 0.0
    whole, frac = s.split(".", 1)
    try:
        whole_i = int(whole)
    except Exception:
        whole_i = 0
    try:
        outs = int(frac[:1])
    except Exception:
        outs = 0
    outs = max(0, min(outs, 2))
    return whole_i + outs / 3.0


def fetch_schedule_for_date(target_date: str) -> pd.DataFrame:
    r = requests.get(
        SCHEDULE_URL,
        params={"sportId": 1, "date": str(target_date)},
        timeout=20,
    )
    r.raise_for_status()
    payload = r.json()

    rows = []
    for d in payload.get("dates", []) or []:
        for g in d.get("games", []) or []:
            rows.append({
                "mlb_game_pk": g.get("gamePk"),
                "game_date": g.get("gameDate"),
                "home_team": ((g.get("teams", {}) or {}).get("home", {}) or {}).get("team", {}).get("name"),
                "away_team": ((g.get("teams", {}) or {}).get("away", {}) or {}).get("team", {}).get("name"),
            })
    df = pd.DataFrame(rows)
    if len(df):
        df["game_date"] = pd.to_datetime(df["game_date"], utc=True, errors="coerce")
    return df


def fetch_live_feed(game_pk: int) -> dict:
    r = requests.get(LIVE_FEED_URL.format(game_pk=int(game_pk)), timeout=20)
    r.raise_for_status()
    return r.json()


def extract_pitcher_rows(feed: dict, sched_row: pd.Series, capture_run_ts: pd.Timestamp) -> list[dict]:
    out = []

    teams = ((((feed or {}).get("liveData", {}) or {}).get("boxscore", {}) or {}).get("teams", {}) or {})
    defense_pitcher = ((((feed or {}).get("liveData", {}) or {}).get("linescore", {}) or {}).get("defense", {}) or {}).get("pitcher", {}) or {}
    active_pitcher_id = defense_pitcher.get("id")

    for side in ("home", "away"):
        players = ((teams.get(side, {}) or {}).get("players", {}) or {})
        for _, rec in players.items():
            person = (rec.get("person", {}) or {})
            full_name = person.get("fullName")
            if not full_name:
                continue

            pitching = ((rec.get("stats", {}) or {}).get("pitching", {}) or {})
            has_pitching = any(
                k in pitching
                for k in ["strikeOuts", "strikeouts", "inningsPitched", "battersFaced", "numberOfPitches"]
            )
            if not has_pitching:
                continue

            pid = person.get("id")
            out.append({
                "capture_run_ts": capture_run_ts,
                "mlb_game_pk": int(sched_row["mlb_game_pk"]),
                "game_date": sched_row["game_date"],
                "home_team": sched_row["home_team"],
                "away_team": sched_row["away_team"],
                "pitcher_name": full_name,
                "pitcher_name_norm": _norm(full_name),
                "mlb_pitcher_id": _safe_int(pid, 0),
                "strikeouts_so_far": _safe_int(pitching.get("strikeOuts", pitching.get("strikeouts", 0)), 0),
                "batters_faced_so_far": _safe_int(pitching.get("battersFaced", pitching.get("battersfaced", 0)), 0),
                "pitches_thrown_so_far": _safe_int(
                    pitching.get("numberOfPitches", pitching.get("pitchesThrown", pitching.get("numberofpitches", 0))),
                    0,
                ),
                "innings_completed": _parse_innings_pitched(
                    pitching.get("inningsPitched", pitching.get("inningspitched", 0))
                ),
                "pitcher_active_flag": int(_safe_int(pid, -1) == _safe_int(active_pitcher_id, -2)),
            })

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture MLB live pitcher state snapshots for temporal alignment.")
    parser.add_argument("--date", required=True)
    parser.add_argument("--data-dir", default="data_rebuild")
    args = parser.parse_args()

    sched = fetch_schedule_for_date(args.date)
    if sched.empty:
        raise SystemExit(f"No MLB schedule rows found for {args.date}")

    capture_run_ts = pd.Timestamp.utcnow()
    rows = []

    for _, r in sched.iterrows():
        try:
            feed = fetch_live_feed(int(r["mlb_game_pk"]))
        except Exception:
            continue
        rows.extend(extract_pitcher_rows(feed, r, capture_run_ts))

    out_df = pd.DataFrame(rows)
    archive_dir = Path(args.data_dir) / "snapshots" / "mlb_statsapi" / "live_state_archive"
    archive_dir.mkdir(parents=True, exist_ok=True)
    archive_path = archive_dir / f"mlb_live_state_{args.date}.parquet"

    if archive_path.exists():
        prev = pd.read_parquet(archive_path)
        out_df = pd.concat([prev, out_df], ignore_index=True, sort=False)

    dedupe_cols = [c for c in [
        "capture_run_ts", "mlb_game_pk", "pitcher_name_norm",
        "strikeouts_so_far", "batters_faced_so_far", "pitches_thrown_so_far",
        "innings_completed", "pitcher_active_flag"
    ] if c in out_df.columns]

    if dedupe_cols:
        out_df = out_df.drop_duplicates(subset=dedupe_cols, keep="last").reset_index(drop=True)

    out_df.to_parquet(archive_path, index=False)

    print("WROTE:", archive_path)
    print("ROWS:", len(out_df))
    if len(out_df):
        print(out_df.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
