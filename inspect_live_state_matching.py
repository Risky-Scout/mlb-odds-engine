from __future__ import annotations

from pathlib import Path
import argparse
import re
import unicodedata

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


def fetch_schedule_for_date(target_date: str) -> pd.DataFrame:
    r = requests.get(SCHEDULE_URL, params={"sportId": 1, "date": str(target_date)}, timeout=20)
    r.raise_for_status()
    payload = r.json()
    rows = []
    for d in payload.get("dates", []) or []:
        for g in d.get("games", []) or []:
            rows.append({
                "gamePk": g.get("gamePk"),
                "gameDate": g.get("gameDate"),
                "home_team": ((g.get("teams", {}) or {}).get("home", {}) or {}).get("team", {}).get("name"),
                "away_team": ((g.get("teams", {}) or {}).get("away", {}) or {}).get("team", {}).get("name"),
            })
    df = pd.DataFrame(rows)
    if len(df):
        df["gameDate"] = pd.to_datetime(df["gameDate"], utc=True, errors="coerce")
        df["home_team_norm"] = df["home_team"].map(_norm)
        df["away_team_norm"] = df["away_team"].map(_norm)
    return df


def fetch_live_feed(game_pk: int) -> dict:
    r = requests.get(LIVE_FEED_URL.format(game_pk=int(game_pk)), timeout=20)
    r.raise_for_status()
    return r.json()


def extract_feed_pitcher_names(feed: dict) -> list[str]:
    out = []
    teams = ((((feed or {}).get("liveData", {}) or {}).get("boxscore", {}) or {}).get("teams", {}) or {})
    for side in ("home", "away"):
        players = ((teams.get(side, {}) or {}).get("players", {}) or {})
        for _, rec in players.items():
            person = (rec.get("person", {}) or {})
            full_name = person.get("fullName")
            pitching = ((rec.get("stats", {}) or {}).get("pitching", {}) or {})
            has_pitching = any(
                k in pitching
                for k in ["strikeOuts", "strikeouts", "inningsPitched", "battersFaced", "numberOfPitches"]
            )
            if full_name and has_pitching:
                out.append(full_name)
    return sorted(set(out))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True)
    parser.add_argument("--data-dir", default="data_rebuild")
    args = parser.parse_args()

    archive_path = Path(args.data_dir) / "snapshots" / "player_props_odds_api" / "live_archive" / f"odds_api_live_capture_{args.date}.parquet"
    quotes = pd.read_parquet(archive_path).copy()
    quotes["player_name_norm"] = quotes["player_name"].map(_norm)
    quotes["home_team_norm"] = quotes["home_team"].map(_norm)
    quotes["away_team_norm"] = quotes["away_team"].map(_norm)
    quotes["commence_time"] = pd.to_datetime(quotes["commence_time"], utc=True, errors="coerce")

    print("QUOTE ROWS:")
    show_cols = [c for c in ["game_id","event_id","home_team","away_team","player_name","vendor","line_value","snapshot_ts","commence_time"] if c in quotes.columns]
    print(quotes[show_cols].drop_duplicates().to_string(index=False))

    sched = fetch_schedule_for_date(args.date)
    print("\nSCHEDULE:")
    print(sched[["gamePk","gameDate","home_team","away_team"]].to_string(index=False))

    print("\nMATCH DIAGNOSTIC:")
    for r in quotes[["game_id","home_team","away_team","home_team_norm","away_team_norm","player_name","player_name_norm","commence_time"]].drop_duplicates().itertuples(index=False):
        print("\n" + "=" * 100)
        print("QUOTE GAME ID:", r.game_id)
        print("QUOTE TEAMS:", r.away_team, "@", r.home_team)
        print("QUOTE PITCHER:", r.player_name)
        print("QUOTE COMMENCE:", r.commence_time)

        candidates = sched.copy()

        team_match = candidates.loc[
            (candidates["home_team_norm"] == r.home_team_norm) &
            (candidates["away_team_norm"] == r.away_team_norm)
        ].copy()

        if len(team_match):
            candidates = team_match
            print("TEAM MATCH CANDIDATES:", len(candidates))
        else:
            print("TEAM MATCH CANDIDATES: 0  -> using all schedule games")

        candidates["time_diff_min"] = ((candidates["gameDate"] - r.commence_time).abs().dt.total_seconds() / 60.0)
        candidates = candidates.sort_values("time_diff_min")

        print("\nTOP SCHEDULE CANDIDATES:")
        print(candidates[["gamePk","gameDate","home_team","away_team","time_diff_min"]].head(5).to_string(index=False))

        for cand in candidates.head(3).itertuples(index=False):
            try:
                feed = fetch_live_feed(int(cand.gamePk))
                names = extract_feed_pitcher_names(feed)
            except Exception as e:
                print(f"\nGAMEPK {cand.gamePk} FEED ERROR: {e}")
                continue

            print(f"\nGAMEPK {cand.gamePk} FEED PITCHERS:")
            for name in names:
                marker = " <== NAME MATCH" if _norm(name) == r.player_name_norm else ""
                print(f" - {name}{marker}")


if __name__ == "__main__":
    main()
