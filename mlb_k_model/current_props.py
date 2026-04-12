from __future__ import annotations

import os
import re
import unicodedata
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests


def _norm(s):
    if s is None:
        return ""
    s = str(s)
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = s.lower().strip()
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s


def _build_player_lookup(players: pd.DataFrame):
    player_name_col = None
    for c in ["full_name", "display_name", "name"]:
        if c in players.columns:
            player_name_col = c
            break
    if player_name_col is None and {"first_name", "last_name"}.issubset(players.columns):
        players = players.copy()
        players["full_name_tmp"] = (
            players["first_name"].fillna("").astype(str).str.strip() + " " +
            players["last_name"].fillna("").astype(str).str.strip()
        )
        player_name_col = "full_name_tmp"

    lookup = (
        players[["id", player_name_col]]
        .dropna()
        .drop_duplicates()
        .assign(player_name_norm=lambda d: d[player_name_col].map(_norm))
    )
    return lookup.groupby("player_name_norm")["id"].apply(list).to_dict()


def _match_player_id(name_to_ids, player_name):
    ids = name_to_ids.get(_norm(player_name), [])
    return int(ids[0]) if ids else None


def _build_game_records(games: pd.DataFrame):
    games = games.copy()
    games["date"] = pd.to_datetime(games["date"], utc=True, errors="coerce")

    team_cols = [c for c in games.columns if any(k in c.lower() for k in ["home", "away", "visitor", "team"])]
    records = []
    for row in games.itertuples(index=False):
        rd = row._asdict()
        strings = set()
        for c in team_cols:
            v = rd.get(c)
            if isinstance(v, str) and v.strip():
                strings.add(_norm(v))
        gid = rd.get("id", rd.get("game_id"))
        if gid is None:
            continue
        records.append({
            "game_id": int(gid),
            "game_date": pd.Timestamp(rd["date"]).date() if pd.notna(rd["date"]) else None,
            "team_strings": strings,
        })
    games_by_date = {}
    for rec in records:
        games_by_date.setdefault(rec["game_date"], []).append(rec)
    return games_by_date


def _match_game_id(games_by_date, commence_time, home_team, away_team):
    dt = pd.to_datetime(commence_time, utc=True, errors="coerce")
    if pd.isna(dt):
        return None
    day = dt.date()
    home_n = _norm(home_team)
    away_n = _norm(away_team)

    candidates = games_by_date.get(day, [])
    if not candidates:
        candidates = games_by_date.get((dt - pd.Timedelta(days=1)).date(), []) + games_by_date.get((dt + pd.Timedelta(days=1)).date(), [])

    for rec in candidates:
        s = rec["team_strings"]
        if home_n in s and away_n in s:
            return rec["game_id"]
    return None


def fetch_current_pitcher_strikeouts_snapshot(data_dir: str | Path, target_date: str, regions: str = "us") -> pd.DataFrame:
    data_dir = Path(data_dir)
    api_key = os.environ.get("ODDS_API_KEY")
    if not api_key:
        raise SystemExit("ODDS_API_KEY is not set.")

    games = pd.read_parquet(data_dir / "games.parquet")
    players = pd.read_parquet(data_dir / "players.parquet")
    name_to_ids = _build_player_lookup(players)
    games_by_date = _build_game_records(games)

    out_dir = data_dir / "snapshots" / "player_props_odds_api"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "odds_api_current_normalized.parquet"

    events_url = "https://api.the-odds-api.com/v4/sports/baseball_mlb/events"
    events_params = {"apiKey": api_key, "dateFormat": "iso"}

    r = requests.get(events_url, params=events_params, timeout=30)
    if r.status_code != 200:
        raise SystemExit(f"Events endpoint failed: {r.status_code} {r.text[:500]}")

    events = r.json()
    ny_tz = ZoneInfo("America/New_York")
    target_events = []
    for ev in events:
        ct = pd.to_datetime(ev.get("commence_time"), utc=True, errors="coerce")
        if pd.isna(ct):
            continue
        local_date = ct.tz_convert(ny_tz).date().isoformat()
        if local_date == target_date:
            target_events.append(ev)

    all_rows = []
    for ev in target_events:
        event_id = ev.get("id")
        odds_url = f"https://api.the-odds-api.com/v4/sports/baseball_mlb/events/{event_id}/odds"
        odds_params = {
            "apiKey": api_key,
            "regions": regions,
            "markets": "pitcher_strikeouts",
            "oddsFormat": "american",
            "dateFormat": "iso",
        }
        r2 = requests.get(odds_url, params=odds_params, timeout=30)
        if r2.status_code != 200:
            continue
        body = r2.json()

        for book in body.get("bookmakers", []) or []:
            vendor = book.get("key") or book.get("title")
            book_last = book.get("last_update")

            for market in book.get("markets", []) or []:
                if market.get("key") != "pitcher_strikeouts":
                    continue
                updated_at = market.get("last_update") or book_last
                grouped = {}
                for outcome in market.get("outcomes", []) or []:
                    side = str(outcome.get("name", "")).strip().lower()
                    player_name = outcome.get("description")
                    line_value = outcome.get("point")
                    price = outcome.get("price")
                    if player_name is None or line_value is None:
                        continue

                    key = (player_name, float(line_value))
                    rec = grouped.setdefault(key, {
                        "event_id": body.get("id"),
                        "game_id": _match_game_id(games_by_date, body.get("commence_time"), body.get("home_team"), body.get("away_team")),
                        "player_name": player_name,
                        "player_id": _match_player_id(name_to_ids, player_name),
                        "vendor": vendor,
                        "prop_type": "pitcher_strikeouts",
                        "line_value": float(line_value),
                        "updated_at": updated_at,
                        "commence_time": body.get("commence_time"),
                        "home_team": body.get("home_team"),
                        "away_team": body.get("away_team"),
                        "over_odds": None,
                        "under_odds": None,
                    })
                    if side == "over":
                        rec["over_odds"] = price
                    elif side == "under":
                        rec["under_odds"] = price

                all_rows.extend(grouped.values())

    snap = pd.DataFrame(all_rows)
    if snap.empty:
        empty = pd.DataFrame(columns=[
            "event_id", "game_id", "player_name", "player_id", "vendor", "prop_type",
            "line_value", "updated_at", "commence_time", "home_team", "away_team",
            "over_odds", "under_odds", "snapshot_ts", "snapshot_date", "is_live_snapshot"
        ])
        empty.to_parquet(out_path, index=False)
        return empty

    snap["updated_at"] = pd.to_datetime(snap["updated_at"], utc=True, errors="coerce")
    snap["commence_time"] = pd.to_datetime(snap["commence_time"], utc=True, errors="coerce")
    snap["snapshot_ts"] = snap["updated_at"]
    snap["snapshot_date"] = snap["snapshot_ts"].dt.normalize()
    snap["is_live_snapshot"] = (snap["snapshot_ts"] >= snap["commence_time"]).astype(int)

    for c in ["game_id", "player_id", "line_value", "over_odds", "under_odds"]:
        snap[c] = pd.to_numeric(snap[c], errors="coerce")

    snap = snap.dropna(subset=["game_id", "player_id", "line_value", "snapshot_ts", "commence_time"])
    snap = snap.drop_duplicates(
        subset=["event_id", "vendor", "prop_type", "player_name", "line_value", "snapshot_ts"],
        keep="last",
    )

    snap.to_parquet(out_path, index=False)
    return snap
