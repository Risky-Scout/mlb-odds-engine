from __future__ import annotations

from pathlib import Path
import json
import re
import unicodedata
import pandas as pd

DATA_DIR = Path("data_rebuild")
RAW_DIR = DATA_DIR / "snapshots" / "odds_api_raw_archive"
OUT_DIR = DATA_DIR / "snapshots" / "player_props_odds_api"
OUT_DIR.mkdir(parents=True, exist_ok=True)

games = pd.read_parquet(DATA_DIR / "games.parquet").copy()
players = pd.read_parquet(DATA_DIR / "players.parquet").copy()

def norm(s):
    if s is None:
        return ""
    s = str(s)
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = s.lower().strip()
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s

# player lookup
player_name_col = None
for c in ["full_name", "display_name", "name"]:
    if c in players.columns:
        player_name_col = c
        break
if player_name_col is None and {"first_name", "last_name"}.issubset(players.columns):
    players["full_name_tmp"] = (
        players["first_name"].fillna("").astype(str).str.strip() + " " +
        players["last_name"].fillna("").astype(str).str.strip()
    )
    player_name_col = "full_name_tmp"

player_lookup = (
    players[["id", player_name_col]]
    .dropna()
    .drop_duplicates()
    .assign(player_name_norm=lambda d: d[player_name_col].map(norm))
)
name_to_player_ids = player_lookup.groupby("player_name_norm")["id"].apply(list).to_dict()

def match_player_id(player_name):
    ids = name_to_player_ids.get(norm(player_name), [])
    return int(ids[0]) if ids else None

# game lookup
games["date"] = pd.to_datetime(games["date"], utc=True, errors="coerce")
team_cols = [c for c in games.columns if any(k in c.lower() for k in ["home", "away", "visitor", "team"])]

records = []
for row in games.itertuples(index=False):
    rd = row._asdict()
    strings = set()
    for c in team_cols:
        v = rd.get(c)
        if isinstance(v, str) and v.strip():
            strings.add(norm(v))
    records.append({
        "game_id": int(rd["id"]),
        "game_date": pd.Timestamp(rd["date"]).date() if pd.notna(rd["date"]) else None,
        "team_strings": strings,
    })

games_by_date = {}
for rec in records:
    games_by_date.setdefault(rec["game_date"], []).append(rec)

def match_game_id(commence_time, home_team, away_team):
    dt = pd.to_datetime(commence_time, utc=True, errors="coerce")
    if pd.isna(dt):
        return None
    day = dt.date()
    home_n = norm(home_team)
    away_n = norm(away_team)

    candidates = games_by_date.get(day, [])
    if not candidates:
        candidates = (
            games_by_date.get((dt - pd.Timedelta(days=1)).date(), []) +
            games_by_date.get((dt + pd.Timedelta(days=1)).date(), [])
        )

    matched = []
    for rec in candidates:
        s = rec["team_strings"]
        if home_n in s and away_n in s:
            matched.append(rec["game_id"])

    return matched[0] if matched else None

def fetched_ts_from_name(path: Path):
    m = re.search(r"__fetched_(\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}Z)\.json$", path.name)
    if not m:
        return pd.NaT
    ts = m.group(1).replace("-", ":", 2).replace("-", ":", 1)  # not used
    raw = m.group(1).replace("T", "T").replace("Z", "Z")
    # easier parse:
    safe = m.group(1)
    safe = safe.replace("T", " ")
    # rebuild YYYY-MM-DD HH:MM:SSZ from YYYY-MM-DDTHH-MM-SSZ
    dpart, tpart = m.group(1).split("T")
    tpart = tpart[:-1]  # drop Z
    hh, mm, ss = tpart.split("-")
    real = f"{dpart}T{hh}:{mm}:{ss}Z"
    return pd.to_datetime(real, utc=True, errors="coerce")

rows = []
files = sorted(RAW_DIR.glob("*.json"))
print(f"Scanning {len(files)} archived raw JSON files")

for fp in files:
    fetched_ts = fetched_ts_from_name(fp)

    try:
        payload = json.loads(fp.read_text())
    except Exception:
        continue

    events = payload if isinstance(payload, list) else [payload]

    for event in events:
        if not isinstance(event, dict):
            continue

        event_id = event.get("id")
        commence_time = event.get("commence_time")
        home_team = event.get("home_team")
        away_team = event.get("away_team")
        game_id = match_game_id(commence_time, home_team, away_team)

        for book in event.get("bookmakers", []) or []:
            vendor = book.get("key") or book.get("title")
            book_last = book.get("last_update")

            for market in book.get("markets", []) or []:
                prop_type = market.get("key")
                if prop_type != "pitcher_strikeouts":
                    continue

                updated_at = market.get("last_update") or book_last
                grouped = {}

                for outcome in market.get("outcomes", []) or []:
                    side = str(outcome.get("name", "")).strip().lower()
                    player_name = outcome.get("description") or outcome.get("participant") or outcome.get("label")
                    line_value = outcome.get("point")
                    price = outcome.get("price")

                    if player_name is None or line_value is None:
                        continue

                    key = (player_name, float(line_value))
                    rec = grouped.setdefault(key, {
                        "event_id": event_id,
                        "game_id": match_game_id(commence_time, home_team, away_team),
                        "player_name": player_name,
                        "player_id": match_player_id(player_name),
                        "vendor": vendor,
                        "prop_type": prop_type,
                        "line_value": float(line_value),
                        "updated_at": updated_at,
                        "fetched_at": fetched_ts,
                        "commence_time": commence_time,
                        "home_team": home_team,
                        "away_team": away_team,
                        "over_odds": None,
                        "under_odds": None,
                    })

                    if side == "over":
                        rec["over_odds"] = price
                    elif side == "under":
                        rec["under_odds"] = price

                rows.extend(grouped.values())

snap = pd.DataFrame(rows)
if snap.empty:
    raise SystemExit("No normalized primary market rows were created.")

snap["updated_at"] = pd.to_datetime(snap["updated_at"], utc=True, errors="coerce")
snap["fetched_at"] = pd.to_datetime(snap["fetched_at"], utc=True, errors="coerce")
snap["commence_time"] = pd.to_datetime(snap["commence_time"], utc=True, errors="coerce")

# Use fetched_at as the true snapshot timestamp for calibration/eval
snap["snapshot_ts"] = snap["fetched_at"]
snap["snapshot_date"] = snap["snapshot_ts"].dt.normalize()
snap["is_live_snapshot"] = (snap["snapshot_ts"] >= snap["commence_time"]).astype(int)

for c in ["game_id", "player_id", "line_value", "over_odds", "under_odds"]:
    snap[c] = pd.to_numeric(snap[c], errors="coerce")

snap = snap.dropna(subset=["game_id", "player_id", "line_value", "snapshot_ts", "commence_time"])
snap = snap.drop_duplicates(
    subset=["event_id", "vendor", "prop_type", "player_name", "line_value", "snapshot_ts"],
    keep="last"
)

out_path = OUT_DIR / "odds_api_historical_normalized.parquet"
snap.to_parquet(out_path, index=False)

print(f"Wrote {out_path}")
print("rows:", len(snap))
print("\nsnapshot dates:")
print(snap["snapshot_date"].dt.strftime("%Y-%m-%d").value_counts().sort_index().to_string())
print("\nvendors:")
print(snap["vendor"].value_counts(dropna=False).head(20).to_string())
