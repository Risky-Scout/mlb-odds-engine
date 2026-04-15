from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple
import math
import re
import unicodedata

import numpy as np
import pandas as pd
import requests


SCHEDULE_URL = "https://statsapi.mlb.com/api/v1/schedule"
LIVE_FEED_URL = "https://statsapi.mlb.com/api/v1/game/{game_pk}/feed/live"


@dataclass
class LivePitcherState:
    quote_game_id: int
    pitcher_name_norm: str
    strikeouts_so_far: int
    batters_faced_so_far: int
    pitches_thrown_so_far: int
    innings_completed: float
    pitcher_active_flag: int
    mlb_game_pk: int | None = None


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


def fetch_schedule_for_date(target_date: str, timeout: int = 20) -> list[dict]:
    r = requests.get(
        SCHEDULE_URL,
        params={"sportId": 1, "date": str(target_date)},
        timeout=timeout,
    )
    r.raise_for_status()
    payload = r.json()

    out = []
    for d in payload.get("dates", []) or []:
        for g in d.get("games", []) or []:
            out.append({
                "gamePk": g.get("gamePk"),
                "gameDate": g.get("gameDate"),
                "home_team": ((g.get("teams", {}) or {}).get("home", {}) or {}).get("team", {}).get("name"),
                "away_team": ((g.get("teams", {}) or {}).get("away", {}) or {}).get("team", {}).get("name"),
            })
    return out


def fetch_live_feed(game_pk: int, timeout: int = 20) -> dict:
    r = requests.get(LIVE_FEED_URL.format(game_pk=int(game_pk)), timeout=timeout)
    r.raise_for_status()
    return r.json()


def _extract_pitchers_from_feed(feed: dict, quote_game_id: int, mlb_game_pk: int) -> Dict[str, LivePitcherState]:
    out: Dict[str, LivePitcherState] = {}

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
            has_pitching = any(k in pitching for k in ["strikeOuts", "strikeouts", "inningsPitched", "battersFaced", "numberOfPitches"])
            if not has_pitching:
                continue

            strikeouts_so_far = _safe_int(pitching.get("strikeOuts", pitching.get("strikeouts", 0)), 0)
            batters_faced_so_far = _safe_int(pitching.get("battersFaced", pitching.get("battersfaced", 0)), 0)
            pitches_thrown_so_far = _safe_int(
                pitching.get("numberOfPitches", pitching.get("pitchesThrown", pitching.get("numberofpitches", 0))),
                0,
            )
            innings_completed = _parse_innings_pitched(
                pitching.get("inningsPitched", pitching.get("inningspitched", 0))
            )

            pid = person.get("id")
            pitcher_active_flag = int(_safe_int(pid, -1) == _safe_int(active_pitcher_id, -2))

            out[_norm(full_name)] = LivePitcherState(
                quote_game_id=int(quote_game_id),
                pitcher_name_norm=_norm(full_name),
                strikeouts_so_far=strikeouts_so_far,
                batters_faced_so_far=batters_faced_so_far,
                pitches_thrown_so_far=pitches_thrown_so_far,
                innings_completed=float(innings_completed),
                pitcher_active_flag=pitcher_active_flag,
                mlb_game_pk=int(mlb_game_pk),
            )

    return out


def build_live_state_lookup_from_quotes(quotes: pd.DataFrame, target_date: str) -> Dict[Tuple[int, str], LivePitcherState]:
    if quotes is None or quotes.empty:
        return {}

    q = quotes.copy()
    for c in ["home_team", "away_team", "player_name"]:
        if c in q.columns:
            q[c] = q[c].astype(str)

    q["player_name_norm"] = q["player_name"].map(_norm)
    if "home_team" in q.columns:
        q["home_team_norm"] = q["home_team"].map(_norm)
    else:
        q["home_team_norm"] = ""
    if "away_team" in q.columns:
        q["away_team_norm"] = q["away_team"].map(_norm)
    else:
        q["away_team_norm"] = ""

    sched = fetch_schedule_for_date(target_date)
    sched_df = pd.DataFrame(sched)
    if sched_df.empty:
        return {}

    sched_df["home_team_norm"] = sched_df["home_team"].map(_norm)
    sched_df["away_team_norm"] = sched_df["away_team"].map(_norm)

    out: Dict[Tuple[int, str], LivePitcherState] = {}

    unique_games = q[["game_id", "home_team_norm", "away_team_norm"]].drop_duplicates()

    for row in unique_games.itertuples(index=False):
        candidates = sched_df.loc[
            (sched_df["home_team_norm"] == row.home_team_norm) &
            (sched_df["away_team_norm"] == row.away_team_norm)
        ].copy()

        if candidates.empty:
            continue

        mlb_game_pk = candidates.iloc[0]["gamePk"]

        try:
            feed = fetch_live_feed(int(mlb_game_pk))
        except Exception:
            continue

        pitcher_states = _extract_pitchers_from_feed(feed, quote_game_id=int(row.game_id), mlb_game_pk=int(mlb_game_pk))
        if not pitcher_states:
            continue

        qg = q.loc[q["game_id"].eq(row.game_id)].copy()
        for r in qg.itertuples(index=False):
            key = (int(r.game_id), str(r.player_name_norm))
            st = pitcher_states.get(str(r.player_name_norm))
            if st is not None:
                out[key] = st

    return out


def _tilt_to_target_mean(base: np.ndarray, values: np.ndarray, target_mean: float) -> np.ndarray:
    base = np.asarray(base, dtype=float)
    values = np.asarray(values, dtype=float)

    base = np.clip(base, 1e-15, None)
    base /= base.sum()

    lo, hi = -20.0, 20.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        w = base * np.exp(mid * values)
        w /= w.sum()
        m = float(np.sum(values * w))
        if m > target_mean:
            hi = mid
        else:
            lo = mid

    w = base * np.exp(0.5 * (lo + hi) * values)
    w /= w.sum()
    return w


def condition_pmf_with_live_state(
    pmf: np.ndarray,
    live_state: LivePitcherState | None,
) -> np.ndarray:
    z = np.asarray(pmf, dtype=float).copy()
    z = np.clip(z, 1e-15, None)
    z /= z.sum()

    if live_state is None:
        return z

    k_so_far = max(0, int(live_state.strikeouts_so_far))
    max_k = len(z) - 1
    k_so_far = min(k_so_far, max_k)

    if live_state.pitcher_active_flag == 0:
        out = np.zeros_like(z)
        out[k_so_far] = 1.0
        return out

    ks = np.arange(len(z), dtype=int)
    z[ks < k_so_far] = 0.0
    if z.sum() <= 0:
        out = np.zeros_like(z)
        out[k_so_far] = 1.0
        return out
    z /= z.sum()

    add = ks - k_so_far
    add[add < 0] = 0

    base_add_mean = float(np.sum(add * z))

    progress_parts = []
    if live_state.innings_completed > 0:
        progress_parts.append(min(1.0, live_state.innings_completed / 9.0))
    if live_state.batters_faced_so_far > 0:
        progress_parts.append(min(1.0, live_state.batters_faced_so_far / 27.0))
    if live_state.pitches_thrown_so_far > 0:
        progress_parts.append(min(1.0, live_state.pitches_thrown_so_far / 95.0))

    progress = float(np.mean(progress_parts)) if progress_parts else 0.0
    progress = float(np.clip(progress, 0.0, 0.98))
    remaining_share = 1.0 - progress
    target_add_mean = base_add_mean * remaining_share

    if target_add_mean <= 1e-8:
        out = np.zeros_like(z)
        out[k_so_far] = 1.0
        return out

    if abs(target_add_mean - base_add_mean) < 1e-8:
        return z

    z = _tilt_to_target_mean(z, add.astype(float), target_add_mean)
    z /= z.sum()
    return z
