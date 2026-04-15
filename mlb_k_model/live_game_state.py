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
    capture_run_ts: str | None = None


def _norm(s) -> str:
    if s is None:
        return ""
    s = str(s)
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = s.lower().strip()
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s


def fetch_schedule_for_date(target_date: str, timeout: int = 20) -> pd.DataFrame:
    r = requests.get(
        SCHEDULE_URL,
        params={"sportId": 1, "date": str(target_date)},
        timeout=timeout,
    )
    r.raise_for_status()
    payload = r.json()

    rows = []
    for d in payload.get("dates", []) or []:
        for g in d.get("games", []) or []:
            rows.append({
                "mlb_game_pk": g.get("gamePk"),
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


def build_quote_to_mlb_game_mapping(quotes: pd.DataFrame, target_date: str) -> pd.DataFrame:
    q = quotes.copy()
    q["home_team_norm"] = q.get("home_team", "").map(_norm) if "home_team" in q.columns else ""
    q["away_team_norm"] = q.get("away_team", "").map(_norm) if "away_team" in q.columns else ""
    q["commence_time"] = pd.to_datetime(q.get("commence_time"), utc=True, errors="coerce")

    sched = fetch_schedule_for_date(target_date)
    if sched.empty:
        return pd.DataFrame(columns=["game_id", "mlb_game_pk"])

    rows = []
    unique_games = q[["game_id", "home_team_norm", "away_team_norm", "commence_time"]].drop_duplicates()

    for r in unique_games.itertuples(index=False):
        candidates = sched.copy()

        if getattr(r, "home_team_norm", "") and getattr(r, "away_team_norm", ""):
            team_match = candidates.loc[
                (candidates["home_team_norm"] == r.home_team_norm) &
                (candidates["away_team_norm"] == r.away_team_norm)
            ].copy()
            if not team_match.empty:
                candidates = team_match

        if pd.notna(getattr(r, "commence_time", pd.NaT)):
            candidates["time_diff"] = (candidates["gameDate"] - r.commence_time).abs()
            candidates = candidates.sort_values("time_diff")

        if len(candidates):
            rows.append({
                "game_id": int(r.game_id),
                "mlb_game_pk": int(candidates.iloc[0]["mlb_game_pk"]),
            })

    return pd.DataFrame(rows).drop_duplicates()


def build_live_state_lookup_from_quotes(quotes: pd.DataFrame, target_date: str, data_dir: str | Path) -> Dict[Tuple[int, str], LivePitcherState]:
    if quotes is None or quotes.empty:
        return {}

    data_dir = Path(data_dir)
    archive_path = data_dir / "snapshots" / "mlb_statsapi" / "live_state_archive" / f"mlb_live_state_{target_date}.parquet"
    if not archive_path.exists():
        return {}

    state_df = pd.read_parquet(archive_path).copy()
    if state_df.empty:
        return {}

    state_df["capture_run_ts"] = pd.to_datetime(state_df["capture_run_ts"], utc=True, errors="coerce")
    state_df["pitcher_name_norm"] = state_df["pitcher_name_norm"].map(_norm)

    q = quotes.copy()
    q["player_name_norm"] = q["player_name"].map(_norm)
    q["snapshot_ts"] = pd.to_datetime(q["snapshot_ts"], utc=True, errors="coerce")

    mapping = build_quote_to_mlb_game_mapping(q, target_date)
    if mapping.empty:
        return {}

    q = q.merge(mapping, on="game_id", how="left")
    q = q.dropna(subset=["mlb_game_pk", "snapshot_ts", "player_name_norm"]).copy()
    q["mlb_game_pk"] = q["mlb_game_pk"].astype(int)

    # Use the latest quote timestamp per (game_id, pitcher_name) and match to the latest prior state snapshot
    quote_groups = (
        q.groupby(["game_id", "mlb_game_pk", "player_name_norm"], as_index=False)["snapshot_ts"]
        .max()
        .rename(columns={"snapshot_ts": "quote_ts"})
    )

    out: Dict[Tuple[int, str], LivePitcherState] = {}
    tolerance = pd.Timedelta(seconds=120)

    for r in quote_groups.itertuples(index=False):
        cand = state_df.loc[
            (state_df["mlb_game_pk"] == int(r.mlb_game_pk)) &
            (state_df["pitcher_name_norm"] == str(r.player_name_norm)) &
            (state_df["capture_run_ts"] <= (r.quote_ts + tolerance))
        ].copy()

        if cand.empty:
            continue

        cand = cand.sort_values("capture_run_ts")
        row = cand.iloc[-1]

        out[(int(r.game_id), str(r.player_name_norm))] = LivePitcherState(
            quote_game_id=int(r.game_id),
            pitcher_name_norm=str(r.player_name_norm),
            strikeouts_so_far=int(row["strikeouts_so_far"]),
            batters_faced_so_far=int(row["batters_faced_so_far"]),
            pitches_thrown_so_far=int(row["pitches_thrown_so_far"]),
            innings_completed=float(row["innings_completed"]),
            pitcher_active_flag=int(row["pitcher_active_flag"]),
            mlb_game_pk=int(row["mlb_game_pk"]),
            capture_run_ts=str(pd.Timestamp(row["capture_run_ts"]).isoformat()),
        )

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
