from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple
import math

import numpy as np
import requests


STATSAPI_LIVE_URL = "https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live"


@dataclass
class LivePitcherState:
    game_id: int
    pitcher_id: int
    strikeouts_so_far: int
    batters_faced_so_far: int
    pitches_thrown_so_far: int
    innings_completed: float
    pitcher_active_flag: int


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


def fetch_live_feed(game_pk: int, timeout: int = 20) -> dict:
    url = STATSAPI_LIVE_URL.format(game_pk=int(game_pk))
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r.json()


def _extract_pitcher_state(feed: dict, game_id: int, pitcher_id: int) -> LivePitcherState | None:
    pitcher_key = f"ID{int(pitcher_id)}"

    teams = (((feed or {}).get("liveData", {}) or {}).get("boxscore", {}) or {}).get("teams", {}) or {}
    player_rec = None
    for side in ("home", "away"):
        players = ((teams.get(side, {}) or {}).get("players", {}) or {})
        if pitcher_key in players:
            player_rec = players[pitcher_key]
            break

    if player_rec is None:
        return None

    pitching = ((player_rec.get("stats", {}) or {}).get("pitching", {}) or {})

    strikeouts_so_far = _safe_int(
        pitching.get("strikeOuts", pitching.get("strikeouts", 0)),
        0,
    )
    batters_faced_so_far = _safe_int(
        pitching.get("battersFaced", pitching.get("battersfaced", 0)),
        0,
    )
    pitches_thrown_so_far = _safe_int(
        pitching.get("numberOfPitches", pitching.get("pitchesThrown", pitching.get("numberofpitches", 0))),
        0,
    )
    innings_completed = _parse_innings_pitched(
        pitching.get("inningsPitched", pitching.get("inningspitched", 0))
    )

    defense_pitcher = ((((feed or {}).get("liveData", {}) or {}).get("linescore", {}) or {}).get("defense", {}) or {}).get("pitcher", {}) or {}
    active_pitcher_id = defense_pitcher.get("id")
    pitcher_active_flag = int(_safe_int(active_pitcher_id, -1) == int(pitcher_id))

    return LivePitcherState(
        game_id=int(game_id),
        pitcher_id=int(pitcher_id),
        strikeouts_so_far=strikeouts_so_far,
        batters_faced_so_far=batters_faced_so_far,
        pitches_thrown_so_far=pitches_thrown_so_far,
        innings_completed=float(innings_completed),
        pitcher_active_flag=pitcher_active_flag,
    )


def build_live_state_lookup(game_ids: Iterable[int], pitcher_ids: Iterable[int]) -> Dict[Tuple[int, int], LivePitcherState]:
    game_ids = sorted({int(g) for g in game_ids if g is not None})
    pitcher_ids = {int(p) for p in pitcher_ids if p is not None}

    out: Dict[Tuple[int, int], LivePitcherState] = {}

    for game_id in game_ids:
        try:
            feed = fetch_live_feed(game_id)
        except Exception:
            continue

        for pitcher_id in pitcher_ids:
            st = _extract_pitcher_state(feed, game_id=game_id, pitcher_id=pitcher_id)
            if st is not None:
                out[(int(game_id), int(pitcher_id))] = st

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
