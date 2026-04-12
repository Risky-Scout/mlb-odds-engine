
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
import requests

from .api_clients import BallDontLieClient


ROLL_WINDOWS = (15, 30, 60)


def _safe_mean(values: Sequence[float]) -> float:
    arr = pd.Series(values, dtype=float).dropna()
    return float(arr.mean()) if not arr.empty else np.nan


def _last_or_nan(values: Sequence[float]) -> float:
    arr = pd.Series(values, dtype=float).dropna()
    return float(arr.iloc[-1]) if not arr.empty else np.nan


def _normalize_side(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip().upper()
    if text.startswith("L"):
        return "L"
    if text.startswith("R"):
        return "R"
    return text[:1] if text else None


def _map_result(result: Optional[str]) -> Dict[str, int]:
    text = str(result or "").lower()
    return {
        "is_strikeout": int("strikeout" in text),
        "is_walk": int("walk" in text and "intentional" not in text),
        "is_hit": int(any(x in text for x in ["single", "double", "triple", "home run", "homered"])),
        "is_hr": int("home run" in text or "homered" in text),
        "is_out": int(any(x in text for x in ["out", "strikeout", "double play", "fielders choice"])),
    }


@dataclass
class DataBuilder:
    bdl: BallDontLieClient
    root: Path

    def collect_games(
        self,
        *,
        seasons: Optional[Iterable[int]] = None,
        dates: Optional[Iterable[str]] = None,
        season_type: str = "regular",
        save_name: str = "games.parquet",
    ) -> pd.DataFrame:
        rows = self.bdl.list_games(seasons=seasons, dates=dates, season_type=season_type)
        games = pd.json_normalize(rows) if rows else pd.DataFrame()
        if not games.empty:
            path = self.root / save_name
            path.parent.mkdir(parents=True, exist_ok=True)
            games.to_parquet(path, index=False)
        return games

    def collect_plate_appearances(self, game_ids: Iterable[int], *, save_name: str = "plate_appearances.parquet") -> pd.DataFrame:
        frames: List[pd.DataFrame] = []

        for game_id in game_ids:
            try:
                rows = self.bdl.get_plate_appearances(int(game_id))
            except requests.HTTPError as e:
                status = getattr(e.response, "status_code", "unknown")
                print(f"Skipping game_id={game_id}: plate_appearances HTTP {status}")
                continue
            except requests.RequestException as e:
                print(f"Skipping game_id={game_id}: plate_appearances request failed: {e}")
                continue

            if not rows:
                continue

            try:
                frames.append(FeatureBuilder.flatten_plate_appearances(rows, int(game_id)))
            except Exception as e:
                print(f"Skipping game_id={game_id}: flatten failed: {e}")
                continue

        pa = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        if not pa.empty:
            path = self.root / save_name
            path.parent.mkdir(parents=True, exist_ok=True)
            pa.to_parquet(path, index=False)
        return pa

    def collect_players(self, player_ids: Iterable[int], *, save_name: str = "players.parquet", batch_size: int = 100) -> pd.DataFrame:
        unique_ids = sorted({int(x) for x in player_ids if pd.notna(x)})
        rows: List[Dict] = []

        for i in range(0, len(unique_ids), batch_size):
            chunk = unique_ids[i : i + batch_size]
            rows.extend(self.bdl.list_players(player_ids=chunk, per_page=100))

        players = pd.json_normalize(rows).drop_duplicates("id") if rows else pd.DataFrame()
        if not players.empty:
            path = self.root / save_name
            path.parent.mkdir(parents=True, exist_ok=True)
            players.to_parquet(path, index=False)
        return players

    def collect_stats(
        self,
        *,
        game_ids: Optional[Iterable[int]] = None,
        player_ids: Optional[Iterable[int]] = None,
        seasons: Optional[Iterable[int]] = None,
        save_name: str = "stats.parquet",
    ) -> pd.DataFrame:
        rows = self.bdl.get_stats(game_ids=game_ids, player_ids=player_ids, seasons=seasons)
        frame = pd.json_normalize(rows) if rows else pd.DataFrame()
        if not frame.empty:
            path = self.root / save_name
            path.parent.mkdir(parents=True, exist_ok=True)
            frame.to_parquet(path, index=False)
        return frame

    def collect_season_stats(self, seasons: Iterable[int], *, season_type: str = "regular", out_subdir: str = "season_stats/players") -> pd.DataFrame:
        frames: List[pd.DataFrame] = []
        for season in seasons:
            rows = self.bdl.get_season_stats(season=int(season), season_type=season_type)
            frame = pd.json_normalize(rows) if rows else pd.DataFrame()
            if frame.empty:
                continue
            frame["season"] = int(season)
            path = self.root / out_subdir / f"{int(season)}.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            frame.to_parquet(path, index=False)
            frames.append(frame)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def collect_team_season_stats(self, seasons: Iterable[int], *, season_type: str = "regular", out_subdir: str = "season_stats/teams") -> pd.DataFrame:
        frames: List[pd.DataFrame] = []
        for season in seasons:
            rows = self.bdl.get_team_season_stats(season=int(season), season_type=season_type)
            frame = pd.json_normalize(rows) if rows else pd.DataFrame()
            if frame.empty:
                continue
            frame["season"] = int(season)
            path = self.root / out_subdir / f"{int(season)}.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            frame.to_parquet(path, index=False)
            frames.append(frame)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def collect_injuries(self, *, player_ids: Optional[Iterable[int]] = None, team_ids: Optional[Iterable[int]] = None, save_name: str = "injuries.parquet") -> pd.DataFrame:
        rows = self.bdl.get_injuries(player_ids=player_ids, team_ids=team_ids)
        frame = pd.json_normalize(rows) if rows else pd.DataFrame()
        if not frame.empty:
            path = self.root / save_name
            path.parent.mkdir(parents=True, exist_ok=True)
            frame.to_parquet(path, index=False)
        return frame

    def snapshot_game_odds(self, date: str, *, out_subdir: str = "snapshots/game_odds") -> pd.DataFrame:
        rows = self.bdl.get_game_odds(dates=[date])
        frame = pd.json_normalize(rows) if rows else pd.DataFrame()
        if not frame.empty:
            frame["snapshot_date"] = date
            path = self.root / out_subdir / f"{date}.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            frame.to_parquet(path, index=False)
        return frame

    def snapshot_player_props(
        self,
        date: str,
        *,
        prop_type: str = "pitcher_strikeouts",
        out_subdir: str = "snapshots/player_props",
    ) -> pd.DataFrame:
        games = self.bdl.list_games(dates=[date], season_type="regular")
        out: List[Dict] = []

        for game in games:
            for quote in self.bdl.get_player_props(int(game["id"]), prop_type=prop_type):
                row = quote.__dict__.copy()
                row["snapshot_date"] = date
                out.append(row)

        frame = pd.DataFrame(out)
        if not frame.empty:
            path = self.root / out_subdir / f"{date}_{prop_type}.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            frame.to_parquet(path, index=False)
        return frame

    def snapshot_lineups(self, date: str, *, out_subdir: str = "snapshots/lineups") -> pd.DataFrame:
        games = self.bdl.list_games(dates=[date], season_type="regular")
        game_ids = [int(g["id"]) for g in games]
        frame = pd.json_normalize(self.bdl.get_lineups(game_ids)) if game_ids else pd.DataFrame()
        if not frame.empty:
            frame["snapshot_date"] = date
            path = self.root / out_subdir / f"{date}.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            frame.to_parquet(path, index=False)
        return frame


class FeatureBuilder:
    @staticmethod
    def flatten_plate_appearances(rows: List[Dict], game_id: int) -> pd.DataFrame:
        out: List[Dict] = []

        for pa in rows:
            pitches = pa.get("pitches") or []
            last_pitch = pitches[-1] if pitches else {}

            pitch_count = len(pitches)
            avg_velo = _safe_mean([p.get("release_speed") for p in pitches])
            avg_spin = _safe_mean([p.get("spin_rate") for p in pitches])
            avg_extension = _safe_mean([p.get("release_extension") for p in pitches])
            avg_plate_x = _safe_mean([p.get("plate_x") for p in pitches])
            avg_plate_z = _safe_mean([p.get("plate_z") for p in pitches])
            avg_ivb = _safe_mean([p.get("induced_vertical_break") for p in pitches])
            avg_hmov = _safe_mean([p.get("horizontal_movement") for p in pitches])
            pitch_types = [str(p.get("pitch_type_code", "")) for p in pitches if p.get("pitch_type_code")]
            fastball_share = float(np.mean([pt in {"FF", "SI", "FC"} for pt in pitch_types])) if pitch_types else np.nan
            breaking_share = float(np.mean([pt in {"SL", "CU", "KC", "SV"} for pt in pitch_types])) if pitch_types else np.nan
            offspeed_share = float(np.mean([pt in {"CH", "FS", "FO", "SC"} for pt in pitch_types])) if pitch_types else np.nan
            strikes_seen = sum(1 for p in pitches if p.get("pitch_call") in {"S", "X"})
            zone_rate = float(np.mean([(p.get("strike_zone") or 0) in {1,2,3,4,5,6,7,8,9} for p in pitches])) if pitches else np.nan

            result_flags = _map_result(pa.get("result"))
            out.append(
                {
                    "game_id": int(game_id),
                    "batter_id": int(pa["batter_id"]) if pa.get("batter_id") is not None else np.nan,
                    "pitcher_id": int(pa["pitcher_id"]) if pa.get("pitcher_id") is not None else np.nan,
                    "inning": pa.get("inning"),
                    "half_inning": pa.get("half_inning"),
                    "pa_number": pa.get("pa_number"),
                    "outs_start": pa.get("outs"),
                    "batter_side": _normalize_side(pa.get("batter_side")),
                    "pitcher_hand": _normalize_side(pa.get("pitcher_hand")),
                    "runner_on_first": pa.get("runner_on_first"),
                    "runner_on_second": pa.get("runner_on_second"),
                    "runner_on_third": pa.get("runner_on_third"),
                    "result": pa.get("result"),
                    "pitch_count_pa": pitch_count,
                    "avg_release_speed": avg_velo,
                    "avg_spin_rate": avg_spin,
                    "avg_release_extension": avg_extension,
                    "avg_plate_x": avg_plate_x,
                    "avg_plate_z": avg_plate_z,
                    "avg_induced_vertical_break": avg_ivb,
                    "avg_horizontal_movement": avg_hmov,
                    "fastball_share_pa": fastball_share,
                    "breaking_share_pa": breaking_share,
                    "offspeed_share_pa": offspeed_share,
                    "zone_rate_pa": zone_rate,
                    "strikes_seen_pa": strikes_seen,
                    "last_pitch_number": last_pitch.get("pitch_number"),
                    "last_balls": last_pitch.get("balls"),
                    "last_strikes": last_pitch.get("strikes"),
                    "last_pitch_type_code": last_pitch.get("pitch_type_code"),
                    "last_release_speed": last_pitch.get("release_speed"),
                    "last_spin_rate": last_pitch.get("spin_rate"),
                    "last_release_extension": last_pitch.get("release_extension"),
                    "last_plate_x": last_pitch.get("plate_x"),
                    "last_plate_z": last_pitch.get("plate_z"),
                    "last_strike_zone": last_pitch.get("strike_zone"),
                    "last_hmov": last_pitch.get("horizontal_movement"),
                    "last_vmov": last_pitch.get("vertical_movement"),
                    "last_ivb": last_pitch.get("induced_vertical_break"),
                    "last_exit_velocity": last_pitch.get("exit_velocity"),
                    "last_launch_angle": last_pitch.get("launch_angle"),
                    "last_xba_contact": last_pitch.get("expected_batting_average"),
                    "last_is_barrel": int(bool(last_pitch.get("is_barrel"))) if last_pitch.get("is_barrel") is not None else 0,
                    "game_pitch_count": last_pitch.get("game_pitch_count"),
                    "pitcher_pitch_count": last_pitch.get("pitcher_pitch_count"),
                    **result_flags,
                }
            )

        frame = pd.DataFrame(out)
        if frame.empty:
            return frame

        frame = frame.sort_values(["game_id", "pa_number"]).reset_index(drop=True)
        return frame

    @staticmethod
    def _attach_game_context(pa_df: pd.DataFrame, games_df: pd.DataFrame) -> pd.DataFrame:
        games = games_df.copy()
        games["game_date"] = pd.to_datetime(games["date"]).dt.normalize()

        keep = games[
            [
                "id",
                "game_date",
                "season",
                "season_type",
                "home_team.id",
                "away_team.id",
                "home_team.abbreviation",
                "away_team.abbreviation",
                "home_team_name",
                "away_team_name",
                "venue",
                "status",
            ]
        ].rename(columns={"id": "game_id"})

        frame = pa_df.merge(keep, on="game_id", how="left")
        frame["is_top_half"] = frame["half_inning"].astype(str).str.lower().eq("top")
        frame["batting_team_id"] = np.where(frame["is_top_half"], frame["away_team.id"], frame["home_team.id"])
        frame["fielding_team_id"] = np.where(frame["is_top_half"], frame["home_team.id"], frame["away_team.id"])
        frame["is_home_pitcher"] = (frame["fielding_team_id"] == frame["home_team.id"]).astype(int)
        return frame

    @staticmethod
    def _rolling_rates(frame: pd.DataFrame, entity: str, windows: Sequence[int]) -> pd.DataFrame:
        frame = frame.sort_values(["game_date", "game_id", "pa_number"]).copy()

        target_cols = ["is_strikeout", "is_walk", "is_hit", "is_out", "pitch_count_pa", "avg_release_speed", "avg_spin_rate", "avg_release_extension", "last_xba_contact", "last_is_barrel"]
        grouped = frame.groupby(entity, group_keys=False)

        for window in windows:
            shifted = grouped[target_cols].shift(1)
            roll = shifted.groupby(frame[entity]).rolling(window, min_periods=max(5, window // 4)).mean().reset_index(level=0, drop=True)

            prefix = "pitcher" if entity == "pitcher_id" else "batter"
            frame[f"{prefix}_k_rate_{window}"] = roll["is_strikeout"]
            frame[f"{prefix}_walk_rate_{window}"] = roll["is_walk"]
            frame[f"{prefix}_hit_rate_{window}"] = roll["is_hit"]
            frame[f"{prefix}_out_rate_{window}"] = roll["is_out"]
            frame[f"{prefix}_pa_pitch_ct_{window}"] = roll["pitch_count_pa"]
            frame[f"{prefix}_velo_{window}"] = roll["avg_release_speed"]
            frame[f"{prefix}_spin_{window}"] = roll["avg_spin_rate"]
            frame[f"{prefix}_extension_{window}"] = roll["avg_release_extension"]
            frame[f"{prefix}_xba_contact_{window}"] = roll["last_xba_contact"]
            frame[f"{prefix}_barrel_rate_{window}"] = roll["last_is_barrel"]

        return frame

    @staticmethod
    def _days_rest(frame: pd.DataFrame) -> pd.DataFrame:
        starts = (
            frame.groupby(["pitcher_id", "game_id", "game_date"], as_index=False)
            .agg(first_pa=("pa_number", "min"))
            .sort_values(["pitcher_id", "game_date", "game_id"])
        )
        starts["days_rest"] = starts.groupby("pitcher_id")["game_date"].diff().dt.days
        return frame.merge(starts[["pitcher_id", "game_id", "days_rest"]], on=["pitcher_id", "game_id"], how="left")

    @staticmethod
    def _game_state_features(frame: pd.DataFrame) -> pd.DataFrame:
        frame = frame.sort_values(["game_id", "pitcher_id", "pa_number"]).copy()
        by_pg = frame.groupby(["game_id", "pitcher_id"], group_keys=False)

        frame["bf_so_far"] = by_pg.cumcount()
        frame["k_so_far"] = by_pg["is_strikeout"].cumsum().shift(1).fillna(0)
        frame["walk_so_far"] = by_pg["is_walk"].cumsum().shift(1).fillna(0)
        frame["hit_so_far"] = by_pg["is_hit"].cumsum().shift(1).fillna(0)
        frame["outs_so_far"] = by_pg["is_out"].cumsum().shift(1).fillna(0)
        frame["pitch_count_so_far"] = by_pg["pitch_count_pa"].cumsum().shift(1).fillna(0)
        frame["avg_pitch_count_so_far"] = frame["pitch_count_so_far"] / frame["bf_so_far"].replace(0, np.nan)

        faced_before = (
            frame.groupby(["game_id", "pitcher_id", "batter_id"]).cumcount()
            .rename("times_faced_batter_before")
        )
        frame["times_faced_batter_before"] = faced_before
        frame["times_through_order"] = np.floor(frame["bf_so_far"] / 9.0).astype(int)
        frame["same_hand"] = (frame["batter_side"] == frame["pitcher_hand"]).astype(int)

        runners = frame[["runner_on_first", "runner_on_second", "runner_on_third"]].fillna(False).astype(int)
        frame["runners_on"] = runners.sum(axis=1)
        frame["is_scoring_position"] = (runners[["runner_on_second", "runner_on_third"]].sum(axis=1) > 0).astype(int)
        return frame

    @staticmethod
    def _lineup_quality(frame: pd.DataFrame) -> pd.DataFrame:
        batter_pre = (
            frame.sort_values(["game_date", "game_id", "pa_number"])
            .groupby(["game_id", "fielding_team_id", "batter_id"], as_index=False)
            .first()
        )

        lineup = (
            batter_pre.groupby(["game_id", "fielding_team_id"], as_index=False)
            .agg(
                lineup_mean_batter_k_rate=("batter_k_rate_60", "mean"),
                lineup_mean_batter_walk_rate=("batter_walk_rate_60", "mean"),
                lineup_mean_batter_hit_rate=("batter_hit_rate_60", "mean"),
                lineup_mean_batter_barrel_rate=("batter_barrel_rate_60", "mean"),
                lineup_mean_batter_xba=("batter_xba_contact_60", "mean"),
                lineup_same_side_share=("same_hand", "mean"),
            )
        )
        return frame.merge(lineup, on=["game_id", "fielding_team_id"], how="left")

    @staticmethod
    def build_training_frame(pa_df: pd.DataFrame, games_df: pd.DataFrame) -> pd.DataFrame:
        if pa_df.empty:
            return pa_df.copy()

        frame = FeatureBuilder._attach_game_context(pa_df, games_df)
        frame["game_date"] = pd.to_datetime(frame["game_date"]).dt.normalize()
        frame = FeatureBuilder._rolling_rates(frame, "pitcher_id", ROLL_WINDOWS)
        frame = FeatureBuilder._rolling_rates(frame, "batter_id", ROLL_WINDOWS)
        frame = FeatureBuilder._days_rest(frame)
        frame = FeatureBuilder._game_state_features(frame)
        frame = FeatureBuilder._lineup_quality(frame)

        frame["days_rest"] = frame["days_rest"].fillna(frame["days_rest"].median())
        frame["avg_pitch_count_so_far"] = frame["avg_pitch_count_so_far"].fillna(frame["pitch_count_pa"].median())
        frame["pitcher_workload_flag"] = ((frame["pitch_count_so_far"] >= 85) | (frame["times_through_order"] >= 2)).astype(int)

        numeric_fill = [
            c
            for c in frame.columns
            if any(x in c for x in ["_rate_", "_xba_", "_velo_", "_spin_", "_extension_", "lineup_mean", "avg_"]) or c.startswith("ext_")
        ]
        for col in numeric_fill:
            frame[col] = frame[col].astype(float)
            frame[col] = frame[col].fillna(frame[col].median())

        return frame.sort_values(["game_date", "game_id", "pa_number"]).reset_index(drop=True)

    @staticmethod
    def build_start_state_frame(train_df: pd.DataFrame) -> pd.DataFrame:
        cols = [
            "game_id",
            "pitcher_id",
            "game_date",
            "season",
            "season_type",
            "pitcher_hand",
            "is_home_pitcher",
            "days_rest",
            "pitcher_k_rate_15",
            "pitcher_k_rate_30",
            "pitcher_k_rate_60",
            "pitcher_walk_rate_60",
            "pitcher_hit_rate_60",
            "pitcher_out_rate_60",
            "pitcher_pa_pitch_ct_60",
            "pitcher_velo_60",
            "pitcher_spin_60",
            "pitcher_extension_60",
            "pitcher_xba_contact_60",
            "pitcher_barrel_rate_60",
            "lineup_mean_batter_k_rate",
            "lineup_mean_batter_walk_rate",
            "lineup_mean_batter_hit_rate",
            "lineup_mean_batter_barrel_rate",
            "lineup_mean_batter_xba",
            "lineup_same_side_share",
        ]
        cols += [c for c in train_df.columns if c.startswith("ext_pitcher_")]
        firsts = train_df.sort_values(["game_date", "game_id", "pa_number"]).groupby(["game_id", "pitcher_id"], as_index=False).first()
        totals = train_df.groupby(["game_id", "pitcher_id"], as_index=False).agg(
            actual_total_k=("is_strikeout", "sum"),
            actual_total_bf=("batter_id", "size"),
        )
        out = firsts[cols].merge(totals, on=["game_id", "pitcher_id"], how="left")
        out["bf_so_far"] = 0
        out["k_so_far"] = 0
        out["walk_so_far"] = 0
        out["hit_so_far"] = 0
        out["outs_so_far"] = 0
        out["pitch_count_so_far"] = 0
        out["avg_pitch_count_so_far"] = out["pitcher_pa_pitch_ct_60"]
        out["times_through_order"] = 0
        out["inning"] = 1
        out["outs_start"] = 0
        out["runners_on"] = 0
        out["is_scoring_position"] = 0
        out["pitcher_workload_flag"] = 0
        out["state_type"] = "pregame"
        return out

    @staticmethod
    def build_state_frame(train_df: pd.DataFrame) -> pd.DataFrame:
        totals = train_df.groupby(["game_id", "pitcher_id"], as_index=False).agg(actual_total_bf=("batter_id", "size"), actual_total_k=("is_strikeout", "sum"))
        cols = [
            "game_id",
            "pitcher_id",
            "game_date",
            "season",
            "season_type",
            "pitcher_hand",
            "is_home_pitcher",
            "inning",
            "outs_start",
            "bf_so_far",
            "k_so_far",
            "walk_so_far",
            "hit_so_far",
            "outs_so_far",
            "pitch_count_so_far",
            "avg_pitch_count_so_far",
            "times_through_order",
            "runners_on",
            "is_scoring_position",
            "days_rest",
            "pitcher_k_rate_15",
            "pitcher_k_rate_30",
            "pitcher_k_rate_60",
            "pitcher_walk_rate_60",
            "pitcher_hit_rate_60",
            "pitcher_out_rate_60",
            "pitcher_pa_pitch_ct_60",
            "pitcher_velo_60",
            "pitcher_spin_60",
            "pitcher_extension_60",
            "pitcher_xba_contact_60",
            "pitcher_barrel_rate_60",
            "lineup_mean_batter_k_rate",
            "lineup_mean_batter_walk_rate",
            "lineup_mean_batter_hit_rate",
            "lineup_mean_batter_barrel_rate",
            "lineup_mean_batter_xba",
            "lineup_same_side_share",
            "pitcher_workload_flag",
        ]
        cols += [c for c in train_df.columns if c.startswith("ext_pitcher_")]
        states = train_df[cols].copy()
        states = states.merge(totals, on=["game_id", "pitcher_id"], how="left")
        states["remaining_bf"] = states["actual_total_bf"] - states["bf_so_far"]
        states["remaining_k"] = states["actual_total_k"] - states["k_so_far"]
        states["state_type"] = "live"
        states = states[states["remaining_bf"] > 0].copy()
        return states.sort_values(["game_date", "game_id", "pitcher_id", "bf_so_far"]).reset_index(drop=True)

    @staticmethod
    def build_survival_long(states: pd.DataFrame, max_horizon: int) -> pd.DataFrame:
        rows: List[Dict] = []

        base_cols = [c for c in states.columns if c not in {"remaining_bf", "remaining_k"}]
        for row in states.itertuples(index=False):
            base = {c: getattr(row, c) for c in base_cols}
            remaining_bf = int(getattr(row, "remaining_bf"))
            for step in range(1, min(max_horizon, remaining_bf) + 1):
                rows.append(
                    {
                        **base,
                        "future_bf_step": step,
                        "stop_here": int(step == remaining_bf),
                    }
                )

        return pd.DataFrame(rows)

    @staticmethod
    def build_future_lineup_frame(
        lineup_df: pd.DataFrame,
        history_pa_df: pd.DataFrame,
        *,
        pitcher_id: int,
        as_of_date: Optional[str] = None,
    ) -> pd.DataFrame:
        lineup = lineup_df.copy()
        if lineup.empty:
            return lineup

        if as_of_date is not None:
            cutoff = pd.Timestamp(as_of_date).normalize()
            hist = history_pa_df[history_pa_df["game_date"] < cutoff].copy()
        else:
            hist = history_pa_df.copy()

        latest_batter = (
            hist.sort_values(["game_date", "game_id", "pa_number"])
            .groupby("batter_id", as_index=False)
            .last()
        )
        latest_pitcher = (
            hist[hist["pitcher_id"].eq(pitcher_id)]
            .sort_values(["game_date", "game_id", "pa_number"])
            .tail(1)
        )

        keep_batter = [
            "batter_id",
            "batter_side",
            "batter_k_rate_15",
            "batter_k_rate_30",
            "batter_k_rate_60",
            "batter_walk_rate_60",
            "batter_hit_rate_60",
            "batter_barrel_rate_60",
            "batter_xba_contact_60",
        ] + [c for c in latest_batter.columns if c.startswith("ext_batter_")]
        lineup = lineup.merge(latest_batter[keep_batter], left_on="player.id", right_on="batter_id", how="left")

        if not latest_pitcher.empty:
            for col in [
                "pitcher_hand",
                "pitcher_k_rate_15",
                "pitcher_k_rate_30",
                "pitcher_k_rate_60",
                "pitcher_walk_rate_60",
                "pitcher_hit_rate_60",
                "pitcher_out_rate_60",
                "pitcher_pa_pitch_ct_60",
                "pitcher_velo_60",
                "pitcher_spin_60",
                "pitcher_extension_60",
                "pitcher_xba_contact_60",
                "pitcher_barrel_rate_60",
                "days_rest",
            ] + [c for c in latest_pitcher.columns if c.startswith("ext_pitcher_")]:
                lineup[col] = latest_pitcher.iloc[0].get(col)

        lineup["batter_side"] = lineup["batter_side"].apply(_normalize_side)
        lineup["pitcher_hand"] = lineup["pitcher_hand"].apply(_normalize_side)
        lineup["same_hand"] = (lineup["batter_side"] == lineup["pitcher_hand"]).astype(int)

        return lineup.sort_values("batting_order").reset_index(drop=True)

    @staticmethod
    def build_live_state_from_pas(
        pa_df: pd.DataFrame,
        *,
        pitcher_id: int,
        history_pa_df: pd.DataFrame,
        current_game_id: int,
        is_home_pitcher: Optional[int] = None,
    ) -> Optional[Dict]:
        g = pa_df[pa_df["pitcher_id"].eq(pitcher_id)].sort_values("pa_number").copy()
        if g.empty:
            return None

        latest_hist = (
            history_pa_df[history_pa_df["pitcher_id"].eq(pitcher_id)]
            .sort_values(["game_date", "game_id", "pa_number"])
            .tail(1)
        )

        last = g.iloc[-1]
        state = {
            "game_id": int(current_game_id),
            "pitcher_id": int(pitcher_id),
            "pitcher_hand": last.get("pitcher_hand") if pd.notna(last.get("pitcher_hand")) else (latest_hist.iloc[0].get("pitcher_hand") if not latest_hist.empty else None),
            "is_home_pitcher": int(is_home_pitcher) if is_home_pitcher is not None else int(last.get("is_home_pitcher", 0)),
            "inning": int(last.get("inning", 1)),
            "outs_start": int(last.get("outs_start", 0)),
            "bf_so_far": int(len(g)),
            "k_so_far": int(g["is_strikeout"].sum()),
            "walk_so_far": int(g["is_walk"].sum()),
            "hit_so_far": int(g["is_hit"].sum()),
            "outs_so_far": int(g["is_out"].sum()),
            "pitch_count_so_far": float(g["pitch_count_pa"].sum()),
            "avg_pitch_count_so_far": float(g["pitch_count_pa"].mean()),
            "times_through_order": int(len(g) // 9),
            "runners_on": int(last[["runner_on_first", "runner_on_second", "runner_on_third"]].fillna(False).astype(int).sum()),
            "is_scoring_position": int(bool(last.get("runner_on_second")) or bool(last.get("runner_on_third"))),
            "pitcher_workload_flag": int((g["pitch_count_pa"].sum() >= 85) or (len(g) >= 18)),
        }

        if not latest_hist.empty:
            hist_row = latest_hist.iloc[0]
            for col in [
                "days_rest",
                "pitcher_k_rate_15",
                "pitcher_k_rate_30",
                "pitcher_k_rate_60",
                "pitcher_walk_rate_60",
                "pitcher_hit_rate_60",
                "pitcher_out_rate_60",
                "pitcher_pa_pitch_ct_60",
                "pitcher_velo_60",
                "pitcher_spin_60",
                "pitcher_extension_60",
                "pitcher_xba_contact_60",
                "pitcher_barrel_rate_60",
                "lineup_mean_batter_k_rate",
                "lineup_mean_batter_walk_rate",
                "lineup_mean_batter_hit_rate",
                "lineup_mean_batter_barrel_rate",
                "lineup_mean_batter_xba",
                "lineup_same_side_share",
            ] + [c for c in latest_hist.columns if c.startswith("ext_pitcher_")]:
                state[col] = hist_row.get(col)

        return state

    @staticmethod
    def build_actual_game_totals(train_df: pd.DataFrame) -> pd.DataFrame:
        return (
            train_df.groupby(["game_id", "pitcher_id", "game_date"], as_index=False)
            .agg(actual_total_k=("is_strikeout", "sum"), actual_total_bf=("batter_id", "size"))
            .sort_values(["game_date", "game_id", "pitcher_id"])
            .reset_index(drop=True)
        )

    @staticmethod
    def load_prop_snapshots(snapshot_dir: Path) -> pd.DataFrame:
        files = sorted(snapshot_dir.glob("*.parquet"))
        if not files:
            return pd.DataFrame()
        return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)

    @staticmethod
    def build_market_training_frame(snapshot_df: pd.DataFrame, actuals_df: pd.DataFrame) -> pd.DataFrame:
        if snapshot_df.empty:
            return snapshot_df.copy()

        snap = snapshot_df.copy()
        snap["snapshot_date"] = pd.to_datetime(snap["snapshot_date"]).dt.normalize()
        snap["line_value"] = snap["line_value"].astype(float)

        latest = (
            snap.sort_values(["game_id", "player_id", "vendor", "snapshot_date", "updated_at"])
            .groupby(["game_id", "player_id", "vendor", "line_value"], as_index=False)
            .last()
        )
        latest = latest.merge(
            actuals_df.rename(columns={"pitcher_id": "player_id"}),
            on=["game_id", "player_id"],
            how="left",
        )
        latest["over_hit"] = (latest["actual_total_k"] > latest["line_value"]).astype(float)
        latest["under_hit"] = (latest["actual_total_k"] < latest["line_value"]).astype(float)
        latest["push"] = (latest["actual_total_k"] == latest["line_value"]).astype(float)
        latest["is_live_snapshot"] = (latest["snapshot_date"] == latest["game_date"]).astype(int) if "game_date" in latest else 0
        return latest
