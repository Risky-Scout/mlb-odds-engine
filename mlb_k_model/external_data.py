
from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from .api_clients import MarketQuote, PyBaseballClient


TEAM_ALIASES = {
    "athletics": "oakland athletics",
    "a's": "oakland athletics",
    "diamondbacks": "arizona diamondbacks",
    "dbacks": "arizona diamondbacks",
    "d-backs": "arizona diamondbacks",
    "white sox": "chicago white sox",
    "red sox": "boston red sox",
    "guardians": "cleveland guardians",
    "indians": "cleveland guardians",
}


def normalize_text(value: object) -> str:
    text = "" if value is None else str(value)
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.lower().strip()
    text = re.sub(r"\b(jr|sr|ii|iii|iv)\b\.?", "", text)
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_team(value: object) -> str:
    text = normalize_text(value)
    return TEAM_ALIASES.get(text, text)


def _infer_mlbam_col(df: pd.DataFrame) -> Optional[str]:
    for col in ["player_id", "mlbam_id", "mlb_id", "key_mlbam", "pitcher", "batter"]:
        if col in df.columns:
            return col
    return None


def _numeric_feature_cols(df: pd.DataFrame) -> List[str]:
    banned = {"season", "year", "pitcher_id", "batter_id", "key_mlbam", "mlbam_id", "player_id", "pitcher", "batter"}
    out = []
    for col in df.columns:
        if col in banned:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            out.append(col)
    return out


def _best_lookup_row(cands: pd.DataFrame, debut_year: Optional[int]) -> Optional[pd.Series]:
    if cands.empty:
        return None

    work = cands.copy()
    for col in ["mlb_played_first", "mlb_played_last"]:
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")

    if debut_year is not None and {"mlb_played_first", "mlb_played_last"}.issubset(work.columns):
        mask = (work["mlb_played_first"].fillna(-9999) <= debut_year) & (work["mlb_played_last"].fillna(9999) >= debut_year)
        if mask.any():
            work = work[mask].copy()

        work["debut_gap"] = (work["mlb_played_first"].fillna(debut_year) - debut_year).abs()
        work = work.sort_values(["debut_gap", "mlb_played_last"], ascending=[True, False])

    return work.iloc[0] if not work.empty else None




def build_player_id_map(players_df: pd.DataFrame, pyb: PyBaseballClient) -> pd.DataFrame:
    if players_df.empty:
        return pd.DataFrame()

    try:
        import pybaseball as pyb_mod
        chadwick = pyb_mod.chadwick_register()
    except Exception:
        chadwick = pd.DataFrame()

    if not chadwick.empty:
        chadwick = chadwick.copy()
        for col in ["name_first", "name_last"]:
            if col not in chadwick.columns:
                chadwick[col] = ""
        chadwick["name_first_norm"] = chadwick["name_first"].map(normalize_text)
        chadwick["name_last_norm"] = chadwick["name_last"].map(normalize_text)
        for col in ["mlb_played_first", "mlb_played_last", "key_mlbam"]:
            if col in chadwick.columns:
                chadwick[col] = pd.to_numeric(chadwick[col], errors="coerce")

    rows: List[Dict] = []
    keep = players_df.drop_duplicates("id").copy()

    for row in keep.itertuples(index=False):
        first_raw = str(getattr(row, "first_name", "") or "").strip()
        last_raw = str(getattr(row, "last_name", "") or "").strip()
        debut_year = getattr(row, "debut_year", None)
        debut_year = int(debut_year) if pd.notna(debut_year) else None

        first_norm = normalize_text(first_raw)
        last_norm = normalize_text(last_raw)

        chosen = None
        method = "unmapped"

        try:
            exact = pyb.playerid_lookup(last_raw, first_raw, fuzzy=False)
        except Exception:
            exact = pd.DataFrame()

        chosen = _best_lookup_row(exact, debut_year)
        if chosen is not None:
            method = "exact_raw"

        if chosen is None and not chadwick.empty:
            cands = chadwick.loc[
                (chadwick["name_first_norm"] == first_norm) &
                (chadwick["name_last_norm"] == last_norm)
            ].copy()
            chosen = _best_lookup_row(cands, debut_year)
            if chosen is not None:
                method = "chadwick_normalized"

        mapped = {
            "bdl_player_id": int(getattr(row, "id")),
            "bdl_full_name": str(getattr(row, "full_name", f"{first_raw} {last_raw}")).strip(),
            "bdl_first_name": first_raw,
            "bdl_last_name": last_raw,
            "bdl_first_name_normalized": first_norm,
            "bdl_last_name_normalized": last_norm,
            "bdl_debut_year": debut_year if debut_year is not None else np.nan,
            "map_method": method,
            "mlbam_id": np.nan,
            "lookup_first": np.nan,
            "lookup_last": np.nan,
            "lookup_name_first": np.nan,
            "lookup_name_last": np.nan,
        }

        if chosen is not None:
            chosen_dict = chosen.to_dict()
            mlbam = chosen_dict.get("key_mlbam")
            mapped["mlbam_id"] = int(mlbam) if pd.notna(mlbam) else np.nan
            mapped["lookup_first"] = chosen_dict.get("mlb_played_first")
            mapped["lookup_last"] = chosen_dict.get("mlb_played_last")
            mapped["lookup_name_first"] = chosen_dict.get("name_first")
            mapped["lookup_name_last"] = chosen_dict.get("name_last")

        rows.append(mapped)

    out = pd.DataFrame(rows)

    overrides = _load_manual_player_overrides()
    if not overrides.empty:
        out = out.merge(overrides, on="bdl_player_id", how="left", suffixes=("", "_override"))
        use_override = out["mlbam_id"].isna() & out["mlbam_id_override"].notna()
        out.loc[use_override, "mlbam_id"] = out.loc[use_override, "mlbam_id_override"]
        out.loc[use_override, "map_method"] = "manual_override"
        out = out.drop(columns=["mlbam_id_override"])

    out["mapped"] = out["mlbam_id"].notna().astype(int)

    unresolved = out.loc[out["mapped"].eq(0), [
        "bdl_player_id",
        "bdl_full_name",
        "bdl_first_name",
        "bdl_last_name",
        "bdl_first_name_normalized",
        "bdl_last_name_normalized",
        "bdl_debut_year",
    ]].copy()

    Path("data").mkdir(parents=True, exist_ok=True)
    if not unresolved.empty:
        unresolved.to_csv("data/unresolved_player_map.csv", index=False)

    out.to_parquet("data/player_id_map.parquet", index=False)
    return out

def _load_manual_player_overrides(path: str = "data/manual_player_map.csv") -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame(columns=["bdl_player_id", "mlbam_id"])
    df = pd.read_csv(p)
    need = {"bdl_player_id", "mlbam_id"}
    if not need.issubset(df.columns):
        raise ValueError(f"manual override file must contain columns: {sorted(need)}")
    df = df[list(need)].dropna().copy()
    df["bdl_player_id"] = pd.to_numeric(df["bdl_player_id"], errors="coerce").astype("Int64")
    df["mlbam_id"] = pd.to_numeric(df["mlbam_id"], errors="coerce").astype("Int64")
    return df.dropna().drop_duplicates("bdl_player_id")


def _standardize_leaderboard(df: pd.DataFrame, season: int, prefix: str) -> pd.DataFrame:
    frame = df.copy()
    mlbam_col = _infer_mlbam_col(frame)
    if mlbam_col is None:
        return pd.DataFrame()

    frame = frame.rename(columns={mlbam_col: "mlbam_id"})
    frame["season"] = int(season)
    numeric_cols = _numeric_feature_cols(frame)
    keep = ["mlbam_id", "season"] + numeric_cols
    frame = frame[keep].copy()
    rename = {c: f"{prefix}{c}" for c in numeric_cols}
    frame = frame.rename(columns=rename)
    return frame.groupby(["mlbam_id", "season"], as_index=False).mean(numeric_only=True)


def collect_savant_pitcher_features(pyb: PyBaseballClient, seasons: Iterable[int]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []

    for season in seasons:
        parts: List[pd.DataFrame] = []

        try:
            parts.append(_standardize_leaderboard(pyb.statcast_pitcher_expected_stats(int(season)), int(season), "ext_pitcher_xstats_"))
        except Exception:
            pass
        try:
            parts.append(_standardize_leaderboard(pyb.statcast_pitcher_exitvelo_barrels(int(season)), int(season), "ext_pitcher_contact_"))
        except Exception:
            pass
        try:
            parts.append(_standardize_leaderboard(pyb.statcast_pitcher_pitch_arsenal(int(season), arsenal_type="average_speed"), int(season), "ext_pitcher_arsenal_speed_"))
        except Exception:
            pass
        try:
            parts.append(_standardize_leaderboard(pyb.statcast_pitcher_pitch_arsenal(int(season), arsenal_type="average_spin"), int(season), "ext_pitcher_arsenal_spin_"))
        except Exception:
            pass
        try:
            parts.append(_standardize_leaderboard(pyb.statcast_pitcher_arsenal_stats(int(season)), int(season), "ext_pitcher_arsenal_stats_"))
        except Exception:
            pass
        try:
            parts.append(_standardize_leaderboard(pyb.statcast_pitcher_percentile_ranks(int(season)), int(season), "ext_pitcher_pct_"))
        except Exception:
            pass
        try:
            parts.append(_standardize_leaderboard(pyb.statcast_pitcher_pitch_movement(int(season), pitch_type="ALL"), int(season), "ext_pitcher_movement_"))
        except Exception:
            pass

        parts = [p for p in parts if not p.empty]
        if not parts:
            continue

        merged = parts[0]
        for part in parts[1:]:
            merged = merged.merge(part, on=["mlbam_id", "season"], how="outer")
        frames.append(merged)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def collect_savant_batter_features(pyb: PyBaseballClient, seasons: Iterable[int]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []

    for season in seasons:
        parts: List[pd.DataFrame] = []
        try:
            parts.append(_standardize_leaderboard(pyb.statcast_batter_expected_stats(int(season)), int(season), "ext_batter_xstats_"))
        except Exception:
            pass
        try:
            parts.append(_standardize_leaderboard(pyb.statcast_batter_exitvelo_barrels(int(season)), int(season), "ext_batter_contact_"))
        except Exception:
            pass

        parts = [p for p in parts if not p.empty]
        if not parts:
            continue

        merged = parts[0]
        for part in parts[1:]:
            merged = merged.merge(part, on=["mlbam_id", "season"], how="outer")
        frames.append(merged)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()



def merge_external_features(
    train_df: pd.DataFrame,
    player_map: pd.DataFrame,
    pitcher_ext: pd.DataFrame,
    batter_ext: pd.DataFrame,
) -> pd.DataFrame:
    df = train_df.copy()

    if df.empty:
        return df

    # Require season-aware merge when possible.
    has_season = "season" in df.columns

    # Map BDL ids -> MLBAM ids
    if not player_map.empty and "bdl_player_id" in player_map.columns and "mlbam_id" in player_map.columns:
        pitcher_map = (
            player_map[["bdl_player_id", "mlbam_id"]]
            .dropna(subset=["bdl_player_id"])
            .drop_duplicates()
            .rename(columns={"bdl_player_id": "pitcher_id", "mlbam_id": "pitcher_mlbam_id"})
        )
        batter_map = (
            player_map[["bdl_player_id", "mlbam_id"]]
            .dropna(subset=["bdl_player_id"])
            .drop_duplicates()
            .rename(columns={"bdl_player_id": "batter_id", "mlbam_id": "batter_mlbam_id"})
        )
        df = df.merge(pitcher_map, on="pitcher_id", how="left")
        df = df.merge(batter_map, on="batter_id", how="left")

    # Pitcher external priors
    if not pitcher_ext.empty:
        p = pitcher_ext.copy()
        if "mlbam_id" not in p.columns:
            raise ValueError("pitcher_ext must contain mlbam_id")
        p["mlbam_id"] = pd.to_numeric(p["mlbam_id"], errors="coerce")
        if "season" in p.columns:
            p["season"] = pd.to_numeric(p["season"], errors="coerce")
        p_cols = ["mlbam_id"] + (["season"] if "season" in p.columns else []) + [c for c in p.columns if c.startswith("ext_pitcher_")]
        p = p[p_cols].drop_duplicates().rename(columns={"mlbam_id": "pitcher_mlbam_id"})

        if has_season and "season" in p.columns and "pitcher_mlbam_id" in df.columns:
            df = df.merge(p, on=["pitcher_mlbam_id", "season"], how="left")
        elif "pitcher_mlbam_id" in df.columns:
            df = df.merge(p.drop(columns=["season"], errors="ignore"), on="pitcher_mlbam_id", how="left")

    # Batter external priors
    if not batter_ext.empty:
        b = batter_ext.copy()
        if "mlbam_id" not in b.columns:
            raise ValueError("batter_ext must contain mlbam_id")
        b["mlbam_id"] = pd.to_numeric(b["mlbam_id"], errors="coerce")
        if "season" in b.columns:
            b["season"] = pd.to_numeric(b["season"], errors="coerce")
        b_cols = ["mlbam_id"] + (["season"] if "season" in b.columns else []) + [c for c in b.columns if c.startswith("ext_batter_")]
        b = b[b_cols].drop_duplicates().rename(columns={"mlbam_id": "batter_mlbam_id"})

        if has_season and "season" in b.columns and "batter_mlbam_id" in df.columns:
            df = df.merge(b, on=["batter_mlbam_id", "season"], how="left")
        elif "batter_mlbam_id" in df.columns:
            df = df.merge(b.drop(columns=["season"], errors="ignore"), on="batter_mlbam_id", how="left")

    return df


def map_odds_api_quotes_to_bdl(*args, **kwargs):
    # Compatibility shim to satisfy package imports.
    # If an equivalent mapper exists, delegate to it.
    for name in (
        "map_market_quotes_to_bdl",
        "map_odds_quotes_to_bdl",
        "map_quotes_to_bdl",
    ):
        fn = globals().get(name)
        if callable(fn):
            return fn(*args, **kwargs)

    # Market snapshot layer is currently empty in this project state,
    # so returning an empty frame is safe for the current retrain path.
    return pd.DataFrame()


def load_all_market_snapshots(data_dir):
    data_dir = Path(data_dir)
    frames = []

    odds_api_dir = data_dir / "snapshots" / "player_props_odds_api"
    if odds_api_dir.exists():
        for p in sorted(odds_api_dir.glob("*.parquet")):
            try:
                df = pd.read_parquet(p)
                if not df.empty:
                    frames.append(df)
            except Exception:
                continue

    legacy_dir = data_dir / "snapshots" / "player_props"
    if legacy_dir.exists():
        for p in sorted(legacy_dir.glob("*.parquet")):
            try:
                df = pd.read_parquet(p)
                if not df.empty:
                    frames.append(df)
            except Exception:
                continue
        for p in sorted(legacy_dir.glob("*.csv")):
            try:
                df = pd.read_csv(p)
                if not df.empty:
                    frames.append(df)
            except Exception:
                continue

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True, sort=False)
    if "updated_at" in out.columns:
        out["updated_at"] = pd.to_datetime(out["updated_at"], utc=True, errors="coerce")
    if "commence_time" in out.columns:
        out["commence_time"] = pd.to_datetime(out["commence_time"], utc=True, errors="coerce")
    if "snapshot_ts" in out.columns:
        out["snapshot_ts"] = pd.to_datetime(out["snapshot_ts"], utc=True, errors="coerce")
    elif "updated_at" in out.columns:
        out["snapshot_ts"] = pd.to_datetime(out["updated_at"], utc=True, errors="coerce")

    if "snapshot_date" not in out.columns and "snapshot_ts" in out.columns:
        out["snapshot_date"] = out["snapshot_ts"].dt.normalize()

    if "is_live_snapshot" not in out.columns and {"snapshot_ts", "commence_time"}.issubset(out.columns):
        out["is_live_snapshot"] = (out["snapshot_ts"] >= out["commence_time"]).astype(int)

    return out


def market_quotes_from_df(df: pd.DataFrame, *, game_id: int, player_id: int, prop_type: str = "pitcher_strikeouts") -> List[MarketQuote]:
    if df.empty:
        return []

    mask = (df["game_id"].eq(int(game_id))) & (df["player_id"].eq(int(player_id)))
    if "prop_type" in df.columns:
        mask &= df["prop_type"].astype(str).eq(prop_type)

    rows = df.loc[mask].copy()
    out: List[MarketQuote] = []
    for row in rows.itertuples(index=False):
        out.append(
            MarketQuote(
                game_id=int(row.game_id),
                player_id=int(row.player_id),
                vendor=str(row.vendor),
                prop_type=str(getattr(row, "prop_type", prop_type)),
                line_value=float(row.line_value),
                over_odds=int(row.over_odds) if pd.notna(getattr(row, "over_odds", np.nan)) else None,
                under_odds=int(row.under_odds) if pd.notna(getattr(row, "under_odds", np.nan)) else None,
                updated_at=getattr(row, "updated_at", None),
            )
        )
    return out


def normalize_odds_api_event_odds(*args, **kwargs):
    # Compatibility shim added to satisfy package imports.
    # The market snapshot layer is currently empty, so returning an empty
    # DataFrame is safe for the current leakage-fix / retrain path.
    return pd.DataFrame()

