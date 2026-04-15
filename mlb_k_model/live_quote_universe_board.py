from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo
import re
import unicodedata

import numpy as np
import pandas as pd

from mlb_k_model.data_pipeline import FeatureBuilder
from mlb_k_model.external_data import merge_external_features
from mlb_k_model.system import StrikeoutBettingSystem
from mlb_k_model.live_game_state import build_live_state_lookup_from_quotes, condition_pmf_with_live_state
from train_system import build_lineup_lookup


STRICT_EV_GATE_MIN_EDGE = 0.20
STRICT_EV_GATE_MIN_CONF_DIST = 0.10


def _norm(s):
    if s is None:
        return ""
    s = str(s)
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = s.lower().strip()
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s


def _threshold_over_prob(pmf: np.ndarray, line_value: float) -> float:
    thr = int(np.floor(float(line_value)) + 1)
    thr = max(0, min(thr, len(pmf)))
    return float(np.asarray(pmf, dtype=float)[thr:].sum())


def _american_to_prob(odds: float) -> float:
    odds = float(odds)
    if odds > 0:
        return 100.0 / (odds + 100.0)
    return abs(odds) / (abs(odds) + 100.0)


def _no_vig_probs(over_odds, under_odds):
    po = _american_to_prob(over_odds)
    pu = _american_to_prob(under_odds)
    z = po + pu
    return po / z, pu / z


def _prob_to_american(p: float) -> float:
    p = float(np.clip(p, 1e-6, 1 - 1e-6))
    if p >= 0.5:
        return -100.0 * p / (1.0 - p)
    return 100.0 * (1.0 - p) / p


def _ev_from_american(prob: float, odds: float) -> float:
    prob = float(np.clip(prob, 1e-6, 1 - 1e-6))
    odds = float(odds)
    if odds > 0:
        profit = odds / 100.0
    else:
        profit = 100.0 / abs(odds)
    return prob * profit - (1.0 - prob)


def _apply_saved_pmf_postcal(system, pmf: np.ndarray) -> np.ndarray:
    tau = getattr(system.true_model, "mainline_pmf_postcal_tau_", None)
    lam = getattr(system.true_model, "mainline_pmf_postcal_lambda_", None)
    if tau is None or lam is None:
        return pmf

    z = np.asarray(pmf, dtype=float).copy()
    z = np.clip(z, 1e-12, None)
    z /= z.sum()
    ks = np.arange(len(z), dtype=float)
    logits = (np.log(z) / float(tau)) + float(lam) * ks
    logits -= logits.max()
    out = np.exp(logits)
    out = np.clip(out, 1e-12, None)
    return out / out.sum()


def _apply_threshold_cal(system, p: float) -> float:
    p = float(np.clip(p, 1e-6, 1 - 1e-6))
    cal = getattr(system, "mainline_threshold_calibrator", None)
    if cal is None:
        return p
    try:
        return float(np.clip(cal.predict(np.asarray([p]))[0], 1e-6, 1 - 1e-6))
    except Exception:
        return p


def _pick_name_col(players: pd.DataFrame) -> str:
    for c in ["full_name", "display_name", "name"]:
        if c in players.columns:
            return c
    if {"first_name", "last_name"}.issubset(players.columns):
        players["__full_name_tmp__"] = (
            players["first_name"].fillna("").astype(str).str.strip() + " " +
            players["last_name"].fillna("").astype(str).str.strip()
        )
        return "__full_name_tmp__"
    raise SystemExit("Could not determine player name column from players.parquet")


def _load_live_quotes(data_dir: Path, target_date: str) -> pd.DataFrame:
    frames = []

    current_path = data_dir / "snapshots" / "player_props_odds_api" / "odds_api_current_normalized.parquet"
    if current_path.exists():
        try:
            frames.append(pd.read_parquet(current_path))
        except Exception:
            pass

    archive_path = data_dir / "snapshots" / "player_props_odds_api" / "live_archive" / f"odds_api_live_capture_{target_date}.parquet"
    if archive_path.exists():
        try:
            frames.append(pd.read_parquet(archive_path))
        except Exception:
            pass

    if not frames:
        return pd.DataFrame()

    q = pd.concat(frames, ignore_index=True, sort=False)

    if "snapshot_ts" not in q.columns and "updated_at" in q.columns:
        q["snapshot_ts"] = q["updated_at"]

    q["snapshot_ts"] = pd.to_datetime(q.get("snapshot_ts"), utc=True, errors="coerce")
    q["commence_time"] = pd.to_datetime(q.get("commence_time"), utc=True, errors="coerce")

    ny = ZoneInfo("America/New_York")
    q["board_date_et"] = q["commence_time"].dt.tz_convert(ny).dt.date.astype(str)

    q = q.loc[
        q.get("prop_type").eq("pitcher_strikeouts")
        & q["board_date_et"].eq(str(target_date))
        & q.get("is_live_snapshot").fillna(0).eq(1)
        & q.get("player_name").notna()
    ].copy()

    for c in ["game_id", "player_id", "line_value", "over_odds", "under_odds"]:
        if c in q.columns:
            q[c] = pd.to_numeric(q[c], errors="coerce")

    q = q.dropna(subset=["game_id", "player_name", "line_value", "over_odds", "under_odds"]).copy()

    q = q.sort_values("snapshot_ts").drop_duplicates(
        subset=["game_id", "player_name", "vendor", "line_value"],
        keep="last",
    )

    return q.reset_index(drop=True)


def build_quote_universe_live_board(data_dir: str | Path, system_path: str | Path, target_date: str):
    data_dir = Path(data_dir)
    system = StrikeoutBettingSystem.load(system_path)

    quotes = _load_live_quotes(data_dir, target_date)
    if quotes.empty:
        return SimpleNamespace(
            summary=pd.DataFrame(),
            candidate_bets=pd.DataFrame(),
            line_grid=pd.DataFrame(),
            exact_pmf=pd.DataFrame(),
        )

    games = pd.read_parquet(data_dir / "games.parquet")
    pa = pd.read_parquet(data_dir / "plate_appearances.parquet")
    players = pd.read_parquet(data_dir / "players.parquet").copy()

    train_df = FeatureBuilder.build_training_frame(pa, games)

    player_map = pd.read_parquet(data_dir / "external" / "player_id_map.parquet")
    pitcher_ext = pd.read_parquet(data_dir / "external" / "savant_pitcher_features.parquet").copy()
    batter_ext = pd.read_parquet(data_dir / "external" / "savant_batter_features.parquet").copy()
    pitcher_ext["season"] = pd.to_numeric(pitcher_ext["season"], errors="coerce") + 1
    batter_ext["season"] = pd.to_numeric(batter_ext["season"], errors="coerce") + 1

    train_df = merge_external_features(train_df, player_map, pitcher_ext, batter_ext)
    start_states = FeatureBuilder.build_start_state_frame(train_df).copy()
    lineup_lookup = build_lineup_lookup(train_df)

    name_col = _pick_name_col(players)
    player_names = players[["id", name_col]].drop_duplicates().rename(columns={"id": "pitcher_id", name_col: "pitcher_name"})
    start_states = start_states.merge(player_names, on="pitcher_id", how="left")
    start_states["pitcher_name_norm"] = start_states["pitcher_name"].map(_norm)
    start_states["game_date"] = pd.to_datetime(start_states["game_date"], utc=True, errors="coerce")

    target_ts = pd.Timestamp(str(target_date)).tz_localize("UTC")
    start_states = start_states.loc[start_states["game_date"] < target_ts].copy()

    if start_states.empty:
        return SimpleNamespace(
            summary=pd.DataFrame(),
            candidate_bets=pd.DataFrame(),
            line_grid=pd.DataFrame(),
            exact_pmf=pd.DataFrame(),
        )

    start_states = start_states.sort_values(["game_date", "game_id"]).reset_index(drop=True)

    latest_by_id = {}
    latest_by_name = {}
    for r in start_states.itertuples(index=False):
        rec = r._asdict()
        latest_by_id[rec["pitcher_id"]] = rec
        if pd.notna(rec.get("pitcher_name")):
            latest_by_name[_norm(rec["pitcher_name"])] = rec

    state_cols = [c for c in start_states.columns if c not in {"actual_total_k", "actual_total_bf"}]

    summary_rows = []
    bet_rows = []
    line_rows = []
    exact_rows = []

    quotes["player_name_norm"] = quotes["player_name"].map(_norm)

    live_state_lookup = build_live_state_lookup_from_quotes(
        quotes=quotes,
        target_date=target_date,
        data_dir=data_dir,
    )

    for (game_id, player_name_norm), qg in quotes.groupby(["game_id", "player_name_norm"], dropna=False):
        player_name = qg["player_name"].dropna().iloc[0] if qg["player_name"].notna().any() else None
        player_id = qg["player_id"].dropna().iloc[0] if qg["player_id"].notna().any() else np.nan

        state_rec = None
        if pd.notna(player_id):
            state_rec = latest_by_id.get(int(player_id))
        if state_rec is None and player_name_norm:
            state_rec = latest_by_name.get(player_name_norm)

        if state_rec is None:
            continue

        hist_key = (int(state_rec["game_id"]), int(state_rec["pitcher_id"]))
        future_lineup = lineup_lookup.get(hist_key)
        if future_lineup is None or len(future_lineup) == 0:
            continue

        state = {c: state_rec.get(c) for c in state_cols}
        state["game_id"] = int(game_id)
        state["pitcher_id"] = int(state_rec["pitcher_id"])

        pmf = system.true_model.predict_pmf(state=state, future_lineup=future_lineup)
        pmf = _apply_saved_pmf_postcal(system, pmf)

        live_state = live_state_lookup.get((int(game_id), str(player_name_norm)))

        pmf = condition_pmf_with_live_state(pmf, live_state)

        ks = np.arange(len(pmf), dtype=float)
        k_mean = float(np.sum(ks * pmf))
        cdf = np.cumsum(pmf)
        k_median = int(np.searchsorted(cdf, 0.50))
        k_p80 = int(np.searchsorted(cdf, 0.80))
        k_p90 = int(np.searchsorted(cdf, 0.90))

        for k, p in enumerate(pmf):
            exact_rows.append({
                "strikeouts": int(k),
                "probability": float(p),
                "fair_american": float(_prob_to_american(p)),
                "mode": "live",
                "game_id": int(game_id),
                "pitcher_id": int(state_rec["pitcher_id"]),
                "pitcher_name": player_name,
            })

        for lv in sorted(qg["line_value"].dropna().unique().tolist()):
            raw_p_over = _threshold_over_prob(pmf, lv)
            calibrated_p_over = _apply_threshold_cal(system, raw_p_over)
            p_over = raw_p_over
            p_under = 1.0 - p_over
            line_rows.append({
                "line_value": float(lv),
                "p_over": float(p_over),
                "p_over_raw": float(raw_p_over),
                "p_over_calibrated": float(calibrated_p_over),
                "p_under": float(p_under),
                "p_push": 0.0,
                "fair_over_american": float(_prob_to_american(p_over)),
                "fair_under_american": float(_prob_to_american(p_under)),
                "mode": "live",
                "game_id": int(game_id),
                "pitcher_id": int(state_rec["pitcher_id"]),
                "pitcher_name": player_name,
            })

        tmp = qg.copy()
        tmp[["market_over_prob", "market_under_prob"]] = tmp.apply(
            lambda r: pd.Series(_no_vig_probs(r["over_odds"], r["under_odds"])),
            axis=1,
        )
        tmp["market_dist_from_50"] = (tmp["market_over_prob"] - 0.5).abs()
        tmp = tmp.sort_values(["market_dist_from_50", "snapshot_ts"])
        main_q = tmp.iloc[0]

        market_mean = float(main_q["line_value"])
        summary_rows.append({
            "mode": "live",
            "game_id": int(game_id),
            "pitcher_id": int(state_rec["pitcher_id"]),
            "pitcher_name": player_name,
            "k_mean": k_mean,
            "k_median": k_median,
            "k_p80": k_p80,
            "k_p90": k_p90,
            "top_quote_count": int(len(qg)),
            "market_mean": market_mean,
            "market_mean_shift": float(market_mean - k_mean),
            "vendor_count": int(qg["vendor"].nunique()),
            "strikeouts_so_far": int(live_state.strikeouts_so_far) if live_state is not None else None,
            "batters_faced_so_far": int(live_state.batters_faced_so_far) if live_state is not None else None,
            "pitches_thrown_so_far": int(live_state.pitches_thrown_so_far) if live_state is not None else None,
            "innings_completed": float(live_state.innings_completed) if live_state is not None else None,
            "pitcher_active_flag": int(live_state.pitcher_active_flag) if live_state is not None else None,
        })

        for r in qg.itertuples(index=False):
            market_over, market_under = _no_vig_probs(r.over_odds, r.under_odds)
            raw_model_over = _threshold_over_prob(pmf, r.line_value)
            calibrated_model_over = _apply_threshold_cal(system, raw_model_over)
            model_over = raw_model_over
            model_under = 1.0 - model_over

            over_ev = _ev_from_american(model_over, r.over_odds)
            under_ev = _ev_from_american(model_under, r.under_odds)
            best_side = "over" if over_ev >= under_ev else "under"
            best_ev = max(over_ev, under_ev)
            conf_dist = abs(model_over - 0.5)
            passes = (best_ev >= STRICT_EV_GATE_MIN_EDGE) and (conf_dist >= STRICT_EV_GATE_MIN_CONF_DIST)

            bet_rows.append({
                "mode": "live",
                "game_id": int(game_id),
                "pitcher_id": int(state_rec["pitcher_id"]),
                "pitcher_name": player_name,
                "vendor": r.vendor,
                "line_value": float(r.line_value),
                "best_side": best_side,
                "best_ev": float(best_ev),
                "over_ev": float(over_ev),
                "under_ev": float(under_ev),
                "model_over_prob": float(model_over),
                "raw_model_over_prob": float(raw_model_over),
                "calibrated_model_over_prob": float(calibrated_model_over),
                "market_over_prob": float(market_over),
                "model_under_prob": float(model_under),
                "market_under_prob": float(market_under),
                "gate_conf_dist": float(conf_dist),
                "gate_min_edge": STRICT_EV_GATE_MIN_EDGE,
                "gate_min_conf_dist": STRICT_EV_GATE_MIN_CONF_DIST,
                "passes_strict_ev_gate": bool(passes),
                "action_tier": "BET" if passes else "PASS",
                "over_odds": float(r.over_odds),
                "under_odds": float(r.under_odds),
                "snapshot_ts": r.snapshot_ts,
                "commence_time": r.commence_time,
                "is_live_snapshot": int(getattr(r, "is_live_snapshot", 1)),
            })

    summary_df = pd.DataFrame(summary_rows).sort_values(["game_id", "pitcher_name"]).reset_index(drop=True) if summary_rows else pd.DataFrame()
    bets_df = pd.DataFrame(bet_rows)
    if not bets_df.empty:
        bets_df = bets_df.sort_values(
            ["passes_strict_ev_gate", "best_ev", "gate_conf_dist"],
            ascending=[False, False, False],
        ).reset_index(drop=True)
    line_df = pd.DataFrame(line_rows).sort_values(["game_id", "pitcher_name", "line_value"]).reset_index(drop=True) if line_rows else pd.DataFrame()
    exact_df = pd.DataFrame(exact_rows).sort_values(["game_id", "pitcher_name", "strikeouts"]).reset_index(drop=True) if exact_rows else pd.DataFrame()

    return SimpleNamespace(
        summary=summary_df,
        candidate_bets=bets_df,
        line_grid=line_df,
        exact_pmf=exact_df,
    )
