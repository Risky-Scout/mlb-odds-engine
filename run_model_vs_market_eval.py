from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

from mlb_k_model.data_pipeline import FeatureBuilder
from mlb_k_model.external_data import load_all_market_snapshots, merge_external_features
from mlb_k_model.system import StrikeoutBettingSystem
from train_system import build_lineup_lookup


def american_to_prob(odds: float) -> float:
    odds = float(odds)
    if odds > 0:
        return 100.0 / (odds + 100.0)
    return abs(odds) / (abs(odds) + 100.0)


def prob_to_american(p: float) -> float:
    p = float(np.clip(p, 1e-6, 1 - 1e-6))
    if p >= 0.5:
        return -100.0 * p / (1.0 - p)
    return 100.0 * (1.0 - p) / p


def ev_from_american(prob: float, odds: float) -> float:
    prob = float(np.clip(prob, 1e-6, 1 - 1e-6))
    odds = float(odds)
    if odds > 0:
        profit = odds / 100.0
    else:
        profit = 100.0 / abs(odds)
    return prob * profit - (1.0 - prob)


def no_vig_probs(over_odds, under_odds):
    po = american_to_prob(over_odds)
    pu = american_to_prob(under_odds)
    z = po + pu
    return po / z, pu / z


def threshold_over_prob(pmf: np.ndarray, line_value: float) -> float:
    line_value = float(line_value)
    # standard K props are typically half-point lines
    thr = int(np.floor(line_value) + 1)
    thr = max(0, min(thr, len(pmf)))
    return float(np.asarray(pmf, dtype=float)[thr:].sum())


def compute_summary(df: pd.DataFrame, label: str) -> dict:
    if df.empty:
        return {
            "segment": label,
            "rows": 0,
        }

    y = df["actual_over"].astype(int).to_numpy()
    mp = np.clip(df["model_over_prob"].astype(float).to_numpy(), 1e-6, 1 - 1e-6)
    mkp = np.clip(df["market_over_prob"].astype(float).to_numpy(), 1e-6, 1 - 1e-6)

    out = {
        "segment": label,
        "rows": int(len(df)),
        "model_log_loss": float(log_loss(y, mp)),
        "market_log_loss": float(log_loss(y, mkp)),
        "delta_log_loss_market_minus_model": float(log_loss(y, mkp) - log_loss(y, mp)),
        "model_brier": float(brier_score_loss(y, mp)),
        "market_brier": float(brier_score_loss(y, mkp)),
        "delta_brier_market_minus_model": float(brier_score_loss(y, mkp) - brier_score_loss(y, mp)),
        "mean_model_over_prob": float(df["model_over_prob"].mean()),
        "mean_market_over_prob": float(df["market_over_prob"].mean()),
        "mean_best_ev": float(df["best_ev"].mean()),
        "pct_best_ev_pos": float((df["best_ev"] > 0).mean()),
        "pct_best_ev_gt_1pct": float((df["best_ev"] > 0.01).mean()),
        "pct_best_ev_gt_2pct": float((df["best_ev"] > 0.02).mean()),
    }

    pos = df.loc[df["best_ev"] > 0.0].copy()
    if not pos.empty:
        won = np.where(pos["best_side"].eq("over"), pos["actual_over"], pos["actual_under"])
        out["n_pos_ev_rows"] = int(len(pos))
        out["pos_ev_hit_rate"] = float(np.mean(won))
        out["pos_ev_mean_ev"] = float(pos["best_ev"].mean())
    else:
        out["n_pos_ev_rows"] = 0
        out["pos_ev_hit_rate"] = np.nan
        out["pos_ev_mean_ev"] = np.nan

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--system-path", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--primary-only", action="store_true", help="restrict evaluation to pitcher_strikeouts rows only")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    system = StrikeoutBettingSystem.load(args.system_path)

    games = pd.read_parquet(data_dir / "games.parquet")
    pa = pd.read_parquet(data_dir / "plate_appearances.parquet")
    train_df = FeatureBuilder.build_training_frame(pa, games)

    player_map = pd.read_parquet(data_dir / "external" / "player_id_map.parquet")
    pitcher_ext = pd.read_parquet(data_dir / "external" / "savant_pitcher_features.parquet").copy()
    batter_ext = pd.read_parquet(data_dir / "external" / "savant_batter_features.parquet").copy()

    # leakage-safe prior-season priors only
    pitcher_ext["season"] = pd.to_numeric(pitcher_ext["season"], errors="coerce") + 1
    batter_ext["season"] = pd.to_numeric(batter_ext["season"], errors="coerce") + 1

    train_df = merge_external_features(train_df, player_map, pitcher_ext, batter_ext)
    start_states = FeatureBuilder.build_start_state_frame(train_df)
    lineup_lookup = build_lineup_lookup(train_df)

    snap = load_all_market_snapshots(data_dir)
    actuals = FeatureBuilder.build_actual_game_totals(train_df)
    market = FeatureBuilder.build_market_training_frame(snap, actuals)

    if market.empty:
        raise SystemExit("Market training frame is empty.")

    # Cleanest first-stable comparison set:
    # - pregame only
    # - two-sided only
    # - no pushes
    # - half-point lines only
    market = market.copy()
    for c in ["snapshot_ts", "commence_time"]:
        if c in market.columns:
            market[c] = pd.to_datetime(market[c], utc=True, errors="coerce")

    if "snapshot_ts" in market.columns and "commence_time" in market.columns:
        market = market.loc[market["snapshot_ts"] < market["commence_time"]].copy()
    elif "is_live_snapshot" in market.columns:
        market = market.loc[market["is_live_snapshot"].eq(0)].copy()

    for c in ["over_odds", "under_odds", "line_value"]:
        market[c] = pd.to_numeric(market[c], errors="coerce")

    market = market.replace([np.inf, -np.inf], np.nan)
    market = market.dropna(subset=["game_id", "player_id", "line_value", "over_odds", "under_odds"])

    if "push" in market.columns:
        market = market.loc[market["push"].fillna(0).eq(0)].copy()

    market = market.loc[np.isclose(market["line_value"] % 1, 0.5)].copy()

    if args.primary_only:
        market = market.loc[market["prop_type"].eq("pitcher_strikeouts")].copy()

    # latest pregame snapshot per row
    dedupe_keys = [c for c in ["game_id", "player_id", "vendor", "prop_type", "line_value"] if c in market.columns]
    sort_col = "snapshot_ts" if "snapshot_ts" in market.columns else "updated_at"
    market = market.sort_values(sort_col).drop_duplicates(subset=dedupe_keys, keep="last").copy()

    # no-vig market probs
    probs = market.apply(lambda r: no_vig_probs(r["over_odds"], r["under_odds"]), axis=1)
    market["market_over_prob"] = [x[0] for x in probs]
    market["market_under_prob"] = [x[1] for x in probs]

    # actual labels
    if "over_hit" in market.columns and "under_hit" in market.columns:
        market["actual_over"] = market["over_hit"].astype(int)
        market["actual_under"] = market["under_hit"].astype(int)
    else:
        market["actual_over"] = (market["actual_total_k"] > market["line_value"]).astype(int)
        market["actual_under"] = 1 - market["actual_over"]

    # build PMF cache per pitcher-game
    state_cols = [c for c in start_states.columns if c not in {"actual_total_k", "actual_total_bf"}]
    state_lookup = {
        (int(r.game_id), int(r.pitcher_id)): {c: getattr(r, c) for c in state_cols}
        for r in start_states.itertuples(index=False)
    }

    pmf_cache = {}
    rows = []

    for r in market.itertuples(index=False):
        key = (int(r.game_id), int(r.player_id))
        state = state_lookup.get(key)
        lineup = lineup_lookup.get(key)

        if state is None or lineup is None or len(lineup) == 0:
            continue

        pmf = pmf_cache.get(key)
        if pmf is None:
            pmf = system.true_model.predict_pmf(state=state, future_lineup=lineup)
            pmf_cache[key] = pmf

        model_over = threshold_over_prob(pmf, r.line_value)
        model_under = 1.0 - model_over

        over_ev = ev_from_american(model_over, r.over_odds)
        under_ev = ev_from_american(model_under, r.under_odds)

        best_side = "over" if over_ev >= under_ev else "under"
        best_ev = max(over_ev, under_ev)

        rows.append({
            "game_id": int(r.game_id),
            "player_id": int(r.player_id),
            "vendor": getattr(r, "vendor", None),
            "prop_type": getattr(r, "prop_type", None),
            "line_value": float(r.line_value),
            "snapshot_ts": getattr(r, "snapshot_ts", pd.NaT),
            "commence_time": getattr(r, "commence_time", pd.NaT),
            "actual_total_k": getattr(r, "actual_total_k", np.nan),
            "actual_over": int(r.actual_over),
            "actual_under": int(r.actual_under),
            "market_over_prob": float(r.market_over_prob),
            "market_under_prob": float(r.market_under_prob),
            "model_over_prob": float(model_over),
            "model_under_prob": float(model_under),
            "over_odds": float(r.over_odds),
            "under_odds": float(r.under_odds),
            "over_edge": float(model_over - r.market_over_prob),
            "under_edge": float(model_under - r.market_under_prob),
            "over_ev": float(over_ev),
            "under_ev": float(under_ev),
            "best_side": best_side,
            "best_ev": float(best_ev),
            "fair_over_american": float(prob_to_american(model_over)),
            "fair_under_american": float(prob_to_american(model_under)),
            "model_log_loss_over": float(log_loss([int(r.actual_over)], [np.clip(model_over, 1e-6, 1 - 1e-6)], labels=[0, 1])),
            "market_log_loss_over": float(log_loss([int(r.actual_over)], [np.clip(r.market_over_prob, 1e-6, 1 - 1e-6)], labels=[0, 1])),
            "model_brier_over": float((int(r.actual_over) - model_over) ** 2),
            "market_brier_over": float((int(r.actual_over) - r.market_over_prob) ** 2),
        })

    result = pd.DataFrame(rows)
    if result.empty:
        raise SystemExit("No comparable market rows survived cleaning.")

    result.to_parquet(out_dir / "model_vs_market_rows.parquet", index=False)
    result.to_csv(out_dir / "model_vs_market_rows.csv", index=False)

    summaries = []
    summaries.append(compute_summary(result, "overall"))
    if "prop_type" in result.columns:
        for key, g in result.groupby("prop_type"):
            summaries.append(compute_summary(g, f"prop_type={key}"))
    if "vendor" in result.columns:
        for key, g in result.groupby("vendor"):
            if len(g) >= 10:
                summaries.append(compute_summary(g, f"vendor={key}"))
    for lv, g in result.groupby("line_value"):
        if len(g) >= 10:
            summaries.append(compute_summary(g, f"line={lv}"))

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(out_dir / "model_vs_market_summary.csv", index=False)

    overall = summary_df.loc[summary_df["segment"].eq("overall")].iloc[0].to_dict()
    with open(out_dir / "model_vs_market_summary.json", "w") as f:
        json.dump(overall, f, indent=2)

    print("\nOVERALL SUMMARY")
    print(pd.DataFrame([overall]).to_string(index=False))
    print(f"\nWrote {out_dir / 'model_vs_market_rows.parquet'}")
    print(f"Wrote {out_dir / 'model_vs_market_summary.csv'}")
    print(f"Wrote {out_dir / 'model_vs_market_summary.json'}")


if __name__ == "__main__":
    main()
