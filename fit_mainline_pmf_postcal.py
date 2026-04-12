from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

from mlb_k_model.data_pipeline import FeatureBuilder
from mlb_k_model.external_data import merge_external_features
from mlb_k_model.system import StrikeoutBettingSystem
from train_system import build_lineup_lookup


def clip_prob(p):
    return np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)


def threshold_over_prob(pmf: np.ndarray, line_value: float) -> float:
    line_value = float(line_value)
    thr = int(np.floor(line_value) + 1)
    thr = max(0, min(thr, len(pmf)))
    return float(np.asarray(pmf, dtype=float)[thr:].sum())


def apply_pmf_postcal(pmf: np.ndarray, tau: float, lam: float) -> np.ndarray:
    z = np.asarray(pmf, dtype=float).copy()
    z = np.clip(z, 1e-12, None)
    z = z / z.sum()

    ks = np.arange(len(z), dtype=float)
    logits = (np.log(z) / float(tau)) + float(lam) * ks
    logits -= logits.max()
    out = np.exp(logits)
    out = np.clip(out, 1e-12, None)
    return out / out.sum()


def metric_pack(y, p, market_p):
    y = np.asarray(y, dtype=int)
    p = clip_prob(p)
    market_p = clip_prob(market_p)

    model_ll = float(log_loss(y, p))
    market_ll = float(log_loss(y, market_p))
    model_br = float(brier_score_loss(y, p))
    market_br = float(brier_score_loss(y, market_p))

    return {
        "model_log_loss": model_ll,
        "market_log_loss": market_ll,
        "delta_log_loss_market_minus_model": float(market_ll - model_ll),
        "model_brier": model_br,
        "market_brier": market_br,
        "delta_brier_market_minus_model": float(market_br - model_br),
        "mean_model_over_prob": float(np.mean(p)),
        "mean_market_over_prob": float(np.mean(market_p)),
    }


def better(a: dict, b: dict) -> bool:
    ll_tol = 1e-4
    br_tol = 1e-4
    gap_tol = 1e-4

    if a["model_log_loss"] < b["model_log_loss"] - ll_tol:
        return True
    if abs(a["model_log_loss"] - b["model_log_loss"]) <= ll_tol:
        if a["model_brier"] < b["model_brier"] - br_tol:
            return True
        if abs(a["model_brier"] - b["model_brier"]) <= br_tol:
            if a["delta_log_loss_market_minus_model"] > b["delta_log_loss_market_minus_model"] + gap_tol:
                return True
    return False


def build_rows_with_pmfs(data_dir: Path, system: StrikeoutBettingSystem, rows_path: Path) -> pd.DataFrame:
    rows = pd.read_parquet(rows_path).copy()
    if rows.empty:
        raise SystemExit("Main-line evaluator rows are empty.")

    games = pd.read_parquet(data_dir / "games.parquet")
    pa = pd.read_parquet(data_dir / "plate_appearances.parquet")
    train_df = FeatureBuilder.build_training_frame(pa, games)

    player_map = pd.read_parquet(data_dir / "external" / "player_id_map.parquet")
    pitcher_ext = pd.read_parquet(data_dir / "external" / "savant_pitcher_features.parquet").copy()
    batter_ext = pd.read_parquet(data_dir / "external" / "savant_batter_features.parquet").copy()

    pitcher_ext["season"] = pd.to_numeric(pitcher_ext["season"], errors="coerce") + 1
    batter_ext["season"] = pd.to_numeric(batter_ext["season"], errors="coerce") + 1

    train_df = merge_external_features(train_df, player_map, pitcher_ext, batter_ext)
    start_states = FeatureBuilder.build_start_state_frame(train_df)
    lineup_lookup = build_lineup_lookup(train_df)

    state_cols = [c for c in start_states.columns if c not in {"actual_total_k", "actual_total_bf"}]
    state_lookup = {
        (int(r.game_id), int(r.pitcher_id)): {c: getattr(r, c) for c in state_cols}
        for r in start_states.itertuples(index=False)
    }

    pmf_cache = {}
    pmf_keys = []
    for r in rows.itertuples(index=False):
        key = (int(r.game_id), int(r.player_id))
        if key not in pmf_cache:
            state = state_lookup.get(key)
            lineup = lineup_lookup.get(key)
            if state is None or lineup is None or len(lineup) == 0:
                pmf_cache[key] = None
            else:
                pmf_cache[key] = system.true_model.predict_pmf(state=state, future_lineup=lineup)
        pmf_keys.append(key)

    rows["pmf_key"] = pmf_keys
    rows["has_pmf"] = rows["pmf_key"].map(lambda k: pmf_cache.get(k) is not None)
    rows = rows.loc[rows["has_pmf"]].copy()
    rows["snapshot_ts"] = pd.to_datetime(rows["snapshot_ts"], utc=True, errors="coerce")
    return rows, pmf_cache


def split_group_holdout(rows: pd.DataFrame):
    # STRICT grouped holdout by unique pitcher-game.
    # No group may appear in both calibration and evaluation.
    groups = (
        rows.groupby(["game_id", "player_id"], as_index=False)
        .agg(snapshot_ts=("snapshot_ts", "max"))
        .sort_values(["snapshot_ts", "game_id", "player_id"])
        .reset_index(drop=True)
    )
    n_groups = len(groups)
    if n_groups < 20:
        raise SystemExit(f"Need at least 20 unique main-line pitcher-game groups, found {n_groups}.")

    n_eval = max(10, int(round(n_groups * 0.30)))
    if n_eval >= n_groups:
        n_eval = max(1, n_groups - 1)

    cal_groups = groups.iloc[:-n_eval].copy()
    eval_groups = groups.iloc[-n_eval:].copy()

    cal_keys = set(zip(cal_groups["game_id"], cal_groups["player_id"]))
    eval_keys = set(zip(eval_groups["game_id"], eval_groups["player_id"]))

    overlap = cal_keys & eval_keys
    if overlap:
        raise SystemExit(f"Split leakage detected: {len(overlap)} overlapping pitcher-game groups")

    key_series = list(zip(rows["game_id"], rows["player_id"]))
    cal_mask = pd.Series([k in cal_keys for k in key_series], index=rows.index)
    eval_mask = pd.Series([k in eval_keys for k in key_series], index=rows.index)

    cal_df = rows.loc[cal_mask].copy()
    eval_df = rows.loc[eval_mask].copy()

    if cal_df.empty or eval_df.empty:
        raise SystemExit("Calibration/evaluation split is empty.")

    # Final guardrail
    cal_final = set(zip(cal_df["game_id"], cal_df["player_id"]))
    eval_final = set(zip(eval_df["game_id"], eval_df["player_id"]))
    overlap_final = cal_final & eval_final
    if overlap_final:
        raise SystemExit(f"Final split leakage detected: {len(overlap_final)} overlapping pitcher-game groups")

    return cal_df, eval_df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--rows-path", required=True)
    ap.add_argument("--system-path", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    rows_path = Path(args.rows_path)
    system_path = Path(args.system_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    system = StrikeoutBettingSystem.load(system_path)

    rows, pmf_cache = build_rows_with_pmfs(data_dir, system, rows_path)
    cal_df, eval_df = split_group_holdout(rows)

    y_cal = cal_df["actual_over"].astype(int).to_numpy()
    y_eval = eval_df["actual_over"].astype(int).to_numpy()
    market_cal = clip_prob(cal_df["market_over_prob"].astype(float).to_numpy())
    market_eval = clip_prob(eval_df["market_over_prob"].astype(float).to_numpy())

    def probs_for(df, tau, lam):
        out = []
        for r in df.itertuples(index=False):
            pmf = pmf_cache[r.pmf_key]
            adj = apply_pmf_postcal(pmf, tau=tau, lam=lam)
            out.append(threshold_over_prob(adj, r.line_value))
        return clip_prob(out)

    # baseline
    base_cal = probs_for(cal_df, 1.0, 0.0)
    base_eval = probs_for(eval_df, 1.0, 0.0)
    baseline_metrics = metric_pack(y_eval, base_eval, market_eval)

    tau_grid = np.round(np.arange(0.70, 1.401, 0.05), 3)
    lam_grid = np.round(np.arange(-0.30, 0.3001, 0.025), 3)

    rows_out = []
    best = None
    best_tau = None
    best_lam = None

    for tau in tau_grid:
        for lam in lam_grid:
            cal_p = probs_for(cal_df, tau=float(tau), lam=float(lam))
            eval_p = probs_for(eval_df, tau=float(tau), lam=float(lam))

            cal_ll = float(log_loss(y_cal, cal_p))
            cal_br = float(brier_score_loss(y_cal, cal_p))
            m = metric_pack(y_eval, eval_p, market_eval)
            m["tau"] = float(tau)
            m["lam"] = float(lam)
            m["cal_log_loss"] = cal_ll
            m["cal_brier"] = cal_br
            rows_out.append(m)

            if best is None:
                best = m
                best_tau = float(tau)
                best_lam = float(lam)
            else:
                # choose by calibration split first, then eval
                prev = (best["cal_log_loss"], best["cal_brier"], best["model_log_loss"], best["model_brier"])
                curr = (m["cal_log_loss"], m["cal_brier"], m["model_log_loss"], m["model_brier"])
                if curr < prev:
                    best = m
                    best_tau = float(tau)
                    best_lam = float(lam)

    sweep = pd.DataFrame(rows_out).sort_values(
        ["cal_log_loss", "cal_brier", "model_log_loss", "model_brier"]
    ).reset_index(drop=True)
    sweep.to_csv(out_dir / "pmf_postcal_sweep.csv", index=False)

    accepted = (
        best["model_log_loss"] < baseline_metrics["model_log_loss"] - 1e-4
        and best["model_brier"] < baseline_metrics["model_brier"] - 1e-4
        and best["delta_log_loss_market_minus_model"] > baseline_metrics["delta_log_loss_market_minus_model"] + 1e-4
    )

    decision = {
        "accepted": bool(accepted),
        "baseline": baseline_metrics,
        "chosen": {
            "tau": best_tau,
            "lam": best_lam,
            **{k: best[k] for k in [
                "cal_log_loss", "cal_brier",
                "model_log_loss", "market_log_loss", "delta_log_loss_market_minus_model",
                "model_brier", "market_brier", "delta_brier_market_minus_model",
                "mean_model_over_prob", "mean_market_over_prob",
            ]}
        },
        "n_rows_total": int(len(rows)),
        "n_rows_cal": int(len(cal_df)),
        "n_rows_eval": int(len(eval_df)),
    }

    if accepted:
        backup_path = system_path.with_suffix(system_path.suffix + ".bak_before_pmf_postcal")
        if not backup_path.exists():
            backup_path.write_bytes(system_path.read_bytes())

        system.true_model.mainline_pmf_postcal_tau_ = float(best_tau)
        system.true_model.mainline_pmf_postcal_lambda_ = float(best_lam)
        system.true_model.mainline_pmf_postcal_meta_ = decision
        system.save(system_path)

        decision["saved_model_path"] = str(system_path)
        decision["backup_path"] = str(backup_path)

    with open(out_dir / "pmf_postcal_decision.json", "w") as f:
        json.dump(decision, f, indent=2)

    print("\nBASELINE")
    print(json.dumps(baseline_metrics, indent=2))
    print("\nBEST")
    print(json.dumps(decision["chosen"], indent=2))
    print("\nDECISION")
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
