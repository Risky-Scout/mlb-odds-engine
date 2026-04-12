from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss


def clip_prob(x):
    return np.clip(np.asarray(x, dtype=float), 1e-6, 1 - 1e-6)


def normalize_side(x: str) -> str:
    s = str(x).strip().lower()
    if s in {"o", "over", "ov"}:
        return "over"
    if s in {"u", "under", "un"}:
        return "under"
    return s


def find_col(df: pd.DataFrame, candidates, required=True):
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise SystemExit(f"Missing required columns from {candidates}")
    return None


def prepare_rows(df: pd.DataFrame) -> pd.DataFrame:
    best_ev_col = find_col(df, ["best_ev", "ev_best", "selected_ev", "best_expected_value"])
    model_over_col = find_col(df, ["model_over_prob", "raw_model_over_prob"])
    market_over_col = find_col(df, ["market_over_prob"])
    actual_over_col = find_col(df, ["actual_over"])
    vendor_col = find_col(df, ["vendor", "book", "bookmaker"], required=False)
    line_col = find_col(df, ["line_value"], required=False)
    side_col = find_col(df, ["best_side", "selected_side", "best_bet_side", "bet_side", "edge_side"], required=False)

    out = df.copy()

    if side_col is None:
        over_ev_col = find_col(df, ["over_ev", "ev_over", "model_over_ev"], required=False)
        under_ev_col = find_col(df, ["under_ev", "ev_under", "model_under_ev"], required=False)
        if over_ev_col is not None and under_ev_col is not None:
            out["_side"] = np.where(
                pd.to_numeric(out[over_ev_col], errors="coerce").fillna(-np.inf)
                >= pd.to_numeric(out[under_ev_col], errors="coerce").fillna(-np.inf),
                "over",
                "under",
            )
        else:
            raise SystemExit("Could not determine selected side.")
    else:
        out["_side"] = out[side_col].map(normalize_side)

    out["_best_ev"] = pd.to_numeric(out[best_ev_col], errors="coerce")
    out["_model_over"] = pd.to_numeric(out[model_over_col], errors="coerce")
    out["_market_over"] = pd.to_numeric(out[market_over_col], errors="coerce")
    out["_actual_over"] = pd.to_numeric(out[actual_over_col], errors="coerce")
    out["_vendor"] = out[vendor_col].astype(str) if vendor_col is not None else "ALL"
    out["_line"] = pd.to_numeric(out[line_col], errors="coerce") if line_col is not None else np.nan
    out["snapshot_ts"] = pd.to_datetime(out["snapshot_ts"], utc=True, errors="coerce")

    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.dropna(subset=["game_id", "player_id", "snapshot_ts", "_best_ev", "_model_over", "_market_over", "_actual_over"]).copy()

    out["_model_sel_prob"] = np.where(out["_side"].eq("over"), out["_model_over"], 1.0 - out["_model_over"])
    out["_market_sel_prob"] = np.where(out["_side"].eq("over"), out["_market_over"], 1.0 - out["_market_over"])
    out["_actual_sel"] = np.where(out["_side"].eq("over"), out["_actual_over"], 1.0 - out["_actual_over"])
    out["_conf_dist"] = np.abs(out["_model_over"] - 0.5)

    out["_model_sel_prob"] = clip_prob(out["_model_sel_prob"])
    out["_market_sel_prob"] = clip_prob(out["_market_sel_prob"])
    return out


def gate_metrics(g: pd.DataFrame, total_rows: int) -> dict:
    y = g["_actual_sel"].astype(int).to_numpy()
    pm = clip_prob(g["_model_sel_prob"].to_numpy())
    pk = clip_prob(g["_market_sel_prob"].to_numpy())

    return {
        "n_selected": int(len(g)),
        "coverage": float(len(g) / max(total_rows, 1)),
        "selected_hit_rate": float(np.mean(y)),
        "selected_mean_best_ev": float(g["_best_ev"].mean()),
        "selected_model_log_loss": float(log_loss(y, pm)),
        "selected_market_log_loss": float(log_loss(y, pk)),
        "delta_log_loss_market_minus_model": float(log_loss(y, pk) - log_loss(y, pm)),
        "selected_model_brier": float(brier_score_loss(y, pm)),
        "selected_market_brier": float(brier_score_loss(y, pk)),
        "delta_brier_market_minus_model": float(brier_score_loss(y, pk) - brier_score_loss(y, pm)),
        "selected_mean_model_sel_prob": float(np.mean(pm)),
        "selected_mean_market_sel_prob": float(np.mean(pk)),
        "selected_mean_conf_dist": float(g["_conf_dist"].mean()),
    }


def split_holdout(df: pd.DataFrame):
    groups = (
        df.groupby(["game_id", "player_id"], as_index=False)
        .agg(snapshot_ts=("snapshot_ts", "max"))
        .sort_values(["snapshot_ts", "game_id", "player_id"])
        .reset_index(drop=True)
    )
    n = len(groups)
    if n < 20:
        raise SystemExit(f"Need at least 20 pitcher-game groups, found {n}")
    n_eval = max(10, int(round(n * 0.30)))
    train = groups.iloc[:-n_eval].copy()
    test = groups.iloc[-n_eval:].copy()

    train_keys = set(zip(train["game_id"], train["player_id"]))
    test_keys = set(zip(test["game_id"], test["player_id"]))

    key_series = list(zip(df["game_id"], df["player_id"]))
    train_mask = pd.Series([k in train_keys for k in key_series], index=df.index)
    test_mask = pd.Series([k in test_keys for k in key_series], index=df.index)

    return df.loc[train_mask].copy(), df.loc[test_mask].copy()


def candidate_grid(df: pd.DataFrame, min_selected: int):
    edge_grid = [0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20]
    conf_grid = [0.00, 0.02, 0.04, 0.06, 0.08, 0.10]
    vendor_counts = df["_vendor"].value_counts()
    vendor_opts = ["ALL"] + sorted(vendor_counts[vendor_counts >= min_selected].index.tolist())
    line_counts = df["_line"].value_counts(dropna=True)
    line_opts = ["ALL"] + sorted(line_counts[line_counts >= min_selected].index.tolist())

    for edge in edge_grid:
        for conf in conf_grid:
            for vendor in vendor_opts:
                for line in line_opts:
                    yield edge, conf, vendor, line


def apply_gate(df: pd.DataFrame, edge: float, conf: float, vendor: str, line):
    mask = (df["_best_ev"] >= edge) & (df["_conf_dist"] >= conf)
    if vendor != "ALL":
        mask &= df["_vendor"].eq(vendor)
    if line != "ALL":
        mask &= df["_line"].eq(float(line))
    return df.loc[mask].copy()


def better(a: dict, b: dict) -> bool:
    if a["delta_log_loss_market_minus_model"] > b["delta_log_loss_market_minus_model"] + 1e-4:
        return True
    if abs(a["delta_log_loss_market_minus_model"] - b["delta_log_loss_market_minus_model"]) <= 1e-4:
        if a["delta_brier_market_minus_model"] > b["delta_brier_market_minus_model"] + 1e-4:
            return True
        if abs(a["delta_brier_market_minus_model"] - b["delta_brier_market_minus_model"]) <= 1e-4:
            if a["selected_hit_rate"] > b["selected_hit_rate"] + 1e-6:
                return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows-path", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--min-selected-train", type=int, default=20)
    ap.add_argument("--min-selected-test", type=int, default=10)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = prepare_rows(pd.read_parquet(args.rows_path))
    train_df, test_df = split_holdout(df)

    train_rows = []
    best_gate = None
    best_train_metrics = None

    for edge, conf, vendor, line in candidate_grid(train_df, args.min_selected_train):
        g_train = apply_gate(train_df, edge, conf, vendor, line)
        if len(g_train) < args.min_selected_train:
            continue

        m = gate_metrics(g_train, len(train_df))
        m["min_edge"] = edge
        m["min_conf_dist"] = conf
        m["vendor_filter"] = vendor
        m["line_filter"] = line
        m["model_beats_market"] = bool(
            m["delta_log_loss_market_minus_model"] > 0 and m["delta_brier_market_minus_model"] > 0
        )
        train_rows.append(m)

        if not m["model_beats_market"]:
            continue

        if best_train_metrics is None or better(m, best_train_metrics):
            best_train_metrics = m
            best_gate = (edge, conf, vendor, line)

    if best_gate is None:
        decision = {
            "accepted": False,
            "reason": "No training gate beat market on both log loss and Brier.",
            "n_rows_total": int(len(df)),
            "n_rows_train": int(len(train_df)),
            "n_rows_test": int(len(test_df)),
        }
        with open(out_dir / "strict_ev_gate_holdout_decision.json", "w") as f:
            json.dump(decision, f, indent=2)
        pd.DataFrame(train_rows).to_csv(out_dir / "strict_ev_gate_holdout_train_sweep.csv", index=False)
        print(json.dumps(decision, indent=2))
        return

    edge, conf, vendor, line = best_gate
    g_test = apply_gate(test_df, edge, conf, vendor, line)
    if len(g_test) < args.min_selected_test:
        decision = {
            "accepted": False,
            "reason": "Chosen training gate did not leave enough test rows.",
            "chosen_gate": {
                "min_edge": edge,
                "min_conf_dist": conf,
                "vendor_filter": vendor,
                "line_filter": line,
            },
            "n_rows_total": int(len(df)),
            "n_rows_train": int(len(train_df)),
            "n_rows_test": int(len(test_df)),
            "n_selected_test": int(len(g_test)),
        }
        with open(out_dir / "strict_ev_gate_holdout_decision.json", "w") as f:
            json.dump(decision, f, indent=2)
        pd.DataFrame(train_rows).to_csv(out_dir / "strict_ev_gate_holdout_train_sweep.csv", index=False)
        print(json.dumps(decision, indent=2))
        return

    test_metrics = gate_metrics(g_test, len(test_df))
    test_metrics["model_beats_market"] = bool(
        test_metrics["delta_log_loss_market_minus_model"] > 0 and test_metrics["delta_brier_market_minus_model"] > 0
    )

    accepted = test_metrics["model_beats_market"]

    decision = {
        "accepted": bool(accepted),
        "n_rows_total": int(len(df)),
        "n_rows_train": int(len(train_df)),
        "n_rows_test": int(len(test_df)),
        "chosen_gate": {
            "min_edge": edge,
            "min_conf_dist": conf,
            "vendor_filter": vendor,
            "line_filter": line,
        },
        "train_metrics": best_train_metrics,
        "test_metrics": test_metrics,
    }

    with open(out_dir / "strict_ev_gate_holdout_decision.json", "w") as f:
        json.dump(decision, f, indent=2)

    pd.DataFrame(train_rows).to_csv(out_dir / "strict_ev_gate_holdout_train_sweep.csv", index=False)
    g_test.to_parquet(out_dir / "strict_ev_gate_holdout_test_selected_rows.parquet", index=False)

    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
